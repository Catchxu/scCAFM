from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class QBGating(nn.Module):
    """
    Quantile-balanced sparse factor gating.

    Each token selects `k` factors via hard top-k on debiased logits
    `logits - beta_qb`, while the output probabilities are computed from the
    selected raw logits only. The factor bias `beta_qb` is updated after each
    training batch and used for the next batch.
    """

    def __init__(
        self,
        num_factors: int,
        k: int,
        beta_momentum: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if num_factors <= 0:
            raise ValueError(f"`num_factors` must be positive, got {num_factors}.")
        if k < 0:
            raise ValueError(f"`k` must be non-negative, got {k}.")
        if not 0.0 < beta_momentum <= 1.0:
            raise ValueError(
                f"`beta_momentum` must be in (0, 1], got {beta_momentum}."
            )

        self.num_factors = num_factors
        self.k = k
        self.beta_momentum = float(beta_momentum)
        self.eps = float(eps)
        self.register_buffer(
            "beta_qb",
            torch.zeros(num_factors, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _kth_largest(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
        if x.shape[dim] == 0:
            raise ValueError("Cannot compute kth largest value on an empty dimension.")

        k_eff = min(max(k, 1), x.shape[dim])
        return torch.kthvalue(-x, k_eff, dim=dim).values.neg()

    def _compute_beta_qb(
        self,
        flat_logits: torch.Tensor,
        beta_old: torch.Tensor,
        k_eff: int,
    ) -> torch.Tensor:
        if flat_logits.ndim != 2:
            raise ValueError(
                f"`flat_logits` must have shape (N, M), got {tuple(flat_logits.shape)}."
            )

        num_tokens, num_factors = flat_logits.shape
        if num_tokens == 0 or k_eff <= 0 or k_eff >= num_factors:
            return beta_old

        scores_minus_beta = flat_logits - beta_old.unsqueeze(0)
        alpha = self._kth_largest(scores_minus_beta, k_eff + 1, dim=-1)

        scores_minus_alpha = flat_logits - alpha.unsqueeze(-1)
        target_load = max(1, int(round(num_tokens * k_eff / num_factors)))
        beta_new = self._kth_largest(
            scores_minus_alpha.transpose(0, 1),
            target_load + 1,
            dim=-1,
        )
        return beta_new.to(dtype=beta_old.dtype)

    def forward(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError(
                f"`logits` must have shape (..., M), got {tuple(logits.shape)}."
            )
        if logits.shape[-1] != self.num_factors:
            raise ValueError(
                f"Expected last dim {self.num_factors}, got {logits.shape[-1]}."
            )

        temperature = max(float(temperature), self.eps)
        original_shape = logits.shape
        flat_logits = logits.reshape(-1, self.num_factors)

        if flat_logits.shape[0] == 0:
            return torch.zeros_like(logits)

        k_eff = min(self.k, self.num_factors)
        if k_eff == 0:
            return torch.zeros_like(logits)
        if k_eff >= self.num_factors:
            return F.softmax(logits / temperature, dim=-1)

        beta_used = self.beta_qb.to(device=logits.device, dtype=logits.dtype)
        scores_adj = flat_logits - beta_used.unsqueeze(0)
        topk_idx = torch.topk(scores_adj, k=k_eff, dim=-1).indices

        selected_logits = flat_logits.gather(dim=-1, index=topk_idx)
        selected_probs = F.softmax(selected_logits / temperature, dim=-1)

        probs = torch.zeros_like(flat_logits)
        probs.scatter_(dim=-1, index=topk_idx, src=selected_probs)
        probs = probs.reshape(original_shape)

        if self.training:
            with torch.no_grad():
                beta_new = self._compute_beta_qb(
                    flat_logits=flat_logits.detach().to(torch.float32),
                    beta_old=self.beta_qb.detach(),
                    k_eff=k_eff,
                )
                self.beta_qb.lerp_(beta_new, weight=self.beta_momentum)

        return probs


class GeneRouter(nn.Module):
    """
    Router that assigns gene tokens to sparse latent factors.

    Shape convention:
    - `G`: gene-token length
    - `L`: full sequence length, where `L = G + 1`

    The output keeps the full gene-token sequence length `(C, G, M)`. Tokens
    excluded by `non_tf_mask` or `padding_mask` receive zero probabilities and
    zero scores.
    """

    def __init__(
        self,
        embed_dim: int,
        num_factors: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        topk: Optional[int] = None,
        temperature: float = 1.0,
        score_clip: float = 10.0,
        beta_momentum: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if num_factors <= 0:
            raise ValueError(f"`num_factors` must be positive, got {num_factors}.")
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")

        self.embed_dim = embed_dim
        self.num_factors = num_factors
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.temperature = float(temperature)
        self.score_clip = float(score_clip)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * num_factors),
        )

        self.gating = (
            QBGating(
                num_factors=num_factors,
                k=topk,
                beta_momentum=beta_momentum,
            )
            if topk is not None
            else None
        )

    @staticmethod
    def _normalize_gene_mask(
        batch_size: int,
        seq_len: int,
        mask: Optional[torch.Tensor],
        name: str,
        device: torch.device,
    ) -> torch.BoolTensor:
        if mask is None:
            return torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        if mask.shape == (batch_size, seq_len):
            return mask

        raise ValueError(
            f"`{name}` must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}."
        )

    @classmethod
    def _resolve_active_mask(
        cls,
        batch_size: int,
        seq_len: int,
        non_tf_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.BoolTensor:
        active_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        if non_tf_mask is not None:
            active_mask = active_mask & cls._normalize_gene_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                mask=non_tf_mask,
                name="non_tf_mask",
                device=device,
            )

        if padding_mask is not None:
            active_mask = active_mask & ~cls._normalize_gene_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                mask=padding_mask,
                name="padding_mask",
                device=device,
            )

        return active_mask

    def forward(
        self,
        x: torch.Tensor,
        non_tf_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if x.ndim != 3:
            raise ValueError(f"`x` must have shape (C, L, D), got {tuple(x.shape)}.")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {x.shape[-1]}."
            )

        batch_size, seq_len_with_cls, _ = x.shape
        gene_tokens = x[:, 1:, :]

        active_mask = self._resolve_active_mask(
            batch_size=batch_size,
            seq_len=seq_len_with_cls - 1,
            non_tf_mask=non_tf_mask,
            padding_mask=padding_mask,
            device=x.device,
        )

        logits = self.mlp(gene_tokens)
        prob_logits, scores = torch.split(logits, self.num_factors, dim=-1)
        scores = scores.clamp(min=-self.score_clip, max=self.score_clip)
        scores = scores.masked_fill(~active_mask.unsqueeze(-1), 0.0)

        probs = torch.zeros_like(prob_logits)
        active_logits = prob_logits[active_mask]
        if active_logits.numel() == 0:
            return probs, scores, active_mask

        if self.gating is not None:
            probs[active_mask] = self.gating(
                active_logits,
                temperature=self.temperature,
            )
        else:
            probs[active_mask] = F.softmax(
                active_logits / max(self.temperature, 1e-12),
                dim=-1,
            )

        return probs, scores, active_mask
