import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import TomoEmbedding
from .backbone import TomoEncoder


class GumbelTopK(nn.Module):
    """
    Differentiable Top-K operator using Gumbel-Softmax on logits.

    Training mode:
        - Adds Gumbel noise to logits, softmax with temperature tau.
        - Approximates Top-K via a smooth mask (differentiable).

    Evaluation mode:
        - Hard Top-K selection on logits, normalized to sum=1.
    """
    def __init__(self, k: int = 30, tau: float = 1.0, softness: float = 1e-2):
        """
        Args:
            k: number of top elements to select
            tau: temperature for Gumbel noise / softmax (smaller=tighter)
            softness: smoothness factor for soft mask in training
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.softness = softness

    def _safe_topk(self, x: torch.Tensor, k: int, dim=-1):
        k = max(0, min(k, x.size(dim)))
        if k == 0:
            # return empty tensors of appropriate shape
            shape_values = list(x.shape)
            shape_values[dim] = 0
            values = x.new_empty(tuple(shape_values))
            shape_idx = list(x.shape)
            shape_idx[dim] = 0
            indices = x.new_empty(tuple(shape_idx), dtype=torch.long)
            return values, indices
        return torch.topk(x, k, dim=dim)

    def sample_gumbel(self, shape, device, dtype=torch.float32, eps=1e-20):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (C, L, m) unnormalized logits

        Returns:
            Tensor of same shape, with only top-k along last dim kept
            (soft in training, hard in eval with sum=1)
        """
        if logits.dim() < 1:
            raise ValueError("logits must be at least 1-D")

        device = logits.device
        _, _, m = logits.shape
        k_eff = min(self.k, m)
        if k_eff <= 0:
            return torch.zeros_like(logits)

        if self.training:
            # ---------- TRAINING: soft Top-K ----------
            gumbel = self.sample_gumbel(logits.shape, device=device, dtype=logits.dtype)
            perturbed = (logits + gumbel) / max(self.tau, 1e-12)
            probs = F.softmax(perturbed, dim=-1)

            # Compute smooth top-k mask
            topk_vals, _ = self._safe_topk(probs, k_eff, dim=-1)
            threshold = topk_vals[..., -1].unsqueeze(-1)
            smooth_mask = torch.sigmoid((probs - threshold) / max(self.softness, 1e-12))
            return probs * smooth_mask

        else:
            # ---------- EVAL: hard Top-K with normalization ----------
            topk_vals, topk_idx = self._safe_topk(logits, k_eff, dim=-1)
            mask = torch.zeros_like(logits)
            if topk_idx.numel() > 0:
                mask.scatter_(-1, topk_idx, 1.0)

            # Keep only top-k logits
            topk_logits = logits * mask

            # Normalize top-k logits to sum=1 along last dim
            sum_topk = topk_logits.sum(dim=-1, keepdim=True)
            sum_topk = sum_topk + 1e-12  # avoid divide by zero
            probs = topk_logits / sum_topk

            return probs


class GeneRouter(nn.Module):
    """
    Gene router that maps gene tokens to assignment probabilities over 
    a set of latent factors, optionally applying differentiable Top-K.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_factors: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.1,
        topk: Optional[int] = None,   # if specified, apply Top-K selection
        tau: float = 1.0,   # temperature for GumbelTopK
        **kwargs
    ):
        """
        Args:
            embed_dim: Dimension of token embeddings (E)
            num_factors: Number of latent factors (m)
            hidden_dim: Hidden dimension of the MLP
            dropout: Dropout probability
            topk: Optional, number of top factors to select per gene
            tau: Temperature for Gumbel-Softmax
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_factors = num_factors
        self.hidden_dim = hidden_dim

        # MLP mapping gene embeddings -> logits over factors
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors)
        )

        # Optional gating via differentiable top-k
        self.topk = topk
        if topk is not None:
            self.gating = GumbelTopK(k=topk, tau=tau)
        else:
            self.gating = None
    
    def forward(
            self,
            x: torch.Tensor,
            gene_subset: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            temperature: float = 1.0
    ):
        """
        Args:
            x: FloatTensor of shape (C, L, E)
            gene_subset: BoolTensor (L,), True = selected token, False = unselected
            key_padding_mask: BoolTensor (C, L), True = valid token, False = masked
            temperature: Softmax temperature for logits

        Returns:
            probs: FloatTensor (C, S, m)
                Assignment probabilities over m factors for selected genes (S)
        """
        if x.dim() != 3:
            raise ValueError("tokens must be shape (C, L, E)")
        
        L = x.shape[1]
        if gene_subset is None:
            gene_subset = torch.ones(L, dtype=torch.bool, device=x.device)
            gene_subset[0] = False

        x = x[:, gene_subset, :]  # (C, S, E)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, gene_subset]

        logits = self.mlp(x) / temperature          # (C, S, m)

        if self.gating is not None:
            probs = self.gating(logits)             # (C, S, m)
        else:
            probs = F.softmax(logits, dim=-1)       # (C, S, m)

        # Zero out invalid <pad> tokens
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            mask = mask.unsqueeze(-1)
            probs = probs * mask.float()

        return probs, key_padding_mask


class SFM(nn.Module):
    """
    Structure Foundation Model (SFM) for cell-specific GRN inference
    """
    def __init__(
        self,
        token_dict,
        embed_dim: int = 512,
        expr_num_bins: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        num_factors: int = 256,
        topk: int = 32,
        **kwargs
    ):
        super().__init__()
        assert (embed_dim % 2) == 0, "embed_dim must be even"

        self.embedding = TomoEmbedding(
            token_dict, 
            D=embed_dim // 2,
            expr_num_bins=expr_num_bins,
            **kwargs
        )

        self.backbone = TomoEncoder(
            num_layers,
            embed_dim,
            num_heads,
            use_rotary=False,
            **kwargs
        )
    
        self.tfrouter = GeneRouter(
            embed_dim,
            num_factors,
            topk=topk
        )

        self.tgrouter = GeneRouter(
            embed_dim,
            num_factors,
            topk=topk
        )

        # apply initialization recursively to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_grn(self, grn, tf_pad, tg_pad):
        if tf_pad is not None:
            tf_valid = ~tf_pad[0]
        else:
            tf_valid = torch.ones(grn.shape[1], dtype=torch.bool, device=grn.device)
        
        if tg_pad is not None:
            tg_valid = ~tg_pad[0]
        else:
            tg_valid = torch.ones(grn.shape[2], dtype=torch.bool, device=grn.device)

        return grn[:, tf_valid, :][:, :, tg_valid]

    def forward(
            self,
            tokens,
            gene_subset: Optional[torch.Tensor] = None,
            **kwargs
    ):
        x, key_padding_mask = self.embedding(tokens)
        x = self.backbone(x, key_padding_mask, causal=False)

        u, tf_pad = self.tfrouter(x, gene_subset, key_padding_mask, **kwargs)
        v, tg_pad = self.tgrouter(x, None, key_padding_mask, **kwargs)

        grn = torch.einsum('cfm,cgm->cfg', u, v)
        return grn, tf_pad, tg_pad




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from ..tokenizer import TomeTokenizer

    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    tokenizer = TomeTokenizer(token_dict, simplify=True)
    tokens = tokenizer(adata[:4, :].copy())

    model = SFM(token_dict)
    grn, tf_pad, tg_pad = model(tokens)
    grn = model.get_grn(grn, tf_pad, tg_pad)

    print(grn.shape)