from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from .attention import AttentionKVCache
from .backbone import TransformerBackbone
from .embedding import ScEmbedding
from .gene_ordering import GeneOrderState
from .initialization import init_module_xavier


@dataclass(slots=True)
class EFMOutput:
    hidden_states: torch.Tensor
    cell_embedding: torch.Tensor
    gene_embeddings: torch.Tensor
    id_logits: torch.Tensor
    expression_pred: torch.Tensor


def _gather_gene_dim(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError(f"Gene-aligned tensor must have at least 2 dims, got {tuple(value.shape)}.")
    if value.shape[:2] != positions.shape:
        raise ValueError(
            f"Gene-aligned tensor leading shape {tuple(value.shape[:2])} "
            f"does not match positions {tuple(positions.shape)}."
        )
    gather_index = positions
    for _ in range(value.ndim - 2):
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand_as(value)
    return value.gather(dim=1, index=gather_index)


def reorder_gene_aligned_tokens(
    tokens: dict[str, torch.Tensor | None],
    positions: torch.Tensor | GeneOrderState,
) -> dict[str, torch.Tensor | None]:
    """
    Reorder token entries aligned to the gene-token dimension.

    `condition_ids` and metadata entries are copied unchanged.
    """

    if isinstance(positions, GeneOrderState):
        positions = positions.positions
    if not torch.is_tensor(positions):
        raise TypeError("`positions` must be a tensor or GeneOrderState.")
    if positions.ndim != 2:
        raise ValueError(f"`positions` must have shape (C, G), got {tuple(positions.shape)}.")

    reordered: dict[str, torch.Tensor | None] = dict(tokens)
    for key in ("input_ids", "expression_values", "non_tf_mask", "padding_mask"):
        value = tokens.get(key)
        if torch.is_tensor(value):
            reordered[key] = _gather_gene_dim(value, positions.to(device=value.device, dtype=torch.long))
    return reordered


def build_efm_targets(
    tokens: dict[str, torch.Tensor | None],
    *,
    eos_token_id: int,
) -> tuple[torch.LongTensor, torch.Tensor, torch.BoolTensor]:
    """
    Build next-token EFM targets over the full sequence length `(C, G + 1)`.

    Slot 0 corresponds to the prefix hidden state and predicts the first active
    gene. Slot `n_active` predicts EoS with expression 0.
    """

    input_ids = tokens.get("input_ids")
    expression_values = tokens.get("expression_values")
    if not torch.is_tensor(input_ids):
        raise TypeError("`tokens['input_ids']` must be a torch.Tensor.")
    if not torch.is_tensor(expression_values):
        raise TypeError("`tokens['expression_values']` must be a torch.Tensor.")
    if input_ids.ndim != 2:
        raise ValueError(f"`input_ids` must have shape (C, G), got {tuple(input_ids.shape)}.")
    if expression_values.shape != input_ids.shape:
        raise ValueError(
            "`expression_values` must match `input_ids` shape, "
            f"got {tuple(expression_values.shape)} vs {tuple(input_ids.shape)}."
        )

    padding_mask = tokens.get("padding_mask")
    if padding_mask is None:
        active_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if not torch.is_tensor(padding_mask):
            raise TypeError("`tokens['padding_mask']` must be a torch.Tensor or None.")
        if padding_mask.shape != input_ids.shape:
            raise ValueError(
                f"`padding_mask` must match input_ids shape, got {tuple(padding_mask.shape)}."
            )
        active_mask = ~padding_mask.to(dtype=torch.bool, device=input_ids.device)

    batch_size, gene_len = input_ids.shape
    seq_len = gene_len + 1
    target_ids = torch.full(
        (batch_size, seq_len),
        fill_value=int(eos_token_id),
        device=input_ids.device,
        dtype=torch.long,
    )
    target_expr = torch.zeros(
        (batch_size, seq_len),
        device=expression_values.device,
        dtype=expression_values.dtype,
    )
    valid_mask = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)

    target_ids[:, :gene_len] = input_ids.to(dtype=torch.long)
    target_expr[:, :gene_len] = expression_values
    valid_mask[:, :gene_len] = active_mask

    active_lengths = active_mask.sum(dim=1).to(torch.long)
    rows = torch.arange(batch_size, device=input_ids.device)
    target_ids[rows, active_lengths] = int(eos_token_id)
    target_expr[rows, active_lengths] = 0.0
    valid_mask[rows, active_lengths] = True
    return target_ids, target_expr, valid_mask


class _ProjectionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EFM(nn.Module):
    """
    Expression foundation model trained autoregressively over SFM-ordered genes.
    """

    def __init__(
        self,
        token_dict: pd.DataFrame,
        cond_vocab_size: int = 4096,
        embed_dim: int = 768,
        expr_num_bins: int = 32,
        expr_hidden_dim: int = 128,
        expr_tau: float = 1.0,
        batch_num_bins: int = 128,
        batch_hidden_dim: int = 128,
        batch_tau: float = 1.0,
        cond_dropout: float = 0.1,
        out_dropout: float = 0.1,
        gene_embedding_ckpt: Optional[str] = None,
        freeze_loaded_gene_embeddings: bool = False,
        num_layers: int = 12,
        num_heads: int = 12,
        backbone_mlp_hidden_dim: Optional[int] = None,
        backbone_mlp_dropout: float = 0.02,
        attention_backend: str = "fa4",
        use_rotary: bool = True,
        qkv_bias: bool = True,
        out_bias: bool = True,
        backbone_mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        self.token_dict = token_dict
        self.embed_dim = int(embed_dim)
        self.vocab_size = int(token_dict["token_index"].max()) + 1

        self.embedding = ScEmbedding(
            token_dict=token_dict,
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            expr_num_bins=expr_num_bins,
            expr_hidden_dim=expr_hidden_dim,
            expr_tau=expr_tau,
            batch_num_bins=batch_num_bins,
            batch_hidden_dim=batch_hidden_dim,
            batch_tau=batch_tau,
            cond_dropout=cond_dropout,
            out_dropout=out_dropout,
            gene_embedding_ckpt=gene_embedding_ckpt,
            freeze_loaded_gene_embeddings=freeze_loaded_gene_embeddings,
        )
        self.backbone = TransformerBackbone(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=backbone_mlp_hidden_dim,
            mlp_dropout=backbone_mlp_dropout,
            attention_backend=attention_backend,
            use_rotary=use_rotary,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            mlp_bias=backbone_mlp_bias,
            norm_eps=norm_eps,
            rotary_base=rotary_base,
            rotary_interleaved=rotary_interleaved,
        )
        hidden_dim = int(head_hidden_dim or embed_dim)
        self.ph_prob = _ProjectionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=self.vocab_size,
            dropout=head_dropout,
        )
        self.ph_exp = _ProjectionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=head_dropout,
        )
        self.apply(init_module_xavier)

    def forward(self, tokens: dict[str, torch.Tensor | None]) -> EFMOutput:
        input_ids = tokens.get("input_ids")
        expression_values = tokens.get("expression_values")
        condition_ids = tokens.get("condition_ids")
        non_tf_mask = tokens.get("non_tf_mask")
        padding_mask = tokens.get("padding_mask")
        if not torch.is_tensor(input_ids):
            raise TypeError("`tokens['input_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(expression_values):
            raise TypeError("`tokens['expression_values']` must be a torch.Tensor.")
        if not torch.is_tensor(condition_ids):
            raise TypeError("`tokens['condition_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(non_tf_mask):
            raise TypeError("`tokens['non_tf_mask']` must be a torch.Tensor.")

        embedding_output = self.embedding(
            input_ids=input_ids,
            expression_values=expression_values,
            condition_ids=condition_ids,
            padding_mask=padding_mask if torch.is_tensor(padding_mask) else None,
            non_tf_mask=non_tf_mask,
        )
        hidden_states = self.backbone(
            embedding_output.embeddings,
            key_padding_mask=embedding_output.key_padding_mask,
            causal=True,
        )
        id_logits = self.ph_prob(hidden_states)
        expression_pred = self.ph_exp(hidden_states).squeeze(-1)
        return EFMOutput(
            hidden_states=hidden_states,
            cell_embedding=hidden_states[:, 0, :],
            gene_embeddings=hidden_states[:, 1:, :],
            id_logits=id_logits,
            expression_pred=expression_pred,
        )

    def _validate_incremental_tokens(
        self,
        tokens: dict[str, torch.Tensor | None],
        expression_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids = tokens.get("input_ids")
        condition_ids = tokens.get("condition_ids")
        non_tf_mask = tokens.get("non_tf_mask")
        padding_mask = tokens.get("padding_mask")
        if not torch.is_tensor(input_ids):
            raise TypeError("`tokens['input_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(condition_ids):
            raise TypeError("`tokens['condition_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(non_tf_mask):
            raise TypeError("`tokens['non_tf_mask']` must be a torch.Tensor.")
        if not torch.is_tensor(expression_values):
            raise TypeError("`expression_values` must be a torch.Tensor.")
        self.embedding._validate_inputs(
            input_ids=input_ids,
            expression_values=expression_values,
            condition_ids=condition_ids,
            padding_mask=padding_mask if torch.is_tensor(padding_mask) else None,
            non_tf_mask=non_tf_mask,
        )
        cond_vocab_size = self.embedding.condition_embedding.cond_embedding.num_embeddings
        if condition_ids.max().item() >= cond_vocab_size:
            raise ValueError(
                "Found `condition_ids` outside the configured `cond_vocab_size`. "
                f"Max value: {condition_ids.max().item()}, "
                f"vocab size: {cond_vocab_size}."
            )
        return input_ids, expression_values, condition_ids, non_tf_mask, padding_mask

    def _embed_prefix_token(
        self,
        tokens: dict[str, torch.Tensor | None],
        expression_values: torch.Tensor,
    ) -> torch.Tensor:
        input_ids, expression_values, condition_ids, non_tf_mask, padding_mask = (
            self._validate_incremental_tokens(tokens, expression_values)
        )
        if padding_mask is not None and padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(torch.bool)
        if non_tf_mask.dtype != torch.bool:
            non_tf_mask = non_tf_mask.to(torch.bool)

        prefix_token = self.embedding.condition_embedding(condition_ids)
        batch_emb = self.embedding.batch_embedding(
            expression_values,
            padding_mask=padding_mask if torch.is_tensor(padding_mask) else None,
        ).unsqueeze(1)
        cell_token = prefix_token.to(batch_emb.dtype) + batch_emb
        cell_token = cell_token.to(self.embedding.prefix_type_embedding.dtype)
        cell_token = cell_token + self.embedding.prefix_type_embedding.to(cell_token.dtype)
        cell_token = self.embedding.final_norm(cell_token)
        return self.embedding.dropout(cell_token)

    def _embed_gene_token(
        self,
        tokens: dict[str, torch.Tensor | None],
        expression_values: torch.Tensor,
        *,
        gene_position: int,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor | None]:
        input_ids, expression_values, _, non_tf_mask, padding_mask = (
            self._validate_incremental_tokens(tokens, expression_values)
        )
        gene_position = int(gene_position)
        if not 0 <= gene_position < input_ids.shape[1]:
            raise ValueError(f"`gene_position` out of range: {gene_position}")
        if non_tf_mask.dtype != torch.bool:
            non_tf_mask = non_tf_mask.to(torch.bool)

        current_input_ids = input_ids[:, gene_position : gene_position + 1]
        current_expression = expression_values[:, gene_position : gene_position + 1]
        current_non_tf_mask = non_tf_mask[:, gene_position : gene_position + 1]
        current_padding_mask = None
        if torch.is_tensor(padding_mask):
            current_padding_mask = padding_mask[:, gene_position : gene_position + 1].to(
                torch.bool
            )
        if key_padding_mask is not None:
            if key_padding_mask.shape != current_input_ids.shape:
                raise ValueError(
                    "`key_padding_mask` must match the single-token gene shape, "
                    f"got {tuple(key_padding_mask.shape)} vs {tuple(current_input_ids.shape)}."
                )
            key_padding_mask = key_padding_mask.to(device=input_ids.device, dtype=torch.bool)
            current_padding_mask = (
                key_padding_mask
                if current_padding_mask is None
                else current_padding_mask | key_padding_mask
            )

        gene_emb = self.embedding.gene_embedding(current_input_ids)
        expr_emb = self.embedding.expr_embedding(current_expression).to(gene_emb.dtype)
        tf_type_emb = self.embedding.tf_type_embedding(current_non_tf_mask.long()).to(
            gene_emb.dtype
        )
        token_emb = gene_emb + expr_emb + tf_type_emb
        token_emb = token_emb + self.embedding.gene_type_embedding.to(token_emb.dtype)
        token_emb = self.embedding.final_norm(token_emb)
        token_emb = self.embedding.dropout(token_emb)
        if current_padding_mask is not None:
            token_emb = token_emb.masked_fill(current_padding_mask.unsqueeze(-1), 0.0)
        return token_emb, current_padding_mask

    def prefill_incremental_cache(
        self,
        tokens: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[AttentionKVCache]]:
        expression_values = tokens.get("expression_values")
        if not torch.is_tensor(expression_values):
            raise TypeError("`tokens['expression_values']` must be a torch.Tensor.")
        prefix_embedding = self._embed_prefix_token(tokens, expression_values)
        return self.backbone.forward_incremental(
            prefix_embedding,
            caches=None,
            key_padding_mask=None,
        )

    def append_incremental_gene(
        self,
        tokens: dict[str, torch.Tensor | None],
        *,
        expression_values: torch.Tensor,
        gene_position: int,
        caches: list[AttentionKVCache],
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[AttentionKVCache]]:
        gene_embedding, current_padding_mask = self._embed_gene_token(
            tokens,
            expression_values,
            gene_position=gene_position,
            key_padding_mask=key_padding_mask,
        )
        return self.backbone.forward_incremental(
            gene_embedding,
            caches=caches,
            key_padding_mask=current_padding_mask,
        )

    def predict_expression_from_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim != 3 or hidden_state.shape[1] != 1:
            raise ValueError(
                "`hidden_state` must have shape (batch, 1, dim), "
                f"got {tuple(hidden_state.shape)}."
            )
        return self.ph_exp(hidden_state).squeeze(-1).squeeze(1)
