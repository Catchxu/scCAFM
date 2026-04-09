from __future__ import annotations

import pandas as pd
import torch

from typing import Optional

from .models.sfm import FactorState


def require_tensor(
    tokens: dict[str, torch.Tensor | None],
    key: str,
) -> torch.Tensor:
    value = tokens.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"`tokens[{key!r}]` must be a torch.Tensor.")
    return value


def build_active_gene_mask(
    input_ids: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.BoolTensor:
    if input_ids.ndim != 2:
        raise ValueError(f"`input_ids` must have shape (C, G), got {tuple(input_ids.shape)}.")

    if padding_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)

    if not torch.is_tensor(padding_mask):
        raise TypeError("`tokens['padding_mask']` must be a torch.Tensor or None.")
    if padding_mask.shape != input_ids.shape:
        raise ValueError(
            "`padding_mask` must match `input_ids` shape, "
            f"got {tuple(padding_mask.shape)} vs {tuple(input_ids.shape)}."
        )
    return ~padding_mask.to(torch.bool)


def build_active_value_mask(
    values: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.BoolTensor:
    if values.ndim != 2:
        raise ValueError(f"`values` must have shape (C, G), got {tuple(values.shape)}.")
    return build_active_gene_mask(
        input_ids=torch.zeros_like(values, dtype=torch.long),
        padding_mask=padding_mask,
    )


def build_tf_mask(
    input_ids: torch.Tensor,
    non_tf_mask: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.BoolTensor:
    if non_tf_mask.shape != input_ids.shape:
        raise ValueError(
            f"`non_tf_mask` must match `input_ids` shape, got {tuple(non_tf_mask.shape)}."
        )
    active_gene_mask = build_active_gene_mask(
        input_ids=input_ids,
        padding_mask=padding_mask,
    )
    return active_gene_mask & ~non_tf_mask.to(torch.bool)


def validate_factor_shapes(
    factors: FactorState,
    input_shape_prefix: torch.Size | tuple[int, ...],
) -> None:
    tensors = {
        "u": factors.u,
        "v": factors.v,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 3:
            raise ValueError(
                f"`factors.{name}` must have shape (C, G, M), got {tuple(tensor.shape)}."
            )
        if tensor.shape[:2] != tuple(input_shape_prefix):
            raise ValueError(
                f"`factors.{name}` has incompatible leading dims {tuple(tensor.shape[:2])}; "
                f"expected {tuple(input_shape_prefix)}."
            )


def build_token_lookup_maps(
    token_dict: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], int]:
    required_columns = {"token_index", "gene_symbol", "gene_id"}
    missing = required_columns.difference(token_dict.columns)
    if missing:
        raise ValueError(
            f"`token_dict` is missing required columns: {sorted(missing)}."
        )

    symbol_to_index: dict[str, int] = {}
    ensembl_to_index: dict[str, int] = {}
    pad_rows = token_dict[token_dict["gene_id"] == "<pad>"]
    if len(pad_rows) == 0:
        raise ValueError("`token_dict` must contain a '<pad>' entry in `gene_id`.")
    pad_index = int(pad_rows["token_index"].iloc[0])

    for _, row in token_dict.iterrows():
        token_index = int(row["token_index"])

        gene_symbol = row["gene_symbol"]
        if pd.notna(gene_symbol) and str(gene_symbol).strip():
            symbol_to_index[str(gene_symbol).strip().upper()] = token_index

        gene_id = row["gene_id"]
        if pd.notna(gene_id) and str(gene_id).strip():
            gene_id_norm = str(gene_id).strip().upper()
            if gene_id_norm.startswith("ENSG"):
                gene_id_norm = gene_id_norm.split(".", 1)[0]
            ensembl_to_index[gene_id_norm] = token_index

    return symbol_to_index, ensembl_to_index, pad_index
