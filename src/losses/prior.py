from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ..models.sfm import FactorState
from ..utils import (
    build_active_gene_mask,
    build_tf_mask,
    build_token_lookup_maps,
    require_tensor,
    validate_factor_shapes,
)


class PriorLoss(nn.Module):
    """
    Supervise top predicted TF->target edges against a prior GRN using raw `u` and `v`.

    Notes:
    - This loss constrains `u` and `v` only.
    - Source positions are restricted to active TF genes; target positions are
      restricted to active genes.
    - Only top predicted edges per cell contribute to the loss, so unpredicted
      prior edges do not incur a penalty.
    """

    def __init__(
        self,
        token_dict: pd.DataFrame,
        true_grn_df: Optional[pd.DataFrame] = None,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        pred_topk: Optional[int] = 256,
    ) -> None:
        super().__init__()

        self.symbol_to_index, self.ensembl_to_index, self.pad_index = (
            build_token_lookup_maps(token_dict=token_dict)
        )
        token_max_1 = max(int(token_dict["token_index"].max()) + 1, 1)
        self._pair_key_base = int(token_max_1)

        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.pred_topk = None if pred_topk is None else int(pred_topk)
        if self.pred_topk is not None and self.pred_topk <= 0:
            raise ValueError(f"`pred_topk` must be positive when provided, got {pred_topk}.")

        self._cached_prior_ref: Optional[pd.DataFrame] = None
        self.register_buffer(
            "_cached_pair_keys",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        if true_grn_df is not None:
            self._prepare_prior_cache(true_grn_df)

    def _map_gene_to_token(self, name: str) -> Optional[int]:
        key = str(name).strip().upper()
        if key.startswith("ENSG"):
            key = key.split(".", 1)[0]
            return self.ensembl_to_index.get(key)
        return self.symbol_to_index.get(key, self.ensembl_to_index.get(key))

    def _prepare_prior_cache(self, true_grn_df: pd.DataFrame) -> None:
        if not isinstance(true_grn_df, pd.DataFrame):
            raise TypeError("`true_grn_df` must be a pandas DataFrame.")
        if not {"Gene1", "Gene2"}.issubset(true_grn_df.columns):
            raise ValueError("`true_grn_df` must contain columns: 'Gene1' and 'Gene2'.")

        pair_keys: list[int] = []
        for src_name, tgt_name in zip(
            true_grn_df["Gene1"].tolist(),
            true_grn_df["Gene2"].tolist(),
        ):
            src_id = self._map_gene_to_token(src_name)
            tgt_id = self._map_gene_to_token(tgt_name)
            if src_id is None or tgt_id is None:
                continue
            if src_id == self.pad_index or tgt_id == self.pad_index:
                continue
            pair_keys.append(src_id * self._pair_key_base + tgt_id)

        if len(pair_keys) == 0:
            cached_pair_keys = torch.empty(0, dtype=torch.long)
        else:
            cached_pair_keys = torch.unique(torch.tensor(pair_keys, dtype=torch.long))

        self._cached_pair_keys = cached_pair_keys
        self._cached_prior_ref = true_grn_df

    def _resolve_prior_edges(
        self,
        true_grn_df: Optional[pd.DataFrame],
        device: torch.device,
    ) -> torch.LongTensor:
        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)
        return self._cached_pair_keys.to(device=device)

    @staticmethod
    def _bound_probability(
        edge_prob: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if not 0.0 < eps < 0.5:
            raise ValueError(f"`eps` must be in (0, 0.5), got {eps}.")

        dtype_eps = torch.finfo(edge_prob.dtype).eps
        eps = max(float(eps), float(dtype_eps))
        if not 0.0 < eps < 0.5:
            raise ValueError(f"Resolved `eps` must be in (0, 0.5), got {eps}.")

        edge_prob = torch.nan_to_num(edge_prob, nan=0.0, posinf=1.0, neginf=0.0)
        edge_prob = edge_prob.clamp(min=0.0, max=1.0)

        # Keep probabilities in the open interval (eps, 1 - eps) without
        # flattening gradients at exact 0/1 values. This matters here because
        # sparse routers can produce exact zero overlap for many candidate edges.
        return edge_prob * (1.0 - 2.0 * eps) + eps

    @staticmethod
    def _select_topk_edges(
        edge_prob: torch.Tensor,
        topk: Optional[int],
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        num_src, num_tgt = edge_prob.shape
        total_count = int(num_src * num_tgt)
        if total_count == 0:
            empty_idx = torch.empty(0, device=edge_prob.device, dtype=torch.long)
            empty_prob = torch.empty(0, device=edge_prob.device, dtype=edge_prob.dtype)
            return empty_idx, empty_idx, empty_prob

        flat_prob = edge_prob.reshape(-1)
        if topk is None:
            selected_flat = torch.arange(total_count, device=edge_prob.device, dtype=torch.long)
        else:
            k = min(int(topk), total_count)
            selected_flat = torch.topk(flat_prob.detach(), k=k, dim=0).indices
        selected_prob = flat_prob[selected_flat]
        return selected_flat // num_tgt, selected_flat % num_tgt, selected_prob

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        factors: FactorState,
        true_grn_df: Optional[pd.DataFrame] = None,
    ) -> torch.Tensor:
        if factors is None:
            raise ValueError("`factors` must be provided.")

        input_ids = require_tensor(tokens, "input_ids").to(torch.long)
        non_tf_mask = require_tensor(tokens, "non_tf_mask")
        padding_mask = tokens.get("padding_mask")
        validate_factor_shapes(factors=factors, input_shape_prefix=input_ids.shape)

        active_gene_mask = build_active_gene_mask(
            input_ids=input_ids,
            padding_mask=padding_mask,
        )
        tf_mask = build_tf_mask(
            input_ids=input_ids,
            non_tf_mask=non_tf_mask,
            padding_mask=padding_mask,
        )
        prior_pair_keys = self._resolve_prior_edges(
            true_grn_df=true_grn_df,
            device=input_ids.device,
        )

        if prior_pair_keys.numel() == 0:
            return factors.u.new_zeros(())

        if not tf_mask.any() or not active_gene_mask.any():
            return factors.u.new_zeros(())

        batch_indices: list[torch.Tensor] = []
        src_indices: list[torch.Tensor] = []
        tgt_indices: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []

        batch_size, _ = input_ids.shape
        for batch_idx in range(batch_size):
            tf_pos = torch.nonzero(tf_mask[batch_idx], as_tuple=False).flatten()
            tgt_pos = torch.nonzero(active_gene_mask[batch_idx], as_tuple=False).flatten()
            if tf_pos.numel() == 0 or tgt_pos.numel() == 0:
                continue

            candidate_prob = torch.matmul(
                factors.u[batch_idx, tf_pos],
                factors.v[batch_idx, tgt_pos].transpose(0, 1),
            )
            candidate_prob = self._bound_probability(candidate_prob)
            top_src_idx, top_tgt_idx, top_prob = self._select_topk_edges(
                candidate_prob,
                topk=self.pred_topk,
            )
            if top_prob.numel() == 0:
                continue

            src_pos = tf_pos[top_src_idx]
            tgt_pos_selected = tgt_pos[top_tgt_idx]
            src_ids = input_ids[batch_idx, src_pos]
            tgt_ids = input_ids[batch_idx, tgt_pos_selected]
            pair_keys = src_ids * self._pair_key_base + tgt_ids
            is_reference = torch.isin(pair_keys, prior_pair_keys)

            batch_indices.append(torch.full_like(src_pos, fill_value=batch_idx))
            src_indices.append(src_pos)
            tgt_indices.append(tgt_pos_selected)
            labels.append(is_reference.to(dtype=factors.u.dtype))
            edge_weights = torch.where(
                is_reference,
                torch.full_like(top_prob, fill_value=self.pos_weight),
                torch.full_like(top_prob, fill_value=self.neg_weight),
            )
            weights.append(edge_weights.to(dtype=factors.u.dtype))

        if not batch_indices:
            return factors.u.new_zeros(())

        batch_idx = torch.cat(batch_indices)
        src_idx = torch.cat(src_indices)
        tgt_idx = torch.cat(tgt_indices)
        target = torch.cat(labels)
        weight = torch.cat(weights)

        edge_prob = (factors.u[batch_idx, src_idx] * factors.v[batch_idx, tgt_idx]).sum(
            dim=-1
        )
        edge_prob = self._bound_probability(edge_prob)
        edge_logits = torch.logit(edge_prob)
        loss = F.binary_cross_entropy_with_logits(edge_logits, target, reduction="none")

        numer = (loss * weight).sum()
        denom = weight.sum()
        return numer / (denom + 1e-8)
