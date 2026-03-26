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
    Supervise TF->target edges against a prior GRN using the raw `u` and `v`.

    Notes:
    - This loss constrains `u` and `v` only.
    - `u_score` and `v_score` are intentionally ignored.
    - Source positions are restricted to active TF genes; target positions are
      restricted to active genes.
    """

    def __init__(
        self,
        token_dict: pd.DataFrame,
        true_grn_df: Optional[pd.DataFrame] = None,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        neg_sample_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.symbol_to_index, self.ensembl_to_index, self.pad_index = build_token_lookup_maps(
            token_dict=token_dict
        )
        token_max_1 = max(int(token_dict["token_index"].max()) + 1, 1)
        self._pair_key_base = int(token_max_1)

        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.neg_sample_ratio = None if neg_sample_ratio is None else float(neg_sample_ratio)

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
        for src_name, tgt_name in zip(true_grn_df["Gene1"].tolist(), true_grn_df["Gene2"].tolist()):
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

    def _resolve_prior_pair_keys(
        self,
        true_grn_df: Optional[pd.DataFrame],
        device: torch.device,
    ) -> torch.LongTensor:
        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)
        return self._cached_pair_keys.to(device=device)

    def _build_targets(
        self,
        input_ids: torch.LongTensor,
        tf_mask: torch.BoolTensor,
        active_gene_mask: torch.BoolTensor,
        prior_pair_keys: torch.LongTensor,
    ) -> tuple[torch.BoolTensor, torch.Tensor]:
        supervise_mask = tf_mask.unsqueeze(2) & active_gene_mask.unsqueeze(1)
        if prior_pair_keys.numel() == 0:
            target = torch.zeros_like(supervise_mask, dtype=torch.float32)
            return supervise_mask, target

        pair_keys = (
            input_ids.unsqueeze(2).to(torch.long) * self._pair_key_base
            + input_ids.unsqueeze(1).to(torch.long)
        )
        positive_mask = torch.isin(pair_keys, prior_pair_keys) & supervise_mask
        target = positive_mask.to(dtype=torch.float32)
        return supervise_mask, target

    @staticmethod
    def _sample_negative_mask(
        supervise_mask: torch.BoolTensor,
        target: torch.Tensor,
        neg_sample_ratio: Optional[float],
    ) -> torch.BoolTensor:
        if neg_sample_ratio is None or neg_sample_ratio <= 0:
            return supervise_mask

        sampled_mask = torch.zeros_like(supervise_mask, dtype=torch.bool)
        batch_size = supervise_mask.shape[0]
        for batch_idx in range(batch_size):
            valid_mask = supervise_mask[batch_idx]
            pos_mask = (target[batch_idx] > 0.5) & valid_mask
            neg_mask = (~(target[batch_idx] > 0.5)) & valid_mask

            pos_count = int(pos_mask.sum().item())
            neg_count = int(neg_mask.sum().item())

            if pos_count == 0 or neg_count == 0:
                sampled_mask[batch_idx] = valid_mask
                continue

            sample_k = min(neg_count, int(pos_count * neg_sample_ratio))
            if sample_k <= 0:
                sampled_mask[batch_idx] = pos_mask
                continue

            neg_indices = torch.nonzero(neg_mask, as_tuple=False)
            chosen = neg_indices[
                torch.randperm(neg_indices.shape[0], device=neg_indices.device)[:sample_k]
            ]
            sampled_neg_mask = torch.zeros_like(neg_mask)
            sampled_neg_mask[chosen[:, 0], chosen[:, 1]] = True
            sampled_mask[batch_idx] = pos_mask | sampled_neg_mask

        return sampled_mask

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
        prior_pair_keys = self._resolve_prior_pair_keys(
            true_grn_df=true_grn_df,
            device=input_ids.device,
        )

        if prior_pair_keys.numel() == 0:
            return factors.u.new_zeros(())

        supervise_mask, target = self._build_targets(
            input_ids=input_ids,
            tf_mask=tf_mask,
            active_gene_mask=active_gene_mask,
            prior_pair_keys=prior_pair_keys,
        )

        if not supervise_mask.any():
            return factors.u.new_zeros(())

        edge_prob = torch.bmm(factors.u, factors.v.transpose(1, 2)).clamp(1e-8, 1.0 - 1e-8)
        loss = F.binary_cross_entropy(edge_prob, target.to(edge_prob.dtype), reduction="none")
        weight = target.to(edge_prob.dtype) * self.pos_weight + (
            1.0 - target.to(edge_prob.dtype)
        ) * self.neg_weight

        sampled_mask = self._sample_negative_mask(
            supervise_mask=supervise_mask,
            target=target,
            neg_sample_ratio=self.neg_sample_ratio,
        )
        sampled_mask_f = sampled_mask.to(dtype=edge_prob.dtype)

        numer = (loss * weight * sampled_mask_f).sum()
        denom = (weight * sampled_mask_f).sum()
        return numer / (denom + 1e-8)
