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

        self.symbol_to_index, self.ensembl_to_index, self.pad_index = (
            build_token_lookup_maps(token_dict=token_dict)
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
        self.register_buffer(
            "_cached_src_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_cached_tgt_ids",
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
        self._cached_src_ids = cached_pair_keys // self._pair_key_base
        self._cached_tgt_ids = cached_pair_keys % self._pair_key_base
        self._cached_prior_ref = true_grn_df

    def _resolve_prior_edges(
        self,
        true_grn_df: Optional[pd.DataFrame],
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)
        return (
            self._cached_src_ids.to(device=device),
            self._cached_tgt_ids.to(device=device),
            self._cached_pair_keys.to(device=device),
        )

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
    def _lookup_positions(
        query_ids: torch.LongTensor,
        candidate_ids: torch.LongTensor,
        candidate_pos: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        missing = torch.full_like(query_ids, fill_value=-1)
        if query_ids.numel() == 0 or candidate_ids.numel() == 0:
            return missing, torch.zeros_like(query_ids, dtype=torch.bool)

        order = torch.argsort(candidate_ids)
        sorted_ids = candidate_ids[order]
        sorted_pos = candidate_pos[order]
        idx = torch.searchsorted(sorted_ids, query_ids)
        in_bounds = idx < sorted_ids.numel()
        safe_idx = idx.clamp(max=sorted_ids.numel() - 1)
        found = in_bounds & sorted_ids[safe_idx].eq(query_ids)

        positions = missing
        positions[found] = sorted_pos[safe_idx[found]]
        return positions, found

    @staticmethod
    def _sample_negative_edges(
        tf_pos: torch.LongTensor,
        tgt_pos: torch.LongTensor,
        positive_pos_keys: torch.LongTensor,
        sample_k: Optional[int],
        seq_len: int,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        total_count = int(tf_pos.numel() * tgt_pos.numel())
        if total_count == 0:
            return tf_pos.new_empty(0), tgt_pos.new_empty(0)

        positive_pos_keys = torch.unique(positive_pos_keys)
        neg_count = total_count - int(positive_pos_keys.numel())
        if neg_count <= 0:
            return tf_pos.new_empty(0), tgt_pos.new_empty(0)

        if sample_k is None:
            flat = torch.arange(total_count, device=tf_pos.device)
            src = tf_pos[flat // tgt_pos.numel()]
            tgt = tgt_pos[flat % tgt_pos.numel()]
            keep = ~torch.isin(src * seq_len + tgt, positive_pos_keys)
            return src[keep], tgt[keep]

        sample_k = min(int(sample_k), neg_count)
        if sample_k <= 0:
            return tf_pos.new_empty(0), tgt_pos.new_empty(0)

        sampled_keys: list[torch.Tensor] = []
        sampled_count = 0
        for _ in range(20):
            draw_count = max((sample_k - sampled_count) * 2, 64)
            flat = torch.randint(total_count, (draw_count,), device=tf_pos.device)
            src = tf_pos[flat // tgt_pos.numel()]
            tgt = tgt_pos[flat % tgt_pos.numel()]
            keys = torch.unique(src * seq_len + tgt)
            keep = ~torch.isin(keys, positive_pos_keys)
            if sampled_keys:
                keep = keep & ~torch.isin(keys, torch.cat(sampled_keys))
            keys = keys[keep]
            if keys.numel() == 0:
                continue
            sampled_keys.append(keys)
            sampled_count += int(keys.numel())
            if sampled_count >= sample_k:
                break

        if sampled_count < sample_k:
            flat = torch.arange(total_count, device=tf_pos.device)
            src = tf_pos[flat // tgt_pos.numel()]
            tgt = tgt_pos[flat % tgt_pos.numel()]
            keys = src * seq_len + tgt
            keep = ~torch.isin(keys, positive_pos_keys)
            if sampled_keys:
                keep = keep & ~torch.isin(keys, torch.cat(sampled_keys))
            sampled_keys.append(keys[keep])

        neg_keys = torch.cat(sampled_keys)[:sample_k]
        return neg_keys // seq_len, neg_keys % seq_len

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
        prior_src_ids, prior_tgt_ids, prior_pair_keys = self._resolve_prior_edges(
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

        batch_size, seq_len = input_ids.shape
        for batch_idx in range(batch_size):
            tf_pos = torch.nonzero(tf_mask[batch_idx], as_tuple=False).flatten()
            tgt_pos = torch.nonzero(active_gene_mask[batch_idx], as_tuple=False).flatten()
            if tf_pos.numel() == 0 or tgt_pos.numel() == 0:
                continue

            src_pos, src_found = self._lookup_positions(
                query_ids=prior_src_ids,
                candidate_ids=input_ids[batch_idx, tf_pos],
                candidate_pos=tf_pos,
            )
            tgt_pos_for_prior, tgt_found = self._lookup_positions(
                query_ids=prior_tgt_ids,
                candidate_ids=input_ids[batch_idx, tgt_pos],
                candidate_pos=tgt_pos,
            )
            positive_mask = src_found & tgt_found
            if not positive_mask.any():
                continue

            pos_src = src_pos[positive_mask]
            pos_tgt = tgt_pos_for_prior[positive_mask]
            positive_pos_keys = torch.unique(pos_src * seq_len + pos_tgt)
            pos_src = positive_pos_keys // seq_len
            pos_tgt = positive_pos_keys % seq_len
            pos_count = int(pos_src.numel())

            if self.neg_sample_ratio is None or self.neg_sample_ratio <= 0:
                neg_sample_k = None
            else:
                neg_sample_k = int(pos_count * self.neg_sample_ratio)
            neg_src, neg_tgt = self._sample_negative_edges(
                tf_pos=tf_pos,
                tgt_pos=tgt_pos,
                positive_pos_keys=positive_pos_keys,
                sample_k=neg_sample_k,
                seq_len=seq_len,
            )

            batch_pos = torch.full_like(pos_src, fill_value=batch_idx)
            batch_indices.append(batch_pos)
            src_indices.append(pos_src)
            tgt_indices.append(pos_tgt)
            labels.append(
                torch.ones(pos_count, device=input_ids.device, dtype=factors.u.dtype)
            )
            weights.append(
                torch.full(
                    (pos_count,),
                    fill_value=self.pos_weight,
                    device=input_ids.device,
                    dtype=factors.u.dtype,
                )
            )

            if neg_src.numel() > 0:
                neg_count = int(neg_src.numel())
                batch_indices.append(torch.full_like(neg_src, fill_value=batch_idx))
                src_indices.append(neg_src)
                tgt_indices.append(neg_tgt)
                labels.append(
                    torch.zeros(neg_count, device=input_ids.device, dtype=factors.u.dtype)
                )
                weights.append(
                    torch.full(
                        (neg_count,),
                        fill_value=self.neg_weight,
                        device=input_ids.device,
                        dtype=factors.u.dtype,
                    )
                )

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
