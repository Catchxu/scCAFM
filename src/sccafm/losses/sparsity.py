from __future__ import annotations

import torch
import torch.nn as nn

from ..models.sfm import FactorState
from ..utils import (
    build_active_gene_mask,
    build_tf_mask,
    require_tensor,
    validate_factor_shapes,
)
from .reduction import distributed_weighted_mean_loss


class SparsityLoss(nn.Module):
    """
    Penalize the mean valid TF->target edge score induced by factor assignments.

    The dense edge matrix is never materialized. For each cell, the sum over all
    valid TF-target scores from `u @ v.T` is equivalent to
    `sum_tf(u) dot sum_target(v)`.
    """

    def __init__(
        self,
        lambda_sparsity: float = 1e-3,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_sparsity = float(lambda_sparsity)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer(
            "last_mean",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        factors: FactorState,
        global_step: int = 0,
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

        with torch.autocast(device_type=factors.u.device.type, enabled=False):
            u = factors.u.float()
            v = factors.v.float()
            tf_mask_f = tf_mask.to(device=u.device, dtype=u.dtype)
            target_mask_f = active_gene_mask.to(device=v.device, dtype=v.dtype)

            tf_factor_sum = (u * tf_mask_f.unsqueeze(-1)).sum(dim=1)
            target_factor_sum = (v * target_mask_f.unsqueeze(-1)).sum(dim=1)
            edge_sum_per_cell = (tf_factor_sum * target_factor_sum).sum(dim=-1)
            local_sum = edge_sum_per_cell.sum()

            tf_count = tf_mask_f.sum(dim=1)
            target_count = target_mask_f.sum(dim=1)
            local_count = (tf_count * target_count).sum()
            sparsity_loss, sparsity_metric = distributed_weighted_mean_loss(
                local_sum=local_sum,
                local_count=local_count,
            )

        self.last_mean.copy_(sparsity_metric.detach().to(torch.float32))

        if int(global_step) < self.warmup_steps:
            return sparsity_loss.to(dtype=factors.u.dtype) * 0.0

        return sparsity_loss.to(dtype=factors.u.dtype) * self.lambda_sparsity
