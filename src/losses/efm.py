from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.efm import EFMOutput


@dataclass(slots=True)
class EFMLossResult:
    total: torch.Tensor
    id_loss: torch.Tensor
    exp_loss: torch.Tensor
    metrics: dict[str, float]


class EFMLoss(nn.Module):
    def __init__(self, lambda_exp: float = 1.0) -> None:
        super().__init__()
        self.lambda_exp = float(lambda_exp)
        if self.lambda_exp < 0.0:
            raise ValueError(f"`lambda_exp` must be non-negative, got {self.lambda_exp}.")

    def forward(
        self,
        output: EFMOutput,
        target_ids: torch.LongTensor,
        target_expression: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ) -> EFMLossResult:
        if output.id_logits.shape[:2] != target_ids.shape:
            raise ValueError(
                "`output.id_logits` leading dims must match `target_ids`, got "
                f"{tuple(output.id_logits.shape[:2])} vs {tuple(target_ids.shape)}."
            )
        if output.expression_pred.shape != target_expression.shape:
            raise ValueError(
                "`output.expression_pred` must match `target_expression`, got "
                f"{tuple(output.expression_pred.shape)} vs {tuple(target_expression.shape)}."
            )
        if valid_mask.shape != target_ids.shape:
            raise ValueError(
                f"`valid_mask` must match target shape, got {tuple(valid_mask.shape)}."
            )

        valid_mask = valid_mask.to(device=output.id_logits.device, dtype=torch.bool)
        if not bool(valid_mask.any()):
            raise ValueError("EFM loss requires at least one valid target position.")

        target_ids = target_ids.to(device=output.id_logits.device, dtype=torch.long)
        target_expression = target_expression.to(
            device=output.expression_pred.device,
            dtype=output.expression_pred.dtype,
        )
        id_logits = output.id_logits[valid_mask]
        id_targets = target_ids[valid_mask]
        id_loss = F.cross_entropy(id_logits, id_targets)

        exp_pred = output.expression_pred[valid_mask]
        exp_targets = target_expression[valid_mask]
        exp_loss = F.mse_loss(exp_pred, exp_targets)
        total = id_loss + self.lambda_exp * exp_loss
        return EFMLossResult(
            total=total,
            id_loss=id_loss,
            exp_loss=exp_loss,
            metrics={
                "efm_total": float(total.detach().item()),
                "efm_id": float(id_loss.detach().item()),
                "efm_exp": float(exp_loss.detach().item()),
            },
        )
