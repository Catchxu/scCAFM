from __future__ import annotations

import torch
import torch.nn as nn

from ..models.heads.vgae import VGAEOutput
from ..utils import build_active_value_mask, require_tensor


class ELBOLoss(nn.Module):
    """
    ELBO objective for outputs produced by `VGAE`.

    The reduction follows the standard per-sample ELBO convention:
    - sum over valid genes within each cell
    - mean over cells in the minibatch
    """

    def __init__(self) -> None:
        super().__init__()

    def zinormal_loglik(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        p_drop: torch.Tensor,
        active_mask: torch.BoolTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        x = x.to(mu.dtype)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1e4, neginf=1e-5)
        p_drop = torch.nan_to_num(p_drop, nan=0.0, posinf=1.0, neginf=0.0)
        sigma = sigma.clamp_min(1e-5)
        p_drop = p_drop.clamp(min=1e-5, max=1.0 - 1e-5)

        normal = torch.distributions.Normal(mu, sigma)
        log_prob_zero = normal.log_prob(torch.zeros_like(mu))
        log_prob = torch.where(
            x == 0,
            torch.log(p_drop + (1.0 - p_drop) * torch.exp(log_prob_zero) + eps),
            torch.log(1.0 - p_drop + eps) + normal.log_prob(x),
        )
        log_prob = log_prob.masked_fill(~active_mask, 0.0)
        return log_prob.sum(dim=-1)

    @staticmethod
    def kl_normal(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        active_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1e4, neginf=1e-5)
        sigma = sigma.clamp_min(1e-5)
        kl = 0.5 * (
            mu.pow(2) + sigma.pow(2) - 1.0 - torch.log(sigma.pow(2))
        )
        kl = kl.masked_fill(~active_mask, 0.0)
        return kl.sum(dim=-1)

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        vgae_output: VGAEOutput,
    ) -> torch.Tensor:
        expression_values = require_tensor(tokens, "expression_values")
        active_mask = build_active_value_mask(
            values=expression_values,
            padding_mask=tokens.get("padding_mask"),
        )

        log_px = self.zinormal_loglik(
            x=expression_values,
            mu=vgae_output.mu_h,
            sigma=vgae_output.sigma_h,
            p_drop=vgae_output.p_drop,
            active_mask=active_mask,
        ).mean()
        kl_z = self.kl_normal(
            mu=vgae_output.mu_z,
            sigma=vgae_output.sigma_z,
            active_mask=active_mask,
        ).mean()
        return -log_px + kl_z
