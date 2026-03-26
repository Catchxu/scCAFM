from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sfm import FactorState
from ...utils import build_active_value_mask, require_tensor, validate_factor_shapes


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Sample from a diagonal Gaussian using the reparameterization trick.
    """

    if mu.shape != sigma.shape:
        raise ValueError(
            f"`mu` and `sigma` must share shape, got {tuple(mu.shape)} and {tuple(sigma.shape)}."
        )
    eps = torch.randn_like(mu)
    return mu + sigma * eps


@dataclass
class VGAEOutput:
    mu_z: torch.Tensor
    sigma_z: torch.Tensor
    mu_h: torch.Tensor
    sigma_h: torch.Tensor
    p_drop: torch.Tensor
    z: torch.Tensor


class _ScalarFeatureEncoder(nn.Module):
    """
    Lift scalar expression values into dense token features.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"`x` must have shape (C, G), got {tuple(x.shape)}.")
        return self.proj(x.unsqueeze(-1))


class VariationalEncoder(nn.Module):
    """
    Infer per-gene latent noise after conditioning on factor-mediated TF context.

    Inputs:
    - `expression_values`: (C, G)
    - `factors.u`, `factors.v`, `factors.u_score`, `factors.v_score`: (C, G, M)
    Outputs:
    - `mu_z`, `sigma_z`: (C, G)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.expr_proj = _ScalarFeatureEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.posterior_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _predict_from_factors(
        expr_features: torch.Tensor,
        factors: FactorState,
    ) -> torch.Tensor:
        u_eff, v_eff = factors.effective_factors()
        tf_summary = torch.einsum("cgm,cgh->cmh", u_eff, expr_features)
        return torch.einsum("cgm,cmh->cgh", v_eff, tf_summary)

    def forward(
        self,
        expression_values: torch.Tensor,
        factors: FactorState,
        active_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expr_features = self.expr_proj(expression_values)
        pred_features = self._predict_from_factors(expr_features, factors)
        posterior_in = torch.cat(
            [expr_features, pred_features, expr_features - pred_features],
            dim=-1,
        )
        posterior_stats = self.posterior_proj(posterior_in)
        mu_z, log_sigma_z = posterior_stats.unbind(dim=-1)
        sigma_z = F.softplus(log_sigma_z) + 1e-6

        mu_z = mu_z.masked_fill(~active_mask, 0.0)
        sigma_z = sigma_z.masked_fill(~active_mask, 1.0)
        return mu_z, sigma_z


class VariationalDecoder(nn.Module):
    """
    Decode latent per-gene variables into expression parameters and dropout.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        fp_steps: int = 3,
        fp_damping: float = 0.5,
    ) -> None:
        super().__init__()
        self.fp_steps = fp_steps
        self.fp_damping = fp_damping

        self.z_proj = _ScalarFeatureEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _propagate_with_factors(
        token_features: torch.Tensor,
        factors: FactorState,
    ) -> torch.Tensor:
        u_eff, v_eff = factors.effective_factors()
        tf_summary = torch.einsum("cgm,cgh->cmh", u_eff, token_features)
        return torch.einsum("cgm,cmh->cgh", v_eff, tf_summary)

    def decode(
        self,
        z: torch.Tensor,
        factors: FactorState,
        active_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_features = self.z_proj(z)

        h = z_features
        for _ in range(max(1, self.fp_steps)):
            propagated = self._propagate_with_factors(h, factors)
            h = z_features + self.fp_damping * propagated

        out = self.out_proj(torch.cat([z_features, h], dim=-1))
        mu_h, log_sigma_h, drop_logits = out.unbind(dim=-1)
        sigma_h = F.softplus(log_sigma_h) + 1e-6
        p_drop = torch.sigmoid(drop_logits)

        mu_h = mu_h.masked_fill(~active_mask, 0.0)
        sigma_h = sigma_h.masked_fill(~active_mask, 1.0)
        p_drop = p_drop.masked_fill(~active_mask, 0.0)
        return mu_h, sigma_h, p_drop

    def forward(
        self,
        mu_z: torch.Tensor,
        sigma_z: torch.Tensor,
        factors: FactorState,
        active_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = reparameterize(mu_z, sigma_z)
        return self.decode(
            z=z,
            factors=factors,
            active_mask=active_mask,
        )


class VGAE(nn.Module):
    """
    Factor-aware VGAE model for the current SFM tokenization pipeline.

    Expected token entries:
    - `expression_values`: (C, G)
    - `padding_mask`: optional (C, G), True where padded
    - `non_tf_mask`: (C, G), True for non-TF genes

    Expected factor entries:
    - `u`, `v`, `u_score`, `v_score`: (C, G, M)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        fp_steps: int = 3,
        fp_damping: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = VariationalEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = VariationalDecoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        factors: FactorState,
    ) -> VGAEOutput:
        if factors is None:
            raise ValueError("`factors` must be provided.")

        expression_values = require_tensor(tokens, "expression_values")
        active_mask = build_active_value_mask(
            values=expression_values,
            padding_mask=tokens.get("padding_mask"),
        )
        validate_factor_shapes(factors=factors, input_shape_prefix=expression_values.shape)

        mu_z, sigma_z = self.encoder(
            expression_values=expression_values,
            factors=factors,
            active_mask=active_mask,
        )
        z = reparameterize(mu_z, sigma_z)
        mu_h, sigma_h, p_drop = self.decoder.decode(
            z=z,
            factors=factors,
            active_mask=active_mask,
        )

        return VGAEOutput(
            mu_z=mu_z,
            sigma_z=sigma_z,
            mu_h=mu_h,
            sigma_h=sigma_h,
            p_drop=p_drop,
            z=z,
        )
