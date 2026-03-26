from __future__ import annotations

import torch
import torch.nn as nn

from ..models.sfm import FactorState


class DAGLoss(nn.Module):
    """
    DAG regularization on factor-space adjacency induced by raw `u` and `v`.

    Notes:
    - This loss constrains `u` and `v` only.
    - `u_score` and `v_score` are intentionally ignored.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        rho: float = 0.1,
        rho_max: float = 1e6,
        update_period: int = 100,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.rho_max = float(rho_max)
        self.update_period = int(update_period)

        self.prev_h_val = float("inf")
        self.step_counter = 0
        self.accumulated_h = 0.0

    @staticmethod
    def _validate_factors(factors: FactorState) -> None:
        tensors = {
            "u": factors.u,
            "v": factors.v,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 3:
                raise ValueError(
                    f"`factors.{name}` must have shape (C, G, M), got {tuple(tensor.shape)}."
                )
        if factors.u.shape != factors.v.shape:
            raise ValueError(
                f"`factors.u` and `factors.v` must share shape, got "
                f"{tuple(factors.u.shape)} and {tuple(factors.v.shape)}."
            )

    @staticmethod
    def _compute_dag_constraint(adj: torch.Tensor) -> torch.Tensor:
        if adj.ndim != 3 or adj.shape[-1] != adj.shape[-2]:
            raise ValueError(
                f"`adj` must have shape (C, M, M), got {tuple(adj.shape)}."
            )

        batch_size, num_factors, _ = adj.shape
        device = adj.device
        adj_sq = adj * adj
        eye = torch.eye(num_factors, device=device, dtype=adj.dtype).unsqueeze(0)
        matrix_poly = eye.expand(batch_size, -1, -1) + adj_sq / num_factors
        res = torch.matrix_power(matrix_poly, num_factors)
        return torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1) - num_factors

    def _auto_update_params(self) -> None:
        avg_h = self.accumulated_h / self.update_period
        if avg_h > 0.25 * self.prev_h_val:
            self.rho = min(self.rho * 10.0, self.rho_max)
        else:
            self.alpha += self.rho * avg_h
        self.prev_h_val = avg_h
        self.accumulated_h = 0.0
        self.step_counter = 0

    def forward(self, factors: FactorState) -> torch.Tensor:
        if factors is None:
            raise ValueError("`factors` must be provided.")
        self._validate_factors(factors)

        adj_factor = torch.bmm(factors.v.transpose(1, 2), factors.u)
        dag_h_batch = self._compute_dag_constraint(adj_factor).mean()

        if self.training:
            self.step_counter += 1
            self.accumulated_h += float(dag_h_batch.detach())
            if self.step_counter >= self.update_period:
                self._auto_update_params()

        return self.alpha * dag_h_batch + 0.5 * self.rho * (dag_h_batch ** 2)
