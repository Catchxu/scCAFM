from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist

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
        self.rho_max = float(rho_max)
        self.update_period = int(update_period)

        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "rho",
            torch.tensor(float(rho), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "prev_h_val",
            torch.tensor(float("inf"), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "step_counter",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "accumulated_h",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_pending_h_sum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_pending_h_weight",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "last_h",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

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
        adj_pos = adj.clamp_min(0.0)
        eye = torch.eye(num_factors, device=device, dtype=adj.dtype).unsqueeze(0)
        matrix_poly = eye.expand(batch_size, -1, -1) + adj_pos / num_factors
        res = torch.matrix_power(matrix_poly, num_factors)
        return torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1) - num_factors

    def _auto_update_params(self) -> None:
        avg_h = self.accumulated_h / float(self.update_period)
        if avg_h.item() > 0.25 * self.prev_h_val.item():
            self.rho.fill_(min(self.rho.item() * 10.0, self.rho_max))
        else:
            self.alpha.add_(self.rho * avg_h)
        self.prev_h_val.copy_(avg_h)
        self.accumulated_h.zero_()
        self.step_counter.zero_()

    def forward(self, factors: FactorState) -> torch.Tensor:
        if factors is None:
            raise ValueError("`factors` must be provided.")
        self._validate_factors(factors)

        adj_factor = torch.bmm(factors.v.transpose(1, 2), factors.u)
        dag_h_batch = self._compute_dag_constraint(adj_factor).mean()
        self.last_h.copy_(dag_h_batch.detach().to(torch.float32))

        if self.training:
            self._pending_h_sum.add_(dag_h_batch.detach().to(torch.float32))
            self._pending_h_weight.add_(1.0)

        alpha = self.alpha.to(device=dag_h_batch.device, dtype=dag_h_batch.dtype)
        rho = self.rho.to(device=dag_h_batch.device, dtype=dag_h_batch.dtype)
        return alpha * dag_h_batch + 0.5 * rho * (dag_h_batch ** 2)

    def apply_constraint_update(self) -> None:
        if self._pending_h_weight.item() <= 0:
            return

        h_sum = self._pending_h_sum.clone()
        h_weight = self._pending_h_weight.clone()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(h_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(h_weight, op=dist.ReduceOp.SUM)

        if h_weight.item() > 0:
            avg_h_step = float((h_sum / h_weight).item())
            self.step_counter.add_(1)
            self.accumulated_h.add_(avg_h_step)
            if self.step_counter.item() >= self.update_period:
                self._auto_update_params()

        self._pending_h_sum.zero_()
        self._pending_h_weight.zero_()
