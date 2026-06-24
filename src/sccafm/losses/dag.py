from __future__ import annotations

import torch
import torch.nn as nn

from ..models.sfm import FactorState


class DAGLoss(nn.Module):
    """
    Direct DAG regularization on factor-space adjacency induced by raw `u` and `v`.
    """

    def __init__(
        self,
        lambda_dag: float = 0.1,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_dag = float(lambda_dag)
        self.warmup_steps = int(warmup_steps)

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

    def forward(self, factors: FactorState, global_step: int = 0) -> torch.Tensor:
        if factors is None:
            raise ValueError("`factors` must be provided.")
        self._validate_factors(factors)

        # Keep the DAG computation in fp32 for stability even if training uses bf16 autocast.
        with torch.autocast(device_type=factors.u.device.type, enabled=False):
            u = factors.u.float()
            v = factors.v.float()
            adj_factor = torch.bmm(v.transpose(1, 2), u)
            dag_h_batch = self._compute_dag_constraint(adj_factor).mean()

        self.last_h.copy_(dag_h_batch.detach().to(torch.float32))

        if int(global_step) < self.warmup_steps:
            return dag_h_batch.to(dtype=factors.u.dtype) * 0.0

        return dag_h_batch.to(dtype=factors.u.dtype) * self.lambda_dag
