from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_weighted_mean_loss(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a differentiable globally normalized loss and detached global mean.

    In DDP/FSDP, gradients are averaged across ranks after backward. Scaling each
    rank's local numerator by world_size / global_count makes the averaged
    gradient match the gradient of a true global mean.
    """

    if local_sum.ndim != 0:
        raise ValueError(f"`local_sum` must be scalar, got shape {tuple(local_sum.shape)}.")
    if local_count.ndim != 0:
        raise ValueError(f"`local_count` must be scalar, got shape {tuple(local_count.shape)}.")

    global_sum = local_sum.detach().to(dtype=torch.float64)
    global_count = local_count.detach().to(device=local_sum.device, dtype=torch.float64)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = float(dist.get_world_size())
    else:
        world_size = 1.0

    if float(global_count.item()) <= 0.0:
        zero_loss = local_sum * 0.0
        zero_metric = global_sum.new_zeros(())
        return zero_loss, zero_metric

    loss = local_sum * (world_size / global_count.to(dtype=local_sum.dtype))
    metric = global_sum / global_count
    return loss, metric
