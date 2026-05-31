from __future__ import annotations

import os
import random

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


_CONTROL_GROUP: dist.ProcessGroup | None = None


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool
    is_main: bool


def initialize_distributed() -> RuntimeContext:
    global _CONTROL_GROUP

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        # Let NCCL initialize lazily instead of forcing an eager single-device
        # connect during process-group setup. This is more robust across Slurm
        # environments where CUDA/NCCL library resolution can be finicky.
        dist.init_process_group(backend=backend)
    if distributed and _CONTROL_GROUP is None:
        _CONTROL_GROUP = (
            dist.group.WORLD
            if dist.get_backend() == "gloo"
            else dist.new_group(backend="gloo")
        )

    return RuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
        is_main=(rank == 0),
    )


def cleanup_distributed() -> None:
    global _CONTROL_GROUP

    if dist.is_available() and dist.is_initialized():
        if _CONTROL_GROUP is not None and _CONTROL_GROUP is not dist.group.WORLD:
            dist.destroy_process_group(_CONTROL_GROUP)
        _CONTROL_GROUP = None
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=_CONTROL_GROUP)


def seed_everything(seed: int, rank: int = 0) -> None:
    seed_value = int(seed) + int(rank)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=src, group=_CONTROL_GROUP)
    return objects[0]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def reduce_tensor_mean(value: torch.Tensor, runtime: RuntimeContext) -> torch.Tensor:
    if not runtime.distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= runtime.world_size
    return reduced


def reduce_scalar_dict(metrics: dict[str, float], runtime: RuntimeContext) -> dict[str, float]:
    if not metrics:
        return {}
    if not runtime.distributed:
        return metrics

    keys = list(metrics.keys())
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=_CONTROL_GROUP)
    values /= runtime.world_size
    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}
