from __future__ import annotations

import os
import random

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool
    is_main: bool


def initialize_distributed() -> RuntimeContext:
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
        init_kwargs: dict[str, Any] = {"backend": backend}
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)

    return RuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
        is_main=(rank == 0),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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
    dist.broadcast_object_list(objects, src=src)
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
        device=runtime.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= runtime.world_size
    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}
