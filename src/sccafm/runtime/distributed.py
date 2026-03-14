import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    initialized_here = False
    rank = 0
    local_rank = 0

    if is_distributed:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            initialized_here = True
        rank = dist.get_rank()
    return is_distributed, initialized_here, rank, local_rank, world_size


def unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


def resolve_device(device: str, is_distributed: bool, local_rank: int):
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"cuda:{local_rank}" if is_distributed else device
    return "cpu"
