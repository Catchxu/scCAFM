from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from .distributed import RuntimeContext, barrier
from .experiment import ExperimentLogger, ExperimentPaths


def _cast_floating_tensors_to_fp32(payload: Any) -> Any:
    if torch.is_tensor(payload):
        if payload.is_floating_point():
            return payload.detach().cpu().float()
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {key: _cast_floating_tensors_to_fp32(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_cast_floating_tensors_to_fp32(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_cast_floating_tensors_to_fp32(value) for value in payload)
    return payload


class CheckpointManager:
    def __init__(
        self,
        paths: ExperimentPaths,
        runtime: RuntimeContext,
        logger: ExperimentLogger,
    ) -> None:
        self.paths = paths
        self.runtime = runtime
        self.logger = logger

    def latest_checkpoint_path(self) -> Path:
        return self.paths.checkpoints / "sfm_latest.pt"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_manager: torch.nn.Module,
        train_state: dict[str, Any],
    ) -> Path:
        checkpoint_path = self.latest_checkpoint_path()

        if isinstance(model, FSDP):
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=state_cfg,
            ):
                model_state = model.state_dict()
            optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
        else:
            model_state = model.state_dict()
            optim_state = optimizer.state_dict()

        if self.runtime.is_main:
            payload = {
                "model_state_dict": _cast_floating_tensors_to_fp32(model_state),
                "optimizer_state_dict": _cast_floating_tensors_to_fp32(optim_state),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_manager_state_dict": loss_manager.state_dict(),
                "train_state": train_state,
            }
            torch.save(payload, checkpoint_path)
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        barrier()
        return checkpoint_path

    def load(self, resume_path: str | None) -> dict[str, Any] | None:
        if resume_path is None:
            return None

        checkpoint_path = Path(resume_path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        if self.runtime.is_main:
            self.logger.info(f"Resuming from checkpoint {checkpoint_path}")
        return payload
