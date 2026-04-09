from __future__ import annotations

import warnings

from pathlib import Path
from typing import Any

import torch

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from .assets import ModelAssets, load_model_state_dict, save_model_state_dict
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
        assets: ModelAssets,
        runtime: RuntimeContext,
        logger: ExperimentLogger,
    ) -> None:
        self.paths = paths
        self.assets = assets
        self.runtime = runtime
        self.logger = logger

    def model_weights_path(self) -> Path:
        return self.assets.sfm_model

    def latest_resume_state_path(self) -> Path:
        return self.paths.resume_state_file

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_manager: torch.nn.Module,
        train_state: dict[str, Any],
    ) -> tuple[Path, Path]:
        model_weights_path = self.model_weights_path()
        resume_state_path = self.latest_resume_state_path()

        if isinstance(model, FSDP):
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated\.",
                    category=FutureWarning,
                )
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    state_dict_config=state_cfg,
                ):
                    model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
        else:
            model_state = model.state_dict()
            optim_state = optimizer.state_dict()

        if self.runtime.is_main:
            save_model_state_dict(
                model_weights_path,
                _cast_floating_tensors_to_fp32(model_state),
            )
            resume_payload = {
                "optimizer_state_dict": _cast_floating_tensors_to_fp32(optim_state),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_manager_state_dict": loss_manager.state_dict(),
                "train_state": train_state,
                "model_weights_path": str(model_weights_path),
            }
            torch.save(resume_payload, resume_state_path)
            self.logger.info("Model weights saved to %s", model_weights_path)
            self.logger.info("Resume state saved to %s", resume_state_path)

        barrier()
        return model_weights_path, resume_state_path

    def load_model_weights(self) -> dict[str, torch.Tensor] | None:
        model_weights_path = self.model_weights_path()
        if not model_weights_path.exists():
            return None
        return load_model_state_dict(model_weights_path)

    def load_resume_state(self, resume_path: str | None) -> dict[str, Any] | None:
        if resume_path is None:
            return None

        resume_state_path = Path(resume_path).expanduser().resolve()
        payload = torch.load(resume_state_path, map_location="cpu")
        if self.runtime.is_main:
            self.logger.info(f"Resuming from local train-state file {resume_state_path}")
        return payload
