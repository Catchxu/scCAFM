from __future__ import annotations

import copy
import warnings

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from .assets import ModelAssets, load_json, save_json, load_model_state_dict, save_model_state_dict
from .data import PretrainingAssets
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
        config: dict[str, Any],
        data_assets: PretrainingAssets,
    ) -> None:
        self.paths = paths
        self.assets = assets
        self.runtime = runtime
        self.logger = logger
        self.config = config
        self.data_assets = data_assets

    def model_weights_path(self) -> Path:
        return self.assets.sfm_model

    def latest_resume_state_path(self) -> Path:
        return self.paths.resume_state_file

    def resume_manifest_path(self) -> Path:
        return self.paths.resume_manifest_file

    @staticmethod
    def _resume_manifest_path_for_resume_file(resume_state_path: Path) -> Path:
        return resume_state_path.parent / "logs" / "resume_manifest.json"

    @staticmethod
    def _current_train_path(train_state: dict[str, Any], train_paths: list[Path]) -> str | None:
        if not train_paths:
            return None

        file_index = int(train_state.get("file_index", 0))
        if file_index < 0:
            file_index = 0
        if file_index >= len(train_paths):
            file_index = len(train_paths) - 1
        return str(train_paths[file_index])

    def _build_resume_manifest(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_manager: torch.nn.Module,
        train_state: dict[str, Any],
    ) -> dict[str, Any]:
        data_cfg = self.config["data"]
        optimizer_cfg = copy.deepcopy(self.config.get("optimizer", {}))
        scheduler_cfg = copy.deepcopy(self.config.get("scheduler", {}))
        trainer_cfg = copy.deepcopy(self.config.get("trainer", {}))
        runtime_cfg = copy.deepcopy(self.config.get("runtime", {}))
        train_paths = self.data_assets.train_paths

        return {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_source": self.assets.model_source,
            "asset_dir": str(self.assets.local_dir),
            "model_weights_path": str(self.model_weights_path()),
            "resume_state_path": str(self.latest_resume_state_path()),
            "train_state": copy.deepcopy(train_state),
            "runtime": {
                "distributed": bool(self.runtime.distributed),
                "world_size": int(self.runtime.world_size),
                "device": str(self.runtime.device),
                "config": runtime_cfg,
            },
            "optimizer": {
                "class_name": optimizer.__class__.__name__,
                "config": optimizer_cfg,
                "param_group_count": len(optimizer.param_groups),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            },
            "scheduler": {
                "class_name": scheduler.__class__.__name__,
                "config": scheduler_cfg,
                "last_epoch": int(getattr(scheduler, "last_epoch", -1)),
            },
            "loss_manager": {
                "class_name": loss_manager.__class__.__name__,
            },
            "data": {
                "num_train_files": len(train_paths),
                "train_paths": [str(path) for path in train_paths],
                "current_train_path": self._current_train_path(train_state, train_paths),
                "batch_size": int(data_cfg["batch_size"]),
                "gradient_accumulation_steps": int(data_cfg.get("gradient_accumulation_steps", 1)),
                "max_length": int(data_cfg["max_length"]),
                "gene_key": data_cfg.get("gene_key"),
                "cond_vocab_size": int(self.data_assets.cond_vocab_size),
            },
            "trainer": trainer_cfg,
        }

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
        resume_manifest_path = self.resume_manifest_path()

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
            resume_manifest = self._build_resume_manifest(
                optimizer=optimizer,
                scheduler=scheduler,
                loss_manager=loss_manager,
                train_state=train_state,
            )
            resume_payload = {
                "optimizer_state_dict": _cast_floating_tensors_to_fp32(optim_state),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_manager_state_dict": loss_manager.state_dict(),
                "train_state": train_state,
                "model_weights_path": str(model_weights_path),
                "resume_manifest_path": str(resume_manifest_path),
            }
            torch.save(resume_payload, resume_state_path)
            save_json(resume_manifest_path, resume_manifest)
            self.logger.info("Model weights saved to %s", model_weights_path)
            self.logger.info("Resume state saved to %s", resume_state_path)
            self.logger.info(
                "Resume manifest saved to %s (global_step=%s, epoch=%s, file_index=%s, optimizer=%s, scheduler=%s)",
                resume_manifest_path,
                int(train_state.get("global_step", 0)),
                int(train_state.get("epoch", 0)),
                int(train_state.get("file_index", 0)),
                optimizer.__class__.__name__,
                scheduler.__class__.__name__,
            )

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
            manifest_path = self._resume_manifest_path_for_resume_file(resume_state_path)
            if manifest_path.exists():
                manifest = load_json(manifest_path)
                if isinstance(manifest, dict):
                    train_state = manifest.get("train_state", {})
                    data_info = manifest.get("data", {})
                    optimizer_info = manifest.get("optimizer", {})
                    scheduler_info = manifest.get("scheduler", {})
                    self.logger.info(
                        "Resume manifest: step=%s, epoch=%s, file_index=%s, current_train_path=%s",
                        int(train_state.get("global_step", 0)),
                        int(train_state.get("epoch", 0)),
                        int(train_state.get("file_index", 0)),
                        data_info.get("current_train_path"),
                    )
                    self.logger.info(
                        "Resume components: optimizer=%s, scheduler=%s, train_files=%s",
                        optimizer_info.get("class_name"),
                        scheduler_info.get("class_name"),
                        data_info.get("num_train_files"),
                    )
        return payload
