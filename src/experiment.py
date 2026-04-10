from __future__ import annotations

import logging
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .distributed import RuntimeContext, barrier, broadcast_object


def format_metric_value(value: Any) -> str:
    value_f = float(value)
    abs_value = abs(value_f)
    if abs_value == 0.0:
        return "0.000000"
    if 1e-4 <= abs_value < 1e4:
        return f"{value_f:.6f}"
    return f"{value_f:.6e}"


@dataclass
class ExperimentPaths:
    root: Path
    logs: Path
    checkpoints: Path
    model_package_dir: Path
    log_file: Path
    resume_manifest_file: Path
    resume_state_file: Path


def prepare_experiment_paths(
    runtime: RuntimeContext,
    resume_path: str | None = None,
) -> ExperimentPaths:
    if resume_path:
        checkpoint_path = Path(resume_path).expanduser().resolve()
        root = checkpoint_path.parent
    elif runtime.is_main:
        root = Path.cwd().resolve()
    else:
        root = None

    root = Path(broadcast_object(str(root) if root is not None else None))
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    model_package_dir = checkpoints / "models"

    if runtime.is_main:
        logs.mkdir(parents=True, exist_ok=True)
        model_package_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    return ExperimentPaths(
        root=root,
        logs=logs,
        checkpoints=checkpoints,
        model_package_dir=model_package_dir,
        log_file=logs / "pretrain.log",
        resume_manifest_file=logs / "resume_manifest.json",
        resume_state_file=root / "sfm_train_state.pt",
    )


class ExperimentLogger:
    def __init__(
        self,
        name: str,
        paths: ExperimentPaths,
        runtime: RuntimeContext,
    ) -> None:
        self.paths = paths
        self.runtime = runtime
        self.logger = logging.getLogger(f"{name}.rank{runtime.rank}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if runtime.is_main:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(
                paths.log_file,
                mode="w",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger.addHandler(logging.NullHandler())

    def info(self, message: str, *args: Any) -> None:
        if self.runtime.is_main:
            self.logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        if self.runtime.is_main:
            self.logger.warning(message, *args)

    def log_metrics(self, split: str, step: int, metrics: dict[str, Any]) -> None:
        if not self.runtime.is_main:
            return
        metric_text = ", ".join(
            f"{key}={format_metric_value(value)}" for key, value in metrics.items()
        )
        self.logger.info("[%s step=%s] %s", split, int(step), metric_text)
