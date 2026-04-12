from __future__ import annotations

import argparse
import contextlib
import gc
from typing import Any

import torch

from ..assets import (
    apply_model_assets_to_runtime_config,
    load_sfm_config,
    materialize_model_package,
    resolve_model_assets,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from .builders import (
    build_model,
    build_optimizer,
    build_scheduler,
    maybe_wrap_fsdp,
)
from ..checkpoint import CheckpointManager
from ..config import load_yaml_config
from ..data import (
    PretrainingAssets,
    PretrainingDataBundle,
    build_data_bundle_for_path,
    build_pretraining_assets,
    estimate_total_training_steps,
)
from ..distributed import (
    RuntimeContext,
    cleanup_distributed,
    initialize_distributed,
    move_batch_to_device,
    reduce_scalar_dict,
)
from ..experiment import ExperimentLogger, format_metric_value, prepare_experiment_paths
from ..losses import PretrainingLossManager
from ..models.router import QBGating


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the wrapped SFM + VGAE model.")
    parser.add_argument("--pretrain-config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _load_resume_states(
    payload: dict[str, Any] | None,
    optimizer: Any,
    scheduler: Any,
    model: Any,
) -> dict[str, Any] | None:
    if payload is None:
        return None

    if isinstance(model, FSDP):
        optim_state = FSDP.optim_state_dict_to_load(
            model,
            optimizer,
            payload["optimizer_state_dict"],
        )
        optimizer.load_state_dict(optim_state)
    else:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload.get("train_state")


def _apply_qb_gating_updates(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, QBGating):
            module.apply_beta_update()


def _release_data_bundle(data_bundle: PretrainingDataBundle | None, runtime: RuntimeContext) -> None:
    if data_bundle is None:
        return

    train_loader = getattr(data_bundle, "train_loader", None)
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "close"):
        dataset.close()

    del train_loader
    gc.collect()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()


def _log_run_summary(
    logger: ExperimentLogger,
    runtime: RuntimeContext,
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    total_steps: int,
) -> None:
    runtime_cfg = config["runtime"]
    data_cfg = config["data"]
    optimizer_cfg = config["optimizer"]
    scheduler_cfg = config["scheduler"]
    trainer_cfg = config["trainer"]
    precision_cfg = runtime_cfg.get("precision", {})
    preprocess_cfg = data_cfg.get("preprocess", {})
    condition_mask_cfg = data_cfg.get("condition_mask", {})
    condition_vocab_cfg = data_cfg.get("condition_vocab", {})
    batch_size = int(data_cfg["batch_size"])
    grad_accum_steps = int(data_cfg.get("gradient_accumulation_steps", 1))
    global_batch_size = batch_size * int(runtime.world_size) * grad_accum_steps

    logger.info("========== Run Summary ==========")
    logger.info(
        "Runtime: distributed=%s, world_size=%s",
        runtime.distributed,
        runtime.world_size,
    )
    logger.info(
        "Precision: model_dtype=%s, autocast_dtype=%s",
        precision_cfg.get("model_dtype", "fp32"),
        precision_cfg.get("autocast_dtype", "fp32"),
    )
    logger.info(
        "Data: num_adata=%s, batch_size=%s, gradient_accumulation_steps=%s, global_batch_size=%s, max_length=%s",
        len(data_assets.train_paths),
        batch_size,
        grad_accum_steps,
        global_batch_size,
        data_cfg["max_length"],
    )
    logger.info(
        "Optimizer: name=%s, lr=%s, betas=%s, weight_decay=%s",
        optimizer_cfg.get("name", "adamw"),
        optimizer_cfg["lr"],
        optimizer_cfg.get("betas"),
        optimizer_cfg.get("weight_decay", 0.0),
    )
    logger.info(
        "Scheduler: name=%s, warmup_ratio=%s, min_lr_ratio=%s, estimated_total_steps=%s",
        scheduler_cfg.get("name", "cosine_with_warmup"),
        scheduler_cfg.get("warmup_ratio", 0.03),
        scheduler_cfg.get("min_lr_ratio", 0.0),
        total_steps,
    )
    logger.info(
        "Trainer: epochs=%s, grad_clip_norm=%s, save_every_epochs=%s, log_every_steps=%s",
        trainer_cfg["epochs"],
        trainer_cfg.get("grad_clip_norm"),
        trainer_cfg.get("save_every_epochs", 1),
        trainer_cfg.get("log_every_steps", 10),
    )
    logger.info(
        "Preprocess: enabled=%s, min_genes=%s, min_cells=%s, n_top_genes=%s, hvg_flavor=%s",
        preprocess_cfg.get("enabled", False),
        preprocess_cfg.get("min_genes"),
        preprocess_cfg.get("min_cells"),
        preprocess_cfg.get("n_top_genes"),
        preprocess_cfg.get("hvg_flavor"),
    )
    logger.info(
        "Condition mask: enabled=%s, unk_ratio=%s, keys=%s",
        condition_mask_cfg.get("enabled", False),
        condition_mask_cfg.get("unk_ratio", 0.1),
        "platform,tissue,disease",
    )
    logger.info(
        "Condition vocab: regenerate=%s, cond_vocab_size=%s",
        condition_vocab_cfg.get("regenerate", False),
        data_assets.cond_vocab_size,
    )
    logger.info("=================================")


class PretrainingTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_manager: PretrainingLossManager,
        data_assets: PretrainingAssets,
        checkpoint_manager: CheckpointManager,
        logger: ExperimentLogger,
        runtime: RuntimeContext,
        config: dict[str, Any],
        train_state: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_manager = loss_manager.to(runtime.device)
        self.data_assets = data_assets
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.runtime = runtime
        self.config = config
        self.train_cfg = config["trainer"]
        self.runtime_cfg = config["runtime"]
        self.train_state = train_state or {
            "file_index": 0,
            "epoch": 0,
            "global_step": 0,
        }

    def _autocast_context(self):
        precision_cfg = self.runtime_cfg.get("precision", {})
        autocast_dtype = str(precision_cfg.get("autocast_dtype", "fp32")).lower()
        if self.runtime.device.type != "cuda" or autocast_dtype == "fp32":
            return contextlib.nullcontext()
        if autocast_dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("`runtime.precision.autocast_dtype=bf16` requires CUDA bf16 support.")
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if autocast_dtype == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        raise ValueError(f"Unsupported `runtime.precision.autocast_dtype`: {autocast_dtype}")

    def _clip_grad_norm(self) -> float | None:
        max_norm = self.train_cfg.get("grad_clip_norm")
        if max_norm is None:
            return None

        if isinstance(self.model, FSDP):
            grad_norm = self.model.clip_grad_norm_(float(max_norm))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
        if torch.is_tensor(grad_norm):
            return float(grad_norm.detach().item())
        return float(grad_norm)

    def _log_step(
        self,
        metrics: dict[str, float],
        current_step: int,
        total_steps: int,
        next_log_step: int,
    ) -> int:
        log_every_steps = int(self.train_cfg.get("log_every_steps", 10))
        if log_every_steps <= 0:
            raise ValueError(f"`trainer.log_every_steps` must be positive, got {log_every_steps}.")

        if current_step < next_log_step:
            return next_log_step

        reduced = reduce_scalar_dict(metrics, self.runtime)
        metric_text = ", ".join(
            f"{key}={format_metric_value(value)}" for key, value in reduced.items()
        )
        self.logger.info(
            "[train step=%s/%s] %s",
            int(current_step),
            int(total_steps),
            metric_text,
        )

        while next_log_step <= current_step:
            next_log_step += log_every_steps
        return next_log_step

    def _maybe_save_checkpoint(self, epoch: int) -> None:
        save_every_epochs = int(self.train_cfg.get("save_every_epochs", 1))
        if epoch % save_every_epochs != 0:
            return

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_manager=self.loss_manager,
            train_state=self.train_state,
        )

    def _grad_sync_context(self, should_sync: bool):
        if should_sync or not isinstance(self.model, FSDP):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def train(self) -> None:
        epochs = int(self.train_cfg["epochs"])
        train_paths = self.data_assets.train_paths
        start_file_index = int(self.train_state.get("file_index", 0))
        grad_accum_steps = int(self.config["data"].get("gradient_accumulation_steps", 1))
        if grad_accum_steps <= 0:
            raise ValueError(
                f"`data.gradient_accumulation_steps` must be positive, got {grad_accum_steps}."
            )

        self.model.train()
        self.loss_manager.train()

        for file_offset, path in enumerate(train_paths[start_file_index:], start=start_file_index):
            file_index = file_offset + 1
            if self.runtime.is_main and file_offset > start_file_index:
                self.logger.info("")
            if self.runtime.is_main:
                self.logger.info(
                    "[adata %s/%s] Start processing %s",
                    file_index,
                    len(train_paths),
                    path,
                )

            data_bundle: PretrainingDataBundle | None = None
            try:
                data_bundle = build_data_bundle_for_path(
                    path=path,
                    assets=self.data_assets,
                    config=self.config,
                    runtime=self.runtime,
                    logger=self.logger,
                    file_index=file_index,
                    num_files=len(train_paths),
                )

                start_epoch = int(self.train_state.get("epoch", 0)) if file_offset == start_file_index else 0
                for epoch in range(start_epoch, epochs):
                    epoch_metric_sums: dict[str, float] = {}
                    epoch_steps = 0

                    if self.runtime.is_main:
                        self.logger.info(
                            "[epoch %s/%s] Train on %s cells from %s",
                            epoch + 1,
                            epochs,
                            data_bundle.train_size,
                            path,
                        )

                    if hasattr(data_bundle.train_sampler, "set_epoch"):
                        data_bundle.train_sampler.set_epoch(epoch)

                    progress = tqdm(
                        data_bundle.train_loader,
                        disable=not self.runtime.is_main,
                        desc=f"adata {file_index}/{len(train_paths)} epoch {epoch + 1}/{epochs}",
                    )
                    try:
                        total_forward_steps = len(data_bundle.train_loader) * int(self.runtime.world_size)
                        next_log_step = int(self.train_cfg.get("log_every_steps", 10))
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_metric_sums: dict[str, float] = {}
                        accum_micro_steps = 0

                        num_batches = len(data_bundle.train_loader)
                        for batch_idx, batch in enumerate(progress, start=1):
                            tokens = move_batch_to_device(batch, self.runtime.device)
                            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == num_batches)

                            with self._grad_sync_context(should_sync=should_step):
                                with self._autocast_context():
                                    model_output = self.model(
                                        tokens,
                                        compute_grn=False,
                                        return_factors=True,
                                    )
                                    loss_result = self.loss_manager(
                                        tokens=tokens,
                                        model_output=model_output,
                                        current_epoch=epoch,
                                        global_step=int(self.train_state["global_step"]),
                                    )

                                (loss_result.total / grad_accum_steps).backward()

                            metrics = dict(loss_result.metrics)
                            accum_micro_steps += 1
                            for key, value in metrics.items():
                                accum_metric_sums[key] = accum_metric_sums.get(key, 0.0) + float(value)

                            epoch_steps += 1
                            for key, value in metrics.items():
                                epoch_metric_sums[key] = epoch_metric_sums.get(key, 0.0) + float(value)

                            global_forward_step = batch_idx * int(self.runtime.world_size)
                            next_log_step = self._log_step(
                                metrics,
                                current_step=global_forward_step,
                                total_steps=total_forward_steps,
                                next_log_step=next_log_step,
                            )

                            if should_step:
                                self._clip_grad_norm()
                                self.optimizer.step()
                                _apply_qb_gating_updates(self.model)
                                self.scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)

                                self.train_state["global_step"] += 1
                                accum_metric_sums = {}
                                accum_micro_steps = 0

                        self.train_state["file_index"] = file_offset
                        self.train_state["epoch"] = epoch + 1
                        epoch_metrics = {
                            key: value / max(epoch_steps, 1)
                            for key, value in epoch_metric_sums.items()
                        }
                        epoch_metrics = reduce_scalar_dict(epoch_metrics, self.runtime)
                        self.logger.log_metrics("train_epoch", self.train_state["global_step"], epoch_metrics)

                        self._maybe_save_checkpoint(epoch + 1)
                    finally:
                        progress.close()

                self.train_state["file_index"] = file_offset + 1
                self.train_state["epoch"] = 0
            finally:
                _release_data_bundle(data_bundle, self.runtime)

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_manager=self.loss_manager,
            train_state=self.train_state,
        )


def main() -> None:
    args = parse_args()
    runtime = initialize_distributed()

    try:
        pretrain_config = load_yaml_config(args.pretrain_config)
        model_source = pretrain_config["model_source"]
        source_assets = resolve_model_assets(model_source=model_source, require_model_weights=False)
        sfm_config = load_sfm_config(source_assets.sfm_config)
        config = apply_model_assets_to_runtime_config(
            {
                "model": sfm_config,
                **pretrain_config,
            },
            source_assets,
            require_model_weights=False,
        )

        paths = prepare_experiment_paths(
            runtime=runtime,
            resume_path=args.resume,
        )
        logger = ExperimentLogger(
            name="pretrain",
            paths=paths,
            runtime=runtime,
        )

        regenerate_condition_vocab = bool(
            config["data"].get("condition_vocab", {}).get("regenerate", False)
        )
        checkpoint_assets = materialize_model_package(
            source_assets=source_assets,
            target_dir=paths.model_package_dir,
            include_model_weights=True,
            include_cond_dict=not regenerate_condition_vocab,
            overwrite=args.resume is None,
        )
        config = apply_model_assets_to_runtime_config(
            config,
            checkpoint_assets,
            require_model_weights=False,
        )

        data_assets = build_pretraining_assets(config=config, runtime=runtime)
        model = build_model(
            sfm_config=config["model"],
            data_bundle=PretrainingDataBundle(
                train_loader=None,
                train_sampler=None,
                token_dict=data_assets.token_dict,
                cond_vocab_size=data_assets.cond_vocab_size,
                train_size=0,
                path=data_assets.train_paths[0],
            ),
            assets=checkpoint_assets,
        )

        total_steps = estimate_total_training_steps(
            paths=data_assets.train_paths,
            config=config,
            runtime=runtime,
        )
        loss_manager = PretrainingLossManager(
            config=config,
            token_dict=data_assets.token_dict,
            total_epochs=int(config["trainer"]["epochs"]),
            total_steps=total_steps,
        )

        checkpoint_manager = CheckpointManager(
            paths=paths,
            assets=checkpoint_assets,
            runtime=runtime,
            logger=logger,
            config=config,
            data_assets=data_assets,
        )
        model_state = checkpoint_manager.load_model_weights()
        resume_payload = checkpoint_manager.load_resume_state(args.resume)
        if resume_payload is not None and model_state is None:
            raise FileNotFoundError(
                f"Full resume requires model weights at {checkpoint_assets.sfm_model}, but none were found."
            )

        if model_state is not None:
            model.load_state_dict(model_state)
        if resume_payload is not None:
            loss_manager.load_state_dict(resume_payload["loss_manager_state_dict"])

        model = maybe_wrap_fsdp(
            model=model,
            config=config,
            runtime=runtime,
            sync_module_states=(model_state is not None or resume_payload is not None),
        )
        optimizer = build_optimizer(model=model, config=config)
        scheduler = build_scheduler(
            optimizer=optimizer,
            config=config,
            total_steps=total_steps,
        )
        train_state = _load_resume_states(
            payload=resume_payload,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
        )

        if runtime.is_main:
            logger.info(f"Output directory: {paths.root}")
            logger.info("")
            _log_run_summary(
                logger=logger,
                runtime=runtime,
                config=config,
                data_assets=data_assets,
                total_steps=total_steps,
            )
            logger.info("")

        trainer = PretrainingTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            data_assets=data_assets,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            runtime=runtime,
            config=config,
            train_state=train_state,
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
