from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import os
import warnings

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm

from ..assets import (
    EFM_CONFIG_NAME,
    EFM_DIR_NAME,
    EFM_MODEL_NAME,
    MODELS_DIR_NAME,
    ModelAssets,
    SFM_CONFIG_NAME,
    SFM_DIR_NAME,
    SFM_MODEL_NAME,
    apply_model_assets_to_runtime_config,
    load_model_state_dict,
    load_sfm_config,
    materialize_model_package,
    resolve_model_assets,
    save_json,
    save_model_state_dict,
    write_release_manifest,
)
from ..config import load_yaml_config
from ..checkpoint import _cast_floating_tensors_to_fp32
from ..data import (
    PretrainingAssets,
    PretrainingDataBundle,
    build_data_bundle_for_path,
    build_pretraining_assets,
    estimate_total_training_steps,
)
from ..distributed import (
    RuntimeContext,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    move_batch_to_device,
    reduce_scalar_dict,
)
from ..experiment import ExperimentLogger, prepare_experiment_paths
from ..losses import EFMLoss
from ..models import EFM, build_efm_targets, reorder_gene_aligned_tokens
from ..models.wrapper import ModelWrapper
from .builders import (
    _resolve_torch_dtype,
    build_model,
    build_optimizer,
    build_scheduler,
    maybe_wrap_fsdp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain EFM with frozen online SFM ordering.")
    parser.add_argument(
        "--efm-pretrain-config",
        "--pretrain-config",
        dest="efm_pretrain_config",
        default="configs/efm_pretrain.yaml",
    )
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _release_data_bundle(data_bundle: PretrainingDataBundle | None, runtime: RuntimeContext) -> None:
    if data_bundle is None:
        return

    train_loader = getattr(data_bundle, "train_loader", None)
    iterator = getattr(train_loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()

    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "close"):
        dataset.close()

    data_bundle.train_loader = None
    data_bundle.train_sampler = None
    del train_loader
    gc.collect()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()


def _autocast_context(runtime: RuntimeContext, runtime_cfg: dict[str, Any]):
    precision_cfg = runtime_cfg.get("precision", {})
    autocast_dtype = str(precision_cfg.get("autocast_dtype", "fp32")).lower()
    if runtime.device.type != "cuda" or autocast_dtype == "fp32":
        return contextlib.nullcontext()
    if autocast_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("`runtime.precision.autocast_dtype=bf16` requires CUDA bf16 support.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if autocast_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported `runtime.precision.autocast_dtype`: {autocast_dtype}")


def _unwrap_sfm(model: ModelWrapper) -> torch.nn.Module:
    try:
        return model.foundation_modules["sfm"]
    except KeyError as exc:
        raise KeyError("Frozen SFM wrapper must contain foundation module 'sfm'.") from exc


def _copy_compatible_module_state(
    *,
    target: torch.nn.Module,
    source: torch.nn.Module,
    module_name: str,
) -> tuple[int, int]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    incompatible: list[str] = []

    for key, value in source_state.items():
        if key not in target_state:
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            incompatible.append(
                f"{module_name}.{key}: source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
            continue
        compatible[key] = value.detach().to(dtype=target_state[key].dtype)

    if incompatible:
        details = "\n".join(incompatible[:20])
        raise ValueError(
            f"Cannot initialize EFM {module_name} from SFM because compatible key shapes differ:\n{details}"
        )

    target.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)


def _initialize_efm_from_sfm(efm: EFM, frozen_sfm: ModelWrapper, logger: ExperimentLogger) -> None:
    sfm = _unwrap_sfm(frozen_sfm)
    if getattr(efm, "embed_dim", None) != getattr(sfm, "embed_dim", None):
        raise ValueError(
            "`efm.init_from_sfm=true` requires matching embed_dim, "
            f"got EFM={getattr(efm, 'embed_dim', None)} and SFM={getattr(sfm, 'embed_dim', None)}."
        )

    embedding_copied, embedding_total = _copy_compatible_module_state(
        target=efm.embedding,
        source=sfm.embedding,
        module_name="embedding",
    )
    backbone_copied, backbone_total = _copy_compatible_module_state(
        target=efm.backbone,
        source=sfm.backbone,
        module_name="backbone",
    )
    logger.info(
        "Initialized EFM from frozen SFM: embedding tensors %s/%s, backbone tensors %s/%s",
        embedding_copied,
        embedding_total,
        backbone_copied,
        backbone_total,
    )


def _resolve_eos_token_id(token_dict) -> int:
    mask = token_dict["gene_id"].astype(str).str.lower() == "<eos>"
    if not bool(mask.any()):
        mask = token_dict["gene_symbol"].astype(str).str.lower() == "<eos>"
    if not bool(mask.any()):
        raise ValueError("`vocab.json` must contain an <eos> token for EFM pretraining.")
    return int(token_dict.loc[mask, "token_index"].iloc[0])


def _build_efm(
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    assets: ModelAssets,
) -> EFM:
    efm_kwargs = copy.deepcopy(config["efm"])
    efm_kwargs.pop("init_from_sfm", None)
    if "attention_backend" in config.get("runtime", {}):
        efm_kwargs["attention_backend"] = config["runtime"]["attention_backend"]
    efm_kwargs.pop("gene_embedding_ckpt", None)
    configured_cond_vocab_size = efm_kwargs.pop("cond_vocab_size", None)
    if configured_cond_vocab_size is not None and int(configured_cond_vocab_size) != int(data_assets.cond_vocab_size):
        raise ValueError(
            "Mismatched `efm.cond_vocab_size` between config "
            f"({configured_cond_vocab_size}) and data assets ({data_assets.cond_vocab_size})."
        )
    return EFM(
        token_dict=data_assets.token_dict,
        cond_vocab_size=data_assets.cond_vocab_size,
        gene_embedding_ckpt=str(assets.vocab_tensors),
        **efm_kwargs,
    )


def _load_frozen_sfm(
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    assets: ModelAssets,
    runtime: RuntimeContext,
    logger: ExperimentLogger,
) -> ModelWrapper:
    frozen_sfm = build_model(
        sfm_config=config["model"],
        data_bundle=PretrainingDataBundle(
            train_loader=None,
            train_sampler=None,
            token_dict=data_assets.token_dict,
            cond_vocab_size=data_assets.cond_vocab_size,
            train_size=0,
            path=data_assets.train_paths[0],
        ),
        assets=assets,
        runtime_config=config.get("runtime", {}),
    )
    checkpoint_path = config.get("frozen_sfm", {}).get("checkpoint_path") or assets.sfm_model
    state_dict = load_model_state_dict(checkpoint_path)
    frozen_sfm.load_state_dict(state_dict, strict=True)

    precision_cfg = config.get("runtime", {}).get("precision", {})
    model_dtype = _resolve_torch_dtype(precision_cfg.get("model_dtype", "fp32"))
    frozen_sfm = frozen_sfm.to(device=runtime.device, dtype=model_dtype)
    frozen_sfm.eval()
    frozen_sfm.requires_grad_(False)
    logger.info("Loaded frozen SFM weights from %s", checkpoint_path)
    return frozen_sfm


def _normalize_checkpoint_frequency(config: dict[str, Any]) -> str:
    frequency = str(
        config.get("trainer", {}).get("checkpoint_frequency", "epoch")
    ).strip().lower()
    frequency = {
        "file": "adata",
        "dataset": "adata",
    }.get(frequency, frequency)
    if frequency not in {"epoch", "adata"}:
        raise ValueError(
            "`trainer.checkpoint_frequency` must be either 'epoch' or 'adata', "
            f"got {frequency!r}."
        )
    return frequency


def _state_dicts_for_save(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(model, FSDP):
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
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
                optim_state_dict_config=optim_state_cfg,
            ):
                return model.state_dict(), FSDP.optim_state_dict(model, optimizer)
    return model.state_dict(), optimizer.state_dict()


def _load_efm_resume_state(
    *,
    resume_path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: ExperimentLogger,
) -> dict[str, Any] | None:
    if resume_path is None:
        return None

    resolved = Path(resume_path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu")
    module = payload.get("module")
    if module not in {None, "efm"}:
        raise ValueError(
            f"Cannot resume EFM pretraining from {resolved}: train-state module is {module!r}."
        )

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
    logger.info("Loaded EFM train state from %s", resolved)
    return payload.get("train_state")


def _save_efm_package(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: dict[str, Any],
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    assets: ModelAssets,
    runtime: RuntimeContext,
    logger: ExperimentLogger,
    checkpoint_dir: Path,
    resume_state_file: Path,
) -> None:
    model_state, optim_state = _state_dicts_for_save(model, optimizer)
    if runtime.is_main:
        efm_dir = checkpoint_dir / MODELS_DIR_NAME / EFM_DIR_NAME
        model_path = save_model_state_dict(efm_dir / EFM_MODEL_NAME, model_state)
        config_payload = {
            "efm": copy.deepcopy(config["efm"]),
            "loss": copy.deepcopy(config.get("loss", {})),
            "runtime": copy.deepcopy(config.get("runtime", {})),
            "data": {
                "max_length": int(config["data"]["max_length"]),
                "cond_vocab_size": int(data_assets.cond_vocab_size),
            },
            "frozen_sfm": {
                "model_source": assets.model_source,
                "config": f"{MODELS_DIR_NAME}/{SFM_DIR_NAME}/{SFM_CONFIG_NAME}",
                "weights": f"{MODELS_DIR_NAME}/{SFM_DIR_NAME}/{SFM_MODEL_NAME}",
            },
        }
        save_json(efm_dir / EFM_CONFIG_NAME, config_payload)
        write_release_manifest(
            resolve_model_assets(checkpoint_dir, require_resources=False)
        )
        torch.save(
            {
                "module": "efm",
                "optimizer_state_dict": _cast_floating_tensors_to_fp32(optim_state),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_state": train_state,
                "model_weights_path": str(model_path),
            },
            resume_state_file,
        )
        try:
            (checkpoint_dir / "efm_metadata.json").unlink()
        except FileNotFoundError:
            pass
        logger.info("EFM weights saved to %s", model_path)
        logger.info("EFM config saved to %s", efm_dir / EFM_CONFIG_NAME)
        logger.info(
            "Train state saved to %s (global_step=%s, epoch=%s, file_index=%s)",
            resume_state_file,
            int(train_state.get("global_step", 0)),
            int(train_state.get("epoch", 0)),
            int(train_state.get("file_index", 0)),
        )
    del model_state, optim_state
    gc.collect()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    barrier()


class EFMPretrainingTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        frozen_sfm: ModelWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: EFMLoss,
        data_assets: PretrainingAssets,
        logger: ExperimentLogger,
        runtime: RuntimeContext,
        config: dict[str, Any],
        assets: ModelAssets,
        checkpoint_dir: Path,
        resume_state_file: Path,
        train_state: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.frozen_sfm = frozen_sfm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(runtime.device)
        self.data_assets = data_assets
        self.logger = logger
        self.runtime = runtime
        self.config = config
        self.assets = assets
        self.checkpoint_dir = checkpoint_dir
        self.resume_state_file = resume_state_file
        self.train_cfg = config["trainer"]
        self.train_state = train_state or {
            "epoch": 0,
            "file_index": 0,
            "global_step": 0,
            "schedule": "epoch_then_file",
        }
        self.eos_token_id = _resolve_eos_token_id(data_assets.token_dict)

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

    def _grad_sync_context(self, should_sync: bool):
        if should_sync or not isinstance(self.model, FSDP):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def _save_checkpoint(self) -> None:
        _save_efm_package(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_state=self.train_state,
            config=self.config,
            data_assets=self.data_assets,
            assets=self.assets,
            runtime=self.runtime,
            logger=self.logger,
            checkpoint_dir=self.checkpoint_dir,
            resume_state_file=self.resume_state_file,
        )

    def train(self) -> None:
        epochs = int(self.train_cfg["epochs"])
        train_paths = self.data_assets.train_paths
        start_epoch = int(self.train_state.get("epoch", 0))
        start_file_index = int(self.train_state.get("file_index", 0))
        checkpoint_frequency = _normalize_checkpoint_frequency(self.config)
        final_state_saved = False
        grad_accum_steps = int(self.config["data"].get("gradient_accumulation_steps", 1))
        if grad_accum_steps <= 0:
            raise ValueError(
                f"`data.gradient_accumulation_steps` must be positive, got {grad_accum_steps}."
            )
        if start_file_index >= len(train_paths):
            start_epoch += 1
            start_file_index = 0
        if start_epoch >= epochs:
            if self.runtime.is_main:
                self.logger.info("EFM pretraining already completed: epoch=%s/%s", start_epoch, epochs)
            return

        self.model.train()
        self.loss_fn.train()
        self.frozen_sfm.eval()

        for epoch in range(start_epoch, epochs):
            file_start_index = start_file_index if epoch == start_epoch else 0
            if self.runtime.is_main:
                self.logger.info("")
                self.logger.info("[epoch %s/%s] Start EFM pretraining pass", epoch + 1, epochs)

            for file_offset, path in enumerate(train_paths[file_start_index:], start=file_start_index):
                file_index = file_offset + 1
                if self.runtime.is_main:
                    self.logger.info(
                        "[epoch %s/%s][adata %s/%s] Start processing %s",
                        epoch + 1,
                        epochs,
                        file_index,
                        len(train_paths),
                        path,
                    )

                data_bundle: PretrainingDataBundle | None = None
                save_after_data_release = False
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
                    if hasattr(data_bundle.train_sampler, "set_epoch"):
                        data_bundle.train_sampler.set_epoch(epoch)

                    epoch_metric_sums: dict[str, float] = {}
                    epoch_steps = 0
                    progress = None
                    try:
                        progress = tqdm(
                            data_bundle.train_loader,
                            disable=not self.runtime.is_main,
                            desc=f"efm epoch {epoch + 1}/{epochs} adata {file_index}/{len(train_paths)}",
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        num_batches = len(data_bundle.train_loader)
                        for batch_idx, batch in enumerate(progress, start=1):
                            tokens = move_batch_to_device(batch, self.runtime.device)
                            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == num_batches)

                            with torch.no_grad():
                                with _autocast_context(self.runtime, self.config["runtime"]):
                                    sfm_output = self.frozen_sfm(
                                        tokens,
                                        compute_order={"sfm": True},
                                        compute_grn=False,
                                        return_factors=False,
                                    )
                                gene_order = sfm_output.foundations["sfm"].gene_order
                                if gene_order is None:
                                    raise RuntimeError("Frozen SFM did not return `gene_order`.")
                                reordered_tokens = reorder_gene_aligned_tokens(tokens, gene_order)
                                target_ids, target_expr, valid_mask = build_efm_targets(
                                    reordered_tokens,
                                    eos_token_id=self.eos_token_id,
                                )

                            with self._grad_sync_context(should_sync=should_step):
                                with _autocast_context(self.runtime, self.config["runtime"]):
                                    output = self.model(reordered_tokens)
                                    loss_result = self.loss_fn(
                                        output=output,
                                        target_ids=target_ids,
                                        target_expression=target_expr,
                                        valid_mask=valid_mask,
                                    )
                                (loss_result.total / grad_accum_steps).backward()

                            metrics = dict(loss_result.metrics)
                            epoch_steps += 1
                            for key, value in metrics.items():
                                epoch_metric_sums[key] = epoch_metric_sums.get(key, 0.0) + float(value)

                            if should_step:
                                self._clip_grad_norm()
                                self.optimizer.step()
                                self.scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                self.train_state["global_step"] += 1

                            del loss_result, output, tokens, reordered_tokens, target_ids, target_expr, valid_mask

                        epoch_metrics = {
                            key: value / max(epoch_steps, 1)
                            for key, value in epoch_metric_sums.items()
                        }
                        epoch_metrics = reduce_scalar_dict(epoch_metrics, self.runtime)
                        self.logger.log_metrics(
                            "train_adata",
                            int(self.train_state["global_step"]),
                            epoch_metrics,
                        )
                    finally:
                        if progress is not None:
                            progress.close()
                            del progress

                    self.train_state["epoch"] = epoch
                    self.train_state["file_index"] = file_offset + 1
                    save_after_data_release = checkpoint_frequency == "adata"
                finally:
                    _release_data_bundle(data_bundle, self.runtime)

                if save_after_data_release:
                    self._save_checkpoint()

            self.train_state["epoch"] = epoch + 1
            self.train_state["file_index"] = 0
            if checkpoint_frequency == "epoch":
                self._save_checkpoint()
                final_state_saved = epoch + 1 == epochs

        if not final_state_saved:
            self._save_checkpoint()


def _log_run_summary(
    logger: ExperimentLogger,
    runtime: RuntimeContext,
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    total_steps: int,
) -> None:
    logger.info("========== EFM Run Summary ==========")
    logger.info(
        "Runtime: distributed=%s, world_size=%s, attention_backend=%s",
        runtime.distributed,
        runtime.world_size,
        config["runtime"].get("attention_backend", config.get("efm", {}).get("attention_backend", "fa4")),
    )
    logger.info(
        "Data: num_adata=%s, batch_size=%s, gradient_accumulation_steps=%s, max_length=%s, cond_vocab_size=%s",
        len(data_assets.train_paths),
        int(config["data"]["batch_size"]),
        int(config["data"].get("gradient_accumulation_steps", 1)),
        int(config["data"]["max_length"]),
        int(data_assets.cond_vocab_size),
    )
    logger.info(
        "EFM: embed_dim=%s, layers=%s, heads=%s, init_from_sfm=%s",
        config["efm"].get("embed_dim"),
        config["efm"].get("num_layers"),
        config["efm"].get("num_heads"),
        config["efm"].get("init_from_sfm", True),
    )
    logger.info(
        "Optimizer: name=%s, lr=%s, estimated_total_steps=%s",
        config["optimizer"].get("name", "adamw"),
        config["optimizer"]["lr"],
        total_steps,
    )
    logger.info("Loss: lambda_exp=%s", config.get("loss", {}).get("lambda_exp", 1.0))
    logger.info(
        "Trainer: epochs=%s, checkpoint_frequency=%s",
        config["trainer"]["epochs"],
        config["trainer"].get("checkpoint_frequency", "epoch"),
    )
    logger.info("=====================================")


def main() -> None:
    args = parse_args()
    runtime = initialize_distributed()

    try:
        pretrain_config = load_yaml_config(args.efm_pretrain_config)
        model_source = pretrain_config["model_source"]
        source_assets = resolve_model_assets(model_source=model_source, require_model_weights=True)
        sfm_config = load_sfm_config(source_assets.sfm_config)
        config = apply_model_assets_to_runtime_config(
            {
                **pretrain_config,
                "model": sfm_config,
            },
            source_assets,
            require_model_weights=True,
        )
        _normalize_checkpoint_frequency(config)

        paths = prepare_experiment_paths(runtime=runtime, resume_path=args.resume)
        paths = replace(
            paths,
            log_file=paths.logs / "efm_pretrain.log",
        )
        logger = ExperimentLogger(name="efm_pretrain", paths=paths, runtime=runtime)
        if runtime.is_main:
            logger.info(
                "Launch environment: CUDA_VISIBLE_DEVICES=%s, NCCL_NET=%s, NCCL_SHM_DISABLE=%s, NCCL_P2P_DISABLE=%s",
                os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                os.environ.get("NCCL_NET", "<unset>"),
                os.environ.get("NCCL_SHM_DISABLE", "<unset>"),
                os.environ.get("NCCL_P2P_DISABLE", "<unset>"),
            )
            if args.resume is None:
                paths.resume_state_file.unlink(missing_ok=True)
            materialize_model_package(
                source_assets=source_assets,
                target_dir=paths.checkpoints,
                include_model_weights=True,
                include_efm_weights=False,
                include_cond_dict=True,
                include_resources=False,
                overwrite=args.resume is None,
            )
        barrier()

        checkpoint_assets = resolve_model_assets(
            model_source=paths.checkpoints,
            require_model_weights=True,
            require_resources=False,
        )
        data_assets = build_pretraining_assets(config=config, runtime=runtime)

        frozen_sfm = _load_frozen_sfm(
            config=config,
            data_assets=data_assets,
            assets=source_assets,
            runtime=runtime,
            logger=logger,
        )
        efm = _build_efm(config=config, data_assets=data_assets, assets=source_assets)
        if args.resume is not None:
            if not checkpoint_assets.efm_model.exists():
                raise FileNotFoundError(
                    f"Full EFM resume requires model weights at {checkpoint_assets.efm_model}."
                )
            efm.load_state_dict(load_model_state_dict(checkpoint_assets.efm_model), strict=True)
            logger.info("Loaded resume EFM weights from %s", checkpoint_assets.efm_model)
        elif bool(config.get("efm", {}).get("init_from_sfm", True)):
            _initialize_efm_from_sfm(efm, frozen_sfm, logger)
        efm = maybe_wrap_fsdp(
            model=efm,
            config=config,
            runtime=runtime,
            sync_module_states=runtime.distributed,
        )

        total_steps = estimate_total_training_steps(
            paths=data_assets.train_paths,
            config=config,
            runtime=runtime,
        )
        optimizer = build_optimizer(model=efm, config=config)
        scheduler = build_scheduler(
            optimizer=optimizer,
            config=config,
            total_steps=total_steps,
        )
        train_state = _load_efm_resume_state(
            resume_path=args.resume,
            model=efm,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
        )
        loss_fn = EFMLoss(lambda_exp=float(config.get("loss", {}).get("lambda_exp", 1.0)))

        if runtime.is_main:
            logger.info("Output directory: %s", paths.root)
            logger.info("")
            _log_run_summary(
                logger=logger,
                runtime=runtime,
                config=config,
                data_assets=data_assets,
                total_steps=total_steps,
            )
            logger.info("")

        trainer = EFMPretrainingTrainer(
            model=efm,
            frozen_sfm=frozen_sfm,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            data_assets=data_assets,
            logger=logger,
            runtime=runtime,
            config=config,
            assets=source_assets,
            checkpoint_dir=paths.checkpoints,
            resume_state_file=paths.resume_state_file,
            train_state=train_state,
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
