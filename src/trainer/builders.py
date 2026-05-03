from __future__ import annotations

import math
from typing import Any

import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from ..assets import ModelAssets
from ..data import PretrainingDataBundle
from ..models.heads.vgae import VGAE
from ..models.sfm import SFM
from ..models.wrapper import ModelWrapper
from ..distributed import RuntimeContext


def _resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype_map[normalized]


def build_model(
    sfm_config: dict[str, Any],
    data_bundle: PretrainingDataBundle,
    assets: ModelAssets,
    *,
    runtime_config: dict[str, Any] | None = None,
) -> ModelWrapper:
    sfm_kwargs = dict(sfm_config["sfm"])
    configured_cond_vocab_size = sfm_kwargs.pop("cond_vocab_size", None)
    if configured_cond_vocab_size is not None and int(configured_cond_vocab_size) != int(data_bundle.cond_vocab_size):
        raise ValueError(
            "Mismatched `cond_vocab_size` between model config "
            f"({configured_cond_vocab_size}) and data bundle ({data_bundle.cond_vocab_size})."
        )
    sfm_kwargs.pop("gene_embedding_ckpt", None)
    if runtime_config is not None and "attention_backend" in runtime_config:
        sfm_kwargs["attention_backend"] = runtime_config["attention_backend"]

    sfm_module = SFM(
        token_dict=data_bundle.token_dict,
        cond_vocab_size=data_bundle.cond_vocab_size,
        gene_embedding_ckpt=str(assets.vocab_tensors),
        **sfm_kwargs,
    )
    vgae_head = VGAE(**sfm_config["vgae"])
    return ModelWrapper(
        foundation_modules={"sfm": sfm_module},
        head_modules={"vgae": vgae_head},
        head_to_foundation={"vgae": "sfm"},
    )


def maybe_wrap_fsdp(
    model: torch.nn.Module,
    config: dict[str, Any],
    runtime: RuntimeContext,
    sync_module_states: bool,
) -> torch.nn.Module:
    precision_cfg = config["runtime"].get("precision", {})
    model_dtype = _resolve_torch_dtype(precision_cfg.get("model_dtype", "fp32"))
    fsdp_cfg = config["runtime"].get("fsdp", {})
    enabled = bool(fsdp_cfg.get("enabled", runtime.distributed))
    if not enabled or not runtime.distributed or runtime.device.type != "cuda":
        return model.to(device=runtime.device, dtype=model_dtype)

    sharding_name = str(fsdp_cfg.get("sharding_strategy", "FULL_SHARD")).upper()
    sharding_strategy = getattr(ShardingStrategy, sharding_name)
    model = model.to(device=runtime.device, dtype=model_dtype)
    return FSDP(
        model,
        device_id=runtime.device,
        sharding_strategy=sharding_strategy,
        use_orig_params=bool(fsdp_cfg.get("use_orig_params", True)),
        sync_module_states=sync_module_states,
        limit_all_gathers=bool(fsdp_cfg.get("limit_all_gathers", True)),
    )


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = config["optimizer"]
    name = optimizer_cfg.get("name", "adamw").lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_cfg = config["scheduler"]
    name = scheduler_cfg.get("name", "cosine_with_warmup").lower()
    warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.03))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    if total_steps <= 0:
        raise ValueError(f"`total_steps` must be positive, got {total_steps}.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"`warmup_ratio` must be in [0, 1], got {warmup_ratio}.")

    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / max(warmup_steps, 1)

        if name == "constant":
            return 1.0
        if name != "cosine_with_warmup":
            raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")

        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_total_training_steps(config: dict[str, Any], steps_per_epoch: int) -> int:
    return int(config["trainer"]["epochs"]) * int(steps_per_epoch)
