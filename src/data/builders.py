from __future__ import annotations

import random

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import anndata as ad
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .collator import ScBatchCollator
from .dataset import PreprocessedScDataset
from .tokenizer import CondTokenizer, ExprTokenizer, GeneTokenizer, ScTokenizer
from ..assets import load_table_json, load_vocab_json, save_table_json
from ..distributed import RuntimeContext, broadcast_object

if TYPE_CHECKING:
    from .preprocess import ScPreprocessor


@dataclass
class PretrainingAssets:
    train_paths: list[Path]
    token_dict: pd.DataFrame
    tokenizer: ScTokenizer
    preprocessor: Optional[ScPreprocessor]
    gene_key: str | None
    cond_vocab_size: int
    collator: ScBatchCollator


@dataclass
class PretrainingDataBundle:
    train_loader: DataLoader
    train_sampler: Any
    token_dict: pd.DataFrame
    cond_vocab_size: int
    train_size: int
    path: Path


def resolve_train_paths(inputs: str | Path | list[str] | list[Path] | None) -> list[Path]:
    if inputs is None:
        return []
    if isinstance(inputs, (str, Path)):
        raw_inputs = [inputs]
    else:
        raw_inputs = list(inputs)

    resolved: list[Path] = []
    for raw_path in raw_inputs:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        if path.is_dir():
            h5ad_paths = sorted(path.rglob("*.h5ad"))
            if not h5ad_paths:
                raise ValueError(f"No `.h5ad` files found under directory: {path}")
            resolved.extend(h5ad_paths)
            continue

        if path.suffix != ".h5ad":
            raise ValueError(
                f"`train_paths` only supports `.h5ad` files or directories containing them, got: {path}"
            )
        resolved.append(path)

    return list(dict.fromkeys(sorted(resolved)))


def _shuffle_train_paths(
    train_paths: list[Path],
    runtime: RuntimeContext,
    enabled: bool,
    seed: int,
) -> list[Path]:
    if not enabled or len(train_paths) <= 1:
        return train_paths

    payload: list[str] | None = None
    if not runtime.distributed or runtime.is_main:
        shuffled = list(train_paths)
        random.Random(int(seed)).shuffle(shuffled)
        payload = [str(path) for path in shuffled]

    resolved = broadcast_object(payload, src=0)
    return [Path(path_str) for path_str in resolved]


def _load_table(path: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        return load_table_json(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        prefix = handle.read(256)
    stripped_prefix = prefix.lstrip()
    if stripped_prefix.startswith("[") or stripped_prefix.startswith("{"):
        return load_table_json(resolved)

    try:
        return pd.read_csv(resolved)
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse table at {resolved}. JSON tables must contain an array of records."
        ) from exc


def _build_preprocessor(config: dict[str, Any], token_dict: pd.DataFrame) -> Optional[ScPreprocessor]:
    preprocess_cfg = config.get("preprocess", {})
    if not preprocess_cfg.get("enabled", False):
        return None

    from .preprocess import ScPreprocessor

    kwargs = {key: value for key, value in preprocess_cfg.items() if key != "enabled"}
    kwargs["token_dict"] = token_dict
    kwargs["gene_key"] = config.get("gene_key")
    return ScPreprocessor(**kwargs)


def _condition_vocab_regenerate_enabled(config: dict[str, Any]) -> bool:
    condition_vocab_cfg = config.get("condition_vocab", {})
    return bool(condition_vocab_cfg.get("regenerate", False))


def _resolve_condition_vocab_table(config: dict[str, Any]) -> pd.DataFrame | None:
    cond_dict_path = config.get("cond_dict_path")
    regenerate = _condition_vocab_regenerate_enabled(config)
    resolved = Path(cond_dict_path).expanduser().resolve() if cond_dict_path else None

    if regenerate:
        if resolved is not None and resolved.exists():
            return _load_table(str(resolved))
        return None

    if not cond_dict_path:
        raise ValueError(
            "`data.cond_dict_path` is required when `data.condition_vocab.regenerate` is false."
        )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Condition vocab not found at {resolved}. Set `data.condition_vocab.regenerate: true` "
            "to rebuild it during pretraining."
        )
    return _load_table(str(resolved))


def _build_tokenizer(config: dict[str, Any], token_dict: pd.DataFrame) -> ScTokenizer:
    cond_dict = _resolve_condition_vocab_table(config)
    human_tfs = _load_table(config["human_tfs_path"])
    mouse_tfs = _load_table(config["mouse_tfs_path"])
    condition_mask_cfg = config.get("condition_mask", {})

    gene_tokenizer = GeneTokenizer(
        token_dict=token_dict,
        species_key=config.get("species_key"),
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        max_length=config["max_length"],
    )
    expr_tokenizer = ExprTokenizer(
        max_length=config["max_length"],
        pad_value=config.get("expr_pad_value", 0.0),
    )
    cond_tokenizer = CondTokenizer(
        cond_dict=cond_dict,
        platform_key=config.get("platform_key"),
        species_key=config.get("species_key"),
        tissue_key=config.get("tissue_key"),
        disease_key=config.get("disease_key"),
        mask_unknown_enabled=condition_mask_cfg.get("enabled", False),
        mask_unknown_ratio=condition_mask_cfg.get("unk_ratio", 0.1),
    )
    return ScTokenizer(
        token_dict=token_dict,
        max_length=config["max_length"],
        gene_tokenizer=gene_tokenizer,
        expr_tokenizer=expr_tokenizer,
        cond_tokenizer=cond_tokenizer,
    )


def _fit_condition_vocab(paths: list[Path], tokenizer: ScTokenizer) -> None:
    if not tokenizer.cond_tokenizer.allow_new_conditions:
        return
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        try:
            tokenizer.cond_tokenizer.fit_obs(adata.obs)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()


def _save_condition_vocab(config: dict[str, Any], tokenizer: ScTokenizer, runtime: RuntimeContext) -> None:
    cond_dict_path = config.get("cond_dict_path")
    if not cond_dict_path or not runtime.is_main or not tokenizer.cond_tokenizer.allow_new_conditions:
        return

    output_path = Path(cond_dict_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_table_json(output_path, tokenizer.cond_tokenizer.cond_dict)


def build_evaluation_assets(config: dict[str, Any], runtime: RuntimeContext) -> PretrainingAssets:
    eval_config = dict(config)
    data_cfg = dict(config["data"])
    data_cfg["condition_vocab"] = {"regenerate": False}
    data_cfg["condition_mask"] = {"enabled": False, "unk_ratio": 0.0}
    eval_config["data"] = data_cfg
    return build_pretraining_assets(config=eval_config, runtime=runtime)


def build_pretraining_assets(config: dict[str, Any], runtime: RuntimeContext) -> PretrainingAssets:
    data_cfg = config["data"]
    token_dict = load_vocab_json(data_cfg["token_dict_path"])
    tokenizer = _build_tokenizer(data_cfg, token_dict=token_dict)
    preprocessor = _build_preprocessor(data_cfg, token_dict=token_dict)
    train_paths = resolve_train_paths(data_cfg.get("train_paths"))
    if not train_paths:
        raise ValueError("At least one `.h5ad` file is required for training.")
    train_paths = _shuffle_train_paths(
        train_paths=train_paths,
        runtime=runtime,
        enabled=bool(data_cfg.get("shuffle_train_paths", False)),
        seed=int(data_cfg.get("shuffle_train_paths_seed", 0)),
    )
    _fit_condition_vocab(train_paths, tokenizer)
    _save_condition_vocab(data_cfg, tokenizer, runtime)
    gene_key = data_cfg.get("gene_key")
    return PretrainingAssets(
        train_paths=train_paths,
        token_dict=token_dict,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        gene_key=gene_key,
        cond_vocab_size=tokenizer.cond_tokenizer.next_index,
        collator=ScBatchCollator(),
    )


def _read_adata(path: Path):
    return ad.read_h5ad(path)


def _build_dataset_for_path(
    path: Path,
    tokenizer: ScTokenizer,
    gene_key: str | None,
    preprocessor: Optional[ScPreprocessor],
    logger: Any | None = None,
    file_index: int | None = None,
    num_files: int | None = None,
) -> PreprocessedScDataset:
    dataset = PreprocessedScDataset(
        adata=_read_adata(path),
        tokenizer=tokenizer,
        gene_key=gene_key,
        preprocessor=preprocessor,
    )
    if logger is not None:
        prefix = ""
        if file_index is not None and num_files is not None:
            prefix = f"[adata {file_index}/{num_files}] "
        logger.info(
            "%sPreprocess summary: %s, cells %s -> %s, genes %s -> %s",
            prefix,
            path,
            dataset.raw_n_obs,
            dataset.processed_n_obs,
            dataset.raw_n_vars,
            dataset.processed_n_vars,
        )
    return dataset


def _build_dataloader_kwargs(data_cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": bool(data_cfg.get("drop_last", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return kwargs


def build_data_bundle_for_path(
    path: Path,
    assets: PretrainingAssets,
    config: dict[str, Any],
    runtime: RuntimeContext,
    logger: Any | None = None,
    file_index: int | None = None,
    num_files: int | None = None,
) -> PretrainingDataBundle:
    data_cfg = config["data"]
    train_dataset = _build_dataset_for_path(
        path=path,
        tokenizer=assets.tokenizer,
        gene_key=assets.gene_key,
        preprocessor=assets.preprocessor,
        logger=logger,
        file_index=file_index,
        num_files=num_files,
    )
    if runtime.distributed:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            drop_last=bool(data_cfg.get("drop_last", False)),
        )
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        sampler=train_sampler,
        shuffle=False,
        collate_fn=assets.collator,
        **_build_dataloader_kwargs(data_cfg),
    )
    return PretrainingDataBundle(
        train_loader=train_loader,
        train_sampler=train_sampler,
        token_dict=assets.token_dict,
        cond_vocab_size=assets.cond_vocab_size,
        train_size=len(train_dataset),
        path=path,
    )


def _count_cells(path: Path) -> int:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def estimate_total_training_steps(
    paths: list[Path],
    config: dict[str, Any],
    runtime: RuntimeContext,
) -> int:
    data_cfg = config["data"]
    batch_size = int(data_cfg["batch_size"])
    drop_last = bool(data_cfg.get("drop_last", False))
    grad_accum_steps = int(data_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError(
            f"`data.gradient_accumulation_steps` must be positive, got {grad_accum_steps}."
        )

    steps_per_epoch = 0
    for path in paths:
        num_cells = _count_cells(path)
        if runtime.distributed:
            if drop_last:
                samples_per_rank = num_cells // runtime.world_size
            else:
                samples_per_rank = (num_cells + runtime.world_size - 1) // runtime.world_size
        else:
            samples_per_rank = num_cells

        if drop_last:
            file_steps = samples_per_rank // batch_size
        else:
            file_steps = (samples_per_rank + batch_size - 1) // batch_size
        update_steps = (file_steps + grad_accum_steps - 1) // grad_accum_steps
        steps_per_epoch += update_steps

    return int(config["trainer"]["epochs"]) * int(steps_per_epoch)
