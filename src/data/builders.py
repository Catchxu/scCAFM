from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import anndata as ad
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

from .collator import ScBatchCollator
from .dataset import ScDataset
from .preprocess import ScPreprocessor
from .tokenizer import CondTokenizer, ExprTokenizer, GeneTokenizer, ScTokenizer
from ..distributed import RuntimeContext


class PreprocessedScDataset(ScDataset):
    def __init__(
        self,
        adata,
        tokenizer: ScTokenizer,
        gene_key: Optional[str] = None,
        preprocessor: Optional[ScPreprocessor] = None,
    ) -> None:
        self.raw_n_obs = int(adata.n_obs)
        self.raw_n_vars = int(adata.n_vars)
        processed = preprocessor(adata) if preprocessor is not None else adata
        self.processed_n_obs = int(processed.n_obs)
        self.processed_n_vars = int(processed.n_vars)
        super().__init__(adata=processed, tokenizer=tokenizer, gene_key=gene_key)


@dataclass
class PretrainingAssets:
    train_paths: list[Path]
    token_dict: pd.DataFrame
    tokenizer: ScTokenizer
    preprocessor: Optional[ScPreprocessor]
    gene_key: str | None
    cond_vocab_size: int


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

    return sorted(dict.fromkeys(resolved))


def _load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser().resolve())


def _build_preprocessor(config: dict[str, Any], token_dict: pd.DataFrame) -> Optional[ScPreprocessor]:
    preprocess_cfg = config.get("preprocess", {})
    if not preprocess_cfg.get("enabled", False):
        return None

    kwargs = {key: value for key, value in preprocess_cfg.items() if key != "enabled"}
    kwargs["token_dict"] = token_dict
    kwargs["gene_key"] = config.get("gene_key")
    return ScPreprocessor(**kwargs)


def _build_tokenizer(config: dict[str, Any], token_dict: pd.DataFrame) -> ScTokenizer:
    cond_dict_path = config.get("cond_dict_path")
    cond_dict = _load_table(cond_dict_path) if cond_dict_path else None
    human_tfs = _load_table(config["human_tfs_path"])
    mouse_tfs = _load_table(config["mouse_tfs_path"])

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
    )
    return ScTokenizer(
        token_dict=token_dict,
        max_length=config["max_length"],
        gene_tokenizer=gene_tokenizer,
        expr_tokenizer=expr_tokenizer,
        cond_tokenizer=cond_tokenizer,
    )


def _fit_condition_vocab(paths: list[Path], tokenizer: ScTokenizer) -> None:
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        try:
            tokenizer.cond_tokenizer.fit_obs(adata.obs)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()


def build_pretraining_assets(config: dict[str, Any]) -> PretrainingAssets:
    data_cfg = config["data"]
    token_dict = _load_table(data_cfg["token_dict_path"])
    tokenizer = _build_tokenizer(data_cfg, token_dict=token_dict)
    preprocessor = _build_preprocessor(data_cfg, token_dict=token_dict)
    train_paths = resolve_train_paths(data_cfg.get("train_paths"))
    if not train_paths:
        raise ValueError("At least one `.h5ad` file is required for training.")
    _fit_condition_vocab(train_paths, tokenizer)
    gene_key = data_cfg.get("gene_key")
    return PretrainingAssets(
        train_paths=train_paths,
        token_dict=token_dict,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        gene_key=gene_key,
        cond_vocab_size=tokenizer.cond_tokenizer.next_index,
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
) -> Dataset:
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
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        collate_fn=ScBatchCollator(),
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
