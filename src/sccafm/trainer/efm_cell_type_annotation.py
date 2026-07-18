"""Single-process EFM fine-tuning for supervised cell-type annotation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from ..assets import (
    EFM_DIR_NAME,
    EFM_MODEL_NAME,
    MODELS_DIR_NAME,
    ModelAssets,
    apply_model_assets_to_runtime_config,
    load_efm_config,
    load_json,
    load_model_state_dict,
    load_sfm_config,
    load_table_json,
    load_vocab_json,
    materialize_model_package,
    resolve_model_assets,
    save_json,
    save_model_state_dict,
)
from ..config import load_yaml_config, save_yaml_config
from ..data import (
    PreprocessedScDataset,
    PretrainingAssets,
    PretrainingDataBundle,
    ScBatchCollator,
    ScTokenizer,
)
from ..distributed import move_batch_to_device, seed_everything
from ..evaluator.cell_type_annotation import CellTypeAnnotationResult, evaluate_cell_type_annotation
from ..models import EFM, EFMCellTypeClassifier, reorder_gene_aligned_tokens
from .builders import _resolve_torch_dtype, build_model, build_scheduler


__all__ = [
    "CellTypeAnnotationResult",
    "CellTypeAnnotationRun",
    "balanced_class_weights",
    "encode_and_split_labels",
    "evaluate",
    "fit",
    "main",
    "predict",
    "prepare",
]


TASK_DIR_NAME = "cell_type_annotation"
CLASSIFIER_MODEL_NAME = "classifier.safetensors"
LABEL_MAP_NAME = "labels.json"
SPLIT_MANIFEST_NAME = "split_manifest.json"
TRAIN_STATE_NAME = "train_state.pt"
METRICS_NAME = "metrics.json"
PREDICTIONS_NAME = "test_predictions.csv"


@dataclass(slots=True)
class _AnnotationSample:
    label_id: int
    obs_name: str


class _LabelledScDataset(Dataset):
    """Attach annotation labels and observation names to cached cell tokens."""

    def __init__(
        self,
        base_dataset: PreprocessedScDataset,
        label_ids: np.ndarray,
        obs_names: pd.Index,
    ) -> None:
        if len(base_dataset) != len(label_ids) or len(base_dataset) != len(obs_names):
            raise ValueError("Dataset, label, and observation-name lengths must match.")
        self.base_dataset = base_dataset
        self.samples = [
            _AnnotationSample(label_id=int(label_id), obs_name=str(obs_name))
            for label_id, obs_name in zip(label_ids.tolist(), obs_names.tolist())
        ]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        sample = self.samples[index]
        item["label_id"] = sample.label_id
        item["obs_name"] = sample.obs_name
        return item


class _CellTypeAnnotationCollator:
    def __init__(self) -> None:
        self._token_collator = ScBatchCollator()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated = self._token_collator(batch)
        collated["label_ids"] = torch.tensor(
            [int(sample["label_id"]) for sample in batch], dtype=torch.long
        )
        collated["obs_names"] = [str(sample["obs_name"]) for sample in batch]
        return collated


@dataclass(slots=True)
class CellTypeAnnotationRun:
    config: dict[str, Any]
    assets: ModelAssets
    input_path: Path | None
    device: torch.device
    model_dtype: torch.dtype
    adata: ad.AnnData
    data_assets: PretrainingAssets
    dataset: _LabelledScDataset
    train_loader: DataLoader
    test_loader: DataLoader
    train_indices: np.ndarray
    test_indices: np.ndarray
    label_ids: np.ndarray
    class_names: list[str]
    class_weights: torch.Tensor
    sfm: torch.nn.Module
    model: EFMCellTypeClassifier


def _load_adata(
    *,
    input_h5ad: str | Path | None,
    adata_value: ad.AnnData | None,
) -> tuple[ad.AnnData, Path | None]:
    if input_h5ad is None and adata_value is None:
        raise ValueError("Provide either `input_h5ad` or `adata`.")
    if input_h5ad is not None and adata_value is not None:
        raise ValueError("Provide only one of `input_h5ad` or `adata`, not both.")
    if input_h5ad is not None:
        path = Path(input_h5ad).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input AnnData file does not exist: {path}")
        return ad.read_h5ad(path), path
    if not isinstance(adata_value, ad.AnnData):
        raise TypeError(f"`adata` must be an AnnData object, got {type(adata_value).__name__}.")
    return adata_value.copy(), None


def _apply_data_overrides(
    data_cfg: dict[str, Any],
    *,
    batch_size: int | None,
    max_length: int | None,
    gene_key: str | None,
    species_key: str | None,
    platform_key: str | None,
    tissue_key: str | None,
    disease_key: str | None,
    condition_defaults: dict[str, str] | None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(data_cfg)
    for name, value in {
        "batch_size": batch_size,
        "max_length": max_length,
        "gene_key": gene_key,
        "species_key": species_key,
        "platform_key": platform_key,
        "tissue_key": tissue_key,
        "disease_key": disease_key,
    }.items():
        if value is not None:
            resolved[name] = value
    if condition_defaults is not None:
        resolved["condition_defaults"] = dict(condition_defaults)
    return resolved


def _apply_condition_defaults(adata_value: ad.AnnData, data_cfg: dict[str, Any]) -> ad.AnnData:
    prepared = adata_value.copy()
    defaults = data_cfg.get("condition_defaults", {}) or {}
    added: dict[str, str] = {}
    for condition_name, key_name in (
        ("species", data_cfg.get("species_key")),
        ("disease", data_cfg.get("disease_key")),
        ("platform", data_cfg.get("platform_key")),
        ("tissue", data_cfg.get("tissue_key")),
    ):
        if key_name is None or key_name in prepared.obs or condition_name not in defaults:
            continue
        prepared.obs[key_name] = str(defaults[condition_name])
        added[str(key_name)] = str(defaults[condition_name])
    if added:
        prepared.uns.setdefault("cell_type_annotation_condition_defaults", {}).update(added)
    return prepared


def _to_dense_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if sp.issparse(value):
        return value.toarray()
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def _validate_preprocessed_adata(
    adata_value: ad.AnnData,
    *,
    max_length: int,
    gene_key: str | None,
) -> None:
    if adata_value.n_obs == 0:
        raise ValueError("Input AnnData has no cells.")
    if adata_value.n_vars == 0:
        raise ValueError("Input AnnData has no genes.")
    if not adata_value.obs_names.is_unique:
        raise ValueError(
            "Input AnnData obs_names must be unique for split artifacts and predictions."
        )
    if int(adata_value.n_vars) > int(max_length) - 1:
        raise ValueError(
            f"Input AnnData has {adata_value.n_vars} genes, but max_length={max_length} "
            "only leaves room for max_length - 1 gene tokens."
        )
    if gene_key is not None and gene_key not in adata_value.var.columns:
        raise KeyError(
            f"`gene_key={gene_key}` not found in adata.var columns: "
            f"{list(adata_value.var.columns)}"
        )
    X = _to_dense_array(adata_value.X)
    if X.ndim != 2:
        raise ValueError(f"`adata.X` must be 2D, got shape {X.shape}.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"`adata.X` must be numeric, got dtype {X.dtype}.")
    if not np.isfinite(X).all():
        raise ValueError("`adata.X` contains NaN or infinite values.")


def _validate_gene_tokenization(adata_value: ad.AnnData, data_assets: PretrainingAssets) -> None:
    gene_tokenizer = data_assets.tokenizer.gene_tokenizer
    gene_names = gene_tokenizer._extract_gene_names(adata_value, gene_key=data_assets.gene_key)
    token_ids, _ = gene_tokenizer._map_gene_names_to_indices(gene_names)
    unknown = [
        str(gene_name)
        for gene_name, token_id in zip(gene_names, token_ids.tolist())
        if int(token_id) == int(gene_tokenizer.pad_index)
    ]
    if unknown:
        preview = ", ".join(unknown[:10])
        suffix = "" if len(unknown) <= 10 else f", ... ({len(unknown)} total)"
        raise ValueError(f"Input AnnData contains genes not found in vocab: {preview}{suffix}")


def encode_and_split_labels(
    labels: pd.Series,
    *,
    train_fraction: float = 0.7,
    split_seed: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically encode labels and make a strict stratified train/test split."""

    if not isinstance(labels, pd.Series):
        raise TypeError("`labels` must be a pandas Series.")
    if labels.empty:
        raise ValueError("At least one cell-type label is required.")
    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(f"Cell-type labels contain {missing} missing value(s).")
    normalized = labels.astype(str).str.strip()
    if (normalized == "").any():
        raise ValueError("Cell-type labels must not be empty strings.")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(f"`train_fraction` must be in (0, 1), got {train_fraction}.")

    class_names = sorted(normalized.unique().tolist())
    if len(class_names) < 2:
        raise ValueError("Cell-type annotation requires at least two distinct labels.")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    label_ids = normalized.map(class_to_id).to_numpy(dtype=np.int64)
    counts = normalized.value_counts().sort_index()
    indices = np.arange(label_ids.size, dtype=np.int64)
    try:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=float(train_fraction),
            random_state=int(split_seed),
            shuffle=True,
            stratify=label_ids,
        )
    except ValueError as exc:
        count_text = ", ".join(f"{name}={int(count)}" for name, count in counts.items())
        raise ValueError(
            "Unable to create a stratified train/test split that represents every cell type. "
            f"Label counts: {count_text}."
        ) from exc

    expected_ids = set(range(len(class_names)))
    train_ids = set(label_ids[train_indices].tolist())
    test_ids = set(label_ids[test_indices].tolist())
    if train_ids != expected_ids or test_ids != expected_ids:
        count_text = ", ".join(f"{name}={int(count)}" for name, count in counts.items())
        missing_train = [class_names[index] for index in sorted(expected_ids - train_ids)]
        missing_test = [class_names[index] for index in sorted(expected_ids - test_ids)]
        raise ValueError(
            "Every cell type must appear in both train and test partitions. "
            f"Missing train labels: {missing_train}; missing test labels: {missing_test}. "
            f"Label counts: {count_text}."
        )
    return class_names, label_ids, np.asarray(train_indices), np.asarray(test_indices)


def balanced_class_weights(label_ids: np.ndarray, num_classes: int) -> torch.Tensor:
    """Return inverse-frequency class weights normalized to mean one."""

    values = np.asarray(label_ids, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("`label_ids` must be a non-empty one-dimensional array.")
    if int(num_classes) < 2:
        raise ValueError("`num_classes` must be at least two.")
    counts = np.bincount(values, minlength=int(num_classes))
    if counts.size != int(num_classes) or np.any(counts == 0):
        raise ValueError("Every class must appear in the training partition.")
    inverse = 1.0 / counts.astype(np.float64)
    normalized = inverse / inverse.mean()
    return torch.tensor(normalized, dtype=torch.float32)


def _load_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        return load_table_json(resolved)
    return pd.read_csv(resolved)


def _build_inference_assets(
    config: dict[str, Any],
    assets: ModelAssets,
    input_path: Path | None,
) -> PretrainingAssets:
    data_cfg = config["data"]
    token_dict = load_vocab_json(data_cfg["token_dict_path"])
    cond_dict = _load_table(data_cfg["cond_dict_path"])
    human_tfs = _load_table(data_cfg["human_tfs_path"])
    mouse_tfs = _load_table(data_cfg["mouse_tfs_path"])
    tokenizer = ScTokenizer(
        token_dict=token_dict,
        cond_dict=cond_dict,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        max_length=int(data_cfg["max_length"]),
        expr_pad_value=float(data_cfg.get("expr_pad_value", 0.0)),
        platform_key=data_cfg.get("platform_key"),
        species_key=data_cfg.get("species_key"),
        tissue_key=data_cfg.get("tissue_key"),
        disease_key=data_cfg.get("disease_key"),
    )
    return PretrainingAssets(
        train_paths=[] if input_path is None else [input_path],
        token_dict=token_dict,
        tokenizer=tokenizer,
        preprocessor=None,
        gene_key=data_cfg.get("gene_key"),
        cond_vocab_size=tokenizer.cond_tokenizer.next_index,
        collator=ScBatchCollator(),
    )


def _build_efm(config: dict[str, Any], data_assets: PretrainingAssets, assets: ModelAssets) -> EFM:
    efm_kwargs = copy.deepcopy(config["efm_model"]["efm"])
    if "attention_backend" in config.get("runtime", {}):
        efm_kwargs["attention_backend"] = config["runtime"]["attention_backend"]
    efm_kwargs.pop("gene_embedding_ckpt", None)
    configured_cond_vocab_size = efm_kwargs.pop("cond_vocab_size", None)
    if configured_cond_vocab_size is not None and int(configured_cond_vocab_size) != int(
        data_assets.cond_vocab_size
    ):
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


def _autocast_context(device: torch.device, runtime_cfg: dict[str, Any]):
    autocast_dtype = str(runtime_cfg.get("precision", {}).get("autocast_dtype", "fp32")).lower()
    if device.type != "cuda" or autocast_dtype == "fp32":
        return contextlib.nullcontext()
    if autocast_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("`runtime.precision.autocast_dtype=bf16` requires CUDA bf16 support.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if autocast_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported `runtime.precision.autocast_dtype`: {autocast_dtype}")


def _dataloader_kwargs(data_cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return kwargs


def prepare(
    *,
    input_h5ad: str | Path | None = None,
    adata: ad.AnnData | None = None,
    model_source: str | Path | None = None,
    config_path: str | Path = "configs/efm_cell_type_annotation.yaml",
    label_key: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    gene_key: str | None = None,
    species_key: str | None = None,
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = None,
    condition_defaults: dict[str, str] | None = None,
) -> CellTypeAnnotationRun:
    """Prepare a preprocessed labelled AnnData object for EFM fine-tuning."""

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "EFM cell-type annotation supports a single process only; do not use torchrun."
        )
    source_adata, input_path = _load_adata(input_h5ad=input_h5ad, adata_value=adata)
    config = load_yaml_config(config_path)
    if model_source is not None:
        config["model_source"] = str(model_source)
    if not config.get("model_source"):
        raise ValueError("`model_source` must be configured or provided to `prepare`.")
    if bool(config.get("runtime", {}).get("fsdp", {}).get("enabled", False)):
        raise ValueError(
            "EFM cell-type annotation is single-process; set `runtime.fsdp.enabled: false`."
        )

    config["data"] = _apply_data_overrides(
        config.get("data", {}),
        batch_size=batch_size,
        max_length=max_length,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
        condition_defaults=condition_defaults,
    )
    task_cfg = config.setdefault("cell_type_annotation", {})
    if label_key is not None:
        task_cfg["label_key"] = str(label_key)
    resolved_label_key = task_cfg.get("label_key", "cell_type")
    if resolved_label_key not in source_adata.obs:
        raise KeyError(
            f"`label_key={resolved_label_key}` not found in adata.obs columns: "
            f"{list(source_adata.obs.columns)}"
        )

    assets = resolve_model_assets(
        model_source=config["model_source"],
        require_model_weights=True,
        require_efm_config=True,
        require_efm_weights=True,
    )
    config = apply_model_assets_to_runtime_config(config, assets, require_model_weights=True)
    config["model"] = load_sfm_config(assets.sfm_config)
    config["efm_model"] = load_efm_config(assets.efm_config)
    prepared_adata = _apply_condition_defaults(source_adata, config["data"])
    _validate_preprocessed_adata(
        prepared_adata,
        max_length=int(config["data"]["max_length"]),
        gene_key=config["data"].get("gene_key"),
    )

    class_names, label_ids, train_indices, test_indices = encode_and_split_labels(
        prepared_adata.obs[resolved_label_key],
        train_fraction=float(task_cfg.get("train_fraction", 0.7)),
        split_seed=int(task_cfg.get("split_seed", 42)),
    )
    data_assets = _build_inference_assets(config, assets, input_path)
    _validate_gene_tokenization(prepared_adata, data_assets)
    base_dataset = PreprocessedScDataset(
        adata=prepared_adata,
        tokenizer=data_assets.tokenizer,
        gene_key=data_assets.gene_key,
        preprocessor=None,
    )
    dataset = _LabelledScDataset(base_dataset, label_ids, prepared_adata.obs_names)
    collator = _CellTypeAnnotationCollator()
    loader_kwargs = _dataloader_kwargs(config["data"])
    split_seed = int(task_cfg.get("split_seed", 42))
    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    test_subset = torch.utils.data.Subset(dataset, test_indices.tolist())
    train_sampler = RandomSampler(
        train_subset,
        generator=torch.Generator().manual_seed(split_seed),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["data"]["batch_size"]),
        sampler=train_sampler,
        drop_last=bool(config["data"].get("drop_last", False)),
        collate_fn=collator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=int(config["data"]["batch_size"]),
        sampler=SequentialSampler(test_subset),
        drop_last=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = _resolve_torch_dtype(
        str(config.get("runtime", {}).get("precision", {}).get("model_dtype", "fp32"))
    )
    model_bundle = PretrainingDataBundle(
        train_loader=None,
        train_sampler=None,
        token_dict=data_assets.token_dict,
        cond_vocab_size=data_assets.cond_vocab_size,
        train_size=len(dataset),
        path=input_path or Path("<memory>"),
    )
    sfm = build_model(
        sfm_config=config["model"],
        data_bundle=model_bundle,
        assets=assets,
        runtime_config=config.get("runtime", {}),
    )
    sfm.load_state_dict(load_model_state_dict(assets.sfm_model), strict=True)
    sfm.to(device=device, dtype=model_dtype)
    sfm.eval()
    sfm.requires_grad_(False)

    efm = _build_efm(config, data_assets, assets)
    efm.load_state_dict(load_model_state_dict(assets.efm_model), strict=True)
    model = EFMCellTypeClassifier(efm=efm, num_classes=len(class_names))
    model.to(device=device, dtype=model_dtype)
    return CellTypeAnnotationRun(
        config=config,
        assets=assets,
        input_path=input_path,
        device=device,
        model_dtype=model_dtype,
        adata=prepared_adata,
        data_assets=data_assets,
        dataset=dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        train_indices=train_indices,
        test_indices=test_indices,
        label_ids=label_ids,
        class_names=class_names,
        class_weights=balanced_class_weights(label_ids[train_indices], len(class_names)),
        sfm=sfm,
        model=model,
    )


def _ordered_tokens(run: CellTypeAnnotationRun, tokens: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad(), _autocast_context(run.device, run.config.get("runtime", {})):
        sfm_output = run.sfm(
            tokens,
            compute_order={"sfm": True},
            compute_grn=False,
            return_factors=False,
        )
    gene_order = sfm_output.foundations["sfm"].gene_order
    if gene_order is None:
        raise RuntimeError("Frozen SFM did not return `gene_order`.")
    return reorder_gene_aligned_tokens(tokens, gene_order)


def _resolve_output_dir(run: CellTypeAnnotationRun, output_dir: str | Path | None) -> Path:
    candidate = output_dir
    if candidate is None:
        candidate = run.config.get("cell_type_annotation", {}).get(
            "output_dir", "results/efm_cell_type_annotation"
        )
    return Path(candidate).expanduser().resolve()


def _task_paths(output_dir: Path) -> dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    task_dir = checkpoint_dir / MODELS_DIR_NAME / TASK_DIR_NAME
    return {
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "task_dir": task_dir,
        "classifier": task_dir / CLASSIFIER_MODEL_NAME,
        "labels": task_dir / LABEL_MAP_NAME,
        "split": output_dir / SPLIT_MANIFEST_NAME,
        "state": checkpoint_dir / TRAIN_STATE_NAME,
        "metrics": output_dir / METRICS_NAME,
        "predictions": output_dir / PREDICTIONS_NAME,
    }


def _save_split_manifest(run: CellTypeAnnotationRun, path: Path) -> None:
    task_cfg = run.config["cell_type_annotation"]
    save_json(
        path,
        {
            "input_h5ad": None if run.input_path is None else str(run.input_path),
            "label_key": str(task_cfg.get("label_key", "cell_type")),
            "train_fraction": float(task_cfg.get("train_fraction", 0.7)),
            "split_seed": int(task_cfg.get("split_seed", 42)),
            "class_names": list(run.class_names),
            "train_obs_names": run.adata.obs_names[run.train_indices].astype(str).tolist(),
            "test_obs_names": run.adata.obs_names[run.test_indices].astype(str).tolist(),
        },
    )


def _save_checkpoint(
    *,
    run: CellTypeAnnotationRun,
    output_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    paths = _task_paths(output_dir)
    efm_path = paths["checkpoint_dir"] / MODELS_DIR_NAME / EFM_DIR_NAME / EFM_MODEL_NAME
    save_model_state_dict(efm_path, run.model.efm.state_dict())
    save_model_state_dict(paths["classifier"], run.model.classifier.state_dict())
    save_json(paths["labels"], {"class_names": list(run.class_names)})
    torch.save(
        {
            "module": "efm_cell_type_annotation",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        paths["state"],
    )


def _initialize_output_bundle(run: CellTypeAnnotationRun, output_dir: Path) -> None:
    paths = _task_paths(output_dir)
    materialize_model_package(
        source_assets=run.assets,
        target_dir=paths["checkpoint_dir"],
        include_model_weights=True,
        include_efm_weights=True,
        include_cond_dict=True,
        include_resources=True,
        overwrite=True,
    )
    save_yaml_config(output_dir / "efm_cell_type_annotation_config.yaml", run.config)
    _save_split_manifest(run, paths["split"])


def _load_resume_state(
    run: CellTypeAnnotationRun,
    resume_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    state_path = Path(resume_path).expanduser().resolve()
    if not state_path.exists():
        raise FileNotFoundError(f"Annotation resume state not found: {state_path}")
    checkpoint_dir = state_path.parent
    output_dir = checkpoint_dir.parent
    paths = _task_paths(output_dir)
    if state_path != paths["state"]:
        raise ValueError(
            f"Resume state must be named {TRAIN_STATE_NAME} under a checkpoints directory."
        )
    labels_payload = load_json(paths["labels"])
    if labels_payload.get("class_names") != run.class_names:
        raise ValueError("Resume label map does not match the current labelled AnnData split.")
    split_payload = load_json(paths["split"])
    if (
        split_payload.get("train_obs_names")
        != run.adata.obs_names[run.train_indices].astype(str).tolist()
        or split_payload.get("test_obs_names")
        != run.adata.obs_names[run.test_indices].astype(str).tolist()
    ):
        raise ValueError("Resume split manifest does not match the current labelled AnnData split.")
    run.model.efm.load_state_dict(
        load_model_state_dict(checkpoint_dir / MODELS_DIR_NAME / EFM_DIR_NAME / EFM_MODEL_NAME),
        strict=True,
    )
    run.model.classifier.load_state_dict(load_model_state_dict(paths["classifier"]), strict=True)
    payload = torch.load(state_path, map_location="cpu")
    if payload.get("module") != "efm_cell_type_annotation":
        raise ValueError(f"Unsupported annotation resume module: {payload.get('module')!r}.")
    return payload, output_dir


def _build_optimizer(run: CellTypeAnnotationRun) -> torch.optim.Optimizer:
    optimizer_cfg = run.config["optimizer"]
    if str(optimizer_cfg.get("name", "adamw")).lower() != "adamw":
        raise ValueError("Only AdamW is supported for EFM cell-type annotation.")
    efm_lr = float(optimizer_cfg.get("efm_lr", optimizer_cfg.get("lr", 1e-5)))
    classifier_lr = float(optimizer_cfg.get("classifier_lr", optimizer_cfg.get("lr", 1e-3)))
    return torch.optim.AdamW(
        [
            {"params": run.model.efm.parameters(), "lr": efm_lr},
            {"params": run.model.classifier.parameters(), "lr": classifier_lr},
        ],
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )


def _clip_grad_norm(run: CellTypeAnnotationRun) -> float | None:
    max_norm = run.config.get("trainer", {}).get("grad_clip_norm")
    if max_norm is None:
        return None
    value = torch.nn.utils.clip_grad_norm_(run.model.parameters(), float(max_norm))
    return float(value.detach().item()) if torch.is_tensor(value) else float(value)


def predict(
    run: CellTypeAnnotationRun,
    split: Literal["test"] = "test",
) -> pd.DataFrame:
    """Return held-out true and predicted labels after fine-tuning."""

    if not isinstance(run, CellTypeAnnotationRun):
        raise TypeError(f"`run` must be a CellTypeAnnotationRun, got {type(run).__name__}.")
    if split != "test":
        raise ValueError("Only the untouched `test` split is available for prediction.")
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    obs_names: list[str] = []
    run.model.eval()
    run.sfm.eval()
    with torch.no_grad():
        for batch in run.test_loader:
            moved = move_batch_to_device(batch, run.device)
            labels = moved.pop("label_ids")
            names = moved.pop("obs_names")
            ordered_tokens = _ordered_tokens(run, moved)
            with _autocast_context(run.device, run.config.get("runtime", {})):
                logits = run.model(ordered_tokens).logits
            true_ids.extend(labels.detach().cpu().to(torch.long).tolist())
            predicted_ids.extend(logits.argmax(dim=-1).detach().cpu().to(torch.long).tolist())
            obs_names.extend(str(name) for name in names)
    if not true_ids:
        raise RuntimeError("The test split produced no predictions.")
    return pd.DataFrame(
        {
            "obs_name": obs_names,
            "true_label": [run.class_names[index] for index in true_ids],
            "predicted_label": [run.class_names[index] for index in predicted_ids],
            "true_label_id": true_ids,
            "predicted_label_id": predicted_ids,
        }
    )


def evaluate(
    run: CellTypeAnnotationRun,
    split: Literal["test"] = "test",
) -> CellTypeAnnotationResult:
    """Evaluate held-out annotations with accuracy and macro-F1."""

    predictions = predict(run, split=split)
    return evaluate_cell_type_annotation(
        predictions["true_label_id"].to_numpy(dtype=np.int64),
        predictions["predicted_label_id"].to_numpy(dtype=np.int64),
        class_names=run.class_names,
    )


def fit(
    run: CellTypeAnnotationRun,
    *,
    output_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> CellTypeAnnotationResult:
    """Fine-tune EFM and its linear classifier, then evaluate the final epoch."""

    if not isinstance(run, CellTypeAnnotationRun):
        raise TypeError(f"`run` must be a CellTypeAnnotationRun, got {type(run).__name__}.")
    seed = int(run.config.get("cell_type_annotation", {}).get("split_seed", 42))
    seed_everything(seed)
    output_path = _resolve_output_dir(run, output_dir)
    optimizer = _build_optimizer(run)
    grad_accum_steps = int(run.config.get("data", {}).get("gradient_accumulation_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("`data.gradient_accumulation_steps` must be positive.")
    steps_per_epoch = max(math.ceil(len(run.train_loader) / grad_accum_steps), 1)
    scheduler = build_scheduler(
        optimizer=optimizer,
        config=run.config,
        total_steps=int(run.config["trainer"]["epochs"]) * steps_per_epoch,
    )
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        payload, resume_output_dir = _load_resume_state(run, resume_path)
        if output_dir is not None and output_path != resume_output_dir:
            raise ValueError(
                "`output_dir` must match the output directory inferred from `resume_path`."
            )
        output_path = resume_output_dir
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
    else:
        _initialize_output_bundle(run, output_path)

    class_weighting = str(
        run.config.get("cell_type_annotation", {}).get("class_weighting", "balanced")
    ).lower()
    if class_weighting != "balanced":
        raise ValueError("Only `cell_type_annotation.class_weighting: balanced` is supported.")
    loss_fn = torch.nn.CrossEntropyLoss(weight=run.class_weights.to(run.device))
    epochs = int(run.config["trainer"]["epochs"])
    if start_epoch > epochs:
        raise ValueError(f"Resume epoch {start_epoch} exceeds configured epochs {epochs}.")

    for epoch in range(start_epoch, epochs):
        run.model.train()
        run.sfm.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        num_batches = len(run.train_loader)
        for batch_index, batch in enumerate(run.train_loader, start=1):
            moved = move_batch_to_device(batch, run.device)
            labels = moved.pop("label_ids")
            moved.pop("obs_names")
            ordered_tokens = _ordered_tokens(run, moved)
            with _autocast_context(run.device, run.config.get("runtime", {})):
                output = run.model(ordered_tokens)
                loss = loss_fn(output.logits, labels)
            (loss / grad_accum_steps).backward()
            loss_total += float(loss.detach().item())
            should_step = batch_index % grad_accum_steps == 0 or batch_index == num_batches
            if should_step:
                _clip_grad_norm(run)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        print(
            f"[efm-cell-type-annotation epoch {epoch + 1}/{epochs}] "
            f"train_loss={loss_total / max(num_batches, 1):.6f}"
        )
        _save_checkpoint(
            run=run,
            output_dir=output_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
        )

    predictions = predict(run)
    result = evaluate_cell_type_annotation(
        predictions["true_label_id"].to_numpy(dtype=np.int64),
        predictions["predicted_label_id"].to_numpy(dtype=np.int64),
        class_names=run.class_names,
    )
    paths = _task_paths(output_path)
    predictions.to_csv(paths["predictions"], index=False)
    save_json(paths["metrics"], result.to_summary())
    print(
        "[efm-cell-type-annotation test] "
        f"accuracy={result.accuracy:.6f}, macro_f1={result.macro_f1:.6f}, "
        f"test_cells={result.test_cell_count}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EFM for cell-type annotation.")
    parser.add_argument(
        "--efm-cell-type-annotation-config",
        "--config",
        dest="config_path",
        default="configs/efm_cell_type_annotation.yaml",
    )
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config_path)
    input_h5ad = args.input_h5ad or config.get("data", {}).get("input_h5ad")
    if input_h5ad is None:
        raise ValueError("Set `data.input_h5ad` in the config or pass `--input-h5ad`.")
    run = prepare(
        input_h5ad=input_h5ad,
        model_source=config.get("model_source"),
        config_path=args.config_path,
        label_key=args.label_key,
    )
    fit(run, output_dir=args.output_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()
