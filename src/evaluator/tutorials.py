from __future__ import annotations

import contextlib

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, SequentialSampler

from ..assets import (
    ModelAssets,
    SFM_MODEL_NAME,
    apply_model_assets_to_runtime_config,
    load_model_state_dict,
    load_sfm_config,
    resolve_model_assets,
)
from ..config import load_yaml_config
from ..data import (
    PreprocessedScDataset,
    PretrainingAssets,
    PretrainingDataBundle,
    ScBatchCollator,
    build_evaluation_assets,
)
from ..distributed import RuntimeContext, move_batch_to_device
from ..trainer.builders import build_model
from .grn import all_gene_random_positive_rate, build_candidate_pair_keys, build_evaluation_grn_cache
from .metrics import summarize_binary_metrics


__all__ = [
    "SFMGRNContext",
    "build_cell_specific_edges",
    "collect_sfm_forward_results",
    "evaluate_dataset_level_forward_results",
    "evaluate_dataset_level_grn",
    "export_cell_specific_grn_edges",
    "prepare_cell_specific_grn_context",
    "prepare_dataset_grn_evaluation_context",
]


@dataclass(slots=True)
class SFMGRNContext:
    config: dict
    assets: ModelAssets
    sfm_config: dict
    checkpoint_file: Path
    device: torch.device
    runtime: RuntimeContext
    data_assets: PretrainingAssets
    raw_adata: ad.AnnData
    processed_adata: ad.AnnData
    dataset: PreprocessedScDataset
    loader: DataLoader
    model: torch.nn.Module
    inference_dtype: torch.dtype
    forward_token_template: dict[str, torch.Tensor | None] | None = None


def prepare_cell_specific_grn_context(
    *,
    input_h5ad: str | Path,
    model_source: str | Path = "assets",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = "configs/eval_grn.yaml",
    batch_size: int = 8,
    max_length: int = 4096,
    n_top_genes: int | None = None,
    gene_key: str | None = None,
    species_key: str | None = "species",
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = "disease",
) -> SFMGRNContext:
    """Prepare model, data, and normal HVG preprocessing for cell-specific GRN export."""

    return _build_context(
        input_h5ad=input_h5ad,
        model_source=model_source,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        max_length=max_length,
        n_top_genes=n_top_genes,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
        preprocess_mode="normal_hvg",
    )


def prepare_dataset_grn_evaluation_context(
    *,
    input_h5ad: str | Path,
    model_source: str | Path = "assets",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = "configs/eval_grn.yaml",
    batch_size: int = 8,
    max_length: int = 4096,
    n_top_genes: int | None = None,
    preserve_tf_species: str = "mouse",
    gene_key: str | None = None,
    species_key: str | None = "species",
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = "disease",
) -> SFMGRNContext:
    """Prepare model, data, and TF-preserving HVG preprocessing for GRN evaluation."""

    return _build_context(
        input_h5ad=input_h5ad,
        model_source=model_source,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        max_length=max_length,
        n_top_genes=n_top_genes,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
        preprocess_mode="grn_inference_hvg",
        preserve_tf_species=preserve_tf_species,
    )


def _resolve_checkpoint_path(checkpoint_path: object, default_model_path: Path) -> Path:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        resolved = default_model_path
    else:
        candidate = Path(str(checkpoint_path)).expanduser().resolve()
        resolved = candidate / SFM_MODEL_NAME if candidate.is_dir() else candidate
    if not resolved.exists():
        raise FileNotFoundError(f"SFM checkpoint not found: {resolved}")
    return resolved


def _runtime_context(device: torch.device) -> RuntimeContext:
    return RuntimeContext(
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
        distributed=False,
        is_main=True,
    )


def _resolve_inference_dtype(config: dict, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    precision_cfg = config.get("runtime", {}).get("precision", {})
    requested = str(precision_cfg.get("autocast_dtype", "bf16")).lower()
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _inference_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_tf_names(path: str | Path) -> list[str]:
    tf_df = pd.read_csv(Path(path).expanduser().resolve())
    if "TF" not in tf_df.columns:
        raise ValueError(f"TF table must contain a 'TF' column: {path}")
    return sorted({str(gene).strip() for gene in tf_df["TF"].tolist() if str(gene).strip()})


def _apply_data_overrides(
    data_cfg: dict,
    *,
    input_h5ad: Path,
    batch_size: int,
    max_length: int,
    gene_key: str | None,
    species_key: str | None,
    platform_key: str | None,
    tissue_key: str | None,
    disease_key: str | None,
) -> dict:
    updated = deepcopy(data_cfg)
    updated["train_paths"] = [str(input_h5ad.resolve())]
    updated["batch_size"] = int(batch_size)
    updated["num_workers"] = 0
    updated["pin_memory"] = torch.cuda.is_available()
    updated["max_length"] = int(max_length)
    updated["gene_key"] = gene_key
    updated["species_key"] = species_key
    updated["platform_key"] = platform_key
    updated["tissue_key"] = tissue_key
    updated["disease_key"] = disease_key
    updated["condition_vocab"] = {"regenerate": False}
    updated["condition_mask"] = {"enabled": False, "unk_ratio": 0.0}
    return updated


def _normal_hvg_preprocess(data_cfg: dict, *, n_top_genes: int | None) -> dict:
    updated = deepcopy(data_cfg)
    preprocess_cfg = dict(updated.get("preprocess", {}))
    if n_top_genes is not None:
        preprocess_cfg["n_top_genes"] = int(n_top_genes)
    preprocess_cfg.update(
        {
            "enabled": True,
            "log1p": True,
            "subset_hvg": True,
            "preserve_gene_names": [],
            "hvg_exclude_preserved_genes": False,
        }
    )
    updated["preprocess"] = preprocess_cfg
    return updated


def _grn_inference_preprocess(
    data_cfg: dict,
    *,
    preserve_gene_names: Iterable[str],
    n_top_genes: int | None,
) -> dict:
    updated = deepcopy(data_cfg)
    preprocess_cfg = dict(updated.get("preprocess", {}))
    if n_top_genes is not None:
        preprocess_cfg["n_top_genes"] = int(n_top_genes)
    preprocess_cfg.update(
        {
            "enabled": True,
            "preserve_gene_names": sorted(
                {str(gene).strip() for gene in preserve_gene_names if str(gene).strip()}
            ),
            "hvg_exclude_preserved_genes": True,
            "subset_hvg": True,
        }
    )
    updated["preprocess"] = preprocess_cfg
    return updated


def _build_context(
    *,
    input_h5ad: str | Path,
    model_source: str | Path,
    checkpoint_path: str | Path | None,
    config_path: str | Path,
    batch_size: int,
    max_length: int,
    n_top_genes: int | None,
    gene_key: str | None,
    species_key: str | None,
    platform_key: str | None,
    tissue_key: str | None,
    disease_key: str | None,
    preprocess_mode: str,
    preserve_tf_species: str | None = None,
) -> SFMGRNContext:
    input_path = Path(input_h5ad).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input AnnData file does not exist: {input_path}")
    if n_top_genes is not None and int(n_top_genes) >= int(max_length):
        raise ValueError(
            "`n_top_genes` must be smaller than `max_length` because one sequence "
            "position is reserved."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = _runtime_context(device)

    base_config = load_yaml_config(config_path)
    base_config["model_source"] = str(model_source)
    base_config["data"] = _apply_data_overrides(
        base_config.get("data", {}),
        input_h5ad=input_path,
        batch_size=batch_size,
        max_length=max_length,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
    )
    base_config.setdefault("evaluator", {})["foundation_name"] = "sfm"
    if checkpoint_path is not None:
        base_config["evaluator"]["checkpoint_path"] = str(checkpoint_path)

    assets = resolve_model_assets(
        model_source=base_config["model_source"],
        require_model_weights=checkpoint_path is None,
    )
    sfm_config = load_sfm_config(assets.sfm_config)
    config = apply_model_assets_to_runtime_config(
        base_config,
        assets,
        require_model_weights=checkpoint_path is None,
    )

    if preprocess_mode == "normal_hvg":
        config["data"] = _normal_hvg_preprocess(
            config["data"],
            n_top_genes=n_top_genes,
        )
    elif preprocess_mode == "grn_inference_hvg":
        if preserve_tf_species == "human":
            preserve_gene_names = _load_tf_names(config["data"]["human_tfs_path"])
        elif preserve_tf_species == "mouse":
            preserve_gene_names = _load_tf_names(config["data"]["mouse_tfs_path"])
        else:
            raise ValueError("`preserve_tf_species` must be 'human' or 'mouse'.")
        config["data"] = _grn_inference_preprocess(
            config["data"],
            preserve_gene_names=preserve_gene_names,
            n_top_genes=n_top_genes,
        )
    else:
        raise ValueError(
            "`preprocess_mode` must be either 'normal_hvg' or 'grn_inference_hvg'."
        )

    checkpoint_file = _resolve_checkpoint_path(
        config.get("evaluator", {}).get("checkpoint_path"),
        assets.sfm_model,
    )
    inference_dtype = _resolve_inference_dtype(config, device)

    data_assets = build_evaluation_assets(config=config, runtime=runtime)
    raw_adata = ad.read_h5ad(input_path)
    processed_adata = (
        data_assets.preprocessor(raw_adata) if data_assets.preprocessor is not None else raw_adata
    )
    dataset = PreprocessedScDataset(
        adata=processed_adata,
        tokenizer=data_assets.tokenizer,
        gene_key=data_assets.gene_key,
        preprocessor=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=SequentialSampler(dataset),
        shuffle=False,
        collate_fn=ScBatchCollator(),
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
    )

    model_bundle = PretrainingDataBundle(
        train_loader=None,
        train_sampler=None,
        token_dict=data_assets.token_dict,
        cond_vocab_size=data_assets.cond_vocab_size,
        train_size=len(dataset),
        path=input_path,
    )
    model = build_model(
        sfm_config=sfm_config,
        data_bundle=model_bundle,
        assets=assets,
        runtime_config=config.get("runtime", {}),
    )
    model.load_state_dict(load_model_state_dict(checkpoint_file))
    model.to(device=device, dtype=inference_dtype)
    model.eval()

    return SFMGRNContext(
        config=config,
        assets=assets,
        sfm_config=sfm_config,
        checkpoint_file=checkpoint_file,
        device=device,
        runtime=runtime,
        data_assets=data_assets,
        raw_adata=raw_adata,
        processed_adata=processed_adata,
        dataset=dataset,
        loader=loader,
        model=model,
        inference_dtype=inference_dtype,
    )


def _token_id_to_gene(token_dict: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for _, row in token_dict.iterrows():
        token_id = int(row["token_index"])
        symbol = row.get("gene_symbol")
        gene_id = row.get("gene_id")
        if pd.notna(symbol) and str(symbol).strip():
            mapping[token_id] = str(symbol)
        elif pd.notna(gene_id) and str(gene_id).strip():
            mapping[token_id] = str(gene_id)
        else:
            mapping[token_id] = str(token_id)
    return mapping


def _edges_from_batch(
    grn: torch.Tensor,
    tokens: dict[str, torch.Tensor | None],
    *,
    cell_offset: int,
    token_id_to_gene: dict[int, str],
    top_k_per_cell: int | None,
    score_threshold: float | None,
) -> list[dict[str, object]]:
    input_ids = tokens["input_ids"].detach().cpu().to(torch.long)
    non_tf_mask = tokens["non_tf_mask"].detach().cpu().to(torch.bool)
    padding_mask = tokens.get("padding_mask")
    if padding_mask is None:
        padding_mask_cpu = torch.zeros_like(non_tf_mask, dtype=torch.bool)
    else:
        padding_mask_cpu = padding_mask.detach().cpu().to(torch.bool)

    grn_cpu = grn.detach().cpu().to(torch.float32)
    rows: list[dict[str, object]] = []

    for batch_cell_idx in range(grn_cpu.shape[0]):
        active_mask = ~padding_mask_cpu[batch_cell_idx]
        source_mask = active_mask & ~non_tf_mask[batch_cell_idx]
        target_mask = active_mask
        candidate_mask = source_mask[:, None] & target_mask[None, :]
        candidate_mask.fill_diagonal_(False)

        src_pos, tgt_pos = candidate_mask.nonzero(as_tuple=True)
        if src_pos.numel() == 0:
            continue

        scores = grn_cpu[batch_cell_idx, src_pos, tgt_pos]
        if score_threshold is not None:
            keep = scores >= float(score_threshold)
            src_pos = src_pos[keep]
            tgt_pos = tgt_pos[keep]
            scores = scores[keep]

        if scores.numel() == 0:
            continue

        if (
            top_k_per_cell is not None
            and int(top_k_per_cell) > 0
            and scores.numel() > int(top_k_per_cell)
        ):
            top_indices = torch.topk(scores, k=int(top_k_per_cell), largest=True).indices
            src_pos = src_pos[top_indices]
            tgt_pos = tgt_pos[top_indices]
            scores = scores[top_indices]

        order = torch.argsort(scores, descending=True)
        src_pos = src_pos[order]
        tgt_pos = tgt_pos[order]

        cell_index = cell_offset + batch_cell_idx
        for src_i, tgt_i in zip(src_pos.tolist(), tgt_pos.tolist()):
            source_token_id = int(input_ids[batch_cell_idx, src_i].item())
            target_token_id = int(input_ids[batch_cell_idx, tgt_i].item())
            rows.append(
                {
                    "cell_index": cell_index,
                    "source_gene": token_id_to_gene.get(source_token_id, str(source_token_id)),
                    "target_gene": token_id_to_gene.get(target_token_id, str(target_token_id)),
                }
            )
    return rows


def _validate_forward_token_template(
    context: SFMGRNContext,
    batch_tokens: dict[str, torch.Tensor | None],
) -> None:
    batch_template = _clone_token_template(batch_tokens)
    if context.forward_token_template is None:
        context.forward_token_template = batch_template
        return
    if not _same_token_template(context.forward_token_template, batch_template):
        raise ValueError(
            "Collecting forward results as a single array requires a fixed gene/token "
            "order within the dataset."
        )


def _active_gene_positions(tokens: dict[str, torch.Tensor | None]) -> torch.LongTensor:
    input_ids = tokens.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise KeyError("Forward-result tokens must contain tensor entry 'input_ids'.")

    padding_mask = tokens.get("padding_mask")
    if padding_mask is None:
        active_mask = torch.ones(input_ids.shape[1], dtype=torch.bool)
    else:
        if not torch.is_tensor(padding_mask):
            raise TypeError("`tokens['padding_mask']` must be a torch.Tensor or None.")
        padding_mask = padding_mask.to(torch.bool)
        first_mask = padding_mask[0]
        if not torch.equal(padding_mask, first_mask.expand_as(padding_mask)):
            raise ValueError(
                "Collecting forward results as a single array requires identical "
                "padding positions for every cell."
            )
        active_mask = ~first_mask.cpu()

    positions = active_mask.nonzero(as_tuple=True)[0].to(torch.long)
    if positions.numel() == 0:
        raise RuntimeError("No active gene positions were available after removing padding.")
    return positions


def _trim_tokens_to_gene_positions(
    tokens: dict[str, torch.Tensor | None],
    gene_positions: torch.LongTensor,
) -> dict[str, torch.Tensor | None]:
    trimmed: dict[str, torch.Tensor | None] = {}
    max_position = int(gene_positions.max().item())
    for key, value in tokens.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] >= max_position + 1:
            trimmed[key] = value.index_select(dim=1, index=gene_positions)
        else:
            trimmed[key] = value
    return trimmed


def collect_sfm_forward_results(context: SFMGRNContext) -> np.ndarray:
    """Collect dense SFM GRN outputs as an array with shape (cells, genes, genes)."""

    grn_batches: list[np.ndarray] = []
    context.forward_token_template = None
    with torch.no_grad(), _inference_autocast_context(
        context.device,
        context.inference_dtype,
    ):
        for batch in context.loader:
            tokens = move_batch_to_device(batch, context.device)
            model_output = context.model(
                tokens,
                compute_grn={"sfm": True},
                return_factors=False,
            )
            grn = model_output.foundations["sfm"].grn
            if grn is None:
                raise RuntimeError("SFM did not return a GRN tensor.")

            cpu_tokens = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in tokens.items()
            }
            gene_positions = _active_gene_positions(cpu_tokens)
            trimmed_tokens = _trim_tokens_to_gene_positions(cpu_tokens, gene_positions)
            _validate_forward_token_template(context, trimmed_tokens)
            trimmed_grn = grn.detach().to(dtype=torch.float32, device="cpu")
            trimmed_grn = trimmed_grn.index_select(dim=1, index=gene_positions)
            trimmed_grn = trimmed_grn.index_select(dim=2, index=gene_positions)
            grn_batches.append(trimmed_grn.numpy())

    if not grn_batches:
        raise RuntimeError("No GRN scores were produced.")
    return np.concatenate(grn_batches, axis=0)


def _require_forward_token_template(context: SFMGRNContext) -> dict[str, torch.Tensor | None]:
    if context.forward_token_template is None:
        raise RuntimeError(
            "No forward token template is available. Run collect_sfm_forward_results(context) "
            "before building edges or evaluating collected forward results."
        )
    return context.forward_token_template


def build_cell_specific_edges(
    forward_results: np.ndarray,
    context: SFMGRNContext,
    *,
    output_csv: str | Path | None = None,
    top_k_per_cell: int | None = 1000,
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """Build a minimal TF-to-target edge table from dense SFM forward results."""

    if forward_results.ndim != 3:
        raise ValueError(
            f"`forward_results` must have shape (cells, genes, genes), got {forward_results.shape}."
        )

    token_id_to_gene = _token_id_to_gene(context.data_assets.token_dict)
    token_template = _require_forward_token_template(context)
    all_rows: list[dict[str, object]] = []
    batch_size = int(context.config.get("data", {}).get("batch_size", 1))
    for cell_start in range(0, int(forward_results.shape[0]), batch_size):
        batch_grn = torch.as_tensor(forward_results[cell_start : cell_start + batch_size])
        repeat_count = int(batch_grn.shape[0])
        batch_tokens = {
            key: (
                value.repeat((repeat_count, 1))
                if torch.is_tensor(value) and value.shape[0] == 1
                else value
            )
            for key, value in token_template.items()
        }
        all_rows.extend(
            _edges_from_batch(
                batch_grn,
                batch_tokens,
                cell_offset=cell_start,
                token_id_to_gene=token_id_to_gene,
                top_k_per_cell=top_k_per_cell,
                score_threshold=score_threshold,
            )
        )

    edges_df = pd.DataFrame(
        all_rows,
        columns=[
            "cell_index",
            "source_gene",
            "target_gene",
        ],
    )
    if output_csv is not None:
        output_path = Path(output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_path, index=False)
    return edges_df


def export_cell_specific_grn_edges(
    *,
    input_h5ad: str | Path,
    output_csv: str | Path,
    model_source: str | Path = "assets",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = "configs/eval_grn.yaml",
    batch_size: int = 8,
    max_length: int = 4096,
    n_top_genes: int | None = None,
    top_k_per_cell: int | None = 1000,
    score_threshold: float | None = None,
    gene_key: str | None = None,
    species_key: str | None = "species",
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = "disease",
) -> pd.DataFrame:
    """Export TF-sourced cell-specific SFM GRN edges after normal HVG preprocessing."""

    context = prepare_cell_specific_grn_context(
        input_h5ad=input_h5ad,
        model_source=model_source,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        max_length=max_length,
        n_top_genes=n_top_genes,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
    )
    forward_results = collect_sfm_forward_results(context)
    return build_cell_specific_edges(
        forward_results,
        context,
        output_csv=output_csv,
        top_k_per_cell=top_k_per_cell,
        score_threshold=score_threshold,
    )


def _clone_token_template(tokens: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    template: dict[str, torch.Tensor | None] = {}
    for key in ("input_ids", "padding_mask", "non_tf_mask"):
        value = tokens.get(key)
        template[key] = value[:1].detach().cpu().clone() if torch.is_tensor(value) else None
    return template


def _same_token_template(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> bool:
    for key in ("input_ids", "padding_mask", "non_tf_mask"):
        left_value = left.get(key)
        right_value = right.get(key)
        if torch.is_tensor(left_value) != torch.is_tensor(right_value):
            return False
        if torch.is_tensor(left_value) and torch.is_tensor(right_value):
            if not torch.equal(left_value, right_value):
                return False
    return True


def evaluate_dataset_level_forward_results(
    forward_results: np.ndarray,
    context: SFMGRNContext,
    *,
    reference_grn_csv: str | Path,
    output_metrics_csv: str | Path | None = None,
    candidate_source_universe: str = "supported_tfs",
    metric_names: Iterable[str] = ("auprc", "auroc", "ep"),
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    reference_grn_name: str | None = None,
) -> pd.DataFrame:
    """Evaluate dataset-level metrics from dense SFM forward results."""

    if forward_results.ndim != 3:
        raise ValueError(
            f"`forward_results` must have shape (cells, genes, genes), got {forward_results.shape}."
        )

    reference_path = Path(reference_grn_csv).expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference GRN file does not exist: {reference_path}")
    reference_grn_df = pd.read_csv(reference_path)
    if not {"Gene1", "Gene2"}.issubset(reference_grn_df.columns):
        raise ValueError("Reference GRN must contain Gene1 and Gene2 columns.")

    reference_cache = build_evaluation_grn_cache(
        token_dict=context.data_assets.token_dict,
        evaluation_grn_df=reference_grn_df,
    )
    token_template = _require_forward_token_template(context)
    cell_count = int(forward_results.shape[0])
    if cell_count <= 0:
        raise RuntimeError("No GRN scores were produced for dataset-level evaluation.")

    dataset_grn = torch.as_tensor(forward_results, dtype=torch.float32).mean(dim=0)
    pair_keys, candidate_mask = build_candidate_pair_keys(
        tokens=token_template,
        cache=reference_cache,
        candidate_source_universe=candidate_source_universe,
    )
    candidate_scores = dataset_grn.unsqueeze(0)[candidate_mask].to(dtype=torch.float64)
    candidate_keys = pair_keys[candidate_mask].to(dtype=torch.long)
    labels = torch.isin(
        candidate_keys,
        reference_cache.pair_keys.to(dtype=torch.long),
    ).to(dtype=torch.float64)

    if candidate_scores.numel() == 0:
        raise RuntimeError("No candidate edges were available for evaluation.")

    selected_metric_names = list(metric_names)
    metrics = summarize_binary_metrics(
        scores=candidate_scores,
        labels=labels,
        metric_names=selected_metric_names,
        random_positive_rate=all_gene_random_positive_rate(
            tokens=token_template,
            cache=reference_cache,
        ),
    )

    resolved_dataset_path = (
        str(Path(dataset_path).expanduser().resolve()) if dataset_path is not None else ""
    )
    metrics_df = pd.DataFrame(
        [
            {
                "aggregation": "dataset",
                "dataset_name": dataset_name or (Path(dataset_path).stem if dataset_path else ""),
                "dataset_path": resolved_dataset_path,
                "evaluation_grn_name": reference_grn_name or reference_path.stem,
                "evaluation_grn_path": str(reference_path),
                "num_cells": int(cell_count),
                "num_candidate_edges": int(candidate_scores.numel()),
                "num_positive_edges": int(labels.sum().item()),
                **metrics,
            }
        ]
    )

    if output_metrics_csv is not None:
        output_path = Path(output_metrics_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
    return metrics_df


def evaluate_dataset_level_grn(
    *,
    input_h5ad: str | Path,
    reference_grn_csv: str | Path,
    output_metrics_csv: str | Path,
    model_source: str | Path = "assets",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = "configs/eval_grn.yaml",
    batch_size: int = 8,
    max_length: int = 4096,
    n_top_genes: int | None = None,
    preserve_tf_species: str = "mouse",
    candidate_source_universe: str = "supported_tfs",
    metric_names: Iterable[str] = ("auprc", "auroc", "ep"),
    dataset_name: str | None = None,
    reference_grn_name: str | None = None,
    gene_key: str | None = None,
    species_key: str | None = "species",
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = "disease",
) -> pd.DataFrame:
    """Evaluate the mean dataset-level SFM GRN against a reference GRN."""

    context = prepare_dataset_grn_evaluation_context(
        input_h5ad=input_h5ad,
        model_source=model_source,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        max_length=max_length,
        n_top_genes=n_top_genes,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
        preserve_tf_species=preserve_tf_species,
    )
    forward_results = collect_sfm_forward_results(context)
    return evaluate_dataset_level_forward_results(
        forward_results,
        context,
        reference_grn_csv=reference_grn_csv,
        output_metrics_csv=output_metrics_csv,
        candidate_source_universe=candidate_source_universe,
        metric_names=metric_names,
        dataset_name=dataset_name,
        dataset_path=input_h5ad,
        reference_grn_name=reference_grn_name,
    )
