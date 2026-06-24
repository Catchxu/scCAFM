from __future__ import annotations

import contextlib

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from torch.utils.data import DataLoader, SequentialSampler

from ..assets import (
    EFM_MODEL_NAME,
    MODELS_DIR_NAME,
    EFM_DIR_NAME,
    ModelAssets,
    apply_model_assets_to_runtime_config,
    load_efm_config,
    load_model_state_dict,
    load_sfm_config,
    load_table_json,
    load_vocab_json,
    resolve_model_assets,
    resolve_sfm_checkpoint_path,
)
from ..config import load_yaml_config
from ..data import (
    PreprocessedScDataset,
    PretrainingAssets,
    PretrainingDataBundle,
    ScBatchCollator,
    ScTokenizer,
)
from ..distributed import RuntimeContext, move_batch_to_device
from ..models import EFM, reorder_gene_aligned_tokens
from ..trainer.builders import build_model


__all__ = [
    "CellFateRun",
    "predict",
    "predict_cell_fate_transition",
    "prepare",
]


GENERATION_MODES = {"iterative_replace", "one_forward"}


@dataclass(slots=True)
class CellFateRun:
    config: dict[str, Any]
    assets: ModelAssets
    sfm_config: dict[str, Any]
    efm_config: dict[str, Any]
    sfm_checkpoint_file: Path
    efm_checkpoint_file: Path
    device: torch.device
    runtime: RuntimeContext
    data_assets: PretrainingAssets
    adata: ad.AnnData
    dataset: PreprocessedScDataset
    loader: DataLoader
    sfm: torch.nn.Module
    efm: torch.nn.Module
    inference_dtype: torch.dtype


def prepare(
    *,
    input_h5ad: str | Path | None = None,
    adata: ad.AnnData | None = None,
    model_source: str | Path = "assets",
    config_path: str | Path = "configs/eval_cell_fate.yaml",
    sfm_checkpoint_path: str | Path | None = None,
    efm_checkpoint_path: str | Path | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    gene_key: str | None = None,
    species_key: str | None = None,
    platform_key: str | None = None,
    tissue_key: str | None = None,
    disease_key: str | None = None,
    condition_defaults: dict[str, str] | None = None,
) -> CellFateRun:
    """
    Prepare pretrained SFM and EFM for cell-fate transition prediction.

    The input AnnData is assumed to be preprocessed. This function does not run
    QC filtering, normalization, HVG selection, or gene filtering.
    """

    if input_h5ad is None and adata is None:
        raise ValueError("Provide either `input_h5ad` or `adata`.")
    if input_h5ad is not None and adata is not None:
        raise ValueError("Provide only one of `input_h5ad` or `adata`, not both.")

    input_path: Path | None = None
    if input_h5ad is not None:
        input_path = Path(input_h5ad).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input AnnData file does not exist: {input_path}")
        source_adata = ad.read_h5ad(input_path)
    else:
        if not isinstance(adata, ad.AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got {type(adata).__name__}.")
        source_adata = adata.copy()

    base_config = load_yaml_config(config_path)
    base_config["model_source"] = str(model_source)
    data_cfg = _apply_data_overrides(
        base_config.get("data", {}),
        batch_size=batch_size,
        max_length=max_length,
        gene_key=gene_key,
        species_key=species_key,
        platform_key=platform_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
        condition_defaults=condition_defaults,
    )
    base_config["data"] = data_cfg
    cell_fate_cfg = base_config.setdefault("cell_fate", {})
    if sfm_checkpoint_path is not None:
        cell_fate_cfg["sfm_checkpoint_path"] = str(sfm_checkpoint_path)
    if efm_checkpoint_path is not None:
        cell_fate_cfg["efm_checkpoint_path"] = str(efm_checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = _runtime_context(device)
    assets = resolve_model_assets(
        model_source=base_config["model_source"],
        require_model_weights=cell_fate_cfg.get("sfm_checkpoint_path") is None,
        require_efm_config=True,
        require_efm_weights=cell_fate_cfg.get("efm_checkpoint_path") is None,
    )
    config = apply_model_assets_to_runtime_config(
        base_config,
        assets,
        require_model_weights=cell_fate_cfg.get("sfm_checkpoint_path") is None,
    )
    sfm_config = load_sfm_config(assets.sfm_config)
    efm_config = load_efm_config(assets.efm_config)
    config["model"] = sfm_config
    config["efm_model"] = efm_config

    prepared_adata = _apply_condition_defaults(
        source_adata,
        data_cfg=config["data"],
    )
    _validate_preprocessed_adata(
        prepared_adata,
        max_length=int(config["data"]["max_length"]),
        gene_key=config["data"].get("gene_key"),
    )

    data_assets = _build_inference_assets(
        config=config,
        assets=assets,
        input_path=input_path,
    )
    _validate_gene_tokenization(
        prepared_adata,
        data_assets=data_assets,
    )

    dataset = PreprocessedScDataset(
        adata=prepared_adata,
        tokenizer=data_assets.tokenizer,
        gene_key=data_assets.gene_key,
        preprocessor=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
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
        path=input_path or Path("<memory>"),
    )
    inference_dtype = _resolve_inference_dtype(config, device)
    sfm = build_model(
        sfm_config=sfm_config,
        data_bundle=model_bundle,
        assets=assets,
        runtime_config=config.get("runtime", {}),
    )
    sfm_checkpoint_file = _resolve_sfm_checkpoint_path(
        config.get("cell_fate", {}).get("sfm_checkpoint_path"),
        assets.sfm_model,
    )
    sfm.load_state_dict(load_model_state_dict(sfm_checkpoint_file), strict=True)
    sfm.to(device=device, dtype=inference_dtype)
    sfm.eval()
    sfm.requires_grad_(False)

    efm = _build_efm(
        config=config,
        data_assets=data_assets,
        assets=assets,
    )
    efm_checkpoint_file = _resolve_efm_checkpoint_path(
        config.get("cell_fate", {}).get("efm_checkpoint_path"),
        assets.efm_model,
    )
    efm.load_state_dict(load_model_state_dict(efm_checkpoint_file), strict=True)
    efm.to(device=device, dtype=inference_dtype)
    efm.eval()
    efm.requires_grad_(False)

    return CellFateRun(
        config=config,
        assets=assets,
        sfm_config=sfm_config,
        efm_config=efm_config,
        sfm_checkpoint_file=sfm_checkpoint_file,
        efm_checkpoint_file=efm_checkpoint_file,
        device=device,
        runtime=runtime,
        data_assets=data_assets,
        adata=prepared_adata,
        dataset=dataset,
        loader=loader,
        sfm=sfm,
        efm=efm,
        inference_dtype=inference_dtype,
    )


def predict(
    run: CellFateRun,
    perturb_gene: str,
    layer_key: str | None = None,
) -> ad.AnnData:
    """Generate perturbed expression profiles and save them in an AnnData layer."""

    if not isinstance(run, CellFateRun):
        raise TypeError(f"`run` must be a CellFateRun, got {type(run).__name__}.")

    resolved_layer_key = layer_key or str(
        run.config.get("cell_fate", {}).get("layer_key", "cell_fate_transition")
    )
    generation_mode = _resolve_generation_mode(run.config)
    perturb_token_id = _resolve_perturb_token_id(
        perturb_gene,
        data_assets=run.data_assets,
    )
    perturb_position = _find_unique_token_position(
        run.dataset.tokenized.input_ids[0],
        token_id=perturb_token_id,
        gene_name=perturb_gene,
    )

    generated_batches: list[np.ndarray] = []
    with torch.no_grad(), _inference_autocast_context(
        run.device,
        run.inference_dtype,
    ):
        for batch in run.loader:
            tokens = move_batch_to_device(batch, run.device)
            generated = _predict_batch(
                run=run,
                tokens=tokens,
                perturb_position=perturb_position,
                generation_mode=generation_mode,
            )
            generated_batches.append(generated.detach().cpu().to(torch.float32).numpy())

    if not generated_batches:
        raise RuntimeError("No expression profiles were generated.")

    generated_matrix = np.concatenate(generated_batches, axis=0)
    generated_matrix = generated_matrix[:, : run.adata.n_vars]
    if generated_matrix.shape != run.adata.shape:
        raise RuntimeError(
            "Generated expression shape does not match AnnData shape: "
            f"{generated_matrix.shape} vs {run.adata.shape}."
        )

    output = run.adata.copy()
    output.layers[resolved_layer_key] = generated_matrix.astype(np.float32, copy=False)
    output.obs[f"{resolved_layer_key}_generated"] = True
    output.var[f"{resolved_layer_key}_generated"] = True
    output.uns[resolved_layer_key] = {
        "perturb_gene": str(perturb_gene),
        "perturb_token_id": int(perturb_token_id),
        "perturb_position": int(perturb_position),
        "model_source": str(run.assets.model_source),
        "sfm_checkpoint": str(run.sfm_checkpoint_file),
        "efm_checkpoint": str(run.efm_checkpoint_file),
        "generation_mode": generation_mode,
        "gene_id_decoding": "fixed_input_gene_ids",
        "use_kv_cache": (
            _use_kv_cache(run.config) if generation_mode == "iterative_replace" else False
        ),
        "expression_scale": "same scale as the preprocessed input AnnData",
    }
    return output


def predict_cell_fate_transition(
    *,
    input_h5ad: str | Path,
    perturb_gene: str,
    output_h5ad: str | Path | None = None,
    layer_key: str | None = None,
    **prepare_kwargs: Any,
) -> ad.AnnData:
    """Convenience wrapper around `prepare(...)` and `predict(...)`."""

    run = prepare(input_h5ad=input_h5ad, **prepare_kwargs)
    output = predict(run, perturb_gene=perturb_gene, layer_key=layer_key)
    if output_h5ad is not None:
        output.write_h5ad(Path(output_h5ad).expanduser().resolve())
    return output


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
    updated = deepcopy(data_cfg)
    if batch_size is not None:
        updated["batch_size"] = int(batch_size)
    if max_length is not None:
        updated["max_length"] = int(max_length)
    for key, value in {
        "gene_key": gene_key,
        "species_key": species_key,
        "platform_key": platform_key,
        "tissue_key": tissue_key,
        "disease_key": disease_key,
    }.items():
        if value is not None:
            updated[key] = value
    defaults = dict(updated.get("condition_defaults", {}) or {})
    if condition_defaults is not None:
        defaults.update({str(k): str(v) for k, v in condition_defaults.items()})
    updated["condition_defaults"] = defaults
    updated["condition_vocab"] = {"regenerate": False}
    updated["condition_mask"] = {"enabled": False, "unk_ratio": 0.0}
    updated["num_workers"] = 0
    updated["pin_memory"] = torch.cuda.is_available()
    return updated


def _runtime_context(device: torch.device) -> RuntimeContext:
    return RuntimeContext(
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
        distributed=False,
        is_main=True,
    )


def _resolve_inference_dtype(config: dict[str, Any], device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    precision_cfg = config.get("runtime", {}).get("precision", {})
    requested = str(precision_cfg.get("autocast_dtype", "bf16")).lower()
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_generation_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("cell_fate", {}).get("generation_mode", "iterative_replace"))
    if mode not in GENERATION_MODES:
        raise ValueError(
            f"Unsupported cell-fate generation_mode={mode!r}. "
            f"Supported modes: {sorted(GENERATION_MODES)}."
        )
    return mode


def _use_kv_cache(config: dict[str, Any]) -> bool:
    return bool(config.get("cell_fate", {}).get("use_kv_cache", False))


def _inference_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _resolve_sfm_checkpoint_path(checkpoint_path: object, default_model_path: Path) -> Path:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        resolved = default_model_path
    else:
        resolved = resolve_sfm_checkpoint_path(Path(str(checkpoint_path)).expanduser().resolve())
    if not resolved.exists():
        raise FileNotFoundError(f"SFM checkpoint not found: {resolved}")
    return resolved


def _resolve_efm_checkpoint_path(checkpoint_path: object, default_model_path: Path) -> Path:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        resolved = default_model_path
    else:
        candidate = Path(str(checkpoint_path)).expanduser().resolve()
        if candidate.is_file():
            resolved = candidate
        else:
            resolved = candidate / MODELS_DIR_NAME / EFM_DIR_NAME / EFM_MODEL_NAME
            if not resolved.exists():
                resolved = candidate / EFM_MODEL_NAME
    if not resolved.exists():
        raise FileNotFoundError(f"EFM checkpoint not found: {resolved}")
    return resolved


def _load_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        return load_table_json(resolved)
    return pd.read_csv(resolved)


def _build_inference_assets(
    *,
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


def _build_efm(
    *,
    config: dict[str, Any],
    data_assets: PretrainingAssets,
    assets: ModelAssets,
) -> EFM:
    efm_kwargs = deepcopy(config["efm_model"]["efm"])
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


def _apply_condition_defaults(
    adata: ad.AnnData,
    *,
    data_cfg: dict[str, Any],
) -> ad.AnnData:
    prepared = adata.copy()
    defaults = data_cfg.get("condition_defaults", {}) or {}
    added: dict[str, str] = {}
    for condition_name, key_name in (
        ("species", data_cfg.get("species_key")),
        ("disease", data_cfg.get("disease_key")),
        ("platform", data_cfg.get("platform_key")),
        ("tissue", data_cfg.get("tissue_key")),
    ):
        if key_name is None or key_name in prepared.obs:
            continue
        if condition_name not in defaults:
            continue
        prepared.obs[key_name] = str(defaults[condition_name])
        added[key_name] = str(defaults[condition_name])
    if added:
        prepared.uns.setdefault("cell_fate_condition_defaults", {}).update(added)
    return prepared


def _to_dense_array(X: Any) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    if sp.issparse(X):
        return X.toarray()
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _validate_preprocessed_adata(
    adata: ad.AnnData,
    *,
    max_length: int,
    gene_key: str | None,
) -> None:
    if adata.n_obs == 0:
        raise ValueError("Input AnnData has no cells.")
    if adata.n_vars == 0:
        raise ValueError("Input AnnData has no genes.")
    if int(adata.n_vars) > int(max_length) - 1:
        raise ValueError(
            f"Input AnnData has {adata.n_vars} genes, but max_length={max_length} "
            "only leaves room for max_length - 1 gene tokens."
        )
    if gene_key is not None and gene_key not in adata.var.columns:
        raise KeyError(
            f"`gene_key={gene_key}` not found in `adata.var.columns`. "
            f"Available columns: {list(adata.var.columns)}"
        )

    X = _to_dense_array(adata.X)
    if X.ndim != 2:
        raise ValueError(f"`adata.X` must be 2D, got shape {X.shape}.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"`adata.X` must be numeric, got dtype {X.dtype}.")
    if not np.isfinite(X).all():
        raise ValueError("`adata.X` contains NaN or infinite values.")


def _validate_gene_tokenization(
    adata: ad.AnnData,
    *,
    data_assets: PretrainingAssets,
) -> None:
    gene_tokenizer = data_assets.tokenizer.gene_tokenizer
    gene_names = gene_tokenizer._extract_gene_names(adata, gene_key=data_assets.gene_key)
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


def _resolve_perturb_token_id(
    perturb_gene: str,
    *,
    data_assets: PretrainingAssets,
) -> int:
    if not str(perturb_gene).strip():
        raise ValueError("`perturb_gene` must be a non-empty gene name.")
    gene_tokenizer = data_assets.tokenizer.gene_tokenizer
    token_id = int(gene_tokenizer.encode_gene_list([perturb_gene])[0].item())
    if token_id == int(gene_tokenizer.pad_index):
        raise ValueError(f"`perturb_gene={perturb_gene}` was not found in model vocab.")
    return token_id


def _find_unique_token_position(
    input_ids: torch.Tensor,
    *,
    token_id: int,
    gene_name: str,
) -> int:
    matches = (input_ids.to(torch.long) == int(token_id)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        raise ValueError(f"`perturb_gene={gene_name}` is not present in the input AnnData genes.")
    if matches.numel() > 1:
        raise ValueError(
            f"`perturb_gene={gene_name}` maps to token {token_id}, which appears "
            f"{matches.numel()} times in the input genes."
        )
    return int(matches.item())


def _active_mask(tokens: dict[str, torch.Tensor | None]) -> torch.BoolTensor:
    expression_values = tokens.get("expression_values")
    if not torch.is_tensor(expression_values):
        raise TypeError("`tokens['expression_values']` must be a tensor.")
    padding_mask = tokens.get("padding_mask")
    if padding_mask is None:
        return torch.ones_like(expression_values, dtype=torch.bool)
    if not torch.is_tensor(padding_mask):
        raise TypeError("`tokens['padding_mask']` must be a tensor or None.")
    return ~padding_mask.to(device=expression_values.device, dtype=torch.bool)


def _perturb_expression_values(
    tokens: dict[str, torch.Tensor | None],
    *,
    perturb_position: int,
) -> dict[str, torch.Tensor | None]:
    expression_values = tokens.get("expression_values")
    if not torch.is_tensor(expression_values):
        raise TypeError("`tokens['expression_values']` must be a tensor.")
    if not 0 <= int(perturb_position) < expression_values.shape[1]:
        raise ValueError(f"`perturb_position` out of range: {perturb_position}")

    active_mask = _active_mask(tokens)
    if not bool(active_mask[:, int(perturb_position)].all()):
        raise ValueError("The perturbation gene position is padded for at least one cell.")
    masked = expression_values.masked_fill(~active_mask, -torch.inf)
    max_values = masked.max(dim=1).values
    if not bool(torch.isfinite(max_values).all()):
        raise ValueError("Unable to compute per-cell expression maxima.")

    perturbed = dict(tokens)
    updated_expr = expression_values.clone()
    updated_expr[:, int(perturb_position)] = max_values.to(updated_expr.dtype)
    perturbed["expression_values"] = updated_expr
    return perturbed


def _inverse_reorder_expression(
    ordered_expression: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if ordered_expression.shape != positions.shape:
        raise ValueError(
            "`ordered_expression` and `positions` must have the same shape, "
            f"got {tuple(ordered_expression.shape)} and {tuple(positions.shape)}."
        )
    restored = torch.empty_like(ordered_expression)
    return restored.scatter(dim=1, index=positions.to(torch.long), src=ordered_expression)


def _predict_batch(
    *,
    run: CellFateRun,
    tokens: dict[str, torch.Tensor | None],
    perturb_position: int,
    generation_mode: str | None = None,
) -> torch.Tensor:
    sfm_output = run.sfm(
        tokens,
        compute_order={"sfm": True},
        compute_grn=False,
        return_factors=False,
    )
    gene_order = sfm_output.foundations["sfm"].gene_order
    if gene_order is None:
        raise RuntimeError("SFM did not return a gene order.")

    perturbed_tokens = _perturb_expression_values(
        tokens,
        perturb_position=perturb_position,
    )
    ordered_tokens = reorder_gene_aligned_tokens(perturbed_tokens, gene_order)
    ordered_expr = ordered_tokens["expression_values"]
    if not torch.is_tensor(ordered_expr):
        raise TypeError("Reordered tokens are missing tensor entry `expression_values`.")

    generated_expr = ordered_expr.clone()
    active_lengths = gene_order.active_lengths.to(device=generated_expr.device, dtype=torch.long)
    ordered_perturb_mask = (
        gene_order.positions.to(device=generated_expr.device, dtype=torch.long)
        == int(perturb_position)
    )
    max_active_length = int(active_lengths.max().item()) if active_lengths.numel() else 0
    clamp_min = run.config.get("cell_fate", {}).get("clamp_min", 0.0)
    mode = generation_mode or _resolve_generation_mode(run.config)

    if mode == "one_forward":
        output = run.efm(ordered_tokens)
        predicted_expr = output.expression_pred[:, : generated_expr.shape[1]].to(
            generated_expr.dtype
        )
        if clamp_min is not None:
            predicted_expr = predicted_expr.clamp_min(float(clamp_min))
        active_mask = _active_mask(ordered_tokens)
        update_mask = active_mask & ~ordered_perturb_mask
        generated_expr[update_mask] = predicted_expr[update_mask]
        return _inverse_reorder_expression(
            generated_expr,
            gene_order.positions.to(device=generated_expr.device, dtype=torch.long),
        )

    if mode != "iterative_replace":
        raise ValueError(
            f"Unsupported cell-fate generation_mode={mode!r}. "
            f"Supported modes: {sorted(GENERATION_MODES)}."
        )

    if _use_kv_cache(run.config):
        return _predict_batch_iterative_cached(
            run=run,
            ordered_tokens=ordered_tokens,
            generated_expr=generated_expr,
            gene_order_positions=gene_order.positions.to(
                device=generated_expr.device,
                dtype=torch.long,
            ),
            active_lengths=active_lengths,
            ordered_perturb_mask=ordered_perturb_mask,
            max_active_length=max_active_length,
            clamp_min=clamp_min,
        )

    for step in range(max_active_length):
        step_rows = active_lengths > step
        if not bool(step_rows.any()):
            continue
        step_tokens = dict(ordered_tokens)
        step_tokens["expression_values"] = generated_expr
        output = run.efm(step_tokens)
        step_pred = output.expression_pred[:, step].to(generated_expr.dtype)
        if clamp_min is not None:
            step_pred = step_pred.clamp_min(float(clamp_min))
        update_rows = step_rows & ~ordered_perturb_mask[:, step]
        generated_expr[update_rows, step] = step_pred[update_rows]

    return _inverse_reorder_expression(
        generated_expr,
        gene_order.positions.to(device=generated_expr.device, dtype=torch.long),
    )


def _predict_batch_iterative_cached(
    *,
    run: CellFateRun,
    ordered_tokens: dict[str, torch.Tensor | None],
    generated_expr: torch.Tensor,
    gene_order_positions: torch.Tensor,
    active_lengths: torch.Tensor,
    ordered_perturb_mask: torch.Tensor,
    max_active_length: int,
    clamp_min: object,
) -> torch.Tensor:
    required_methods = (
        "prefill_incremental_cache",
        "append_incremental_gene",
        "predict_expression_from_hidden",
    )
    missing = [name for name in required_methods if not hasattr(run.efm, name)]
    if missing:
        raise TypeError(f"EFM model does not support KV-cache decoding: missing {missing}.")

    hidden, caches = run.efm.prefill_incremental_cache(ordered_tokens)
    for step in range(max_active_length):
        step_rows = active_lengths > step
        if bool(step_rows.any()):
            step_pred = run.efm.predict_expression_from_hidden(hidden).to(generated_expr.dtype)
            if clamp_min is not None:
                step_pred = step_pred.clamp_min(float(clamp_min))
            update_rows = step_rows & ~ordered_perturb_mask[:, step]
            generated_expr[update_rows, step] = step_pred[update_rows]

        if step + 1 >= max_active_length:
            continue

        token_padding_mask = (active_lengths <= step).unsqueeze(1)
        hidden, caches = run.efm.append_incremental_gene(
            ordered_tokens,
            expression_values=generated_expr,
            gene_position=step,
            caches=caches,
            key_padding_mask=token_padding_mask,
        )

    return _inverse_reorder_expression(generated_expr, gene_order_positions)
