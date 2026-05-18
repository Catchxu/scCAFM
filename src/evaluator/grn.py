from __future__ import annotations

import argparse
import contextlib
import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

from ..assets import (
    ModelAssets,
    SFM_MODEL_NAME,
    apply_model_assets_to_runtime_config,
    load_model_state_dict,
    load_sfm_config,
    resolve_model_assets,
)
from .metrics import summarize_binary_metrics
from ..config import load_yaml_config
from ..data import (
    PretrainingDataBundle,
    build_data_bundle_for_path,
    build_evaluation_assets,
)
from ..distributed import (
    RuntimeContext,
    barrier,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
    move_batch_to_device,
)
from ..experiment import ExperimentLogger, ExperimentPaths
from ..models.wrapper import ModelWrapperOutput
from ..trainer.builders import build_model, maybe_wrap_fsdp
from ..utils import build_active_gene_mask, build_tf_mask, build_token_lookup_maps, require_tensor


@dataclass
class EvaluationGRNCache:
    pair_keys: torch.LongTensor
    supported_token_ids: torch.LongTensor
    source_token_ids: torch.LongTensor
    target_token_ids: torch.LongTensor
    pair_key_base: int
    raw_edge_count: int = 0
    mapped_edge_count: int = 0
    unmapped_edge_count: int = 0
    self_loop_edge_count: int = 0
    duplicate_edge_count: int = 0


@dataclass
class EvaluationGRNSpec:
    name: str
    path: Path
    dataframe: pd.DataFrame
    cache: EvaluationGRNCache


@dataclass
class DenseDatasetGRNAccumulator:
    score_sum: torch.Tensor | None = None
    cell_count: int = 0
    token_template: dict[str, torch.Tensor | None] | None = None


@dataclass
class FinalizedDatasetGRN:
    score_mean: torch.Tensor
    cell_count: int
    token_template: dict[str, torch.Tensor | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted cell-specific GRNs.")
    parser.add_argument(
        "--eval-grn-config",
        "--eval-config",
        dest="eval_grn_config",
        default="configs/eval_grn.yaml",
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        default=None,
        help=(
            "Optional model checkpoint directory or weight file. Directories are "
            f"resolved as <dir>/{SFM_MODEL_NAME}."
        ),
    )
    return parser.parse_args()


def _autocast_context(config: dict[str, object], runtime: RuntimeContext):
    precision_cfg = config.get("runtime", {}).get("precision", {})
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


def _map_gene_to_token(
    name: str,
    symbol_to_index: dict[str, int],
    ensembl_to_index: dict[str, int],
) -> Optional[int]:
    key = str(name).strip().upper()
    if key.startswith("ENSG"):
        key = key.split(".", 1)[0]
        return ensembl_to_index.get(key)
    return symbol_to_index.get(key, ensembl_to_index.get(key))


def build_evaluation_grn_cache(
    token_dict: pd.DataFrame,
    evaluation_grn_df: pd.DataFrame,
) -> EvaluationGRNCache:
    if not isinstance(evaluation_grn_df, pd.DataFrame):
        raise TypeError("`evaluation_grn_df` must be a pandas DataFrame.")
    if not {"Gene1", "Gene2"}.issubset(evaluation_grn_df.columns):
        raise ValueError("`evaluation_grn_df` must contain columns: 'Gene1' and 'Gene2'.")

    symbol_to_index, ensembl_to_index, pad_index = build_token_lookup_maps(token_dict)
    pair_key_base = max(int(token_dict["token_index"].max()) + 1, 1)

    pair_keys: list[int] = []
    supported_token_ids: set[int] = set()
    source_token_ids: set[int] = set()
    target_token_ids: set[int] = set()
    raw_edge_count = int(len(evaluation_grn_df))
    unmapped_edge_count = 0
    self_loop_edge_count = 0
    for src_name, tgt_name in zip(
        evaluation_grn_df["Gene1"].tolist(),
        evaluation_grn_df["Gene2"].tolist(),
    ):
        src_id = _map_gene_to_token(src_name, symbol_to_index, ensembl_to_index)
        tgt_id = _map_gene_to_token(tgt_name, symbol_to_index, ensembl_to_index)
        if src_id is None or tgt_id is None:
            unmapped_edge_count += 1
            continue
        if src_id == pad_index or tgt_id == pad_index:
            unmapped_edge_count += 1
            continue
        if src_id == tgt_id:
            self_loop_edge_count += 1
            continue
        pair_keys.append(src_id * pair_key_base + tgt_id)
        supported_token_ids.add(src_id)
        supported_token_ids.add(tgt_id)
        source_token_ids.add(src_id)
        target_token_ids.add(tgt_id)

    pair_key_tensor = (
        torch.unique(torch.tensor(pair_keys, dtype=torch.long))
        if pair_keys
        else torch.empty(0, dtype=torch.long)
    )
    supported_token_tensor = (
        torch.unique(torch.tensor(sorted(supported_token_ids), dtype=torch.long))
        if supported_token_ids
        else torch.empty(0, dtype=torch.long)
    )
    source_token_tensor = (
        torch.unique(torch.tensor(sorted(source_token_ids), dtype=torch.long))
        if source_token_ids
        else torch.empty(0, dtype=torch.long)
    )
    target_token_tensor = (
        torch.unique(torch.tensor(sorted(target_token_ids), dtype=torch.long))
        if target_token_ids
        else torch.empty(0, dtype=torch.long)
    )

    return EvaluationGRNCache(
        pair_keys=pair_key_tensor,
        supported_token_ids=supported_token_tensor,
        source_token_ids=source_token_tensor,
        target_token_ids=target_token_tensor,
        pair_key_base=pair_key_base,
        raw_edge_count=raw_edge_count,
        mapped_edge_count=len(pair_keys),
        unmapped_edge_count=unmapped_edge_count,
        self_loop_edge_count=self_loop_edge_count,
        duplicate_edge_count=max(len(pair_keys) - int(pair_key_tensor.numel()), 0),
    )


def build_reference_grn(
    tokens: dict[str, torch.Tensor | None],
    cache: EvaluationGRNCache,
    candidate_source_universe: str = "supported_tfs",
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    input_ids = require_tensor(tokens, "input_ids").to(torch.long)
    padding_mask = tokens.get("padding_mask")
    non_tf_mask = tokens.get("non_tf_mask")

    if input_ids.ndim != 2:
        raise ValueError(f"`tokens['input_ids']` must have shape (C, G), got {tuple(input_ids.shape)}.")

    active_gene_mask = build_active_gene_mask(input_ids=input_ids, padding_mask=padding_mask)
    if torch.is_tensor(non_tf_mask):
        source_mask = build_tf_mask(
            input_ids=input_ids,
            non_tf_mask=non_tf_mask,
            padding_mask=padding_mask,
        )
    else:
        source_mask = active_gene_mask

    if cache.supported_token_ids.numel() == 0:
        empty_mask = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]),
            dtype=torch.bool,
            device=input_ids.device,
        )
        return empty_mask, empty_mask

    supported_token_ids = cache.supported_token_ids.to(device=input_ids.device)
    supported_gene_mask = torch.isin(input_ids, supported_token_ids) & active_gene_mask
    candidate_source_universe = _normalize_candidate_source_universe(candidate_source_universe)
    if candidate_source_universe == "supported_tfs":
        source_token_ids = cache.source_token_ids.to(device=input_ids.device)
        source_mask = source_mask & torch.isin(input_ids, source_token_ids)

    candidate_mask = source_mask.unsqueeze(2) & supported_gene_mask.unsqueeze(1)
    diagonal_mask = torch.eye(
        input_ids.shape[1],
        dtype=torch.bool,
        device=input_ids.device,
    ).unsqueeze(0)
    candidate_mask = candidate_mask & ~diagonal_mask

    if cache.pair_keys.numel() == 0:
        target = torch.zeros_like(candidate_mask)
        return target, candidate_mask

    pair_keys = (
        input_ids.unsqueeze(2).to(torch.long) * cache.pair_key_base
        + input_ids.unsqueeze(1).to(torch.long)
    )
    positive_mask = torch.isin(pair_keys, cache.pair_keys.to(device=input_ids.device))
    target = positive_mask & candidate_mask
    return target, candidate_mask


def build_candidate_pair_keys(
    tokens: dict[str, torch.Tensor | None],
    cache: EvaluationGRNCache,
    candidate_source_universe: str = "supported_tfs",
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    input_ids = require_tensor(tokens, "input_ids").to(torch.long)
    padding_mask = tokens.get("padding_mask")
    non_tf_mask = tokens.get("non_tf_mask")

    if input_ids.ndim != 2:
        raise ValueError(f"`tokens['input_ids']` must have shape (C, G), got {tuple(input_ids.shape)}.")

    active_gene_mask = build_active_gene_mask(input_ids=input_ids, padding_mask=padding_mask)
    if torch.is_tensor(non_tf_mask):
        source_mask = build_tf_mask(
            input_ids=input_ids,
            non_tf_mask=non_tf_mask,
            padding_mask=padding_mask,
        )
    else:
        source_mask = active_gene_mask

    if cache.supported_token_ids.numel() == 0:
        empty_mask = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]),
            dtype=torch.bool,
            device=input_ids.device,
        )
        empty_keys = torch.zeros_like(empty_mask, dtype=torch.long)
        return empty_keys, empty_mask

    supported_token_ids = cache.supported_token_ids.to(device=input_ids.device)
    supported_gene_mask = torch.isin(input_ids, supported_token_ids) & active_gene_mask
    candidate_source_universe = _normalize_candidate_source_universe(candidate_source_universe)
    if candidate_source_universe == "supported_tfs":
        source_token_ids = cache.source_token_ids.to(device=input_ids.device)
        source_mask = source_mask & torch.isin(input_ids, source_token_ids)
    candidate_mask = source_mask.unsqueeze(2) & supported_gene_mask.unsqueeze(1)
    diagonal_mask = torch.eye(
        input_ids.shape[1],
        dtype=torch.bool,
        device=input_ids.device,
    ).unsqueeze(0)
    candidate_mask = candidate_mask & ~diagonal_mask

    pair_keys = (
        input_ids.unsqueeze(2).to(torch.long) * cache.pair_key_base
        + input_ids.unsqueeze(1).to(torch.long)
    )
    return pair_keys, candidate_mask


def evaluate_cell_specific_grns(
    pred_grn: torch.Tensor,
    tokens: dict[str, torch.Tensor | None],
    token_dict: pd.DataFrame,
    evaluation_grn_df: pd.DataFrame,
    candidate_source_universe: str = "supported_tfs",
) -> pd.DataFrame:
    if pred_grn.ndim != 3:
        raise ValueError(f"`pred_grn` must have shape (C, G, G), got {tuple(pred_grn.shape)}.")

    input_ids = require_tensor(tokens, "input_ids")
    if pred_grn.shape[:2] != input_ids.shape or pred_grn.shape[2] != input_ids.shape[1]:
        raise ValueError(
            f"`pred_grn` shape {tuple(pred_grn.shape)} is incompatible with input_ids {tuple(input_ids.shape)}."
        )

    cache = build_evaluation_grn_cache(
        token_dict=token_dict,
        evaluation_grn_df=evaluation_grn_df,
    )
    return evaluate_cell_specific_grns_with_cache(
        pred_grn=pred_grn,
        tokens=tokens,
        cache=cache,
        candidate_source_universe=candidate_source_universe,
    )


def evaluate_cell_specific_grns_with_cache(
    pred_grn: torch.Tensor,
    tokens: dict[str, torch.Tensor | None],
    cache: EvaluationGRNCache,
    candidate_source_universe: str = "supported_tfs",
) -> pd.DataFrame:
    if pred_grn.ndim != 3:
        raise ValueError(f"`pred_grn` must have shape (C, G, G), got {tuple(pred_grn.shape)}.")

    input_ids = require_tensor(tokens, "input_ids")
    if pred_grn.shape[:2] != input_ids.shape or pred_grn.shape[2] != input_ids.shape[1]:
        raise ValueError(
            f"`pred_grn` shape {tuple(pred_grn.shape)} is incompatible with input_ids {tuple(input_ids.shape)}."
        )

    target, candidate_mask = build_reference_grn(
        tokens=tokens,
        cache=cache,
        candidate_source_universe=candidate_source_universe,
    )

    pred_cpu = pred_grn.detach().to(dtype=torch.float32, device="cpu")
    target_cpu = target.detach().to(dtype=torch.bool, device="cpu")
    candidate_mask_cpu = candidate_mask.detach().to(dtype=torch.bool, device="cpu")

    rows: list[dict[str, float | int]] = []
    for cell_idx in range(pred_cpu.shape[0]):
        cell_mask = candidate_mask_cpu[cell_idx]
        scores = pred_cpu[cell_idx][cell_mask]
        labels = target_cpu[cell_idx][cell_mask].to(dtype=torch.float32)

        if scores.numel() == 0:
            metrics = {
                "auprc": float("nan"),
                "auprc_ratio": float("nan"),
                "auroc": float("nan"),
                "ep": float("nan"),
                "ep_ratio": float("nan"),
            }
        else:
            metrics = summarize_binary_metrics(scores=scores, labels=labels)

        rows.append(
            {
                "cell_index": cell_idx,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def prepare_evaluation_paths(
    runtime: RuntimeContext,
) -> ExperimentPaths:
    root_candidate = None
    if runtime.is_main:
        root_candidate = Path.cwd().resolve()

    root = Path(broadcast_object(str(root_candidate) if root_candidate is not None else None))
    logs = root / "logs"
    results = root / "results"
    checkpoints = root / "checkpoints"
    model_package_dir = checkpoints / "models"

    if runtime.is_main:
        logs.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
    barrier()

    return ExperimentPaths(
        root=root,
        logs=logs,
        checkpoints=checkpoints,
        model_package_dir=model_package_dir,
        log_file=logs / "evaluate_grn.log",
        resume_manifest_file=checkpoints / "resume_manifest.json",
        resume_state_file=checkpoints / "sfm_train_state.pt",
    )


def _prepare_evaluation_config(config: dict[str, object]) -> dict[str, object]:
    prepared = deepcopy(config)
    data_cfg = dict(prepared.get("data", {}))
    data_cfg["condition_vocab"] = {"regenerate": False}
    data_cfg["condition_mask"] = {"enabled": False, "unk_ratio": 0.0}
    prepared["data"] = data_cfg
    return prepared


def _load_evaluation_grn(path: str | Path) -> pd.DataFrame:
    grn_path = Path(path).expanduser().resolve()
    df = pd.read_csv(grn_path)
    if not {"Gene1", "Gene2"}.issubset(df.columns):
        raise ValueError(f"Evaluation GRN file must contain columns 'Gene1' and 'Gene2': {grn_path}")
    return df


def _normalize_evaluation_grn_paths(raw_paths: object) -> list[Path]:
    if raw_paths is None:
        raise ValueError("`evaluator.evaluation_grn_path` is required.")
    if isinstance(raw_paths, (str, Path)):
        paths = [raw_paths]
    elif isinstance(raw_paths, list):
        paths = raw_paths
    else:
        raise TypeError(
            "`evaluator.evaluation_grn_path` must be a path string or a list of path strings."
        )

    resolved_paths: list[Path] = []
    for index, raw_path in enumerate(paths):
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(
                f"`evaluator.evaluation_grn_path[{index}]` must be a path string."
            )
        resolved_paths.append(Path(raw_path).expanduser().resolve())
    if not resolved_paths:
        raise ValueError("`evaluator.evaluation_grn_path` must contain at least one path.")
    return resolved_paths


def _load_evaluation_grn_specs(
    raw_paths: object,
    token_dict: pd.DataFrame,
) -> list[EvaluationGRNSpec]:
    specs: list[EvaluationGRNSpec] = []
    for path in _normalize_evaluation_grn_paths(raw_paths):
        dataframe = _load_evaluation_grn(path)
        cache = build_evaluation_grn_cache(
            token_dict=token_dict,
            evaluation_grn_df=dataframe,
        )
        specs.append(
            EvaluationGRNSpec(
                name=path.stem,
                path=path,
                dataframe=dataframe,
                cache=cache,
            )
        )
    return specs


def _resolve_checkpoint_path(
    checkpoint_path: object,
    default_model_path: Path,
) -> Path:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        resolved = default_model_path.expanduser().resolve()
    else:
        candidate = Path(str(checkpoint_path)).expanduser().resolve()
        resolved = candidate / SFM_MODEL_NAME if candidate.is_dir() else candidate

    if not resolved.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Model checkpoint path must be a file: {resolved}")
    return resolved


def _extract_predicted_grn(
    model_output: ModelWrapperOutput,
    foundation_name: str,
) -> torch.Tensor:
    if foundation_name not in model_output.foundations:
        raise KeyError(
            f"Foundation output {foundation_name!r} not found. "
            f"Available foundations: {list(model_output.foundations.keys())}."
        )
    grn = model_output.foundations[foundation_name].grn
    if grn is None:
        raise ValueError(f"Foundation {foundation_name!r} did not return a GRN tensor.")
    return grn


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = {}
    for column in ["auprc", "auprc_ratio", "auroc", "ep", "ep_ratio"]:
        summary[column] = float(metrics_df[column].mean()) if column in metrics_df.columns else float("nan")
    return pd.DataFrame([summary])


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


def _resolve_aggregation_modes(raw_mode: object) -> list[str]:
    mode = str(raw_mode or "dataset").strip().lower()
    if mode == "both":
        return ["cell", "dataset"]
    if mode in {"cell", "dataset"}:
        return [mode]
    raise ValueError(
        "`evaluator.aggregation` must be one of 'cell', 'dataset', or 'both', "
        f"got {raw_mode!r}."
    )


def _normalize_candidate_source_universe(raw_value: object) -> str:
    value = str(raw_value or "supported_tfs").strip().lower()
    aliases = {
        "all": "all_tfs",
        "all_tf": "all_tfs",
        "tf": "all_tfs",
        "tfs": "all_tfs",
        "supported": "supported_tfs",
        "supported_tf": "supported_tfs",
        "grn_tfs": "supported_tfs",
        "eval_tfs": "supported_tfs",
    }
    value = aliases.get(value, value)
    if value not in {"all_tfs", "supported_tfs"}:
        raise ValueError(
            "`evaluator.candidate_source_universe` must be either 'all_tfs' or "
            f"'supported_tfs', got {raw_value!r}."
        )
    return value


def _init_cell_metric_accumulators(
    eval_specs: list[EvaluationGRNSpec],
    metric_names: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, int]]:
    metric_sums = {
        str(spec.path): {name: 0.0 for name in metric_names}
        for spec in eval_specs
    }
    metric_counts = {
        str(spec.path): {name: 0 for name in metric_names}
        for spec in eval_specs
    }
    cell_counts = {str(spec.path): 0 for spec in eval_specs}
    return metric_sums, metric_counts, cell_counts


def _clone_token_template(
    tokens: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor | None]:
    template: dict[str, torch.Tensor | None] = {}
    for key in ("input_ids", "padding_mask", "non_tf_mask"):
        value = tokens.get(key)
        if torch.is_tensor(value):
            template[key] = value[:1].detach().to(device="cpu").clone()
        else:
            template[key] = None
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


def _init_dense_dataset_accumulator() -> DenseDatasetGRNAccumulator:
    return DenseDatasetGRNAccumulator()


def _accumulate_dense_dataset_grn_batch(
    accumulator: DenseDatasetGRNAccumulator,
    pred_grn: torch.Tensor,
    tokens: dict[str, torch.Tensor | None],
) -> None:
    if pred_grn.ndim != 3:
        raise ValueError(f"`pred_grn` must have shape (C, G, G), got {tuple(pred_grn.shape)}.")

    input_ids = require_tensor(tokens, "input_ids")
    if pred_grn.shape[:2] != input_ids.shape or pred_grn.shape[2] != input_ids.shape[1]:
        raise ValueError(
            f"`pred_grn` shape {tuple(pred_grn.shape)} is incompatible with input_ids {tuple(input_ids.shape)}."
        )

    token_template = _clone_token_template(tokens)
    if accumulator.token_template is None:
        accumulator.token_template = token_template
    elif not _same_token_template(accumulator.token_template, token_template):
        raise ValueError(
            "Dataset-level GRN aggregation requires a fixed gene/token order within each h5ad. "
            "Evaluate changing gene universes as separate input paths."
        )

    batch_sum = pred_grn.detach().sum(dim=0, dtype=torch.float32).to(device="cpu")
    if accumulator.score_sum is None:
        accumulator.score_sum = torch.zeros_like(batch_sum, dtype=torch.float32)
    accumulator.score_sum.add_(batch_sum)
    accumulator.cell_count += int(input_ids.shape[0])


def _finalize_dense_dataset_grn_accumulator(
    accumulator: DenseDatasetGRNAccumulator,
    runtime: RuntimeContext,
) -> FinalizedDatasetGRN | None:
    local_shape = tuple(accumulator.score_sum.shape) if accumulator.score_sum is not None else None
    local_template = accumulator.token_template

    if dist.is_available() and dist.is_initialized():
        shapes: list[tuple[int, int] | None] = [None for _ in range(runtime.world_size)]
        dist.all_gather_object(shapes, local_shape)
        global_shape = next((shape for shape in shapes if shape is not None), None)

        templates: list[dict[str, torch.Tensor | None] | None] = [
            None for _ in range(runtime.world_size)
        ]
        dist.all_gather_object(templates, local_template)
        token_template = next((template for template in templates if template is not None), None)
    else:
        global_shape = local_shape
        token_template = local_template

    if global_shape is None or token_template is None:
        return None

    if accumulator.score_sum is None:
        score_sum = torch.zeros(global_shape, dtype=torch.float32)
    else:
        if tuple(accumulator.score_sum.shape) != tuple(global_shape):
            raise ValueError(
                "Dataset-level GRN aggregation received different GRN shapes across ranks: "
                f"local={tuple(accumulator.score_sum.shape)}, expected={global_shape}."
            )
        score_sum = accumulator.score_sum.to(dtype=torch.float32)

    count_tensor = torch.tensor([float(accumulator.cell_count)], dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        score_sum_device = score_sum.to(device=runtime.device)
        count_tensor = count_tensor.to(device=runtime.device)
        dist.all_reduce(score_sum_device, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        score_sum = score_sum_device.to(device="cpu")
        count_tensor = count_tensor.to(device="cpu")

    cell_count = int(round(float(count_tensor.item())))
    if cell_count <= 0:
        return None

    return FinalizedDatasetGRN(
        score_mean=score_sum.div(float(cell_count)),
        cell_count=cell_count,
        token_template=token_template,
    )


def _build_cell_summary_rows(
    eval_specs: list[EvaluationGRNSpec],
    metric_sums: dict[str, dict[str, float]],
    metric_counts: dict[str, dict[str, int]],
    cell_counts: dict[str, int],
    runtime: RuntimeContext,
) -> list[dict[str, object]]:
    metric_names = ["auprc", "auprc_ratio", "auroc", "ep", "ep_ratio"]
    totals_values: list[float] = []
    for spec in eval_specs:
        spec_key = str(spec.path)
        totals_values.extend(metric_sums[spec_key].get(name, 0.0) for name in metric_names)
        totals_values.extend(float(metric_counts[spec_key].get(name, 0)) for name in metric_names)
        totals_values.append(float(cell_counts.get(spec_key, 0)))
    totals = torch.tensor(totals_values, dtype=torch.float64, device=runtime.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    if not runtime.is_main:
        return []

    rows: list[dict[str, object]] = []
    offset = 0
    values_per_spec = len(metric_names) * 2 + 1
    for spec in eval_specs:
        summary: dict[str, object] = {
            "aggregation": "cell",
            "evaluation_grn_name": spec.name,
            "evaluation_grn_path": str(spec.path),
            "num_eval_edges": int(spec.cache.pair_keys.numel()),
            "num_raw_eval_edges": int(spec.cache.raw_edge_count),
            "num_unmapped_eval_edges": int(spec.cache.unmapped_edge_count),
            "num_self_loop_eval_edges": int(spec.cache.self_loop_edge_count),
            "num_duplicate_eval_edges": int(spec.cache.duplicate_edge_count),
            "num_cells": int(round(float(totals[offset + values_per_spec - 1].item()))),
            "num_candidate_edges": float("nan"),
        }
        for idx, name in enumerate(metric_names):
            total = float(totals[offset + idx].item())
            count = int(round(float(totals[offset + idx + len(metric_names)].item())))
            summary[name] = float("nan") if count <= 0 else total / count
        rows.append(summary)
        offset += values_per_spec
    return rows


def _build_dense_dataset_summary_rows(
    eval_specs: list[EvaluationGRNSpec],
    finalized_datasets: list[FinalizedDatasetGRN],
    runtime: RuntimeContext,
    candidate_source_universe: str,
) -> list[dict[str, object]]:
    if not runtime.is_main:
        return []

    rows: list[dict[str, object]] = []
    for spec in eval_specs:
        score_sums: dict[int, float] = {}
        score_counts: dict[int, int] = {}
        total_cells = 0

        for dataset_grn in finalized_datasets:
            pair_keys, candidate_mask = build_candidate_pair_keys(
                tokens=dataset_grn.token_template,
                cache=spec.cache,
                candidate_source_universe=candidate_source_universe,
            )
            if not candidate_mask.any():
                total_cells += dataset_grn.cell_count
                continue

            selected_keys = pair_keys[candidate_mask].to(dtype=torch.long)
            selected_scores = dataset_grn.score_mean.unsqueeze(0)[candidate_mask].to(
                dtype=torch.float64
            )
            unique_keys, inverse = torch.unique(selected_keys, sorted=False, return_inverse=True)
            weighted_scores = selected_scores * float(dataset_grn.cell_count)
            local_score_sums = torch.zeros(unique_keys.numel(), dtype=torch.float64)
            local_score_sums.scatter_add_(0, inverse, weighted_scores)
            local_score_counts = (
                torch.bincount(inverse, minlength=unique_keys.numel()).to(dtype=torch.long)
                * int(dataset_grn.cell_count)
            )

            for key, score_sum, score_count in zip(
                unique_keys.tolist(),
                local_score_sums.tolist(),
                local_score_counts.tolist(),
            ):
                key_int = int(key)
                score_sums[key_int] = score_sums.get(key_int, 0.0) + float(score_sum)
                score_counts[key_int] = score_counts.get(key_int, 0) + int(score_count)
            total_cells += dataset_grn.cell_count

        pair_key_values = sorted(score_sums)
        if pair_key_values:
            key_tensor = torch.tensor(pair_key_values, dtype=torch.long)
            score_tensor = torch.tensor(
                [
                    score_sums[key] / max(score_counts.get(key, 0), 1)
                    for key in pair_key_values
                ],
                dtype=torch.float64,
            )
            label_tensor = torch.isin(key_tensor, spec.cache.pair_keys.to(dtype=torch.long)).to(
                dtype=torch.float64
            )
            metrics = summarize_binary_metrics(scores=score_tensor, labels=label_tensor)
        else:
            metrics = {
                "auprc": float("nan"),
                "auprc_ratio": float("nan"),
                "auroc": float("nan"),
                "ep": float("nan"),
                "ep_ratio": float("nan"),
            }

        rows.append(
            {
                "aggregation": "dataset",
                "evaluation_grn_name": spec.name,
                "evaluation_grn_path": str(spec.path),
                "num_eval_edges": int(spec.cache.pair_keys.numel()),
                "num_raw_eval_edges": int(spec.cache.raw_edge_count),
                "num_unmapped_eval_edges": int(spec.cache.unmapped_edge_count),
                "num_self_loop_eval_edges": int(spec.cache.self_loop_edge_count),
                "num_duplicate_eval_edges": int(spec.cache.duplicate_edge_count),
                "num_cells": int(total_cells),
                "num_candidate_edges": int(len(pair_key_values)),
                **metrics,
            }
        )
    return rows


def _save_summary_metrics(
    rows: list[dict[str, object]],
    paths: ExperimentPaths,
    runtime: RuntimeContext,
    logger: ExperimentLogger,
) -> None:
    if not runtime.is_main:
        return

    results_dir = paths.root / "results"
    pd.DataFrame(rows).to_csv(results_dir / "summary_metrics.csv", index=False)
    logger.info("Saved evaluation summary to %s", results_dir / "summary_metrics.csv")


def _log_run_summary(
    logger: ExperimentLogger,
    runtime: RuntimeContext,
    num_paths: int,
    eval_specs: list[EvaluationGRNSpec],
    checkpoint_path: Path,
    aggregation_modes: list[str],
    candidate_source_universe: str,
) -> None:
    logger.info("========== Evaluation Summary ==========")
    logger.info(
        "Runtime: distributed=%s, world_size=%s, device=%s",
        runtime.distributed,
        runtime.world_size,
        runtime.device,
    )
    logger.info("Data: num_adata=%s", num_paths)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Aggregation: %s", ",".join(aggregation_modes))
    logger.info("Candidate source universe: %s", candidate_source_universe)
    logger.info("Evaluation GRNs: %s", len(eval_specs))
    for spec in eval_specs:
        logger.info(
            (
                "Evaluation GRN: name=%s, evaluable_edges=%s, raw_edges=%s, "
                "unmapped_edges=%s, self_loop_edges=%s, duplicate_edges=%s, path=%s"
            ),
            spec.name,
            int(spec.cache.pair_keys.numel()),
            int(spec.cache.raw_edge_count),
            int(spec.cache.unmapped_edge_count),
            int(spec.cache.self_loop_edge_count),
            int(spec.cache.duplicate_edge_count),
            spec.path,
        )
    logger.info("Output root: %s", Path.cwd().resolve())
    logger.info("========================================")


def run_evaluation(
    sfm_config: dict[str, object],
    eval_grn_config: dict[str, object],
    assets: ModelAssets,
    runtime: RuntimeContext,
) -> None:
    eval_grn_config = _prepare_evaluation_config(eval_grn_config)
    config = {
        **eval_grn_config,
        "model": sfm_config,
    }
    evaluator_cfg = config["evaluator"]
    checkpoint_path = _resolve_checkpoint_path(
        checkpoint_path=evaluator_cfg.get("checkpoint_path"),
        default_model_path=assets.sfm_model,
    )
    foundation_name = str(evaluator_cfg.get("foundation_name", "sfm"))
    aggregation_modes = _resolve_aggregation_modes(evaluator_cfg.get("aggregation", "dataset"))
    candidate_source_universe = _normalize_candidate_source_universe(
        evaluator_cfg.get("candidate_source_universe", "supported_tfs")
    )
    model_state = load_model_state_dict(checkpoint_path)

    paths = prepare_evaluation_paths(runtime=runtime)
    logger = ExperimentLogger(name="evaluate_grn", paths=paths, runtime=runtime)
    results_dir = paths.root / "results"
    if runtime.is_main:
        logger.info(
            "Launch environment: CUDA_VISIBLE_DEVICES=%s, NCCL_NET=%s, NCCL_SHM_DISABLE=%s, NCCL_P2P_DISABLE=%s",
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            os.environ.get("NCCL_NET", "<unset>"),
            os.environ.get("NCCL_SHM_DISABLE", "<unset>"),
            os.environ.get("NCCL_P2P_DISABLE", "<unset>"),
        )

    data_assets = build_evaluation_assets(config=config, runtime=runtime)
    eval_specs = _load_evaluation_grn_specs(
        raw_paths=evaluator_cfg.get("evaluation_grn_path"),
        token_dict=data_assets.token_dict,
    )

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
        assets=assets,
        runtime_config=config.get("runtime", {}),
    )
    model.load_state_dict(model_state)
    model = maybe_wrap_fsdp(
        model=model,
        config=config,
        runtime=runtime,
        sync_module_states=True,
    )
    model.eval()

    if runtime.is_main:
        _log_run_summary(
            logger=logger,
            runtime=runtime,
            num_paths=len(data_assets.train_paths),
            eval_specs=eval_specs,
            checkpoint_path=checkpoint_path,
            aggregation_modes=aggregation_modes,
            candidate_source_universe=candidate_source_universe,
        )

    metric_names = ["auprc", "auprc_ratio", "auroc", "ep", "ep_ratio"]
    use_cell_metrics = "cell" in aggregation_modes
    use_dataset_metrics = "dataset" in aggregation_modes
    if use_cell_metrics:
        cell_metric_sums, cell_metric_counts, cell_counts = _init_cell_metric_accumulators(
            eval_specs=eval_specs,
            metric_names=metric_names,
        )
    else:
        cell_metric_sums, cell_metric_counts, cell_counts = {}, {}, {}
    finalized_dataset_grns: list[FinalizedDatasetGRN] = []

    with torch.no_grad():
        for file_index, path in enumerate(data_assets.train_paths, start=1):
            data_bundle = None
            dataset_accumulator = (
                _init_dense_dataset_accumulator() if use_dataset_metrics else None
            )
            try:
                data_bundle = build_data_bundle_for_path(
                    path=path,
                    assets=data_assets,
                    config=config,
                    runtime=runtime,
                    logger=logger,
                    file_index=file_index,
                    num_files=len(data_assets.train_paths),
                )

                batch_offset = 0
                for batch_idx, batch in enumerate(data_bundle.train_loader, start=1):
                    tokens = move_batch_to_device(batch, runtime.device)
                    with _autocast_context(config=config, runtime=runtime):
                        model_output = model(
                            tokens,
                            compute_grn={foundation_name: True},
                            return_factors=False,
                        )
                    pred_grn = _extract_predicted_grn(model_output, foundation_name=foundation_name)
                    if dataset_accumulator is not None:
                        _accumulate_dense_dataset_grn_batch(
                            accumulator=dataset_accumulator,
                            pred_grn=pred_grn,
                            tokens=tokens,
                        )
                    for spec in eval_specs:
                        spec_key = str(spec.path)
                        if use_cell_metrics:
                            batch_metrics = evaluate_cell_specific_grns_with_cache(
                                pred_grn=pred_grn,
                                tokens=tokens,
                                cache=spec.cache,
                                candidate_source_universe=candidate_source_universe,
                            )
                            for name in metric_names:
                                series = batch_metrics[name].dropna()
                                cell_metric_sums[spec_key][name] += float(series.sum())
                                cell_metric_counts[spec_key][name] += int(series.shape[0])
                            cell_counts[spec_key] += int(len(batch_metrics))
                            del batch_metrics
                    batch_offset += int(require_tensor(tokens, "input_ids").shape[0])

                    del pred_grn, model_output, tokens
            finally:
                _release_data_bundle(data_bundle, runtime=runtime)

            if dataset_accumulator is not None:
                finalized_dataset_grn = _finalize_dense_dataset_grn_accumulator(
                    accumulator=dataset_accumulator,
                    runtime=runtime,
                )
                if finalized_dataset_grn is not None and runtime.is_main:
                    finalized_dataset_grns.append(finalized_dataset_grn)

    summary_rows: list[dict[str, object]] = []
    if use_cell_metrics:
        summary_rows.extend(
            _build_cell_summary_rows(
                eval_specs=eval_specs,
                metric_sums=cell_metric_sums,
                metric_counts=cell_metric_counts,
                cell_counts=cell_counts,
                runtime=runtime,
            )
        )
    if use_dataset_metrics:
        summary_rows.extend(
            _build_dense_dataset_summary_rows(
                eval_specs=eval_specs,
                finalized_datasets=finalized_dataset_grns,
                runtime=runtime,
                candidate_source_universe=candidate_source_universe,
            )
        )

    _save_summary_metrics(
        rows=summary_rows,
        paths=paths,
        runtime=runtime,
        logger=logger,
    )

    if runtime.is_main:
        logger.info("Evaluation finished. Results are under %s", results_dir)


def main() -> None:
    args = parse_args()
    runtime = initialize_distributed()
    try:
        eval_grn_config = load_yaml_config(args.eval_grn_config)
        if args.checkpoint_path is not None:
            eval_grn_config.setdefault("evaluator", {})["checkpoint_path"] = args.checkpoint_path
        requested_checkpoint_path = eval_grn_config.get("evaluator", {}).get("checkpoint_path")
        checkpoint_overrides_model_source = (
            requested_checkpoint_path is not None
            and str(requested_checkpoint_path).strip() != ""
        )
        assets = resolve_model_assets(
            model_source=eval_grn_config["model_source"],
            require_model_weights=not checkpoint_overrides_model_source,
        )
        sfm_config = load_sfm_config(assets.sfm_config)
        eval_runtime_config = apply_model_assets_to_runtime_config(
            eval_grn_config,
            assets,
            require_model_weights=not checkpoint_overrides_model_source,
        )
        run_evaluation(
            sfm_config=sfm_config,
            eval_grn_config=eval_runtime_config,
            assets=assets,
            runtime=runtime,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
