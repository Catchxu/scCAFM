from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

from ..assets import (
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
    pair_key_base: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted cell-specific GRNs.")
    parser.add_argument("--eval-grn-config", default="configs/eval_grn.yaml")
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
    for src_name, tgt_name in zip(
        evaluation_grn_df["Gene1"].tolist(),
        evaluation_grn_df["Gene2"].tolist(),
    ):
        src_id = _map_gene_to_token(src_name, symbol_to_index, ensembl_to_index)
        tgt_id = _map_gene_to_token(tgt_name, symbol_to_index, ensembl_to_index)
        if src_id is None or tgt_id is None:
            continue
        if src_id == pad_index or tgt_id == pad_index:
            continue
        pair_keys.append(src_id * pair_key_base + tgt_id)
        supported_token_ids.add(src_id)
        supported_token_ids.add(tgt_id)

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

    return EvaluationGRNCache(
        pair_keys=pair_key_tensor,
        supported_token_ids=supported_token_tensor,
        pair_key_base=pair_key_base,
    )


def build_reference_grn(
    tokens: dict[str, torch.Tensor | None],
    cache: EvaluationGRNCache,
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


def evaluate_cell_specific_grns(
    pred_grn: torch.Tensor,
    tokens: dict[str, torch.Tensor | None],
    token_dict: pd.DataFrame,
    evaluation_grn_df: pd.DataFrame,
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
    target, candidate_mask = build_reference_grn(tokens=tokens, cache=cache)

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
                "auprc_ratio": float("nan"),
                "auroc": float("nan"),
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
        model_package_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    return ExperimentPaths(
        root=root,
        logs=logs,
        checkpoints=checkpoints,
        model_package_dir=model_package_dir,
        log_file=logs / "evaluate_grn.log",
        resume_manifest_file=logs / "resume_manifest.json",
        resume_state_file=root / "sfm_train_state.pt",
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
    for column in ["auprc_ratio", "auroc", "ep_ratio"]:
        summary[column] = float(metrics_df[column].mean()) if column in metrics_df.columns else float("nan")
    return pd.DataFrame([summary])


def _release_data_bundle(data_bundle: PretrainingDataBundle | None) -> None:
    if data_bundle is None:
        return

    train_loader = getattr(data_bundle, "train_loader", None)
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "close"):
        dataset.close()


def _save_summary_metrics(
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
    paths: ExperimentPaths,
    runtime: RuntimeContext,
    logger: ExperimentLogger,
) -> None:
    metric_names = ["auprc_ratio", "auroc", "ep_ratio"]
    totals = torch.tensor(
        [
            metric_sums.get(name, 0.0) for name in metric_names
        ] + [
            float(metric_counts.get(name, 0)) for name in metric_names
        ],
        dtype=torch.float64,
        device=runtime.device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    if not runtime.is_main:
        return

    results_dir = paths.root / "results"
    summary = {}
    for idx, name in enumerate(metric_names):
        total = float(totals[idx].item())
        count = int(round(float(totals[idx + len(metric_names)].item())))
        summary[name] = float("nan") if count <= 0 else total / count

    pd.DataFrame([summary]).to_csv(results_dir / "summary_metrics.csv", index=False)
    logger.info("Saved evaluation summary to %s", results_dir / "summary_metrics.csv")


def _log_run_summary(
    logger: ExperimentLogger,
    runtime: RuntimeContext,
    num_paths: int,
    num_eval_edges: int,
) -> None:
    logger.info("========== Evaluation Summary ==========")
    logger.info(
        "Runtime: distributed=%s, world_size=%s, device=%s",
        runtime.distributed,
        runtime.world_size,
        runtime.device,
    )
    logger.info("Data: num_adata=%s", num_paths)
    logger.info("Evaluation GRN edges: %s", num_eval_edges)
    logger.info("Output root: %s", Path.cwd().resolve())
    logger.info("========================================")


def run_evaluation(
    sfm_config: dict[str, object],
    eval_grn_config: dict[str, object],
    runtime: RuntimeContext,
) -> None:
    eval_grn_config = _prepare_evaluation_config(eval_grn_config)
    config = {
        "model": sfm_config,
        **eval_grn_config,
    }
    evaluator_cfg = config["evaluator"]
    checkpoint_path = Path(str(evaluator_cfg["checkpoint_path"])).expanduser().resolve()
    foundation_name = str(evaluator_cfg.get("foundation_name", "sfm"))
    model_state = load_model_state_dict(checkpoint_path)

    paths = prepare_evaluation_paths(runtime=runtime)
    logger = ExperimentLogger(name="evaluate_grn", paths=paths, runtime=runtime)
    results_dir = paths.root / "results"

    data_assets = build_evaluation_assets(config=config, runtime=runtime)
    evaluation_grn_df = _load_evaluation_grn(evaluator_cfg["evaluation_grn_path"])
    eval_cache = build_evaluation_grn_cache(
        token_dict=data_assets.token_dict,
        evaluation_grn_df=evaluation_grn_df,
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
        assets=resolve_model_assets(config["model_source"], require_model_weights=True),
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
            num_eval_edges=int(eval_cache.pair_keys.numel()),
        )

    metric_names = ["auprc_ratio", "auroc", "ep_ratio"]
    metric_sums = {name: 0.0 for name in metric_names}
    metric_counts = {name: 0 for name in metric_names}
    with torch.no_grad():
        for file_index, path in enumerate(data_assets.train_paths, start=1):
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
                batch_metrics = evaluate_cell_specific_grns(
                    pred_grn=pred_grn,
                    tokens=tokens,
                    token_dict=data_assets.token_dict,
                    evaluation_grn_df=evaluation_grn_df,
                )
                for name in metric_names:
                    series = batch_metrics[name].dropna()
                    metric_sums[name] += float(series.sum())
                    metric_counts[name] += int(series.shape[0])
                batch_offset += len(batch_metrics)

            _release_data_bundle(data_bundle)

    _save_summary_metrics(
        metric_sums=metric_sums,
        metric_counts=metric_counts,
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
        assets = resolve_model_assets(
            model_source=eval_grn_config["model_source"],
            require_model_weights=True,
        )
        sfm_config = load_sfm_config(assets.sfm_config)
        eval_runtime_config = apply_model_assets_to_runtime_config(
            eval_grn_config,
            assets,
            require_model_weights=True,
        )
        run_evaluation(
            sfm_config=sfm_config,
            eval_grn_config=eval_runtime_config,
            runtime=runtime,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
