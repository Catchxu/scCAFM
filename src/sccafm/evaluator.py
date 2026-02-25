import gc
import logging
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .models import SFM
from .tokenizer import TomeDataset, TomeTokenizer, tome_collate_fn


def _setup_logger(
    output_dir: str,
    log_dir: Optional[str],
    log_name: str,
    log_overwrite: bool = True,
):
    if log_dir is None:
        log_dir = output_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("sccafm.eval.grn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w" if log_overwrite else "a")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _setup_distributed(device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    initialized_here = False
    rank = 0
    local_rank = 0

    if is_distributed:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            initialized_here = True
        rank = dist.get_rank()
    return is_distributed, initialized_here, rank, local_rank, world_size


def _map_gene_to_token(name: str, symbol2id: Dict[str, int], id2id: Dict[str, int]) -> Optional[int]:
    s = str(name)
    if s.startswith("ENSG") and s in id2id:
        return int(id2id[s])
    if s in symbol2id:
        return int(symbol2id[s])
    if s in id2id:
        return int(id2id[s])
    return None


def _build_gt_lookup(eval_grn_df: pd.DataFrame, tokenizer: TomeTokenizer):
    if not {"Gene1", "Gene2"}.issubset(eval_grn_df.columns):
        raise ValueError("eval_grn_df must contain columns: Gene1, Gene2")

    symbol2id = tokenizer.gene_tokenizer.symbol2id
    id2id = tokenizer.gene_tokenizer.id2id

    src_to_tg: Dict[int, set] = {}
    mapped = 0
    for g1, g2 in zip(eval_grn_df["Gene1"].tolist(), eval_grn_df["Gene2"].tolist()):
        src = _map_gene_to_token(g1, symbol2id, id2id)
        tgt = _map_gene_to_token(g2, symbol2id, id2id)
        if src is None or tgt is None:
            continue
        src_to_tg.setdefault(src, set()).add(tgt)
        mapped += 1

    unique_pairs = sum(len(v) for v in src_to_tg.values())
    return src_to_tg, unique_pairs, mapped


def _build_labels(tf_ids: np.ndarray, tg_ids: np.ndarray, src_to_tg: Dict[int, set]) -> np.ndarray:
    tf_n = tf_ids.shape[0]
    tg_n = tg_ids.shape[0]
    labels = np.zeros((tf_n, tg_n), dtype=np.uint8)
    if tf_n == 0 or tg_n == 0:
        return labels

    for i, src in enumerate(tf_ids.tolist()):
        tg_set = src_to_tg.get(int(src))
        if not tg_set:
            continue
        labels[i, :] = np.isin(tg_ids, np.fromiter(tg_set, dtype=np.int64), assume_unique=False)
    return labels


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    pos = int(y.sum())
    neg = int(y.shape[0] - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores)[::-1]
    y_sorted = y[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    tpr = np.concatenate(([0.0], tps / max(pos, 1), [1.0]))
    fpr = np.concatenate(([0.0], fps / max(neg, 1), [1.0]))
    return float(np.trapz(tpr, fpr))


def _safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(scores)[::-1]
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    denom = np.arange(1, y_sorted.shape[0] + 1)
    precision = tp / denom

    pos_idx = np.where(y_sorted == 1)[0]
    if pos_idx.shape[0] == 0:
        return float("nan")
    return float(np.mean(precision[pos_idx]))


def _safe_nan_stats(values: List[float]) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std()), int(valid.size)


def _safe_ratio(numer: float, denom: float) -> float:
    if np.isnan(numer) or np.isnan(denom) or denom <= 0:
        return float("nan")
    return float(numer / denom)


def _resolve_species_for_tf(adata, cond_species_key: Optional[str]) -> str:
    # 1) Prefer tokenizer species key from adata.obs if provided.
    if cond_species_key is not None and cond_species_key in adata.obs:
        vals = (
            adata.obs[cond_species_key]
            .astype(str)
            .str.lower()
            .replace("nan", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            raise ValueError(
                f"Found multiple species values in adata.obs['{cond_species_key}']: {vals[:10]}. "
                "GRN evaluation currently expects one species per dataset file."
            )

    # 2) Fallback to adata.uns['species'] if available.
    species_uns = adata.uns.get("species", None)
    if species_uns is not None:
        return str(species_uns).lower()

    # 3) Last fallback.
    return "human"


def evaluate_grn(
    model: SFM,
    adata_files: Union[str, List[str]],
    tokenizer: TomeTokenizer,
    eval_grn_df: pd.DataFrame,
    human_tfs: Optional[pd.DataFrame] = None,
    mouse_tfs: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    device: str = "cuda",
    output_dir: str = "./eval/grn",
    log_dir: Optional[str] = None,
    log_name: str = "grn_eval.log",
    log_interval: int = 100,
    metric: Union[str, List[str]] = "auprc",
    preprocess: bool = True,
    platform_key: Optional[str] = None,
    cond_species_key: Optional[str] = None,
    tissue_key: Optional[str] = None,
    disease_key: Optional[str] = None,
    batch_key: Optional[Union[str, List[str]]] = None,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    log_overwrite: bool = True,
):
    if isinstance(adata_files, str):
        adata_files = [adata_files]

    is_distributed, initialized_here, rank, local_rank, world_size = _setup_distributed(device)
    rank0 = rank == 0

    os.makedirs(output_dir, exist_ok=True)
    logger = _setup_logger(output_dir, log_dir, log_name, log_overwrite=log_overwrite) if rank0 else None

    if isinstance(metric, str):
        selected_metrics = [metric.lower()]
    elif isinstance(metric, list) and all(isinstance(m, str) for m in metric):
        selected_metrics = [m.lower() for m in metric]
    else:
        raise ValueError("`metric` must be a string or a list of strings.")

    if len(selected_metrics) == 0:
        raise ValueError("`metric` must contain at least one metric.")
    # Preserve user order and deduplicate.
    selected_metrics = list(dict.fromkeys(selected_metrics))

    supported_metrics = {"auprc", "auroc", "auprc_ratio", "auroc_ratio"}
    invalid_metrics = [m for m in selected_metrics if m not in supported_metrics]
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metric(s) {invalid_metrics}. Use one of: {sorted(supported_metrics)}."
        )
    primary_metric = selected_metrics[0]

    amp_dtype = amp_dtype.lower()
    if amp_dtype not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}. Use 'bf16' or 'fp16'.")

    if device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"

    amp_enabled = bool(use_amp and device.startswith("cuda"))
    if amp_enabled and amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "use_amp=True with amp_dtype='bf16' requires CUDA bf16 support on this GPU/runtime."
        )

    model = model.to(device)
    model.eval()
    tokenizer.set_condition_keys(
        platform_key=platform_key,
        species_key=cond_species_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
    )
    tokenizer.set_batch_key(batch_key=batch_key)

    src_to_tg, unique_pairs, mapped_pairs = _build_gt_lookup(eval_grn_df, tokenizer)
    if logger:
        logger.info(
            (
                "GRN eval start | files=%d | batch_size=%d | device=%s | ddp=%s | world_size=%d | metrics=%s | "
                "gt_mapped_edges=%d | gt_unique_edges=%d | tokenizer_keys="
                "platform=%s cond_species=%s tissue=%s disease=%s batch=%s"
            ),
            len(adata_files),
            batch_size,
            device,
            is_distributed,
            world_size,
            ",".join(selected_metrics),
            mapped_pairs,
            unique_pairs,
            platform_key,
            cond_species_key,
            tissue_key,
            disease_key,
            batch_key,
        )

    records = []
    processed = 0

    for file_idx, file_path in enumerate(adata_files):
        if logger:
            logger.info("[File %d/%d] Loading: %s", file_idx + 1, len(adata_files), file_path)

        adata = sc.read_h5ad(file_path)
        obs_names = adata.obs_names.tolist()

        with torch.no_grad():
            tokens_dict = tokenizer(adata, preprocess=preprocess)
        token_cells = int(tokens_dict["gene"].shape[0])
        if len(obs_names) != token_cells:
            if logger:
                logger.warning(
                    "obs_names size (%d) != tokenized cells (%d). Using index-based names after preprocessing.",
                    len(obs_names),
                    token_cells,
                )
            obs_names = [str(i) for i in range(token_cells)]

        dataset = TomeDataset(tokens_dict)
        sampler = None
        sample_indices = list(range(len(dataset)))
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            sample_indices = list(iter(sampler))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=tome_collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        species = _resolve_species_for_tf(adata, cond_species_key)

        if species == "human":
            model.update_tfs(human_tfs)
        elif species == "mouse":
            model.update_tfs(mouse_tfs)
        else:
            raise ValueError(f"{species} isn't supported!")

        sample_ptr = 0
        with torch.no_grad():
            for step, batch in enumerate(loader, start=1):
                batch_cpu = {k: v.clone() for k, v in batch.items()}
                batch_dev = {k: v.to(device) for k, v in batch.items()}

                autocast_ctx = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16,
                    )
                    if amp_enabled
                    else nullcontext()
                )

                with autocast_ctx:
                    grn, b_tf, b_tg = model(batch_dev, return_factors=False, compute_grn=True)

                grn = grn.detach().float().cpu()
                b_tf = b_tf.detach().cpu().bool()
                b_tg = b_tg.detach().cpu().bool()
                genes = batch_cpu["gene"].cpu().long()

                batch_n = genes.shape[0]
                batch_indices = sample_indices[sample_ptr: sample_ptr + batch_n]
                sample_ptr += batch_n
                for i in range(batch_n):
                    tf_mask = b_tf[i]
                    tg_mask = b_tg[i]
                    tf_n = int(tf_mask.sum().item())
                    tg_n = int(tg_mask.sum().item())

                    if grn[i].shape != (tf_n, tg_n):
                        if logger:
                            logger.warning(
                                "Skip cell due to shape mismatch | file=%s cell_index=%d grn_shape=%s tf_n=%d tg_n=%d",
                                file_path,
                                int(batch_indices[i]),
                                tuple(grn[i].shape),
                                tf_n,
                                tg_n,
                            )
                        continue

                    tf_ids = genes[i][tf_mask].numpy().astype(np.int64, copy=False)
                    tg_ids = genes[i][tg_mask].numpy().astype(np.int64, copy=False)
                    logits = grn[i].numpy().astype(np.float64, copy=False)

                    labels = _build_labels(tf_ids, tg_ids, src_to_tg)
                    y_true = labels.reshape(-1)
                    y_score = logits.reshape(-1)

                    auprc = _safe_auprc(y_true, y_score)
                    auroc = _safe_auroc(y_true, y_score)

                    pos_edges = int(y_true.sum())
                    all_edges = int(y_true.shape[0])
                    neg_edges = all_edges - pos_edges

                    # Random baseline: positive prevalence for AUPRC and 0.5 for AUROC.
                    auprc_random = float(pos_edges / all_edges) if all_edges > 0 else float("nan")
                    auroc_random = 0.5 if pos_edges > 0 and neg_edges > 0 else float("nan")

                    auprc_ratio = _safe_ratio(auprc, auprc_random)
                    auroc_ratio = _safe_ratio(auroc, auroc_random)

                    metric_map = {
                        "auprc": auprc,
                        "auroc": auroc,
                        "auprc_ratio": auprc_ratio,
                        "auroc_ratio": auroc_ratio,
                    }

                    abs_idx = int(batch_indices[i])
                    row = {
                        "file_idx": file_idx,
                        "file_path": file_path,
                        "cell_index_in_file": abs_idx,
                        "cell_name": obs_names[abs_idx] if abs_idx < len(obs_names) else str(abs_idx),
                        "num_tf": tf_n,
                        "num_tg": tg_n,
                        "num_edges": int(tf_n * tg_n),
                        "num_pos_edges": pos_edges,
                    }
                    for m in selected_metrics:
                        row[m] = metric_map[m]
                    records.append(row)

                processed += batch_n
                if logger and log_interval > 0 and step % log_interval == 0:
                    metric_vals = [r.get(primary_metric, float("nan")) for r in records]
                    mean_metric, _, valid_n = _safe_nan_stats(metric_vals)
                    logger.info(
                        "progress | file=%d step=%d processed_cells=%d metric=%s mean=%.6f valid=%d",
                        file_idx + 1,
                        step,
                        processed,
                        primary_metric,
                        mean_metric,
                        valid_n,
                    )

        del adata, tokens_dict, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_distributed:
        gathered_records: List[List[Dict[str, Union[int, str, float]]]] = [None] * world_size
        dist.all_gather_object(gathered_records, records)
        if not rank0:
            if initialized_here:
                dist.destroy_process_group()
            return {}, pd.DataFrame()
        parts = [pd.DataFrame(part) for part in gathered_records if part]
    else:
        parts = [pd.DataFrame(records)] if records else []

    if len(parts) == 0:
        raise RuntimeError("No valid cells were evaluated. Please check your data and model outputs.")

    df = pd.concat(parts, ignore_index=True)
    # DistributedSampler may pad data on some ranks. Keep one record per cell.
    df = df.sort_values(["file_idx", "cell_index_in_file"]).drop_duplicates(
        subset=["file_idx", "cell_index_in_file"], keep="first"
    ).reset_index(drop=True)
    cell_csv = os.path.join(output_dir, "grn_eval_per_cell.csv")
    df.to_csv(cell_csv, index=False)

    summary = {
        "num_cells": int(df.shape[0]),
        "metrics": ",".join(selected_metrics),
        "gt_unique_edges_mapped": int(unique_pairs),
        "gt_edges_total_mapped_rows": int(mapped_pairs),
        "per_cell_csv": cell_csv,
    }
    for m in selected_metrics:
        mean_m, std_m, valid_m = _safe_nan_stats(df[m].tolist())
        summary[f"{m}_mean"] = mean_m
        summary[f"{m}_std"] = std_m
        summary[f"{m}_valid_cells"] = valid_m

    if len(selected_metrics) == 1:
        only = selected_metrics[0]
        summary["metric_name"] = only
        summary["metric_mean"] = summary[f"{only}_mean"]
        summary["metric_std"] = summary[f"{only}_std"]
        summary["metric_valid_cells"] = summary[f"{only}_valid_cells"]
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(output_dir, "grn_eval_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    if logger:
        metric_log_parts = [f"{m}_mean={summary[f'{m}_mean']:.6f}" for m in selected_metrics]
        logger.info(
            "GRN eval done | cells=%d | %s",
            summary["num_cells"],
            " | ".join(metric_log_parts),
        )
        logger.info("Saved per-cell metrics: %s", cell_csv)
        logger.info("Saved summary metrics: %s", summary_csv)

    if initialized_here:
        dist.destroy_process_group()

    return summary, df
