import gc
import logging
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

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


def evaluate_grn(
    model: SFM,
    adata_files: Union[str, List[str]],
    tokenizer: TomeTokenizer,
    eval_grn_df: pd.DataFrame,
    human_tfs: Optional[pd.DataFrame] = None,
    mouse_tfs: Optional[pd.DataFrame] = None,
    species_key: str = "species",
    batch_size: int = 32,
    device: str = "cuda",
    output_dir: str = "./eval/grn",
    log_dir: Optional[str] = None,
    log_name: str = "grn_eval.log",
    log_interval: int = 100,
    metric: str = "auprc",
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

    os.makedirs(output_dir, exist_ok=True)
    logger = _setup_logger(output_dir, log_dir, log_name, log_overwrite=log_overwrite)

    metric = metric.lower()
    if metric not in {"auprc", "auroc"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'auprc' or 'auroc'.")

    amp_dtype = amp_dtype.lower()
    if amp_dtype not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}. Use 'bf16' or 'fp16'.")

    if device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda"
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
    logger.info(
        (
            "GRN eval start | files=%d | batch_size=%d | device=%s | metric=%s | "
            "gt_mapped_edges=%d | gt_unique_edges=%d | tokenizer_keys="
            "platform=%s cond_species=%s tissue=%s disease=%s batch=%s"
        ),
        len(adata_files),
        batch_size,
        device,
        metric,
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
    auprc_all: List[float] = []
    auroc_all: List[float] = []

    for file_idx, file_path in enumerate(adata_files):
        logger.info("[File %d/%d] Loading: %s", file_idx + 1, len(adata_files), file_path)

        adata = sc.read_h5ad(file_path)
        obs_names = adata.obs_names.tolist()

        with torch.no_grad():
            tokens_dict = tokenizer(adata, preprocess=preprocess)
        token_cells = int(tokens_dict["gene"].shape[0])
        if len(obs_names) != token_cells:
            logger.warning(
                "obs_names size (%d) != tokenized cells (%d). Using index-based names after preprocessing.",
                len(obs_names),
                token_cells,
            )
            obs_names = [str(i) for i in range(token_cells)]

        dataset = TomeDataset(tokens_dict)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=tome_collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        try:
            species = adata.uns[species_key]
        except Exception:
            species = "human"

        if species == "human":
            model.update_tfs(human_tfs)
        elif species == "mouse":
            model.update_tfs(mouse_tfs)
        else:
            raise ValueError(f"{species} isn't supported!")

        cell_offset = 0
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
                for i in range(batch_n):
                    tf_mask = b_tf[i]
                    tg_mask = b_tg[i]
                    tf_n = int(tf_mask.sum().item())
                    tg_n = int(tg_mask.sum().item())

                    if grn[i].shape != (tf_n, tg_n):
                        logger.warning(
                            "Skip cell due to shape mismatch | file=%s cell_offset=%d grn_shape=%s tf_n=%d tg_n=%d",
                            file_path,
                            cell_offset + i,
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
                    primary = auprc if metric == "auprc" else auroc

                    auprc_all.append(auprc)
                    auroc_all.append(auroc)

                    abs_idx = cell_offset + i
                    records.append(
                        {
                            "file_idx": file_idx,
                            "file_path": file_path,
                            "cell_index_in_file": abs_idx,
                            "cell_name": obs_names[abs_idx] if abs_idx < len(obs_names) else str(abs_idx),
                            "num_tf": tf_n,
                            "num_tg": tg_n,
                            "num_edges": int(tf_n * tg_n),
                            "num_pos_edges": int(y_true.sum()),
                            "auprc": auprc,
                            "auroc": auroc,
                            "metric_name": metric,
                            "metric_value": primary,
                        }
                    )

                processed += batch_n
                cell_offset += batch_n
                if log_interval > 0 and step % log_interval == 0:
                    metric_vals = [r["metric_value"] for r in records]
                    mean_metric, _, valid_n = _safe_nan_stats(metric_vals)
                    logger.info(
                        "progress | file=%d step=%d processed_cells=%d metric=%s mean=%.6f valid=%d",
                        file_idx + 1,
                        step,
                        processed,
                        metric,
                        mean_metric,
                        valid_n,
                    )

        del adata, tokens_dict, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(records) == 0:
        raise RuntimeError("No valid cells were evaluated. Please check your data and model outputs.")

    df = pd.DataFrame(records)
    cell_csv = os.path.join(output_dir, "grn_eval_per_cell.csv")
    df.to_csv(cell_csv, index=False)

    mean_metric, std_metric, valid_metric_n = _safe_nan_stats(df["metric_value"].tolist())
    mean_auprc, std_auprc, valid_auprc_n = _safe_nan_stats(auprc_all)
    mean_auroc, std_auroc, valid_auroc_n = _safe_nan_stats(auroc_all)

    summary = {
        "num_cells": int(df.shape[0]),
        "metric_name": metric,
        "metric_mean": mean_metric,
        "metric_std": std_metric,
        "metric_valid_cells": valid_metric_n,
        "auprc_mean": mean_auprc,
        "auprc_std": std_auprc,
        "auprc_valid_cells": valid_auprc_n,
        "auroc_mean": mean_auroc,
        "auroc_std": std_auroc,
        "auroc_valid_cells": valid_auroc_n,
        "gt_unique_edges_mapped": int(unique_pairs),
        "gt_edges_total_mapped_rows": int(mapped_pairs),
        "per_cell_csv": cell_csv,
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(output_dir, "grn_eval_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    logger.info(
        "GRN eval done | cells=%d | %s_mean=%.6f | auprc_mean=%.6f | auroc_mean=%.6f",
        summary["num_cells"],
        metric,
        summary["metric_mean"],
        summary["auprc_mean"],
        summary["auroc_mean"],
    )
    logger.info("Saved per-cell metrics: %s", cell_csv)
    logger.info("Saved summary metrics: %s", summary_csv)

    return summary, df
