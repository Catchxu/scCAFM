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

from ..models import SFM
from ..tokenizer import TomeDataset, TomeTokenizer, tome_collate_fn
from .metric import compute_selected_metrics, normalize_metric_selection


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

    pair_set = set()
    token_to_name: Dict[int, str] = {}
    mapped = 0
    for g1, g2 in zip(eval_grn_df["Gene1"].tolist(), eval_grn_df["Gene2"].tolist()):
        src = _map_gene_to_token(g1, symbol2id, id2id)
        tgt = _map_gene_to_token(g2, symbol2id, id2id)
        if src is None or tgt is None:
            continue
        pair_set.add((int(src), int(tgt)))
        token_to_name.setdefault(int(src), str(g1))
        token_to_name.setdefault(int(tgt), str(g2))
        mapped += 1

    gt_tf_ids = sorted({s for s, _ in pair_set})
    gt_gene_ids = sorted({s for s, _ in pair_set} | {t for _, t in pair_set})
    tf_to_idx = {tid: i for i, tid in enumerate(gt_tf_ids)}
    gene_to_idx = {gid: j for j, gid in enumerate(gt_gene_ids)}

    labels = np.zeros((len(gt_tf_ids) * len(gt_gene_ids),), dtype=np.uint8)
    g_n = len(gt_gene_ids)
    for s, t in pair_set:
        i = tf_to_idx.get(s, None)
        j = gene_to_idx.get(t, None)
        if i is None or j is None:
            continue
        labels[i * g_n + j] = 1

    unique_pairs = len(pair_set)
    return (
        mapped,
        gt_tf_ids,
        gt_gene_ids,
        labels,
        unique_pairs,
        token_to_name,
    )


def _map_ids_to_sorted_universe(ids: np.ndarray, universe: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ids.ndim != 1:
        ids = ids.reshape(-1)
    if universe.shape[0] == 0:
        return np.zeros_like(ids, dtype=np.int64), np.zeros_like(ids, dtype=bool)
    idx = np.searchsorted(universe, ids)
    valid = (idx >= 0) & (idx < universe.shape[0])
    valid &= universe[idx] == ids
    return idx, valid


def _accumulate_dense_edges(
    tf_ids: np.ndarray,
    tg_ids: np.ndarray,
    edge_scores: np.ndarray,
    gt_tf_ids_arr: np.ndarray,
    gt_gene_ids_arr: np.ndarray,
    score_sum: np.ndarray,
    score_cnt: np.ndarray,
) -> None:
    if edge_scores.shape != (tf_ids.shape[0], tg_ids.shape[0]):
        raise ValueError(
            f"edge_scores shape mismatch: expected {(tf_ids.shape[0], tg_ids.shape[0])}, got {edge_scores.shape}"
        )

    tf_idx, tf_keep = _map_ids_to_sorted_universe(tf_ids, gt_tf_ids_arr)
    tg_idx, tg_keep = _map_ids_to_sorted_universe(tg_ids, gt_gene_ids_arr)

    g_n = int(gt_gene_ids_arr.shape[0])
    if tf_keep.any() and tg_keep.any():
        tf_idx_kept = tf_idx[tf_keep]
        tg_idx_kept = tg_idx[tg_keep]
        flat_idx = (tf_idx_kept[:, None] * g_n + tg_idx_kept[None, :]).reshape(-1)
        vals = edge_scores[np.ix_(tf_keep, tg_keep)].reshape(-1)
        np.add.at(score_sum, flat_idx, vals)
        np.add.at(score_cnt, flat_idx, 1)


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
    gene_key: Optional[str] = None,
    platform_key: Optional[str] = None,
    cond_species_key: Optional[str] = None,
    tissue_key: Optional[str] = None,
    disease_key: Optional[str] = None,
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

    selected_metrics = normalize_metric_selection(metric)

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
    tokenizer.set_gene_key(gene_key)
    tokenizer.set_condition_keys(
        platform_key=platform_key,
        species_key=cond_species_key,
        tissue_key=tissue_key,
        disease_key=disease_key,
    )

    (
        mapped_pairs,
        gt_tf_ids,
        gt_gene_ids,
        gt_labels,
        unique_pairs,
        token_to_name,
    ) = _build_gt_lookup(eval_grn_df, tokenizer)
    gt_tf_ids_arr = np.asarray(gt_tf_ids, dtype=np.int64)
    gt_gene_ids_arr = np.asarray(gt_gene_ids, dtype=np.int64)
    g_n = int(gt_gene_ids_arr.shape[0])
    n_edges = int(gt_labels.shape[0])
    if n_edges == 0:
        raise RuntimeError("No mapped GT edges/universe after token mapping; cannot evaluate.")

    score_sum = np.zeros((n_edges,), dtype=np.float64)
    score_cnt = np.zeros((n_edges,), dtype=np.int64)
    processed_cells = 0
    if logger:
        logger.info(
            "GRN eval start | files=%d | batch_size=%d | device=%s | ddp=%s | world_size=%d | metrics=%s",
            len(adata_files),
            batch_size,
            device,
            is_distributed,
            world_size,
            ",".join(selected_metrics),
        )

    for file_idx, file_path in enumerate(adata_files):
        if logger:
            logger.info("[File %d/%d] Loading: %s", file_idx + 1, len(adata_files), file_path)

        adata = sc.read_h5ad(file_path)
        raw_cells, raw_genes = int(adata.n_obs), int(adata.n_vars)

        with torch.no_grad():
            tokens_dict = tokenizer(adata, preprocess=preprocess)
        token_cells = int(tokens_dict["gene"].shape[0])
        if token_cells > 0:
            token_genes = int((~tokens_dict["pad"][0].bool()).sum().item())
        else:
            token_genes = 0
        if logger:
            logger.info(
                "Preprocess result | file=%d | raw_cells=%d raw_genes=%d -> post_cells=%d post_genes=%d",
                file_idx + 1,
                raw_cells,
                raw_genes,
                token_cells,
                token_genes,
            )

        dataset = TomeDataset(tokens_dict)
        sampler = None
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
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
                    grn, factors = model(batch_dev, return_factors=True, compute_grn=True)

                grn = grn.detach().float().cpu()
                b_tf = factors.binary_tf.detach().cpu().bool()
                b_tg = factors.binary_tg.detach().cpu().bool()
                genes = batch_cpu["gene"].cpu().long()

                batch_n = genes.shape[0]
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
                                int(sample_ptr + i),
                                tuple(grn[i].shape),
                                tf_n,
                                tg_n,
                            )
                        continue

                    tf_ids = genes[i][tf_mask].numpy().astype(np.int64, copy=False)
                    tg_ids = genes[i][tg_mask].numpy().astype(np.int64, copy=False)
                    logits = np.abs(grn[i].numpy().astype(np.float64, copy=False))

                    _accumulate_dense_edges(
                        tf_ids=tf_ids,
                        tg_ids=tg_ids,
                        edge_scores=logits,
                        gt_tf_ids_arr=gt_tf_ids_arr,
                        gt_gene_ids_arr=gt_gene_ids_arr,
                        score_sum=score_sum,
                        score_cnt=score_cnt,
                    )
                    processed_cells += 1
                sample_ptr += batch_n

                if logger and log_interval > 0 and step % log_interval == 0:
                    logger.info(
                        "progress | file=%d step=%d processed_cells=%d",
                        file_idx + 1,
                        step,
                        processed_cells * world_size if is_distributed else processed_cells,
                    )

        del adata, tokens_dict, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_distributed:
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            {"sum": score_sum, "cnt": score_cnt, "cells": processed_cells},
        )
        if not rank0:
            if initialized_here:
                dist.destroy_process_group()
            return {}, pd.DataFrame()
        total_sum = np.zeros_like(score_sum)
        total_cnt = np.zeros_like(score_cnt)
        total_cells = 0
        for part in gathered:
            total_sum += part["sum"]
            total_cnt += part["cnt"]
            total_cells += int(part["cells"])
        score_sum = total_sum
        score_cnt = total_cnt
        processed_cells = total_cells

    preds = np.full((n_edges,), -1.0, dtype=np.float64)
    observed = score_cnt > 0
    preds[observed] = score_sum[observed] / np.maximum(score_cnt[observed], 1)
    metric_map = compute_selected_metrics(gt_labels, preds, selected_metrics)

    tf_idx = np.repeat(np.arange(len(gt_tf_ids_arr), dtype=np.int64), g_n)
    tg_idx = np.tile(np.arange(g_n, dtype=np.int64), len(gt_tf_ids_arr))
    edge_df = pd.DataFrame(
        {
            "Gene1": [token_to_name.get(int(gt_tf_ids_arr[i]), str(int(gt_tf_ids_arr[i]))) for i in tf_idx],
            "Gene2": [token_to_name.get(int(gt_gene_ids_arr[j]), str(int(gt_gene_ids_arr[j]))) for j in tg_idx],
            "EdgeWeight": preds,
            "ObservedCount": score_cnt.astype(np.int64),
        }
    )
    edge_df = edge_df[edge_df["ObservedCount"] > 0].copy()
    edge_df = edge_df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
    edge_df = edge_df[["Gene1", "Gene2", "EdgeWeight"]]
    edge_tsv = os.path.join(output_dir, "grn_estimated.tsv")
    edge_df.to_csv(edge_tsv, sep="\t", index=False)

    summary = {
        "num_cells": int(processed_cells),
        "metrics": ",".join(selected_metrics),
        "gt_unique_edges_mapped": int(unique_pairs),
        "gt_edges_total_mapped_rows": int(mapped_pairs),
        "estimated_grn_tsv": edge_tsv,
    }
    for m in selected_metrics:
        val = float(metric_map[m])
        summary[f"{m}_mean"] = val
        summary[f"{m}_std"] = 0.0
        summary[f"{m}_valid_cells"] = 1

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
        logger.info("Saved estimated GRN: %s", edge_tsv)
        logger.info("Saved summary metrics: %s", summary_csv)

    if initialized_here:
        dist.destroy_process_group()

    return summary, edge_df
