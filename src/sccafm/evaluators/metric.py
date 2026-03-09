from typing import Dict, Iterable, List, Tuple

import numpy as np


SUPPORTED_METRICS = {"auprc", "auroc", "auprc_ratio", "auroc_ratio"}


def normalize_metric_selection(metric) -> List[str]:
    if isinstance(metric, str):
        selected_metrics = [metric.lower()]
    elif isinstance(metric, list) and all(isinstance(m, str) for m in metric):
        selected_metrics = [m.lower() for m in metric]
    else:
        raise ValueError("`metric` must be a string or a list of strings.")

    if len(selected_metrics) == 0:
        raise ValueError("`metric` must contain at least one metric.")

    selected_metrics = list(dict.fromkeys(selected_metrics))
    invalid_metrics = [m for m in selected_metrics if m not in SUPPORTED_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metric(s) {invalid_metrics}. Use one of: {sorted(SUPPORTED_METRICS)}."
        )
    return selected_metrics


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
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


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
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


def safe_nan_stats(values: Iterable[float]) -> Tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std()), int(valid.size)


def safe_ratio(numer: float, denom: float) -> float:
    if np.isnan(numer) or np.isnan(denom) or denom <= 0:
        return float("nan")
    return float(numer / denom)


def compute_selected_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    selected_metrics: List[str],
) -> Dict[str, float]:
    need_auprc = "auprc" in selected_metrics or "auprc_ratio" in selected_metrics
    need_auroc = "auroc" in selected_metrics or "auroc_ratio" in selected_metrics

    metric_map: Dict[str, float] = {}

    pos_edges = int(y_true.sum())
    all_edges = int(y_true.shape[0])
    neg_edges = all_edges - pos_edges

    auprc = safe_auprc(y_true, y_score) if need_auprc else float("nan")
    auroc = safe_auroc(y_true, y_score) if need_auroc else float("nan")

    if "auprc" in selected_metrics:
        metric_map["auprc"] = auprc
    if "auroc" in selected_metrics:
        metric_map["auroc"] = auroc
    if "auprc_ratio" in selected_metrics:
        auprc_random = float(pos_edges / all_edges) if all_edges > 0 else float("nan")
        metric_map["auprc_ratio"] = safe_ratio(auprc, auprc_random)
    if "auroc_ratio" in selected_metrics:
        auroc_random = 0.5 if pos_edges > 0 and neg_edges > 0 else float("nan")
        metric_map["auroc_ratio"] = safe_ratio(auroc, auroc_random)

    return metric_map
