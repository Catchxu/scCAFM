from __future__ import annotations

import math

import torch


def _validate_binary_inputs(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            f"`scores` and `labels` must be 1D, got {tuple(scores.shape)} and {tuple(labels.shape)}."
        )
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"`scores` and `labels` must have the same length, got {scores.shape[0]} and {labels.shape[0]}."
        )

    scores = scores.detach().to(dtype=torch.float64, device="cpu")
    labels = labels.detach().to(dtype=torch.float64, device="cpu")
    return scores, labels


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    if pos_total <= 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    tp = torch.cumsum(labels_sorted, dim=0)
    precision = tp / torch.arange(1, labels_sorted.numel() + 1, dtype=torch.float64)
    recall = tp / float(pos_total)

    recall_prev = torch.cat([torch.zeros(1, dtype=torch.float64), recall[:-1]], dim=0)
    ap = torch.sum((recall - recall_prev) * precision)
    return float(ap.item())


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    neg_total = int(labels.numel() - pos_total)
    if pos_total <= 0 or neg_total <= 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1.0 - labels_sorted, dim=0)

    tpr = torch.cat([torch.zeros(1, dtype=torch.float64), tp / float(pos_total)], dim=0)
    fpr = torch.cat([torch.zeros(1, dtype=torch.float64), fp / float(neg_total)], dim=0)
    auc = torch.trapz(tpr, fpr)
    return float(auc.item())


def early_precision_ratio(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk: int | None = None,
) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    total = int(labels.numel())
    if pos_total <= 0 or total <= 0:
        return float("nan")

    k = pos_total if topk is None else int(topk)
    k = max(min(k, total), 1)

    order = torch.argsort(scores, descending=True)[:k]
    precision_at_k = float(labels[order].sum().item()) / float(k)
    random_precision = float(pos_total) / float(total)
    if random_precision <= 0.0:
        return float("nan")
    return precision_at_k / random_precision


def summarize_binary_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk: int | None = None,
) -> dict[str, float]:
    scores, labels = _validate_binary_inputs(scores, labels)
    positive_count = int(labels.sum().item())
    total_count = int(labels.numel())
    random_auprc = (
        float(positive_count) / float(total_count)
        if total_count > 0 and positive_count > 0
        else float("nan")
    )

    auprc = binary_auprc(scores, labels)
    auroc = binary_auroc(scores, labels)
    ep_ratio = early_precision_ratio(scores, labels, topk=topk)

    if math.isnan(auprc) or math.isnan(random_auprc) or random_auprc <= 0.0:
        auprc_ratio = float("nan")
    else:
        auprc_ratio = auprc / random_auprc

    return {
        "auprc_ratio": auprc_ratio,
        "auroc": auroc,
        "ep_ratio": ep_ratio,
    }
