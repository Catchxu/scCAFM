from __future__ import annotations

import math

import torch


DEFAULT_BINARY_METRICS = ("auprc", "auprc_ratio", "auroc", "ep", "ep_ratio")


def _validate_similarity_inputs(
    target: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "`target` and `reference` must be 2D matrices, got "
            f"{tuple(target.shape)} and {tuple(reference.shape)}."
        )
    if target.shape[1] != reference.shape[1]:
        raise ValueError(
            "`target` and `reference` must have the same number of features, got "
            f"{target.shape[1]} and {reference.shape[1]}."
        )
    if target.shape[0] == 0:
        raise ValueError("`target` must contain at least one cell.")
    if reference.shape[0] == 0:
        raise ValueError("`reference` must contain at least one cell.")

    target = target.detach().to(dtype=torch.float64, device="cpu")
    reference = reference.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(target).all():
        raise ValueError("`target` must contain only finite values.")
    if not torch.isfinite(reference).all():
        raise ValueError("`reference` must contain only finite values.")
    return target, reference


def median_similarity_distribution(
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    similarity: str = "cosine",
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return one median similarity per target cell.

    For each target row, this computes similarity to every reference row and
    returns the median of those similarities. The output has shape
    ``(target_cells,)``.
    """

    target, reference = _validate_similarity_inputs(target, reference)
    normalized_similarity = str(similarity).lower()
    if normalized_similarity != "cosine":
        raise ValueError(
            f"`similarity` must be 'cosine', got {similarity!r}."
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")

    reference_norm = torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    reference_norm = torch.where(
        reference_norm == 0.0,
        torch.ones_like(reference_norm),
        reference_norm,
    )
    reference = reference / reference_norm

    medians: list[torch.Tensor] = []
    for start in range(0, int(target.shape[0]), chunk_size):
        target_chunk = target[start : start + chunk_size]
        target_norm = torch.linalg.vector_norm(target_chunk, dim=1, keepdim=True)
        target_norm = torch.where(
            target_norm == 0.0,
            torch.ones_like(target_norm),
            target_norm,
        )
        similarity_matrix = (target_chunk / target_norm) @ reference.T
        medians.append(torch.quantile(similarity_matrix, 0.5, dim=1))

    return torch.cat(medians, dim=0)


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
    if not torch.isfinite(scores).all():
        raise ValueError("`scores` must contain only finite values.")
    if not torch.isfinite(labels).all():
        raise ValueError("`labels` must contain only finite values.")
    if not torch.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("`labels` must be binary values in {0, 1}.")
    return scores, labels


def _sort_by_score_desc(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(scores, descending=True, stable=True)
    return scores[order], labels[order]


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average-precision style AUPRC, computed over score thresholds.

    Tied scores are evaluated as one threshold group, so the result does not
    depend on arbitrary ordering within tied predictions.
    """
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    if pos_total <= 0:
        return float("nan")

    scores_sorted, labels_sorted = _sort_by_score_desc(scores, labels)
    _, counts = torch.unique_consecutive(scores_sorted, return_counts=True)
    group_pos = torch.tensor(
        [float(chunk.sum().item()) for chunk in labels_sorted.split(counts.tolist())],
        dtype=torch.float64,
    )
    group_count = counts.to(dtype=torch.float64)

    cum_tp = torch.cumsum(group_pos, dim=0)
    cum_count = torch.cumsum(group_count, dim=0)
    precision = cum_tp / cum_count
    delta_recall = group_pos / float(pos_total)
    ap = torch.sum(delta_recall * precision)
    return float(ap.item())


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Tie-aware AUROC using average ranks for equal scores."""
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    neg_total = int(labels.numel() - pos_total)
    if pos_total <= 0 or neg_total <= 0:
        return float("nan")

    order = torch.argsort(scores, descending=False, stable=True)
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    _, counts = torch.unique_consecutive(scores_sorted, return_counts=True)
    count_values = counts.to(dtype=torch.float64)
    group_ends = torch.cumsum(count_values, dim=0)
    group_starts = group_ends - count_values + 1.0
    average_ranks = (group_starts + group_ends) / 2.0
    ranks_sorted = torch.repeat_interleave(average_ranks, counts)

    positive_rank_sum = torch.sum(ranks_sorted * labels_sorted)
    auc = (positive_rank_sum - (pos_total * (pos_total + 1) / 2.0)) / (
        float(pos_total) * float(neg_total)
    )
    return float(auc.item())


def early_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk: int | None = None,
) -> float:
    """Tie-aware precision at k.

    If the cutoff falls inside a score tie, the tied boundary contributes its
    expected positive fraction rather than an arbitrary subset.
    """
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    total = int(labels.numel())
    if pos_total <= 0 or total <= 0:
        return float("nan")

    k = pos_total if topk is None else int(topk)
    k = max(min(k, total), 1)

    scores_sorted, labels_sorted = _sort_by_score_desc(scores, labels)
    _, counts = torch.unique_consecutive(scores_sorted, return_counts=True)
    group_pos = torch.tensor(
        [float(chunk.sum().item()) for chunk in labels_sorted.split(counts.tolist())],
        dtype=torch.float64,
    )
    group_count = counts.to(dtype=torch.float64)
    group_ends = torch.cumsum(group_count, dim=0)
    group_starts = group_ends - group_count
    k_float = float(k)

    fully_selected = group_ends <= k_float
    selected_pos = torch.sum(group_pos[fully_selected])

    boundary = torch.nonzero((group_starts < k_float) & (group_ends > k_float), as_tuple=False)
    if boundary.numel() > 0:
        idx = int(boundary[0].item())
        remaining = k_float - float(group_starts[idx].item())
        selected_pos = selected_pos + group_pos[idx] * (remaining / group_count[idx])

    return float((selected_pos / k_float).item())


def early_precision_ratio(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk: int | None = None,
    random_precision: float | None = None,
) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    pos_total = int(labels.sum().item())
    total = int(labels.numel())
    if pos_total <= 0 or total <= 0:
        return float("nan")

    ep = early_precision(scores, labels, topk=topk)
    baseline = float(pos_total) / float(total) if random_precision is None else float(random_precision)
    return float("nan") if baseline <= 0.0 else ep / baseline


def summarize_binary_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk: int | None = None,
    metric_names: list[str] | tuple[str, ...] | None = None,
    random_positive_rate: float | None = None,
) -> dict[str, float]:
    scores, labels = _validate_binary_inputs(scores, labels)
    requested_metrics = list(DEFAULT_BINARY_METRICS if metric_names is None else metric_names)
    unknown_metrics = sorted(set(requested_metrics) - set(DEFAULT_BINARY_METRICS))
    if unknown_metrics:
        raise ValueError(
            "`metric_names` contains unsupported binary metrics: "
            + ", ".join(unknown_metrics)
        )

    positive_count = int(labels.sum().item())
    total_count = int(labels.numel())
    random_auprc = float(random_positive_rate) if random_positive_rate is not None else (
        float(positive_count) / float(total_count)
        if total_count > 0 and positive_count > 0
        else float("nan")
    )

    result: dict[str, float] = {}
    if "auprc" in requested_metrics or "auprc_ratio" in requested_metrics:
        auprc = binary_auprc(scores, labels)
        if "auprc" in requested_metrics:
            result["auprc"] = auprc
        if "auprc_ratio" in requested_metrics:
            if math.isnan(auprc) or math.isnan(random_auprc) or random_auprc <= 0.0:
                result["auprc_ratio"] = float("nan")
            else:
                result["auprc_ratio"] = auprc / random_auprc

    if "auroc" in requested_metrics:
        result["auroc"] = binary_auroc(scores, labels)
    if "ep" in requested_metrics:
        result["ep"] = early_precision(scores, labels, topk=topk)
    if "ep_ratio" in requested_metrics:
        result["ep_ratio"] = early_precision_ratio(
            scores,
            labels,
            topk=topk,
            random_precision=random_positive_rate,
        )

    return {name: result[name] for name in requested_metrics}
