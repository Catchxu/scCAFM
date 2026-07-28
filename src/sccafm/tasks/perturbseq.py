from __future__ import annotations

import numbers

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from anndata import AnnData
from scipy.stats import wasserstein_distance

from .grn import PooledGRN


@dataclass(frozen=True, slots=True)
class PerturbSeqEvaluation:
    """Perturb-seq validation results for ranked pooled-GRN edges."""

    top_k_edges: int
    n_candidates: int
    n_evaluated_edges: int
    n_control_cells: int
    n_perturbed_cells: int
    n_perturbed_tfs: int
    mean_wasserstein_distance: float
    median_wasserstein_distance: float
    _edges: pd.DataFrame = field(repr=False, compare=False)

    def to_edge_table(self) -> pd.DataFrame:
        """Return one validation row per ranked TF-to-target edge."""

        return self._edges.copy()

    def to_dict(self) -> dict[str, float | int]:
        """Return the compact summary as a plain dictionary."""

        return {
            "top_k_edges": self.top_k_edges,
            "n_candidates": self.n_candidates,
            "n_evaluated_edges": self.n_evaluated_edges,
            "n_control_cells": self.n_control_cells,
            "n_perturbed_cells": self.n_perturbed_cells,
            "n_perturbed_tfs": self.n_perturbed_tfs,
            "mean_wasserstein_distance": self.mean_wasserstein_distance,
            "median_wasserstein_distance": self.median_wasserstein_distance,
        }


def _normalize_gene(value: object) -> str:
    return str(value).strip().upper()


def _validate_top_k_edges(top_k_edges: int) -> int:
    if isinstance(top_k_edges, bool) or not isinstance(
        top_k_edges, numbers.Integral
    ):
        raise TypeError("`top_k_edges` must be a positive integer.")
    normalized = int(top_k_edges)
    if normalized <= 0:
        raise ValueError("`top_k_edges` must be a positive integer.")
    return normalized


def _validate_expression(adata: AnnData) -> None:
    expression = adata.X
    if sp.issparse(expression):
        if not np.issubdtype(expression.dtype, np.number) or np.issubdtype(
            expression.dtype, np.complexfloating
        ):
            raise TypeError("`adata.X` must contain real numeric values.")
        values = np.asarray(expression.data)
    else:
        values = np.asarray(expression)
        if values.ndim != 2:
            raise ValueError("`adata.X` must be two-dimensional.")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise TypeError("`adata.X` must contain real numeric values.")
    if values.size > 0 and not np.isfinite(values).all():
        raise ValueError("`adata.X` must contain only finite values.")


def _expression_column(
    expression: object,
    row_indices: np.ndarray,
    column_index: int,
) -> np.ndarray:
    if sp.issparse(expression):
        values = expression[row_indices, column_index].toarray()
    else:
        values = np.asarray(expression)[row_indices, column_index]
    return np.asarray(values, dtype=np.float64).reshape(-1)


def evaluate_perturbseq_grn(
    grn: PooledGRN,
    adata: AnnData,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = "non-targeting",
    top_k_edges: int = 100,
    gene_key: str | None = None,
) -> PerturbSeqEvaluation:
    """Validate top pooled-GRN edges using held-out Perturb-seq responses.

    Candidate edges require a source TF with perturbation cells and a target
    measured in ``adata``. Self-edges are excluded before the top-k ranking.
    For every selected TF-to-target edge, the Wasserstein distance compares
    target expression in control cells with target expression after perturbing
    the source TF.
    """

    if not isinstance(grn, PooledGRN):
        raise TypeError(f"`grn` must be a PooledGRN, got {type(grn).__name__}.")
    if not isinstance(adata, AnnData):
        raise TypeError(f"`adata` must be an AnnData, got {type(adata).__name__}.")
    if grn.score_threshold is not None or grn.top_k_edges is not None:
        raise ValueError(
            "Perturb-seq evaluation requires an unfiltered PooledGRN; infer "
            "with `score_threshold=None` and `top_k_edges=None`."
        )
    top_k_edges = _validate_top_k_edges(top_k_edges)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("`adata` must contain at least one cell and one gene.")
    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"`perturbation_key={perturbation_key}` not found in `adata.obs`."
        )
    if gene_key is not None and gene_key not in adata.var.columns:
        raise KeyError(f"`gene_key={gene_key}` not found in `adata.var`.")
    _validate_expression(adata)

    raw_gene_names = (
        adata.var_names.tolist()
        if gene_key is None
        else adata.var[gene_key].tolist()
    )
    gene_names = [_normalize_gene(value) for value in raw_gene_names]
    if any(not name for name in gene_names):
        raise ValueError("Perturb-seq gene identifiers must not be empty.")
    if len(set(gene_names)) != len(gene_names):
        raise ValueError("Perturb-seq gene identifiers must be unique.")
    gene_to_index = {name: index for index, name in enumerate(gene_names)}

    perturbations = np.asarray(
        [_normalize_gene(value) for value in adata.obs[perturbation_key]],
        dtype=object,
    )
    normalized_control = _normalize_gene(control_label)
    control_indices = np.flatnonzero(perturbations == normalized_control)
    if control_indices.size == 0:
        raise ValueError(
            f"No control cells match `control_label={control_label}`."
        )

    source_names = [_normalize_gene(value) for value in grn.source_genes]
    target_names = [_normalize_gene(value) for value in grn.target_genes]
    perturbation_indices = {
        source: np.flatnonzero(perturbations == source)
        for source in source_names
    }
    eligible_sources = torch.tensor(
        [
            source in gene_to_index and perturbation_indices[source].size > 0
            for source in source_names
        ],
        dtype=torch.bool,
    )
    eligible_targets = torch.tensor(
        [target in gene_to_index for target in target_names],
        dtype=torch.bool,
    )
    non_self = torch.tensor(
        [
            [source_name != target_name for target_name in target_names]
            for source_name in source_names
        ],
        dtype=torch.bool,
    )
    candidate_mask = (
        eligible_sources.unsqueeze(1)
        & eligible_targets.unsqueeze(0)
        & non_self
    )
    n_candidates = int(candidate_mask.sum().item())
    if n_candidates == 0:
        raise ValueError(
            "No non-self edges connect perturbed source TFs to measured targets."
        )

    flat_candidate_indices = torch.nonzero(
        candidate_mask.reshape(-1),
        as_tuple=False,
    ).flatten()
    candidate_scores = grn.scores.reshape(-1)[flat_candidate_indices]
    order = torch.argsort(candidate_scores, descending=True, stable=True)
    selected_count = min(top_k_edges, n_candidates)
    selected_flat_indices = flat_candidate_indices[order[:selected_count]]

    target_count = len(target_names)
    rows: list[dict[str, float | int | str]] = []
    control_cache: dict[int, np.ndarray] = {}
    for rank, flat_index in enumerate(selected_flat_indices.tolist(), start=1):
        source_index, target_index = divmod(flat_index, target_count)
        source_name = source_names[source_index]
        target_name = target_names[target_index]
        expression_target_index = gene_to_index[target_name]
        if expression_target_index not in control_cache:
            control_cache[expression_target_index] = _expression_column(
                adata.X,
                control_indices,
                expression_target_index,
            )
        perturbed_indices = perturbation_indices[source_name]
        perturbed_values = _expression_column(
            adata.X,
            perturbed_indices,
            expression_target_index,
        )
        distance = float(
            wasserstein_distance(
                control_cache[expression_target_index],
                perturbed_values,
            )
        )
        rows.append(
            {
                "rank": rank,
                "Gene1": grn.source_genes[source_index],
                "Gene2": grn.target_genes[target_index],
                "score": float(grn.scores[source_index, target_index].item()),
                "n_control": int(control_indices.size),
                "n_perturbed": int(perturbed_indices.size),
                "wasserstein_distance": distance,
            }
        )

    edges = pd.DataFrame(
        rows,
        columns=[
            "rank",
            "Gene1",
            "Gene2",
            "score",
            "n_control",
            "n_perturbed",
            "wasserstein_distance",
        ],
    )
    distances = edges["wasserstein_distance"].to_numpy(dtype=float)
    eligible_source_names = {
        source
        for source, indices in perturbation_indices.items()
        if source in gene_to_index and indices.size > 0
    }
    perturbed_mask = np.isin(
        perturbations,
        np.asarray(sorted(eligible_source_names), dtype=object),
    )
    return PerturbSeqEvaluation(
        top_k_edges=top_k_edges,
        n_candidates=n_candidates,
        n_evaluated_edges=len(edges),
        n_control_cells=int(control_indices.size),
        n_perturbed_cells=int(perturbed_mask.sum()),
        n_perturbed_tfs=len(eligible_source_names),
        mean_wasserstein_distance=float(np.mean(distances)),
        median_wasserstein_distance=float(np.median(distances)),
        _edges=edges,
    )
