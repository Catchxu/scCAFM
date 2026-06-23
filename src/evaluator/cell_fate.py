from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .metrics import median_similarity_distribution


AnnDataLike = ad.AnnData | str | Path


@dataclass(frozen=True)
class CellFateSimilarityResult:
    distribution: pd.Series
    average: float
    median: float
    perturbed_count: int
    target_count: int
    gene_count: int
    expression_key: str
    target_key: str
    similarity: str
    genes: list[str]

    def to_summary(self) -> dict[str, Any]:
        return {
            "average": self.average,
            "median": self.median,
            "perturbed_count": self.perturbed_count,
            "target_count": self.target_count,
            "gene_count": self.gene_count,
            "expression_key": self.expression_key,
            "target_key": self.target_key,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class CellFateMedianSimilarityResult:
    unperturbed: CellFateSimilarityResult
    perturbed: CellFateSimilarityResult
    genes: list[str]
    perturbed_layer_key: str
    gene_selection: str
    deg_method: str | None = None
    top_n_degs: int | None = None

    @property
    def unperturbed_distribution(self) -> pd.Series:
        return self.unperturbed.distribution

    @property
    def perturbed_distribution(self) -> pd.Series:
        return self.perturbed.distribution

    @property
    def deg_genes(self) -> list[str]:
        return self.genes

    def to_summary(self) -> dict[str, Any]:
        return {
            "unperturbed_average": self.unperturbed.average,
            "perturbed_average": self.perturbed.average,
            "unperturbed_median": self.unperturbed.median,
            "perturbed_median": self.perturbed.median,
            "target_count": self.unperturbed.target_count,
            "perturbed_count": self.unperturbed.perturbed_count,
            "gene_count": len(self.genes),
            "perturbed_layer_key": self.perturbed_layer_key,
            "gene_selection": self.gene_selection,
            "deg_method": self.deg_method,
            "similarity": self.unperturbed.similarity,
        }


def select_target_up_degs(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    top_n_degs: int = 20,
    deg_method: str = "wilcoxon",
    cell_type_key: str | None = None,
) -> list[str]:
    """Return top target-up DEGs from `perturbed_h5ad.X` vs `target_h5ad.X`."""

    top_n_degs = int(top_n_degs)
    if top_n_degs <= 0:
        raise ValueError(f"`top_n_degs` must be positive, got {top_n_degs}.")

    perturbed = _load_adata(perturbed_h5ad)
    target = _load_adata(target_h5ad)
    _validate_single_cell_type(perturbed, cell_type_key=cell_type_key, role="perturbed")
    _validate_single_cell_type(target, cell_type_key=cell_type_key, role="target")
    perturbed, target, _ = _align_genes(perturbed, target)

    deg_genes = _target_up_degs(
        perturbed,
        target,
        top_n_degs=top_n_degs,
        method=deg_method,
    )
    return deg_genes


def evaluate_median_similarity(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    perturbed_layer_key: str,
    genes: list[str] | tuple[str, ...] | pd.Index | None = None,
    top_n_degs: int | None = None,
    deg_method: str = "wilcoxon",
    cell_type_key: str | None = None,
    similarity: str = "cosine",
    chunk_size: int = 128,
) -> CellFateMedianSimilarityResult:
    """Evaluate unperturbed and perturbed cells against target cells.

    `perturbed_h5ad.X` is treated as the initial expression, and
    `perturbed_h5ad.layers[perturbed_layer_key]` is treated as the generated
    perturbed expression. Both are compared against `target_h5ad.X`.

    If `top_n_degs` is set, target-up DEGs are selected from
    `perturbed_h5ad.X` vs `target_h5ad.X`; otherwise `genes` is used when
    provided, or all shared genes are used.
    """

    perturbed = _load_adata(perturbed_h5ad)
    target = _load_adata(target_h5ad)
    _validate_single_cell_type(perturbed, cell_type_key=cell_type_key, role="perturbed")
    _validate_single_cell_type(target, cell_type_key=cell_type_key, role="target")
    perturbed, target, common_genes = _align_genes(perturbed, target)

    if top_n_degs is not None:
        if genes is not None:
            raise ValueError("Pass either `genes` or `top_n_degs`, not both.")
        top_n_degs = int(top_n_degs)
        if top_n_degs <= 0:
            raise ValueError(f"`top_n_degs` must be positive, got {top_n_degs}.")
        selected_genes = _target_up_degs(
            perturbed,
            target,
            top_n_degs=top_n_degs,
            method=deg_method,
        )
        gene_selection = "target_up_degs"
    else:
        selected_genes = _resolve_eval_genes(common_genes, genes)
        gene_selection = "provided_genes" if genes is not None else "shared_genes"

    perturbed = perturbed[:, selected_genes].copy()
    target = target[:, selected_genes].copy()

    unperturbed = _evaluate_similarity_pair(
        perturbed=perturbed,
        target=target,
        expression_key="X",
        target_key="X",
        expression_label="unperturbed",
        similarity=similarity,
        chunk_size=chunk_size,
        genes=selected_genes,
    )
    perturbed_result = _evaluate_similarity_pair(
        perturbed=perturbed,
        target=target,
        expression_key=perturbed_layer_key,
        target_key="X",
        expression_label="perturbed",
        similarity=similarity,
        chunk_size=chunk_size,
        genes=selected_genes,
    )
    return CellFateMedianSimilarityResult(
        unperturbed=unperturbed,
        perturbed=perturbed_result,
        genes=selected_genes,
        perturbed_layer_key=str(perturbed_layer_key),
        gene_selection=gene_selection,
        deg_method=str(deg_method) if top_n_degs is not None else None,
        top_n_degs=top_n_degs,
    )


def _evaluate_similarity_pair(
    *,
    perturbed: ad.AnnData,
    target: ad.AnnData,
    expression_key: str,
    target_key: str,
    expression_label: str,
    similarity: str,
    chunk_size: int,
    genes: list[str],
) -> CellFateSimilarityResult:
    perturbed_matrix = _matrix_to_tensor(
        _get_matrix(perturbed, expression_key, role=expression_label)
    )
    target_matrix = _matrix_to_tensor(_get_matrix(target, target_key, role="target"))
    distribution_tensor = median_similarity_distribution(
        perturbed_matrix,
        target_matrix,
        similarity=similarity,
        chunk_size=chunk_size,
    )
    distribution = pd.Series(
        distribution_tensor.numpy(),
        index=perturbed.obs_names.astype(str),
        name=f"{expression_label}_median_similarity",
    )
    return CellFateSimilarityResult(
        distribution=distribution,
        average=float(distribution.mean()),
        median=float(distribution.median()),
        perturbed_count=int(perturbed.n_obs),
        target_count=int(target.n_obs),
        gene_count=len(genes),
        expression_key=str(expression_key),
        target_key=str(target_key),
        similarity=str(similarity),
        genes=list(genes),
    )


def _load_adata(source: AnnDataLike) -> ad.AnnData:
    if isinstance(source, ad.AnnData):
        return source
    return ad.read_h5ad(Path(source).expanduser())


def _subset_obs(
    adata: ad.AnnData,
    *,
    obs_key: str | None,
    obs_value: object | None,
    role: str,
) -> ad.AnnData:
    if obs_key is None:
        return adata
    if obs_key not in adata.obs:
        raise KeyError(
            f"`{role}_obs_key={obs_key}` not found in {role} AnnData obs columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    mask = adata.obs[obs_key].astype(str).to_numpy() == str(obs_value)
    if not bool(mask.any()):
        raise ValueError(
            f"No {role} cells found with `{obs_key}` == {obs_value!r}."
        )
    return adata[mask].copy()


def _validate_single_cell_type(
    adata: ad.AnnData,
    *,
    cell_type_key: str | None,
    role: str,
) -> None:
    if cell_type_key is None:
        return
    if cell_type_key not in adata.obs:
        raise KeyError(
            f"`cell_type_key={cell_type_key}` not found in {role} AnnData obs columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    values = pd.Index(adata.obs[cell_type_key].astype(str).unique()).dropna()
    if len(values) != 1:
        raise ValueError(
            f"{role} AnnData must contain exactly one `{cell_type_key}` value; "
            f"found {len(values)} values: {values.tolist()}"
        )


def _align_genes(
    target: ad.AnnData,
    reference: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    if not target.var_names.is_unique:
        raise ValueError("Target AnnData var_names must be unique.")
    if not reference.var_names.is_unique:
        raise ValueError("Reference AnnData var_names must be unique.")

    reference_genes = set(reference.var_names.astype(str))
    common_genes = [
        str(gene)
        for gene in target.var_names.astype(str)
        if str(gene) in reference_genes
    ]
    if not common_genes:
        raise ValueError("Target and reference AnnData have no shared genes.")
    return target[:, common_genes].copy(), reference[:, common_genes].copy(), common_genes


def _resolve_eval_genes(
    common_genes: list[str],
    genes: list[str] | tuple[str, ...] | pd.Index | None,
) -> list[str]:
    if genes is None:
        return list(common_genes)
    common = set(common_genes)
    selected = [str(gene) for gene in genes]
    missing = [gene for gene in selected if gene not in common]
    if missing:
        raise ValueError(
            "Requested evaluation genes are not shared by perturbed and target AnnData: "
            + ", ".join(missing[:10])
        )
    if not selected:
        raise ValueError("`genes` must contain at least one gene.")
    return selected


def _target_up_degs(
    perturbed: ad.AnnData,
    target: ad.AnnData,
    *,
    top_n_degs: int,
    method: str,
) -> list[str]:
    import scanpy as sc

    combined = ad.concat(
        {"perturbed": perturbed, "target": target},
        label="_cell_fate_group",
        join="inner",
        index_unique="-",
    )
    sc.tl.rank_genes_groups(
        combined,
        groupby="_cell_fate_group",
        groups=["target"],
        reference="perturbed",
        method=method,
        n_genes=combined.n_vars,
        rankby_abs=False,
    )
    result = combined.uns["rank_genes_groups"]
    names = [str(name) for name in result["names"]["target"].tolist()]
    scores = np.asarray(result["scores"]["target"], dtype=float)
    if "logfoldchanges" in result:
        logfoldchanges = np.asarray(result["logfoldchanges"]["target"], dtype=float)
        keep = np.isfinite(scores) & np.isfinite(logfoldchanges) & (logfoldchanges > 0.0)
    else:
        keep = np.isfinite(scores) & (scores > 0.0)

    selected = [name for name, keep_gene in zip(names, keep, strict=True) if keep_gene]
    if len(selected) < top_n_degs:
        selected = selected + [
            name
            for name, score in zip(names, scores, strict=True)
            if np.isfinite(score) and name not in selected
        ]
    selected = selected[:top_n_degs]
    if len(selected) < top_n_degs:
        raise ValueError(
            f"Only found {len(selected)} target-up genes, fewer than top_n_degs={top_n_degs}."
        )
    return selected


def _get_matrix(adata: ad.AnnData, key: str, *, role: str):
    if str(key) == "X":
        return adata.X
    if key not in adata.layers:
        raise KeyError(
            f"`{role}_key={key}` not found in {role} AnnData layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    return adata.layers[key]


def _matrix_to_tensor(matrix: object) -> torch.Tensor:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expression matrix must be 2D, got shape {array.shape}.")
    return torch.as_tensor(array, dtype=torch.float64)
