from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .metrics import median_similarity_distribution


AnnDataLike = ad.AnnData | str | Path
GeneCollection = list[str] | tuple[str, ...] | pd.Index


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

    def to_summary(self) -> dict[str, object]:
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

    def to_summary(self) -> dict[str, object]:
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
            "top_n_degs": self.top_n_degs,
            "similarity": self.unperturbed.similarity,
        }


@dataclass(frozen=True)
class _PreparedCellFateData:
    perturbed: ad.AnnData
    target: ad.AnnData
    shared_genes: list[str]

    def subset_genes(self, genes: list[str]) -> _PreparedCellFateData:
        return _PreparedCellFateData(
            perturbed=self.perturbed[:, genes].copy(),
            target=self.target[:, genes].copy(),
            shared_genes=list(genes),
        )


@dataclass(frozen=True)
class _GeneSelection:
    genes: list[str]
    name: str
    deg_method: str | None = None
    top_n_degs: int | None = None


def select_target_up_degs(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    top_n_degs: int = 20,
    deg_method: str = "wilcoxon",
    cell_type_key: str | None = None,
) -> list[str]:
    """Return top target-up DEGs from `perturbed_h5ad.X` vs `target_h5ad.X`."""

    data = _prepare_cell_fate_data(
        perturbed_h5ad=perturbed_h5ad,
        target_h5ad=target_h5ad,
        cell_type_key=cell_type_key,
    )
    return _find_target_up_degs(
        data,
        top_n_degs=_positive_int(top_n_degs, name="top_n_degs"),
        method=deg_method,
    )


def evaluate_median_similarity(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    perturbed_layer_key: str,
    genes: GeneCollection | None = None,
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

    data = _prepare_cell_fate_data(
        perturbed_h5ad=perturbed_h5ad,
        target_h5ad=target_h5ad,
        cell_type_key=cell_type_key,
    )
    selection = _select_evaluation_genes(
        data,
        genes=genes,
        top_n_degs=top_n_degs,
        deg_method=deg_method,
    )
    data = data.subset_genes(selection.genes)

    unperturbed = _evaluate_similarity_pair(
        data=data,
        expression_key="X",
        expression_label="unperturbed",
        similarity=similarity,
        chunk_size=chunk_size,
    )
    perturbed_result = _evaluate_similarity_pair(
        data=data,
        expression_key=perturbed_layer_key,
        expression_label="perturbed",
        similarity=similarity,
        chunk_size=chunk_size,
    )
    return CellFateMedianSimilarityResult(
        unperturbed=unperturbed,
        perturbed=perturbed_result,
        genes=selection.genes,
        perturbed_layer_key=str(perturbed_layer_key),
        gene_selection=selection.name,
        deg_method=selection.deg_method,
        top_n_degs=selection.top_n_degs,
    )


def _evaluate_similarity_pair(
    *,
    data: _PreparedCellFateData,
    expression_key: str,
    expression_label: str,
    similarity: str,
    chunk_size: int,
) -> CellFateSimilarityResult:
    perturbed_matrix = _matrix_to_tensor(
        _get_matrix(data.perturbed, expression_key, role=expression_label)
    )
    target_matrix = _matrix_to_tensor(_get_matrix(data.target, "X", role="target"))
    distribution_tensor = median_similarity_distribution(
        perturbed_matrix,
        target_matrix,
        similarity=similarity,
        chunk_size=chunk_size,
    )
    distribution = pd.Series(
        distribution_tensor.numpy(),
        index=data.perturbed.obs_names.astype(str),
        name=f"{expression_label}_median_similarity",
    )
    return CellFateSimilarityResult(
        distribution=distribution,
        average=float(distribution.mean()),
        median=float(distribution.median()),
        perturbed_count=int(data.perturbed.n_obs),
        target_count=int(data.target.n_obs),
        gene_count=len(data.shared_genes),
        expression_key=str(expression_key),
        target_key="X",
        similarity=str(similarity),
        genes=list(data.shared_genes),
    )


def _prepare_cell_fate_data(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    cell_type_key: str | None,
) -> _PreparedCellFateData:
    perturbed = _load_adata(perturbed_h5ad)
    target = _load_adata(target_h5ad)
    _validate_single_cell_type(perturbed, cell_type_key=cell_type_key, role="perturbed")
    _validate_single_cell_type(target, cell_type_key=cell_type_key, role="target")
    perturbed, target, shared_genes = _align_shared_genes(perturbed, target)
    return _PreparedCellFateData(
        perturbed=perturbed,
        target=target,
        shared_genes=shared_genes,
    )


def _load_adata(source: AnnDataLike) -> ad.AnnData:
    if isinstance(source, ad.AnnData):
        return source
    return ad.read_h5ad(Path(source).expanduser())


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
    values = pd.Index(adata.obs[cell_type_key].dropna().astype(str).unique())
    if len(values) != 1:
        raise ValueError(
            f"{role} AnnData must contain exactly one `{cell_type_key}` value; "
            f"found {len(values)} values: {values.tolist()}"
        )


def _align_shared_genes(
    perturbed: ad.AnnData,
    target: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    if not perturbed.var_names.is_unique:
        raise ValueError("Perturbed AnnData var_names must be unique.")
    if not target.var_names.is_unique:
        raise ValueError("Target AnnData var_names must be unique.")

    target_genes = set(target.var_names.astype(str))
    shared_genes = [
        str(gene)
        for gene in perturbed.var_names.astype(str)
        if str(gene) in target_genes
    ]
    if not shared_genes:
        raise ValueError("Perturbed and target AnnData have no shared genes.")
    return (
        perturbed[:, shared_genes].copy(),
        target[:, shared_genes].copy(),
        shared_genes,
    )


def _select_evaluation_genes(
    data: _PreparedCellFateData,
    *,
    genes: GeneCollection | None,
    top_n_degs: int | None,
    deg_method: str,
) -> _GeneSelection:
    if genes is not None and top_n_degs is not None:
        raise ValueError("Pass either `genes` or `top_n_degs`, not both.")
    if top_n_degs is not None:
        count = _positive_int(top_n_degs, name="top_n_degs")
        return _GeneSelection(
            genes=_find_target_up_degs(data, top_n_degs=count, method=deg_method),
            name="target_up_degs",
            deg_method=str(deg_method),
            top_n_degs=count,
        )
    if genes is None:
        return _GeneSelection(genes=list(data.shared_genes), name="shared_genes")
    return _GeneSelection(
        genes=_validate_requested_genes(data.shared_genes, genes),
        name="provided_genes",
    )


def _validate_requested_genes(
    shared_genes: list[str],
    genes: GeneCollection,
) -> list[str]:
    shared = set(shared_genes)
    selected = [str(gene) for gene in genes]
    missing = [gene for gene in selected if gene not in shared]
    if missing:
        raise ValueError(
            "Requested evaluation genes are not shared by perturbed and target AnnData: "
            + ", ".join(missing[:10])
        )
    if not selected:
        raise ValueError("`genes` must contain at least one gene.")
    return selected


def _find_target_up_degs(
    data: _PreparedCellFateData,
    *,
    top_n_degs: int,
    method: str,
) -> list[str]:
    import scanpy as sc

    combined = ad.concat(
        {"perturbed": data.perturbed, "target": data.target},
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


def _positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"`{name}` must be positive, got {value}.")
    return value


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
