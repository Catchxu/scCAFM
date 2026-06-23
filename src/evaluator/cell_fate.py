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
    target_count: int
    reference_count: int
    gene_count: int
    target_key: str
    reference_key: str
    similarity: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "average": self.average,
            "median": self.median,
            "target_count": self.target_count,
            "reference_count": self.reference_count,
            "gene_count": self.gene_count,
            "target_key": self.target_key,
            "reference_key": self.reference_key,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class CellFateDEGSimilarityResult:
    unperturbed: pd.Series
    perturbed: pd.Series
    deg_genes: list[str]
    perturbed_layer_key: str
    target_count: int
    perturbed_count: int
    similarity: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "unperturbed_average": float(self.unperturbed.mean()),
            "perturbed_average": float(self.perturbed.mean()),
            "unperturbed_median": float(self.unperturbed.median()),
            "perturbed_median": float(self.perturbed.median()),
            "target_count": self.target_count,
            "perturbed_count": self.perturbed_count,
            "deg_count": len(self.deg_genes),
            "perturbed_layer_key": self.perturbed_layer_key,
            "similarity": self.similarity,
        }


def evaluate_deg_median_similarity(
    *,
    perturbed_h5ad: AnnDataLike,
    target_h5ad: AnnDataLike,
    perturbed_layer_key: str,
    top_n_degs: int = 20,
    deg_method: str = "wilcoxon",
    cell_type_key: str | None = None,
    similarity: str = "cosine",
    chunk_size: int = 128,
) -> CellFateDEGSimilarityResult:
    """Evaluate initial and generated perturbed cells on target-up DEGs.

    The two input AnnData objects are expected to contain one cell type each.
    DEGs are computed from `perturbed.X` and `target.X`; the top target-up genes
    are then used to compute two median-similarity distributions:

    - unperturbed: `perturbed.X` vs `target.X`
    - perturbed: `perturbed.layers[perturbed_layer_key]` vs `target.X`
    """

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
    perturbed_degs = perturbed[:, deg_genes].copy()
    target_degs = target[:, deg_genes].copy()

    reference_matrix = _matrix_to_tensor(_get_matrix(target_degs, "X", role="target"))
    unperturbed_tensor = median_similarity_distribution(
        _matrix_to_tensor(_get_matrix(perturbed_degs, "X", role="perturbed")),
        reference_matrix,
        similarity=similarity,
        chunk_size=chunk_size,
    )
    perturbed_tensor = median_similarity_distribution(
        _matrix_to_tensor(
            _get_matrix(perturbed_degs, perturbed_layer_key, role="perturbed")
        ),
        reference_matrix,
        similarity=similarity,
        chunk_size=chunk_size,
    )
    index = perturbed_degs.obs_names.astype(str)
    return CellFateDEGSimilarityResult(
        unperturbed=pd.Series(
            unperturbed_tensor.numpy(),
            index=index,
            name="unperturbed_median_similarity",
        ),
        perturbed=pd.Series(
            perturbed_tensor.numpy(),
            index=index,
            name="perturbed_median_similarity",
        ),
        deg_genes=deg_genes,
        perturbed_layer_key=str(perturbed_layer_key),
        target_count=int(target_degs.n_obs),
        perturbed_count=int(perturbed_degs.n_obs),
        similarity=str(similarity),
    )


def evaluate_median_similarity(
    *,
    target_h5ad: AnnDataLike,
    reference_h5ad: AnnDataLike,
    target_key: str = "X",
    reference_key: str = "X",
    target_obs_key: str | None = None,
    target_obs_value: object | None = None,
    reference_obs_key: str | None = None,
    reference_obs_value: object | None = None,
    similarity: str = "cosine",
    chunk_size: int = 128,
) -> CellFateSimilarityResult:
    """Evaluate target cells against reference cells with median similarity.

    `target_key` and `reference_key` use `"X"` for `.X`; any other value is
    read from `.layers[key]`. The returned distribution contains one value per
    target cell: the median similarity from that target cell to all reference
    cells.
    """

    target = _load_adata(target_h5ad)
    reference = _load_adata(reference_h5ad)
    target = _subset_obs(
        target,
        obs_key=target_obs_key,
        obs_value=target_obs_value,
        role="target",
    )
    reference = _subset_obs(
        reference,
        obs_key=reference_obs_key,
        obs_value=reference_obs_value,
        role="reference",
    )
    target, reference, common_genes = _align_genes(target, reference)

    target_matrix = _matrix_to_tensor(_get_matrix(target, target_key, role="target"))
    reference_matrix = _matrix_to_tensor(
        _get_matrix(reference, reference_key, role="reference")
    )
    distribution_tensor = median_similarity_distribution(
        target_matrix,
        reference_matrix,
        similarity=similarity,
        chunk_size=chunk_size,
    )
    distribution = pd.Series(
        distribution_tensor.numpy(),
        index=target.obs_names.astype(str),
        name="median_similarity",
    )
    return CellFateSimilarityResult(
        distribution=distribution,
        average=float(distribution.mean()),
        median=float(distribution.median()),
        target_count=int(target.n_obs),
        reference_count=int(reference.n_obs),
        gene_count=len(common_genes),
        target_key=str(target_key),
        reference_key=str(reference_key),
        similarity=str(similarity),
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
