from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from typing import Optional

from ..assets import load_vocab_json

class ScPreprocessor:
    """
    Single-cell preprocessing pipeline:
    - quality control metrics
    - cell/gene filtering
    - qc gene annotation/removal
    - HVG selection
    - normalization
    """

    def __init__(
        self,
        min_genes: Optional[int] = 200,
        min_cells: Optional[int] = 3,
        max_pct_counts_mt: Optional[float] = 20.0,
        target_sum: float = 1e4,
        log1p: bool = True,
        n_top_genes: Optional[int] = 2000,
        hvg_flavor: str = "seurat",
        subset_hvg: bool = True,
        remove_mito_genes: bool = True,
        remove_ribo_genes: bool = False,
        remove_hb_genes: bool = False,
        token_dict: Optional[pd.DataFrame] = None,
        gene_key: Optional[str] = None,
        sanitize_X: bool = True,
        X_fill_value: float = 0.0,
        inplace: bool = False,
    ) -> None:
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_pct_counts_mt = max_pct_counts_mt
        self.target_sum = target_sum
        self.log1p = log1p
        self.n_top_genes = n_top_genes
        self.hvg_flavor = hvg_flavor
        self.subset_hvg = subset_hvg
        self.remove_mito_genes = remove_mito_genes
        self.remove_ribo_genes = remove_ribo_genes
        self.remove_hb_genes = remove_hb_genes
        self.token_dict = token_dict
        self.gene_key = gene_key
        self.sanitize_X = sanitize_X
        self.X_fill_value = float(X_fill_value)
        self.inplace = inplace

    def _coerce_expression_value(self, value) -> float:
        if value is None:
            return self.X_fill_value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "nan", "na", "none", "null", "<na>"}:
                return self.X_fill_value
            try:
                value = float(normalized)
            except ValueError as exc:
                raise ValueError(f"Unable to convert expression value {value!r} to float.") from exc

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unable to convert expression value {value!r} to float.") from exc

        if not np.isfinite(value):
            return self.X_fill_value

        return value

    def _sanitize_expression_matrix(self, adata) -> None:
        if not self.sanitize_X:
            return

        X = adata.X

        if sp.issparse(X):
            if X.dtype == object:
                dense = np.asarray(X.toarray(), dtype=object)
                vectorized = np.vectorize(self._coerce_expression_value, otypes=[float])
                adata.X = sp.csr_matrix(vectorized(dense))
                return

            X = X.copy()
            X.data = np.nan_to_num(
                X.data,
                nan=self.X_fill_value,
                posinf=self.X_fill_value,
                neginf=self.X_fill_value,
            )
            adata.X = X
            return

        X = np.asarray(X, dtype=object if getattr(X, "dtype", None) == object else None)
        if X.dtype == object:
            vectorized = np.vectorize(self._coerce_expression_value, otypes=[float])
            adata.X = vectorized(X)
            return

        adata.X = np.nan_to_num(
            X,
            nan=self.X_fill_value,
            posinf=self.X_fill_value,
            neginf=self.X_fill_value,
        )

    def _ensure_dense_output(self, adata) -> None:
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        else:
            adata.X = np.asarray(adata.X)

    def _get_gene_names(self, adata) -> list[str]:
        if self.gene_key is None:
            return adata.var_names.astype(str).tolist()

        if self.gene_key not in adata.var.columns:
            raise KeyError(
                f"`gene_key={self.gene_key}` not found in `adata.var.columns`. "
                f"Available columns: {list(adata.var.columns)}"
            )

        return adata.var[self.gene_key].astype(str).tolist()

    def _detect_gene_name_type(self, gene_names: list[str]) -> str:
        if len(gene_names) == 0:
            raise ValueError("No genes found in input.")
        if all(str(g).strip().upper().startswith("ENSG") for g in gene_names):
            return "ensembl"
        return "symbol"

    @staticmethod
    def _normalize_symbol(x) -> str:
        return str(x).strip().upper()

    @staticmethod
    def _normalize_ensembl(x) -> str:
        value = str(x).strip().upper()
        if value.startswith("ENSG"):
            value = value.split(".", 1)[0]
        return value

    def _filter_genes_by_token_dict(self, adata) -> None:
        if self.token_dict is None:
            return

        if not isinstance(self.token_dict, pd.DataFrame):
            raise TypeError("`token_dict` must be a pandas DataFrame.")
        required_columns = {"gene_symbol", "gene_id"}
        missing = required_columns - set(self.token_dict.columns)
        if missing:
            raise ValueError(
                f"`token_dict` is missing required columns: {sorted(missing)}. "
                "Expected at least ['gene_symbol', 'gene_id']."
            )

        gene_names = self._get_gene_names(adata)
        gene_name_type = self._detect_gene_name_type(gene_names)

        if gene_name_type == "ensembl":
            valid_genes = {
                self._normalize_ensembl(gene_id)
                for gene_id in self.token_dict["gene_id"].dropna().tolist()
                if str(gene_id).strip() and not str(gene_id).startswith("<")
            }
            keep_mask = np.array(
                [self._normalize_ensembl(gene_name) in valid_genes for gene_name in gene_names],
                dtype=bool,
            )
        else:
            valid_genes = {
                self._normalize_symbol(symbol)
                for symbol in self.token_dict["gene_symbol"].dropna().tolist()
                if str(symbol).strip()
            }
            keep_mask = np.array(
                [self._normalize_symbol(gene_name) in valid_genes for gene_name in gene_names],
                dtype=bool,
            )

        if not keep_mask.any():
            raise ValueError("No genes in `adata` were found in `token_dict`.")

        adata._inplace_subset_var(keep_mask)

    def _annotate_qc_genes(self, adata) -> None:
        gene_names = np.asarray(self._get_gene_names(adata), dtype=str)
        gene_index = pd.Index(gene_names)
        adata.var["mt"] = gene_index.str.startswith(("MT-", "mt-"))
        adata.var["ribo"] = gene_index.str.startswith(("RPS", "RPL"))
        # Match canonical hemoglobin genes while excluding pseudogenes like HBP*.
        adata.var["hb"] = gene_index.str.contains(r"^HB(?!P)", regex=True)

    def _compute_qc_metrics(self, adata) -> None:
        qc_vars = [key for key in ("mt", "ribo", "hb") if key in adata.var.columns]
        if len(qc_vars) == 0:
            qc_vars = None
        sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None, inplace=True)

    def _filter_cells_and_genes(self, adata) -> None:
        if self.min_genes is not None:
            sc.pp.filter_cells(adata, min_genes=self.min_genes)
        if self.min_cells is not None:
            sc.pp.filter_genes(adata, min_cells=self.min_cells)
        if self.max_pct_counts_mt is not None and "pct_counts_mt" in adata.obs.columns:
            adata._inplace_subset_obs(adata.obs["pct_counts_mt"] <= self.max_pct_counts_mt)

    def _remove_qc_genes(self, adata) -> None:
        remove_mask = np.zeros(adata.n_vars, dtype=bool)

        if self.remove_mito_genes and "mt" in adata.var.columns:
            remove_mask |= adata.var["mt"].to_numpy()
        if self.remove_ribo_genes and "ribo" in adata.var.columns:
            remove_mask |= adata.var["ribo"].to_numpy()
        if self.remove_hb_genes and "hb" in adata.var.columns:
            remove_mask |= adata.var["hb"].to_numpy()

        if remove_mask.any():
            adata._inplace_subset_var(~remove_mask)

    def _normalize(self, adata) -> None:
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        if self.log1p:
            sc.pp.log1p(adata)

    def _select_hvg(self, adata) -> None:
        if self.n_top_genes is None:
            return

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.n_top_genes,
            flavor=self.hvg_flavor,
            subset=self.subset_hvg,
            inplace=True,
        )

    def __call__(self, adata):
        if not self.inplace:
            adata = adata.copy()

        self._filter_genes_by_token_dict(adata)
        self._sanitize_expression_matrix(adata)
        self._annotate_qc_genes(adata)
        self._compute_qc_metrics(adata)
        self._filter_cells_and_genes(adata)
        self._remove_qc_genes(adata)
        if self.hvg_flavor == "seurat_v3":
            self._select_hvg(adata)
            self._normalize(adata)
        else:
            self._normalize(adata)
            self._select_hvg(adata)
        self._ensure_dense_output(adata)

        return adata


def preprocess_adata(
    adata,
    min_genes: Optional[int] = 200,
    min_cells: Optional[int] = 3,
    max_pct_counts_mt: Optional[float] = 20.0,
    target_sum: float = 1e4,
    log1p: bool = True,
    n_top_genes: Optional[int] = 2000,
    hvg_flavor: str = "seurat",
    subset_hvg: bool = True,
    remove_mito_genes: bool = True,
    remove_ribo_genes: bool = True,
    remove_hb_genes: bool = True,
    token_dict: Optional[pd.DataFrame] = None,
    gene_key: Optional[str] = None,
    sanitize_X: bool = True,
    X_fill_value: float = 0.0,
    inplace: bool = False,
):
    """
    Convenience wrapper for `ScPreprocessor`.
    """

    return ScPreprocessor(
        min_genes=min_genes,
        min_cells=min_cells,
        max_pct_counts_mt=max_pct_counts_mt,
        target_sum=target_sum,
        log1p=log1p,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        subset_hvg=subset_hvg,
        remove_mito_genes=remove_mito_genes,
        remove_ribo_genes=remove_ribo_genes,
        remove_hb_genes=remove_hb_genes,
        token_dict=token_dict,
        gene_key=gene_key,
        sanitize_X=sanitize_X,
        X_fill_value=X_fill_value,
        inplace=inplace,
    )(adata)




if __name__ == "__main__":
    from pathlib import Path

    from anndata import AnnData

    from ..assets import resolve_model_assets

    root_dir = Path(__file__).resolve().parents[2]
    assets = resolve_model_assets(root_dir / "assets")
    token_dict = load_vocab_json(assets.vocab)

    X = np.array(
        [
            [8.0, 2.0, 0.0, 4.0, 1.0, 0.0],
            [7.0, 1.0, 0.0, 3.0, 0.0, 2.0],
            [1.0, 8.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 7.0, 0.0, 1.0, 6.0, 0.0],
            [3.0, 0.0, 9.0, 2.0, 0.0, 4.0],
            [2.0, 0.0, 8.0, 1.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(
        index=["MT-ND1", "TSPAN6", "TNMD", "ACTB", "MALAT1", "UNKNOWN_GENE"],
    )
    adata = AnnData(X=X, obs=obs, var=var)

    processed = preprocess_adata(
        adata,
        token_dict=token_dict,
        min_genes=1,
        min_cells=1,
        max_pct_counts_mt=100.0,
        n_top_genes=3,
        hvg_flavor="seurat",
        remove_mito_genes=True,
        remove_ribo_genes=True,
        remove_hb_genes=True,
    )

    print("shape:", processed.shape)
    print("var_names:", processed.var_names.tolist())
    print("highly_variable:", processed.var["highly_variable"].tolist())
    print("obs_columns:", list(processed.obs.columns))
    print("X_type:", type(processed.X).__name__)
    print("X:")
    print(processed.X)
