from __future__ import annotations

import torch

import numpy as np
import pandas as pd
import scipy.sparse as sp

from anndata import AnnData
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any


class BasicTokenizer:
    """
    Shared utilities for padded tokenizer implementations.

    Shape convention:
    - `G`: gene-token length after reserving one prefix slot in the full model sequence
    """

    RESERVED_TOKENS = 1

    def __init__(
        self,
        max_length: int = 2048,
    ) -> None:
        self._validate_sequence_config(max_length)
        self.max_length = max_length
        self.sequence_length = max_length - self.RESERVED_TOKENS

    @staticmethod
    def _validate_sequence_config(max_length: int) -> None:
        if max_length <= 0:
            raise ValueError(f"`max_length` must be positive, got {max_length}.")
        if BasicTokenizer.RESERVED_TOKENS >= max_length:
            raise ValueError(
                f"The reserved prefix token count ({BasicTokenizer.RESERVED_TOKENS}) "
                f"must be smaller than "
                f"`max_length` ({max_length})."
            )

    def _validate_feature_count(self, n_features: int, feature_name: str) -> None:
        if n_features == 0:
            raise ValueError(f"No {feature_name} found in input.")
        if n_features > self.sequence_length:
            raise ValueError(
                f"Number of {feature_name} ({n_features}) exceeds available sequence length "
                f"({self.sequence_length}). Increase `max_length` or reduce {feature_name}."
            )

    @staticmethod
    def _validate_adata(adata: AnnData) -> AnnData:
        if not isinstance(adata, AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got {type(adata).__name__}.")
        return adata

    def _build_padding_mask(self, n_cells: int, n_features: int) -> torch.BoolTensor:
        padding_mask = torch.ones((n_cells, self.sequence_length), dtype=torch.bool)
        padding_mask[:, :n_features] = False
        return padding_mask

    @staticmethod
    def _finalize_padding_mask(
        padding_mask: torch.BoolTensor,
    ) -> Optional[torch.BoolTensor]:
        if not padding_mask.any():
            return None
        return padding_mask


@dataclass
class GeneTokenizerOutput:
    input_ids: torch.LongTensor
    padding_mask: Optional[torch.BoolTensor]
    non_tf_mask: torch.BoolTensor
    gene_name_type: str


class GeneTokenizer(BasicTokenizer):
    """
    Tokenize genes into integer token IDs for single-cell models.

    Design:
    - Input genes can be gene symbols or Ensembl gene IDs.
    - Output is a padded token matrix of shape (C, G), where:
        C = number of cells
        G = max_length - 1
    - Unknown genes are mapped to `pad_index`.

    Expected token_dict columns:
    - token_index
    - gene_symbol
    - gene_id
    """

    REQUIRED_COLUMNS = ("token_index", "gene_symbol", "gene_id")

    def __init__(
        self,
        token_dict: pd.DataFrame,
        pad_token: str = "<pad>",
        species_key: Optional[str] = None,
        human_tfs: Optional[pd.DataFrame] = None,
        mouse_tfs: Optional[pd.DataFrame] = None,
        max_length: int = 2048,
    ) -> None:
        super().__init__(max_length=max_length)
        self._validate_token_dict(token_dict)

        token_df = token_dict.copy()

        # Normalize token_index
        token_df["token_index"] = token_df["token_index"].astype(int)

        self.pad_index = self._lookup_special_index(token_df, pad_token, special_name="pad_token")

        # Canonical mappings
        self.symbol_to_index: Dict[str, int] = {}
        self.ensembl_to_index: Dict[str, int] = {}
        self.ensembl_to_symbol: Dict[str, str] = {}
        self.species_key = species_key
        self.human_tf_symbols = self._build_tf_symbol_set(human_tfs)
        self.mouse_tf_symbols = self._build_tf_symbol_set(mouse_tfs)

        for _, row in token_df.iterrows():
            token_index = int(row["token_index"])

            symbol = row["gene_symbol"]
            if pd.notna(symbol) and str(symbol).strip():
                norm_symbol = self._normalize_symbol(symbol)
                self.symbol_to_index[norm_symbol] = token_index

            gene_id = row["gene_id"]
            if pd.notna(gene_id) and str(gene_id).strip():
                norm_gene_id = self._normalize_ensembl(gene_id)
                self.ensembl_to_index[norm_gene_id] = token_index

                if pd.notna(symbol) and str(symbol).strip():
                    self.ensembl_to_symbol[norm_gene_id] = self._normalize_symbol(symbol)

    @staticmethod
    def _build_tf_symbol_set(tf_df: Optional[pd.DataFrame]) -> set[str]:
        if tf_df is None:
            return set()
        if not isinstance(tf_df, pd.DataFrame):
            raise TypeError("TF tables must be pandas DataFrames.")
        if "TF" not in tf_df.columns:
            raise ValueError("TF tables must contain a 'TF' column.")

        return {
            str(tf).strip().upper()
            for tf in tf_df["TF"].dropna().tolist()
            if str(tf).strip()
        }

    @staticmethod
    def _validate_token_dict(token_dict: pd.DataFrame) -> None:
        if not isinstance(token_dict, pd.DataFrame):
            raise TypeError("`token_dict` must be a pandas DataFrame.")

        missing = [c for c in GeneTokenizer.REQUIRED_COLUMNS if c not in token_dict.columns]
        if missing:
            raise ValueError(
                f"`token_dict` is missing required columns: {missing}. "
                f"Expected columns: {GeneTokenizer.REQUIRED_COLUMNS}."
            )

        if token_dict.empty:
            raise ValueError("`token_dict` is empty.")

        if token_dict["token_index"].isna().any():
            raise ValueError("`token_dict['token_index']` contains missing values.")

        duplicated_indices = token_dict["token_index"].duplicated()
        if duplicated_indices.any():
            dup_vals = token_dict.loc[duplicated_indices, "token_index"].tolist()
            raise ValueError(f"Duplicate token_index values found: {dup_vals[:10]}")

    @staticmethod
    def _normalize_symbol(x: Any) -> str:
        return str(x).strip().upper()

    @staticmethod
    def _normalize_ensembl(x: Any) -> str:
        s = str(x).strip().upper()
        if s.startswith("ENSG"):
            s = s.split(".", 1)[0]
        return s

    @staticmethod
    def _looks_like_ensembl_vector(gene_names: Sequence[Any]) -> bool:
        if len(gene_names) == 0:
            return False
        return all(str(g).strip().upper().startswith("ENSG") for g in gene_names)

    @staticmethod
    def _lookup_special_index(
        token_df: pd.DataFrame,
        token_value: Optional[str],
        special_name: str,
    ) -> int:
        if token_value is None:
            raise ValueError(f"`{special_name}` cannot be None here.")

        rows = token_df[token_df["gene_id"] == token_value]
        if len(rows) == 0:
            raise ValueError(f"{special_name}={token_value!r} not found in token_dict['gene_id'].")

        return int(rows["token_index"].iloc[0])

    def _map_gene_names_to_indices(
        self,
        gene_names: Sequence[Any],
        gene_name_type: Optional[str] = None,
    ) -> tuple[torch.LongTensor, str]:
        if gene_name_type is None:
            gene_name_type = "ensembl" if self._looks_like_ensembl_vector(gene_names) else "symbol"

        if gene_name_type not in {"symbol", "ensembl"}:
            raise ValueError(
                f"`gene_name_type` must be one of {{'symbol', 'ensembl'}}, got {gene_name_type!r}."
            )

        if gene_name_type == "ensembl":
            token_ids = torch.tensor(
                [
                    self.ensembl_to_index.get(self._normalize_ensembl(g), self.pad_index)
                    for g in gene_names
                ],
                dtype=torch.long,
            )
        else:
            token_ids = torch.tensor(
                [
                    self.symbol_to_index.get(self._normalize_symbol(g), self.pad_index)
                    for g in gene_names
                ],
                dtype=torch.long,
            )

        return token_ids, gene_name_type

    def _extract_gene_names(
        self,
        adata: AnnData,
        gene_key: Optional[str] = None,
    ) -> list[str]:
        self._validate_adata(adata)

        if gene_key is None:
            return adata.var_names.tolist()

        if gene_key not in adata.var.columns:
            raise KeyError(
                f"`gene_key={gene_key}` not found in `adata.var.columns`. "
                f"Available columns: {list(adata.var.columns)}"
            )

        return adata.var[gene_key].tolist()

    @staticmethod
    def _normalize_species_name(value: Any) -> str:
        species = str(value).strip().lower()
        if species in {"human", "homo sapiens", "hs"}:
            return "human"
        if species in {"mouse", "mus musculus", "mm"}:
            return "mouse"
        return species

    def _get_species_values(self, adata: AnnData) -> list[str]:
        if self.species_key is None:
            return ["human"] * int(adata.n_obs)
        if self.species_key not in adata.obs.columns:
            raise KeyError(
                f"`species_key={self.species_key}` not found in `adata.obs.columns`. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return [self._normalize_species_name(x) for x in adata.obs[self.species_key].tolist()]

    def _build_gene_symbol_vector(
        self,
        gene_names: Sequence[Any],
        gene_name_type: str,
    ) -> list[Optional[str]]:
        if gene_name_type == "symbol":
            return [self._normalize_symbol(gene_name) for gene_name in gene_names]

        return [
            self.ensembl_to_symbol.get(self._normalize_ensembl(gene_name))
            for gene_name in gene_names
        ]

    def _build_non_tf_mask(
        self,
        adata: AnnData,
        gene_names: Sequence[Any],
        gene_name_type: str,
    ) -> torch.BoolTensor:
        n_cells = int(adata.n_obs)
        n_genes = len(gene_names)
        non_tf_mask = torch.ones((n_cells, self.sequence_length), dtype=torch.bool)
        gene_symbols = self._build_gene_symbol_vector(gene_names, gene_name_type)
        species_values = self._get_species_values(adata)

        human_non_tf = torch.tensor(
            [
                gene_symbol not in self.human_tf_symbols
                if gene_symbol is not None
                else True
                for gene_symbol in gene_symbols
            ],
            dtype=torch.bool,
        )
        mouse_non_tf = torch.tensor(
            [
                gene_symbol not in self.mouse_tf_symbols
                if gene_symbol is not None
                else True
                for gene_symbol in gene_symbols
            ],
            dtype=torch.bool,
        )

        for row_idx, species in enumerate(species_values):
            if species == "human":
                non_tf_mask[row_idx, :n_genes] = human_non_tf
            elif species == "mouse":
                non_tf_mask[row_idx, :n_genes] = mouse_non_tf
            else:
                non_tf_mask[row_idx, :n_genes] = True

        return non_tf_mask

    def __call__(
        self,
        adata: AnnData,
        gene_key: Optional[str] = None,
    ) -> GeneTokenizerOutput:
        """
        Tokenize genes for each cell in `adata`.

        Args:
            adata:
                AnnData-like object with:
                - `n_obs`
                - `var_names`
                - `var`
            gene_key:
                If provided, use `adata.var[gene_key]` as gene names.
                Otherwise use `adata.var_names`.

        Returns:
            GeneTokenizerOutput with:
            - input_ids:      (C, G)
            - padding_mask:   (C, G), True where padded
            - gene_name_type: inferred or provided type
        """
        self._validate_adata(adata)

        gene_names = self._extract_gene_names(adata, gene_key=gene_key)
        n_genes = len(gene_names)
        n_cells = int(adata.n_obs)

        self._validate_feature_count(n_genes, feature_name="genes")

        base_token_ids, resolved_type = self._map_gene_names_to_indices(
            gene_names,
        )

        input_ids = torch.full(
            (n_cells, self.sequence_length),
            fill_value=self.pad_index,
            dtype=torch.long,
        )
        input_ids[:, :n_genes] = base_token_ids[None, :]

        padding_mask = self._finalize_padding_mask(
            self._build_padding_mask(n_cells, n_genes)
        )
        non_tf_mask = self._build_non_tf_mask(adata, gene_names, resolved_type)

        return GeneTokenizerOutput(
            input_ids=input_ids,
            padding_mask=padding_mask,
            non_tf_mask=non_tf_mask,
            gene_name_type=resolved_type,
        )

    def encode_gene_list(
        self,
        gene_names: Sequence[str],
    ) -> torch.LongTensor:
        """
        Encode a single ordered gene list into token IDs without cell padding.
        """
        token_ids, _ = self._map_gene_names_to_indices(gene_names)
        return token_ids

    def get_vocab_size(self) -> int:
        return max(
            [self.pad_index] +
            list(self.symbol_to_index.values()) +
            list(self.ensembl_to_index.values())
        ) + 1


@dataclass
class ExprTokenizerOutput:
    expression_values: torch.FloatTensor
    padding_mask: Optional[torch.BoolTensor]


class ExprTokenizer(BasicTokenizer):
    """
    Tokenize expression matrix from AnnData into padded per-cell expression sequences.

    Design:
    - Input is `adata.X` with shape (C, G), where:
        C = number of cells
        G = number of genes
    - Output is padded to shape (C, G), where:
        G = max_length - 1
    - Padding values are filled with `pad_value` (default: 0.0)

    Notes:
    - This tokenizer assumes a fixed gene universe shared across cells.
    - It does not itself decide which genes to keep; it only reorders and pads.
    """

    def __init__(
        self,
        max_length: int = 2048,
        pad_value: float = 0.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(max_length=max_length)
        self.pad_value = float(pad_value)
        self.dtype = dtype

    @staticmethod
    def _to_dense_array(X: Any) -> np.ndarray:
        """
        Convert AnnData matrix-like object to a dense NumPy array.
        """
        if isinstance(X, np.ndarray):
            return X

        if sp.issparse(X):
            return X.toarray()

        if hasattr(X, "toarray"):
            return X.toarray()

        return np.asarray(X)

    @staticmethod
    def _validate_expression_matrix(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("Expression matrix must be convertible to a NumPy array.")

        if X.ndim != 2:
            raise ValueError(f"Expression matrix must be 2D, got shape {X.shape}.")

        if np.issubdtype(X.dtype, np.integer):
            X = X.astype(np.float32, copy=False)

        return X

    def _to_expression_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = X.astype(self.dtype, copy=False)
        return torch.as_tensor(X)

    def __call__(
        self,
        adata: AnnData,
    ) -> ExprTokenizerOutput:
        """
        Convert `adata.X` into padded expression sequences.

        Args:
            adata:
                AnnData-like object with `.X`

        Returns:
            ExprTokenizerOutput with:
            - expression_values: (C, G)
            - padding_mask:      (C, G), True where padded
        """
        self._validate_adata(adata)

        X = self._to_dense_array(adata.X)
        X = self._validate_expression_matrix(X)

        n_cells, n_genes = X.shape

        self._validate_feature_count(n_genes, feature_name="genes")

        X = self._to_expression_tensor(X)

        expression_values = torch.full(
            (n_cells, self.sequence_length),
            fill_value=self.pad_value,
            dtype=X.dtype,
        )
        expression_values[:, :n_genes] = X

        padding_mask = self._finalize_padding_mask(
            self._build_padding_mask(n_cells, n_genes)
        )

        return ExprTokenizerOutput(expression_values=expression_values, padding_mask=padding_mask)


class CondTokenizer:
    """
    Tokenize per-cell condition metadata into four condition tokens.

    Inputs:
    - `adata.obs`
    - condition keys: `platform_key`, `species_key`, `tissue_key`, `disease_key`

    Output:
    - `torch.long` tensor of shape `(C, 4)`
    """

    REQUIRED_COLUMNS = ("cond_value", "token_index")
    DEFAULT_CONDITION_NAMES = ("platform", "species", "tissue", "disease")

    def __init__(
        self,
        cond_dict: Optional[pd.DataFrame] = None,
        platform_key: Optional[str] = None,
        species_key: Optional[str] = None,
        tissue_key: Optional[str] = None,
        disease_key: Optional[str] = None,
    ) -> None:
        if cond_dict is None:
            cond_dict = pd.DataFrame({"cond_value": ["<unk>"], "token_index": [0]})

        self.cond_dict = self._validate_cond_dict(cond_dict)
        self.cond_to_index = {
            str(row["cond_value"]).strip().lower(): int(row["token_index"])
            for _, row in self.cond_dict.iterrows()
        }
        self.next_index = int(self.cond_dict["token_index"].max()) + 1

        self.set_condition_keys(
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

    @classmethod
    def _validate_cond_dict(cls, cond_dict: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(cond_dict, pd.DataFrame):
            raise TypeError("`cond_dict` must be a pandas DataFrame.")

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in cond_dict.columns]
        if missing:
            raise ValueError(
                f"`cond_dict` is missing required columns: {missing}. "
                f"Expected columns: {cls.REQUIRED_COLUMNS}."
            )

        if cond_dict.empty:
            raise ValueError("`cond_dict` must not be empty.")

        cond_dict = cond_dict.copy()
        cond_dict["cond_value"] = cond_dict["cond_value"].astype(str).str.strip().str.lower()

        if cond_dict["token_index"].isna().any():
            raise ValueError("`cond_dict['token_index']` contains missing values.")

        cond_dict["token_index"] = cond_dict["token_index"].astype(int)

        duplicated_values = cond_dict["cond_value"].duplicated()
        if duplicated_values.any():
            dup_vals = cond_dict.loc[duplicated_values, "cond_value"].tolist()
            raise ValueError(f"Duplicate cond_value entries found: {dup_vals[:10]}")

        duplicated_indices = cond_dict["token_index"].duplicated()
        if duplicated_indices.any():
            dup_vals = cond_dict.loc[duplicated_indices, "token_index"].tolist()
            raise ValueError(f"Duplicate token_index values found: {dup_vals[:10]}")

        unk_rows = cond_dict[cond_dict["cond_value"] == "<unk>"]
        if len(unk_rows) == 0:
            raise ValueError("`cond_dict` must contain '<unk>' in `cond_value`.")
        if int(unk_rows["token_index"].iloc[0]) != 0:
            raise ValueError("`cond_dict` must map '<unk>' to token_index 0.")

        return cond_dict.sort_values("token_index").reset_index(drop=True)

    def set_condition_keys(
        self,
        platform_key: Optional[str] = None,
        species_key: Optional[str] = None,
        tissue_key: Optional[str] = None,
        disease_key: Optional[str] = None,
    ) -> None:
        self.condition_keys = [platform_key, species_key, tissue_key, disease_key]

    def set_condition_key_group(self, key_group: Any) -> None:
        if isinstance(key_group, (list, tuple)):
            if len(key_group) != 4:
                raise ValueError("`key_group` list/tuple must have length 4.")
            self.condition_keys = list(key_group)
            return

        if isinstance(key_group, dict):
            self.condition_keys = [
                key_group.get("platform_key"),
                key_group.get("species_key"),
                key_group.get("tissue_key"),
                key_group.get("disease_key"),
            ]
            return

        raise ValueError("`key_group` must be a list/tuple of length 4 or a dict.")

    @staticmethod
    def _normalize_condition_value(value: Any) -> str:
        value = str(value).strip().lower()
        if value in {"", "nan", "none", "<na>", "na", "null"}:
            return "<unk>"
        return value

    def _fetch_or_add(self, value: Any) -> int:
        value = self._normalize_condition_value(value)

        if value in self.cond_to_index:
            return self.cond_to_index[value]

        token_index = self.next_index
        self.next_index += 1
        self.cond_to_index[value] = token_index
        self.cond_dict = pd.concat(
            [
                self.cond_dict,
                pd.DataFrame({"cond_value": [value], "token_index": [token_index]}),
            ],
            ignore_index=True,
        )
        return token_index

    def _get_condition_values(self, obs: pd.DataFrame, key: Optional[str], condition_idx: int) -> list[str]:
        if key is None:
            if condition_idx == 1:
                return ["human"] * len(obs)
            return ["<unk>"] * len(obs)

        if key not in obs.columns:
            return ["<unk>"] * len(obs)

        return obs[key].astype(str).tolist()

    def fit_obs(self, obs: pd.DataFrame) -> None:
        condition_columns = [
            self._get_condition_values(obs, key, idx)
            for idx, key in enumerate(self.condition_keys)
        ]
        for values in condition_columns:
            for value in values:
                self._fetch_or_add(value)

    def fit_adata(self, adata: AnnData) -> None:
        if not isinstance(adata, AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got {type(adata).__name__}.")
        self.fit_obs(adata.obs)

    def __call__(self, adata: AnnData) -> torch.LongTensor:
        if not isinstance(adata, AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got {type(adata).__name__}.")

        obs = adata.obs
        n_cells = int(adata.n_obs)
        condition_columns = [
            self._get_condition_values(obs, key, idx)
            for idx, key in enumerate(self.condition_keys)
        ]

        condition_ids = torch.zeros((n_cells, 4), dtype=torch.long)
        for col_idx, values in enumerate(condition_columns):
            for row_idx, value in enumerate(values):
                condition_ids[row_idx, col_idx] = self._fetch_or_add(value)

        return condition_ids


@dataclass
class ScTokenizerOutput:
    input_ids: torch.LongTensor
    expression_values: torch.Tensor
    condition_ids: torch.LongTensor
    padding_mask: Optional[torch.BoolTensor]
    non_tf_mask: torch.BoolTensor
    gene_name_type: str


class ScTokenizer:
    """
    Integrated tokenizer for single-cell inputs.

    It composes:
    - `GeneTokenizer`
    - `ExprTokenizer`
    - `CondTokenizer`

    Shape convention:
    - `G`: gene-token length
    """

    def __init__(
        self,
        token_dict: pd.DataFrame,
        cond_dict: Optional[pd.DataFrame] = None,
        pad_token: str = "<pad>",
        species_key: Optional[str] = None,
        human_tfs: Optional[pd.DataFrame] = None,
        mouse_tfs: Optional[pd.DataFrame] = None,
        max_length: int = 2048,
        expr_pad_value: float = 0.0,
        expr_dtype: np.dtype = np.float32,
        platform_key: Optional[str] = None,
        tissue_key: Optional[str] = None,
        disease_key: Optional[str] = None,
        gene_tokenizer: Optional[GeneTokenizer] = None,
        expr_tokenizer: Optional[ExprTokenizer] = None,
        cond_tokenizer: Optional[CondTokenizer] = None,
    ) -> None:
        self.gene_tokenizer = gene_tokenizer or GeneTokenizer(
            token_dict=token_dict,
            pad_token=pad_token,
            species_key=species_key,
            human_tfs=human_tfs,
            mouse_tfs=mouse_tfs,
            max_length=max_length,
        )
        self.expr_tokenizer = expr_tokenizer or ExprTokenizer(
            max_length=max_length,
            pad_value=expr_pad_value,
            dtype=expr_dtype,
        )
        self.cond_tokenizer = cond_tokenizer or CondTokenizer(
            cond_dict=cond_dict,
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

    def set_condition_keys(
        self,
        platform_key: Optional[str] = None,
        species_key: Optional[str] = None,
        tissue_key: Optional[str] = None,
        disease_key: Optional[str] = None,
    ) -> None:
        self.cond_tokenizer.set_condition_keys(
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

    def set_condition_key_group(self, key_group: Any) -> None:
        self.cond_tokenizer.set_condition_key_group(key_group)

    def __call__(
        self,
        adata: AnnData,
        gene_key: Optional[str] = None,
    ) -> ScTokenizerOutput:
        BasicTokenizer._validate_adata(adata)

        gene_output = self.gene_tokenizer(
            adata,
            gene_key=gene_key,
        )
        expr_output = self.expr_tokenizer(
            adata,
        )
        condition_ids = self.cond_tokenizer(adata)

        if gene_output.padding_mask is None and expr_output.padding_mask is None:
            padding_mask = None
        elif gene_output.padding_mask is not None and expr_output.padding_mask is not None:
            if not torch.equal(gene_output.padding_mask, expr_output.padding_mask):
                raise ValueError("GeneTokenizer and ExprTokenizer produced different padding masks.")
            padding_mask = gene_output.padding_mask
        else:
            raise ValueError("GeneTokenizer and ExprTokenizer produced different padding masks.")

        return ScTokenizerOutput(
            input_ids=gene_output.input_ids,
            expression_values=expr_output.expression_values,
            condition_ids=condition_ids,
            padding_mask=padding_mask,
            non_tf_mask=gene_output.non_tf_mask,
            gene_name_type=gene_output.gene_name_type,
        )




if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[2]
    token_dict = pd.read_csv(root_dir / "resources" / "token_dict.csv")
    human_tfs = pd.read_csv(root_dir / "resources" / "human_tfs.csv")
    mouse_tfs = pd.read_csv(root_dir / "resources" / "mouse_tfs.csv")

    obs = pd.DataFrame(
        {
            "platform": ["10X", "smart-seq"],
            "species": ["human", "human"],
            "tissue": ["lung", "brain"],
            "disease": ["healthy", "tumor"],
        },
        index=["cell_0", "cell_1"],
    )
    var_names = pd.Index(["ANKS3", "TNMD"])
    adata = AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=var_names),
    )

    tokenizer = ScTokenizer(
        token_dict=token_dict,
        species_key="species",
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        platform_key="platform",
        tissue_key="tissue",
        disease_key="disease",
        max_length=3,
    )
    output = tokenizer(
        adata,
    )

    print("input_ids:")
    print(output.input_ids)
    print("expression_values:")
    print(output.expression_values)
    print("condition_ids:")
    print(output.condition_ids)
    print("padding_mask:")
    print(output.padding_mask)
    print("non_tf_mask:")
    print(output.non_tf_mask)
    print("gene_name_type:")
    print(output.gene_name_type)
