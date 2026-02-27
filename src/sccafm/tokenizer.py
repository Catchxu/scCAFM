import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import scanpy as sc


class GeneTokenizer:
    """
    Gene tokenizer: Convert gene names into token indices.
    Output shape: (C, L-1)
    """
    def __init__(self, token_dict: pd.DataFrame, pad_token="<pad>", max_length=4096):
        """
        token_dict: DataFrame with columns: token_index, gene_symbol, gene_id
        """
        self.max_length = max_length - 1  # remain one position for cond+batch

        # pad token index
        pad_row = token_dict[token_dict["gene_id"] == pad_token]
        if len(pad_row) == 0:
            raise ValueError("pad token not found in token_dict")
        self.pad_index = int(pad_row["token_index"].iloc[0])

        # build two lookup tables for fast search
        self.symbol2id = dict(zip(token_dict["gene_symbol"], token_dict["token_index"]))
        self.id2id = dict(zip(token_dict["gene_id"], token_dict["token_index"]))
        self.id2symbol = dict(zip(token_dict["gene_id"], token_dict["gene_symbol"]))
        self._symbol2id_upper = {
            str(k).strip().upper(): int(v) for k, v in self.symbol2id.items()
        }
        self._id2id_norm = {}
        self._idnorm2symbol = {}
        for k, v in self.id2id.items():
            ks = str(k).strip().upper()
            if ks.startswith("ENSG"):
                ks = ks.split(".", 1)[0]
            self._id2id_norm[ks] = int(v)
            sym = self.id2symbol.get(k, None)
            if sym is not None and str(sym).strip():
                self._idnorm2symbol[ks] = str(sym).strip().upper()

    @staticmethod
    def _norm_symbol(x):
        return str(x).strip().upper()

    @staticmethod
    def _norm_ensembl(x):
        s = str(x).strip().upper()
        if s.startswith("ENSG"):
            return s.split(".", 1)[0]
        return s

    def __call__(self, adata, gene_key=None, order_matrix=None):
        """
        Args:
            adata: AnnData, shape (C, G)
            gene_key: if not None, fetch gene names from adata.var[gene_key]
            order_matrix: (C, G) matrix giving order per cell. If provided, 
                rearrange gene names in each row accordingly.
        """
        # ---- 1. Get gene names ----
        if gene_key is None:
            gene_names = adata.var_names.tolist()
        else:
            gene_names = adata.var[gene_key].tolist()

        G = len(gene_names)
        if G > self.max_length:
            raise ValueError(f"G={G} exceeds max_length={self.max_length}")

        # ---- 2. Detect gene name type: symbol or ENSG id ----
        # If all names start with 'ENSG', treat as gene_id
        use_gene_id = all(str(name).strip().upper().startswith("ENSG") for name in gene_names)

        # Select lookup table
        if use_gene_id:
            base_order_idx = np.array(
                [self._id2id_norm.get(self._norm_ensembl(g), self.pad_index) for g in gene_names]
            )  # shape G
        else:
            base_order_idx = np.array(
                [self._symbol2id_upper.get(self._norm_symbol(g), self.pad_index) for g in gene_names]
            )  # shape G

        C = adata.n_obs
        tokens = np.full((C, self.max_length), self.pad_index, dtype=np.int64)

        # ---- 4. Apply per-cell ordering if provided ----
        # order_matrix is assumed to be C×G, containing column indices
        if order_matrix is not None:
            assert order_matrix.shape == (C, G)
            for i in range(C):
                ordered_indices = order_matrix[i]  # index of genes
                row = base_order_idx[ordered_indices]
                tokens[i, :G] = row
        else:
            # no special ordering, broadcast the same sequence to all cells
            tokens[:, :G] = base_order_idx

        # ---- 5. Pad mask: 1 means padded ----
        pad_mask = np.zeros((C, self.max_length), dtype=np.bool_)
        pad_mask[:, G:] = True

        return torch.tensor(tokens), torch.tensor(pad_mask)


class ExprTokenizer:
    """
    Expression tokenizer: convert adata.X (C, G) into padded matrix.
    Output shape: (C, L-1), pad filled with 0.
    """
    def __init__(self, max_length=4096):
        self.max_length = max_length - 1  # remain one position for cond+batch

    def __call__(self, adata, order_matrix=None):
        X = adata.X  # C×G
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        C, G = X.shape
        if G > self.max_length:
            raise ValueError(f"G={G} exceeds max_length={self.max_length}")

        # ---- 1. Per-cell ordering if provided ----
        if order_matrix is not None:
            assert order_matrix.shape == (C, G)
            X_ordered = np.zeros_like(X)
            for i in range(C):
                X_ordered[i] = X[i, order_matrix[i]]
            X = X_ordered

        # ---- 2. Pad to max_length ----
        expr = np.zeros((C, self.max_length), dtype=np.float32)
        expr[:, :G] = X

        pad_mask = np.zeros((C, self.max_length), dtype=np.bool_)
        pad_mask[:, G:] = True

        return torch.tensor(expr), torch.tensor(pad_mask)




class CondTokenizer:
    """
    Condition tokenizer:
    Inputs: adata + obs keys: platform_key, species_key, tissue_key, disease_key
    Output: (C, 4) tensor (4 condition tokens per cell)
    """
    def __init__(
        self,
        cond_dict=None,
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
    ):
        """
        cond_dict: DataFrame with columns: cond_value, token_index
        """

        # If no dict provided, create an empty one with a reserved 0 token
        if cond_dict is None:
            cond_dict = pd.DataFrame(
                {"cond_value": ["<unk>"], "token_index": [0]}
            )

        # Ensure pad token exists
        if "<unk>" not in cond_dict["cond_value"].values:
            raise ValueError("cond_dict must contain '<unk>' as token_index=0")

        self.cond_dict = cond_dict
        self.set_condition_keys(
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

    def set_condition_keys(
        self,
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
    ):
        """
        Set the obs keys used to build condition tokens.
        """
        self.condition_keys = [platform_key, species_key, tissue_key, disease_key]

    def set_condition_key_group(self, key_group):
        """
        Set condition keys from a sequence/dict.

        Args:
            key_group:
                - list/tuple with length 4 in order:
                  [platform_key, species_key, tissue_key, disease_key]
                - dict with optional keys:
                  platform_key, species_key, tissue_key, disease_key
        """
        if isinstance(key_group, (list, tuple)):
            if len(key_group) != 4:
                raise ValueError("key_group list/tuple must have length 4.")
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
        raise ValueError("key_group must be a list/tuple(length=4) or dict.")

    def _get_next_index(self):
        """Return next available token index."""
        return int(self.cond_dict["token_index"].max()) + 1

    def _fetch_or_add(self, value):
        """
        Lowercase the value, check exist.
        If not exist, add new token row.
        """
        value = str(value).strip().lower()

        # missing or nan -> return 0 token
        if value in {"", "nan", "none", "<na>", "na", "null"}:
            return 0

        df = self.cond_dict
        hit = df[df["cond_value"] == value]

        if len(hit) > 0:
            return int(hit["token_index"].iloc[0])

        # add new token
        new_idx = self._get_next_index()
        new_row = pd.DataFrame({"cond_value": [value], "token_index": [new_idx]})
        self.cond_dict = pd.concat([self.cond_dict, new_row], ignore_index=True)
        return new_idx

    def __call__(self, adata):
        C = adata.n_obs

        obs = adata.obs

        keys = self.condition_keys
        cond_values = []

        for j, key in enumerate(keys):
            if key is None:
                # Species defaults to human only when species_key is not configured.
                if j == 1:
                    cond_values.append(["human"] * C)
                else:
                    cond_values.append(["nan"] * C)
            elif key not in obs:
                # Ensure missing configured keys exist in adata.obs as NaN.
                obs[key] = np.nan
                cond_values.append(obs[key].astype(str).tolist())
            else:
                cond_values.append(obs[key].astype(str).tolist())

        # cond_values is list of 4 lists, each length C
        out = np.zeros((C, 4), dtype=np.int64)

        # For each condition type
        for j in range(4):
            for i in range(C):
                out[i, j] = self._fetch_or_add(cond_values[j][i])

        return torch.tensor(out, dtype=torch.long)


class TomeTokenizer:
    """
    Unified Transcriptome Tokenizer:
    Internally manages GeneTokenizer, ExprTokenizer, and CondTokenizer.
    """
    def __init__(
        self,
        token_dict,           # token dictionary DataFrame for GeneTokenizer
        max_length=2048,      # max length for gene/expression sequences
        gene_key=None,        # default adata.var key for gene ids/symbols
        cond_dict=None,       # optional pre-existing cond_dict DataFrame
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
        **kwargs
    ):
        self.gene_tokenizer = GeneTokenizer(token_dict, max_length=max_length)
        self.expr_tokenizer = ExprTokenizer(max_length=max_length)
        self.cond_tokenizer = CondTokenizer(
            cond_dict=cond_dict,
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )
        self.gene_key = gene_key

        self.prep_cfg = {
            'min_genes_per_cell': 200,
            'min_cells_per_gene': 3,
            'log_norm': True,
            'n_top_genes': 2000,
            'remove_mito_genes': True,
        }
        kwargs.pop("gene_key", None)
        self.prep_cfg.update(kwargs)
        self._token_symbol_set = set(self.gene_tokenizer._symbol2id_upper.keys())
        self._token_id_set = set(self.gene_tokenizer._id2id_norm.keys())

    def set_condition_keys(
        self,
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
    ):
        """
        Update condition keys for a new dataset without rebuilding tokenizer.
        """
        self.cond_tokenizer.set_condition_keys(
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

    def set_condition_key_group(self, key_group):
        """
        Update condition keys from a list/tuple/dict.
        """
        self.cond_tokenizer.set_condition_key_group(key_group)

    def set_gene_key(self, gene_key=None):
        """
        Update default gene key for adata.var lookup.
        """
        self.gene_key = gene_key

    def _check_pad_consistency(self, gene_pad: torch.Tensor, expr_pad: torch.Tensor):
        """
        Check if gene_pad and expr_pad are identical.
        """
        if not torch.equal(gene_pad, expr_pad):
            diff_idx = torch.nonzero(gene_pad != expr_pad)
            raise ValueError(
                f"gene_pad and expr_pad are not identical! "
                f"First differences at indices: {diff_idx[:10]}"
            )
    
    def _resolve_gene_names(self, adata, gene_key=None):
        if gene_key is None:
            return adata.var_names.astype(str).tolist()
        if gene_key not in adata.var:
            raise ValueError(f"gene_key='{gene_key}' not found in adata.var.")
        return adata.var[gene_key].astype(str).tolist()

    def _build_mito_mask(self, gene_names):
        mito = np.zeros(len(gene_names), dtype=bool)
        idnorm2symbol = self.gene_tokenizer._idnorm2symbol
        for i, g in enumerate(gene_names):
            s = str(g).strip().upper()
            if s.startswith("MT-"):
                mito[i] = True
                continue
            if s.startswith("ENSG"):
                sid = s.split(".", 1)[0]
                sym = idnorm2symbol.get(sid, "")
                if sym.startswith("MT-"):
                    mito[i] = True
        return mito

    def _preprocess(self, adata, gene_key=None):
        adata = adata.copy()

        min_cells_per_gene = self.prep_cfg["min_cells_per_gene"]
        min_genes_per_cell = self.prep_cfg["min_genes_per_cell"]
        log_norm = self.prep_cfg["log_norm"]
        n_top_genes = self.prep_cfg["n_top_genes"]
        remove_mito_genes = self.prep_cfg["remove_mito_genes"]

        # Remove mitochondrial genes before downstream filtering/normalization.
        if remove_mito_genes:
            gene_names = self._resolve_gene_names(adata, gene_key=gene_key)
            mito_mask = self._build_mito_mask(gene_names)
            if mito_mask.any():
                adata = adata[:, ~mito_mask].copy()

        if min_cells_per_gene:
            sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

        if min_genes_per_cell:
            sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)

        if log_norm:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        if n_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
        
        return adata

    @staticmethod
    def _norm_symbol(x):
        return str(x).strip().upper()

    @staticmethod
    def _norm_ensembl(x):
        s = str(x).strip().upper()
        if s.startswith("ENSG"):
            return s.split(".", 1)[0]
        return s

    def _intersect_with_token_dict(self, adata, gene_key=None):
        """
        Keep only genes that can be mapped by token_dict, using robust matching:
        - gene_symbol: case-insensitive
        - gene_id (ENSG*): version suffix tolerant (e.g., ENSG... .1)
        """
        if gene_key is None:
            raw_names = adata.var_names.astype(str).tolist()
        else:
            if gene_key not in adata.var:
                raise ValueError(f"gene_key='{gene_key}' not found in adata.var.")
            raw_names = adata.var[gene_key].astype(str).tolist()

        keep = []
        for g in raw_names:
            s_sym = self._norm_symbol(g)
            s_id = self._norm_ensembl(g)
            keep.append((s_sym in self._token_symbol_set) or (s_id in self._token_id_set))

        keep = np.asarray(keep, dtype=bool)
        if keep.sum() == 0:
            raise ValueError(
                "No overlapping genes between input adata and token_dict after robust symbol/ENSG matching."
            )
        return adata[:, keep].copy()

    def __call__(
        self,
        adata,
        gene_key=None,
        order_matrix=None,
        # backward-compatible override (prefer using set_condition_keys in init/runtime)
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
        preprocess=True,
        return_obs_names: bool = False,
    ):
        """
        Tokenize adata and return a dict of tensors.
        Output shapes:
            gene: (C, L-1)
            expr: (C, L-1)
            cond: (C, 4)
            pad: (C, L-1)
        """
        if gene_key is None:
            gene_key = self.gene_key

        # Always align genes to token_dict first, then optional preprocessing.
        adata = self._intersect_with_token_dict(adata, gene_key=gene_key)
        if preprocess:
            adata = self._preprocess(adata, gene_key=gene_key)

        gene_tokens, gene_pad = self.gene_tokenizer(
            adata,
            gene_key=gene_key,
            order_matrix=order_matrix
        )

        expr_tokens, expr_pad = self.expr_tokenizer(
            adata,
            order_matrix=order_matrix
        )

        if any(k is not None for k in [platform_key, species_key, tissue_key, disease_key]):
            self.set_condition_keys(
                platform_key=platform_key,
                species_key=species_key,
                tissue_key=tissue_key,
                disease_key=disease_key,
            )
        cond_tokens = self.cond_tokenizer(adata)

        self._check_pad_consistency(gene_pad, expr_pad)

        tokens = {
            "gene": gene_tokens,
            "expr": expr_tokens,
            "cond": cond_tokens,
            "pad": gene_pad
        }
        if return_obs_names:
            return tokens, adata.obs_names.astype(str).tolist()
        return tokens


class TomeDataset(Dataset):
    def __init__(self, tokens_dict):
        self.tokens = tokens_dict
        self._check_consistency()

        # Use the first available tensor to define the number of samples
        self.num_samples = next(iter(tokens_dict.values())).shape[0]

    def _check_consistency(self):
        """
        Ensures all tensors in the dictionary have the same first dimension.
        """
        it = iter(self.tokens.items())
        # Get the first item as a reference
        first_key, first_tensor = next(it)
        expected_size = first_tensor.shape[0]

        for key, tensor in it:
            if tensor.shape[0] != expected_size:
                raise ValueError(
                    f"Dimension mismatch in tokens_dict! "
                    f"Key '{first_key}' has size {expected_size}, "
                    f"but key '{key}' has size {tensor.shape[0]}."
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a single cell's tokens as a dict
        return {k: v[idx].clone() for k, v in self.tokens.items()}


def tome_collate_fn(batch):
    """
    Combines individual cell dicts into a batch dict.
    """
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}




if __name__ == "__main__":
    from torch.utils.data import DataLoader

    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    tokenizer = TomeTokenizer(token_dict)
    tokens = tokenizer(adata)
    dataset = TomeDataset(tokens)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,              # Number of cells per batch
        shuffle=True,               # Shuffle cells within this adata
        collate_fn=tome_collate_fn, # Uses your custom logic to stack dicts
        num_workers=4,              # Parallel data loading
        pin_memory=True             # Speed up tensor transfer to GPU
    )

    for batch in train_loader:
        print(batch)
