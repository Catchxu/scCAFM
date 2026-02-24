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
        use_gene_id = all(name.startswith("ENSG") for name in gene_names)

        # Select lookup table
        lookup = self.id2id if use_gene_id else self.symbol2id

        # ---- 3. Resolve token index for each gene ----
        # Unknown genes are mapped to pad
        base_order_idx = np.array([lookup.get(g, self.pad_index) for g in gene_names])  # shape G

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
        simplify=False,
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
    ):
        """
        cond_dict: DataFrame with columns: cond_value, token_index
        simplify: if True -> always return 0 token for all 4 features
        """
        self.simplify = simplify

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
        value = str(value).lower()

        # missing or nan -> return 0 token
        if value == "nan":
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

        # If simplify mode: return all zero tokens
        if self.simplify:
            return torch.zeros((C, 4), dtype=torch.long)

        obs = adata.obs

        keys = self.condition_keys
        cond_values = []

        for key in keys:
            if key is None or key not in obs:
                # missing key -> use pad (0)
                cond_values.append(["nan"] * C)
            else:
                cond_values.append(obs[key].astype(str).tolist())

        # cond_values is list of 4 lists, each length C
        out = np.zeros((C, 4), dtype=np.int64)

        # For each condition type
        for j in range(4):
            for i in range(C):
                out[i, j] = self._fetch_or_add(cond_values[j][i])

        return torch.tensor(out, dtype=torch.long)


class BatchTokenizer:
    """
    Batch tokenizer:
    Maps batch metadata from adata.obs to stable token IDs.
    Output: (C, 1) tensor.
    """
    def __init__(self, batch_dict=None, simplify=False, batch_key=None):
        self.simplify = simplify
        if batch_dict is None:
            batch_dict = pd.DataFrame(
                {"batch_value": ["<unk>"], "token_index": [0]}
            )
        if "<unk>" not in batch_dict["batch_value"].values:
            raise ValueError("batch_dict must contain '<unk>' as token_index=0")
        self.batch_dict = batch_dict
        self.set_batch_key(batch_key=batch_key)

    def set_batch_key(self, batch_key=None):
        """
        Set batch key(s) from adata.obs.

        batch_key can be:
        - str: single obs column
        - list/tuple[str]: multiple columns, joined as one batch label
        - None: all cells mapped to unknown batch token
        """
        if batch_key is None:
            self.batch_keys = None
        elif isinstance(batch_key, str):
            self.batch_keys = [batch_key]
        elif isinstance(batch_key, (list, tuple)):
            self.batch_keys = list(batch_key)
        else:
            raise ValueError("batch_key must be None, str, list, or tuple.")

    def set_batch_key_group(self, key_group):
        """
        Alias setter for consistency with CondTokenizer APIs.
        """
        self.set_batch_key(batch_key=key_group)

    def _get_next_index(self):
        return int(self.batch_dict["token_index"].max()) + 1

    def _fetch_or_add(self, value):
        value = str(value).lower()
        if value == "nan":
            return 0
        hit = self.batch_dict[self.batch_dict["batch_value"] == value]
        if len(hit) > 0:
            return int(hit["token_index"].iloc[0])
        new_idx = self._get_next_index()
        new_row = pd.DataFrame({"batch_value": [value], "token_index": [new_idx]})
        self.batch_dict = pd.concat([self.batch_dict, new_row], ignore_index=True)
        return new_idx

    def __call__(self, adata):
        C = adata.n_obs

        # In simplify mode: always output zero token
        if self.simplify:
            return torch.zeros((C, 1), dtype=torch.long)

        if self.batch_keys is None:
            values = ["nan"] * C
        else:
            obs = adata.obs
            cols = []
            for key in self.batch_keys:
                if key is None or key not in obs:
                    cols.append(["nan"] * C)
                else:
                    cols.append(obs[key].astype(str).tolist())
            if len(cols) == 1:
                values = cols[0]
            else:
                values = ["|".join(items) for items in zip(*cols)]

        out = np.zeros((C, 1), dtype=np.int64)
        for i in range(C):
            out[i, 0] = self._fetch_or_add(values[i])
        return torch.tensor(out, dtype=torch.long)


class TomeTokenizer:
    """
    Unified Transcriptome Tokenizer:
    Internally manages GeneTokenizer, ExprTokenizer, CondTokenizer, BatchTokenizer.
    """
    def __init__(
        self,
        token_dict,           # token dictionary DataFrame for GeneTokenizer
        max_length=2048,      # max length for gene/expression sequences
        cond_dict=None,       # optional pre-existing cond_dict DataFrame
        batch_dict=None,      # optional pre-existing batch_dict DataFrame
        simplify=False,       # if True -> cond and batch tokens simplified
        platform_key=None,
        species_key=None,
        tissue_key=None,
        disease_key=None,
        batch_key=None,
        **kwargs
    ):
        self.gene_tokenizer = GeneTokenizer(token_dict, max_length=max_length)
        self.expr_tokenizer = ExprTokenizer(max_length=max_length)
        self.cond_tokenizer = CondTokenizer(
            cond_dict=cond_dict,
            simplify=simplify,
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )
        self.batch_tokenizer = BatchTokenizer(
            batch_dict=batch_dict,
            simplify=simplify,
            batch_key=batch_key,
        )

        self.prep_cfg = {
            'min_genes_per_cell': 200,
            'min_cells_per_gene': 3,
            'log_norm': True,
            'n_top_genes': 3000
        }
        self.prep_cfg.update(kwargs)

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

    def set_batch_key(self, batch_key=None):
        """
        Update batch key(s) for a new dataset without rebuilding tokenizer.
        """
        self.batch_tokenizer.set_batch_key(batch_key=batch_key)

    def set_batch_key_group(self, key_group):
        """
        Update batch key(s) from a list/tuple/string.
        """
        self.batch_tokenizer.set_batch_key_group(key_group)

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
    
    def _preprocess(self, adata):
        adata = adata.copy()

        min_genes_per_cell = self.prep_cfg["min_genes_per_cell"]
        min_cells_per_gene = self.prep_cfg["min_cells_per_gene"]
        log_norm = self.prep_cfg["log_norm"]
        n_top_genes = self.prep_cfg["n_top_genes"]

        if min_genes_per_cell:
            sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
        
        if min_cells_per_gene:
            sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

        if log_norm:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        if n_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
        
        return adata

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
        batch_key=None,
        preprocess=True
    ):
        """
        Tokenize adata and return a dict of tensors.
        Output shapes:
            gene: (C, L-1)
            expr: (C, L-1)
            cond: (C, 4)
            batch: (C, 1)
            pad: (C, L-1)
        """
        if preprocess:
            adata = self._preprocess(adata)

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
        if batch_key is not None:
            self.set_batch_key(batch_key=batch_key)

        cond_tokens = self.cond_tokenizer(adata)

        batch_tokens = self.batch_tokenizer(adata)

        self._check_pad_consistency(gene_pad, expr_pad)

        return {
            "gene": gene_tokens,
            "expr": expr_tokens,
            "cond": cond_tokens,
            "batch": batch_tokens,
            "pad": gene_pad
        }


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

    tokenizer = TomeTokenizer(token_dict, simplify=True)
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
