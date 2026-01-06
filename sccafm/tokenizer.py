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
    def __init__(self, cond_dict=None, simplify=False):
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

    def __call__(
            self, 
            adata, 
            platform_key=None, 
            species_key=None,
            tissue_key=None, 
            disease_key=None
    ):
        C = adata.n_obs

        # If simplify mode: return all zero tokens
        if self.simplify:
            return torch.zeros((C, 4), dtype=torch.long)

        obs = adata.obs

        keys = [platform_key, species_key, tissue_key, disease_key]
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
    Assign a unique batch ID (integer counter) per adata input.
    Output: (C, 1) tensor filled with batch index.
    """
    def __init__(self, simplify=False):
        self.counter = 0
        self.simplify = simplify

    def __call__(self, adata):
        C = adata.n_obs

        # In simplify mode: always output zero token
        if self.simplify:
            return torch.zeros((C, 1), dtype=torch.long)

        # Normal mode: assign increasing batch_id
        batch_id = self.counter
        self.counter += 1

        out = np.full((C, 1), batch_id, dtype=np.int64)
        return torch.tensor(out)


class TomeTokenizer:
    """
    Unified Transcriptome Tokenizer:
    Internally manages GeneTokenizer, ExprTokenizer, CondTokenizer, BatchTokenizer.
    """
    def __init__(
            self,
            token_dict,           # token dictionary DataFrame for GeneTokenizer
            max_length=4096,      # max length for gene/expression sequences
            cond_dict=None,       # optional pre-existing cond_dict DataFrame
            simplify=False        # if True -> cond and batch tokens simplified
    ):
        self.gene_tokenizer = GeneTokenizer(token_dict, max_length=max_length)
        self.expr_tokenizer = ExprTokenizer(max_length=max_length)
        self.cond_tokenizer = CondTokenizer(cond_dict=cond_dict, simplify=simplify)
        self.batch_tokenizer = BatchTokenizer(simplify=simplify)

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
    
    def _preprocess(
            self, 
            adata, 
            min_genes_per_cell=200, 
            min_cells_per_gene=3,
            log_norm=True, 
            n_top_genes=3000
    ):
        adata = adata.copy()

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
            platform_key=None,
            species_key=None,
            tissue_key=None,
            disease_key=None,
            preprocess=True,
            **kwargs
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
            adata = self._preprocess(adata, **kwargs)

        gene_tokens, gene_pad = self.gene_tokenizer(
            adata,
            gene_key=gene_key,
            order_matrix=order_matrix
        )

        expr_tokens, expr_pad = self.expr_tokenizer(
            adata,
            order_matrix=order_matrix
        )

        cond_tokens = self.cond_tokenizer(
            adata,
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

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