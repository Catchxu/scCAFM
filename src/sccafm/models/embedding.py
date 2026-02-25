import torch
import torch.nn as nn
import torch.nn.functional as F


class ExprMapping(nn.Module):
    """
    Map expression tokens (C, L) to (C, L, D) embeddings via bin embeddings and a small MLP.

    Note:
        - expr == 0: assign special embedding bin 0
        - expr > 0: 
            1. Encode token value to a D-dimensional vector via MLP
            2. Compute similarity to bin embeddings (bins 1..N-1)
            3. Softmax over bins -> probability vector
            4. Weighted sum over bin embeddings -> final embedding
    """
    def __init__(self, num_bins: int, embedding_dim: int, hidden_dim=128, dropout=0.1):
        """
        Args:
            num_bins: total number of bins including 0 bin
            embedding_dim: D, embedding dimension
            hidden_dim: hidden size of the MLP encoder
        """
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        self.bin_embeddings = nn.Parameter(
            torch.randn(num_bins, embedding_dim)
        )

        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, expr_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expr_tokens: (C, L) LongTensor

        Returns:
            (C, L, D) tensor
        """
        C, L = expr_tokens.shape
        device = expr_tokens.device

        zero_mask = (expr_tokens == 0)  # C, L
        nonzero_mask = ~zero_mask       # C, L

        out = None
        if nonzero_mask.any():
            nonzero_values = expr_tokens[nonzero_mask].float().unsqueeze(-1)
            encoded = self.encoder(nonzero_values)

            bin_embs = self.bin_embeddings[1:]
            sim = torch.matmul(encoded, bin_embs.t())
            probs = F.softmax(sim, dim=-1)

            emb = torch.matmul(probs, bin_embs)
            out = torch.zeros(C, L, self.embedding_dim, device=device, dtype=emb.dtype)
            out[nonzero_mask] = emb
        else:
            out = torch.zeros(
                C,
                L,
                self.embedding_dim,
                device=device,
                dtype=self.bin_embeddings.dtype,
            )

        if zero_mask.any():
            out[zero_mask] = self.bin_embeddings[0].to(out.dtype)

        return out


class TomoEmbedding(nn.Module):
    """
    Convert tokens dict to embeddings.

    Input: dict of tensors
    ```
    {
        "gene": gene_tokens,    # (C, L-1)
        "expr": expr_tokens,    # (C, L-1)
        "cond": cond_tokens,    # (C, 4)
        "batch": batch_tokens,  # (C, 1)
        "pad": gene_pad         # (C, L-1)
    }
    ```

    Output: (C, L, 2D) embedding
    """

    def __init__(
        self, 
        token_dict, 
        D=256,
        expr_num_bins=32, 
        cond_max_item=1024, 
        batch_max_item=256,
        **kwargs
    ):
        super().__init__()
        self.D = D

        # ---- Gene embedding ----
        num_gene_tokens = token_dict["token_index"].max() + 1
        pad_idx_row = token_dict[token_dict["gene_id"]=="<pad>"]
        pad_idx = int(pad_idx_row["token_index"].iloc[0]) if not pad_idx_row.empty else 0

        self.gene_embedding = nn.Embedding(num_gene_tokens, D, padding_idx=pad_idx)

        # ---- Expr embedding via ExprMapping ----
        self.expr_mapping = ExprMapping(num_bins=expr_num_bins, embedding_dim=D)

        # ---- Cond embedding ----
        self.cond_embedding = nn.Embedding(cond_max_item, D//4)

        # ---- Batch embedding ----
        self.batch_embedding = nn.Embedding(batch_max_item, D)
    
    def _safety_check(self, tokens):
        cond_max_item = self.cond_embedding.num_embeddings-1
        batch_max_item = self.batch_embedding.num_embeddings-1

        if tokens["cond"].max() > cond_max_item:
            raise ValueError(
                f"Found cond token index exceeding cond_max_item={cond_max_item}"
            )

        if tokens["batch"].max() > batch_max_item:
            raise ValueError(
                f"Found batch token index exceeding batch_max_item={batch_max_item}"
            )

    def forward(self, tokens: dict):
        """
        Convert tokens dict to final (C, L, 2D) embedding
        """
        C, _ = tokens["gene"].shape

        self._safety_check(tokens)

        gene_emb = self.gene_embedding(tokens["gene"])
        expr_emb = self.expr_mapping(tokens["expr"])

        cond_emb = self.cond_embedding(tokens["cond"]) 
        cond_emb = cond_emb.view(C, 1, self.D)

        batch_emb = self.batch_embedding(tokens["batch"].squeeze(-1))
        batch_emb = batch_emb.unsqueeze(1)

        cb_emb = torch.cat([cond_emb, batch_emb], dim=-1)  # (C, 1, 2D)
        ge_emb = torch.cat([gene_emb, expr_emb], dim=-1)   # (C, L-1, 2D)
        final_emb = torch.cat([cb_emb, ge_emb], dim=1)     # (C, L, 2D)

        # ---- Build key_padding_mask ----
        # tokens["pad"] is (C, L-1), need to prepend False for cond+batch
        pad_mask = tokens["pad"]  # (C, L-1), dtype=bool
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        first_col = torch.zeros((C, 1), dtype=torch.bool, device=pad_mask.device)
        key_padding_mask = torch.cat([first_col, pad_mask], dim=1)

        return final_emb, key_padding_mask




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from ..tokenizer import TomeTokenizer

    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    tokenizer = TomeTokenizer(token_dict, simplify=True)
    tokens = tokenizer(adata)

    embedding_layer = TomoEmbedding(token_dict)
    final_emb, key_padding_mask = embedding_layer(tokens)

    print(final_emb.shape)
    print(key_padding_mask.shape)
