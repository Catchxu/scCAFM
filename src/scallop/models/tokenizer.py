import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExprQuantizer(nn.Module):
    """
    Quantizes expression values into discrete bins:
        - expr_value == 0 -> bin 0
        - expr_value != 0 -> bins 1..num_bins-1 via small MLP
    """
    def __init__(self, num_bins: int = 10, hidden_dim: int = 64):
        super().__init__()
        assert num_bins >= 2, 'num_bins must be >= 2'
        self.num_bins = num_bins

        # MLP maps scalar -> logits for bins 1..num_bins-1
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_bins-1)
        )

    def forward(self, expr_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            expr_value: FloatTensor of shape (B, G)

        Returns:
            probs: (B, G, num_bins) probabilities
            mask_expr: (B, G) float mask where 1 = nonzero expr, 0 = zero expr
        Notes:
            - For positions where expr_value == 0, output is [1, 0, 0, ..., 0]
            - For positions where expr_value != 0, output is [0, softmax(logits)] across bins 1, ..., num_bins-1
        """
        if expr_value.dim() != 2:
            raise ValueError("expr_value must be (B, G)")

        B, G = expr_value.shape
        device = expr_value.device

        # mask: 1 for nonzero expr, 0 for zero expr
        mask_expr = (expr_value != 0).float()  # (B, G)

        # initialize all probs as [1, 0, ..., 0]
        probs = torch.zeros((B, G, self.num_bins), device=device, dtype=torch.float32)
        probs[..., 0] = 1.0

        # compute only nonzero positions
        if mask_expr.any():
            x_nonzero = expr_value[mask_expr.bool()].unsqueeze(-1).float()  # (K,1)
            logits = self.mlp(x_nonzero)                                    # (K,num_bins-1)
            probs_nonzero = F.softmax(logits, dim=-1)                       # (K,num_bins-1)

            # assign these to probs[..., 1:]
            probs[..., 1:][mask_expr.bool()] = probs_nonzero
            probs[..., 0][mask_expr.bool()] = 0.0  # since we now replaced zeros with nonzeros

        return probs, mask_expr  # (B,G,num_bins), (B,G)


class Tokenizer(nn.Module):
    """
    Tokenizer for single-cell samples (AnnData).
    """
    def __init__(self, num_genes: int, num_bins: int = 20, embed_dim: int = 256, num_conditions: Optional[int] = None):
        """
        Args:
            num_genes (int): number of genes
            num_bins (int): number of bins for expression values
            embed_dim (int): dimension of embedding space
            num_conditions (Optional[int]): number of condition types
        """
        super().__init__()
        self.num_genes = num_genes
        self.num_conditions = num_conditions
        self.embed_dim = embed_dim

        self.quantizer = ExprQuantizer(num_bins)

        # Embedding layers
        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        self.bin_embed = nn.Embedding(num_bins, embed_dim)
        self.cond_embed = nn.Embedding(num_conditions, embed_dim)

        # register gene and bin indices as buffer so they move with model.device
        self.register_buffer('bin_idx', torch.arange(num_bins, dtype=torch.long))
        self.register_buffer('gene_idx', torch.arange(num_genes, dtype=torch.long))

    def expr_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map expression values x (C, G) to expression embeddings (C, G, E)
        """
        bin_idx = self.bin_idx.to(x.device)       # (B,)
        embed_bank = self.bin_embed(bin_idx)      # (B, E)
        weights = self.quantizer(x)               # (C, G, B)

        # weighted sum over bins -> (C, G, E)
        out = torch.einsum('cgb,be->cge', weights, embed_bank)
        return out

    def forward(self, cond_idx: torch.Tensor, expr: torch.Tensor):
        """
        Args:
            cond_idx: LongTensor of shape (C,)
            expr: FloatTensor of shape (C, G)

        Returns:
            out: FloatTensor of shape (C, G+1, E) where first token is the condition embedding
        """
        device = expr.device

        # condition embedding -> (C, 1, E)
        cond_idx = cond_idx.to(device).long()
        cond_emb = self.cond_embed(cond_idx).unsqueeze(1)  # (C, 1, E)

        # gene embedding using range indices -> (C, G, E)
        gene_emb = self.gene_embed(self.gene_idx.to(device))    # (G, E)
        gene_emb = gene_emb.unsqueeze(0).expand(expr.size(0), -1, -1)  # (C, G, E)

        # expression embedding -> (C, G, E)
        expr_emb = self.expr_mapping(expr.to(device))    # (C, G, E)

        out = gene_emb + expr_emb
        out = torch.cat([cond_emb, out], dim=1)
        return out




if __name__ == '__main__':
    # small functional test
    num_genes = 5
    num_bins = 10
    embed_dim = 8
    num_conditions = 3
    batch_size = 2  # number of samples

    tokenizer = Tokenizer(
        num_genes=num_genes,
        num_bins=num_bins,
        embed_dim=embed_dim,
        num_conditions=num_conditions
    )

    cond_idx = torch.randint(low=0, high=num_conditions, size=(batch_size,))

    expr = torch.randn(batch_size, num_genes)
    expr[expr.abs() < 0.5] = 0.0

    out = tokenizer(cond_idx, expr)

    print("Condition indices:", cond_idx)
    print("Expression matrix:\n", expr)
    print("Output shape:", out.shape)
    print("Output tensor:\n", out)
