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
            expr_value: FloatTensor of shape (C, G)

        Returns:
            probs: (C, G, B) probabilities, where B = num_bins
            mask_expr: (C, G) float mask where 1 = nonzero expr, 0 = zero expr

        Notes:
            - For positions where expr_value == 0, output is [1, 0, 0, ..., 0]
            - For positions where expr_value != 0, output is [0, softmax(logits)] across bins 1, ..., B-1
        """
        if expr_value.dim() != 2:
            raise ValueError("expr_value must be (C, G)")

        C, G = expr_value.shape
        device = expr_value.device

        # mask: 1 for nonzero expr, 0 for zero expr
        mask_expr = (expr_value != 0).float()  # (C, G)

        # initialize all probs as [1, 0, ..., 0]
        probs = torch.zeros((C, G, self.num_bins), device=device, dtype=torch.float32)
        probs[..., 0] = 1.0

        # compute only nonzero positions
        if mask_expr.any():
            x_nonzero = expr_value[mask_expr.bool()].unsqueeze(-1).float()  # (CG, 1)
            logits = self.mlp(x_nonzero)                                    # (CG, B-1)
            probs_nonzero = F.softmax(logits, dim=-1)                       # (CG, B-1)

            # assign these to probs[..., 1:]
            probs[..., 1:][mask_expr.bool()] = probs_nonzero
            probs[..., 0][mask_expr.bool()] = 0.0  # since we now replaced zeros with nonzeros

        return probs, mask_expr  # (C, G, B), (C, G)


class Tokenizer(nn.Module):
    """
    Unified tokenizer for single-cell AnnData samples.
    """
    def __init__(self, 
        num_genes: int, 
        num_bins: int = 20, 
        embed_dim: int = 256, 
        num_conditions: Optional[int] = None
    ):
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

    def _expr_mapping(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map expression values (C, G) to expression embeddings (C, G, E)
        """
        bin_idx = self.bin_idx.to(x.device)       # (B,)
        embed_bank = self.bin_embed(bin_idx)      # (B, E)
        weights, mask_expr = self.quantizer(x)    # (C, G, B), (C, G)

        # weighted sum over bins -> (C, G, E)
        out = torch.einsum('cgb,be->cge', weights, embed_bank)
        return out, mask_expr

    def forward(self, cond_idx: torch.Tensor, expr: torch.Tensor):
        """
        Args:
            cond_idx: LongTensor of shape (C,)
            expr: FloatTensor of shape (C, G)

        Returns:
            out: FloatTensor of shape (C, G+1, E)
            mask_expr: FloatTensor of shape (C, G+1)
                - mask_expr[:, 0] = 0 (condition token)
                - mask_expr[:, 1:] = 1 if expr != 0 else 0
        """
        device = expr.device
        C, _ = expr.shape

        # condition embedding -> (C, 1, E)
        cond_idx = cond_idx.to(device).long()
        cond_emb = self.cond_embed(cond_idx).unsqueeze(1)  # (C,1,E)

        # gene embedding -> (C, G, E)
        gene_emb = self.gene_embed(self.gene_idx.to(device))  # (G,E)
        gene_emb = gene_emb.unsqueeze(0).expand(C, -1, -1)    # (C,G,E)

        # expression embedding & mask -> (C, G, E), (C, G)
        expr_emb, mask_expr = self._expr_mapping(expr.to(device))

        # combine embeddings
        out = gene_emb + expr_emb  # (C, G, E)
        out = torch.cat([cond_emb, out], dim=1)  # (C, G+1, E)

        # prepend 0-column to mask_expr -> (C, G+1)
        zero_col = torch.zeros((C, 1), device=device, dtype=mask_expr.dtype)
        mask_expr = torch.cat([zero_col, mask_expr], dim=1)

        return out, mask_expr




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

    out, mask_expr = tokenizer(cond_idx, expr)

    print("Condition indices:", cond_idx)
    print("Expression tensor:\n", expr)
    print("Output shape:", out.shape)
    print("Output tensor:\n", out)
    print("Mask shape:", mask_expr.shape)
    print("Mask tensor:\n", mask_expr)
