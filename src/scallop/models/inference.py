import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Router(nn.Module):
    """
    Generic router that maps gene tokens to assignment probabilities over 
    a set of latent factors.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_factors: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Dimension of token embeddings (E)
            n_factors: Number of latent factors (m)
            hidden_dim: Hidden dimension of the MLP
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_factors = num_factors
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors)
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        gene_idx: Optional[torch.Tensor] = None,
        mask_expr: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            tokens: FloatTensor of shape (C, L=G+1, E)
                - C: number of conditions or batch size
                - L: total token count (1 condition + G gene tokens)
                - E: embedding dimension
            gene_idx: LongTensor of shape (S,), indices in [0, G-1]
                specifying which genes to route
            mask_expr: BoolTensor of shape (C, L), where
                True = valid token, False = masked (e.g. zero expression)
            temperature: Softmax temperature for controlling distribution sharpness

        Returns:
            probs: FloatTensor of shape (C, S, m)
                - Assignment probabilities over m factors for selected genes
        """
        if tokens.dim() != 3:
            raise ValueError("tokens must be shape (C, L, E)")

        C, L, _ = tokens.shape
        G = L - 1
        device = tokens.device

        # Select gene tokens only (exclude condition token)
        gene_tokens = tokens[:, 1:, :]  # (C, G, E)

        # Process expression mask if provided
        if mask_expr is not None:
            if mask_expr.shape != (C, L):
                raise ValueError(f"mask_expr must be of shape (C, L={L})")
            mask_expr = mask_expr[:, 1:]  # remove condition column
        else:
            mask_expr = torch.ones((C, G), dtype=torch.bool, device=device)

        # Select specific genes if indices are provided
        if gene_idx is not None:
            if gene_idx.dim() != 1 or gene_idx.max() >= G:
                raise ValueError(f"gene_idx must be 1-D tensor and contain values in [0, {G-1}]")
            sel_idx = gene_idx.to(device)
            gene_tokens = gene_tokens[:, sel_idx, :]  # (C, S, E)
            mask_expr = mask_expr[:, sel_idx]         # (C, S)

        # Compute logits and apply temperature-scaled softmax
        logits = self.mlp(gene_tokens)                # (C, S, m)
        probs = F.softmax(logits / temperature, dim=-1)

        # Zero out invalid (masked) tokens
        probs = probs * mask_expr.unsqueeze(-1)       # (C, S, m)

        return probs




import torch

if __name__ == "__main__":
    # -------------------------------
    # Test configuration
    # -------------------------------
    C = 2            # batch size (conditions)
    G = 6            # number of genes
    E = 8            # embedding dimension
    m = 4            # number of latent factors

    # Create fake token embeddings: (C, L=G+1, E)
    tokens = torch.randn(C, G + 1, E)

    # Define some example gene indices to select
    gene_idx = torch.tensor([0, 2, 4])  # select 3 genes (subset of G)

    # Create a random binary expression mask (True = valid)
    mask_expr = torch.randint(0, 2, (C, G + 1), dtype=torch.bool)
    print("mask_expr:\n", mask_expr)

    # Instantiate Router
    router = Router(embed_dim=E, num_factors=m, hidden_dim=16, dropout=0.1)

    # -------------------------------
    # Case 1: full routing (no gene_idx, no mask)
    # -------------------------------
    print("\n[Case 1] Full routing without mask:")
    probs_full = router(tokens)
    print("Output shape:", probs_full.shape)  # expected (C, G, m)
    print("Sum over factors (first sample):\n", probs_full[0].sum(dim=-1))

    # -------------------------------
    # Case 2: masked routing
    # -------------------------------
    print("\n[Case 2] Routing with expression mask:")
    probs_masked = router(tokens, mask_expr=mask_expr)
    print("Output shape:", probs_masked.shape)
    print("Masked positions (should be all zeros):\n", probs_masked[~mask_expr[:, 1:]])

    # -------------------------------
    # Case 3: routing with gene selection
    # -------------------------------
    print("\n[Case 3] Routing with selected genes (gene_idx):")
    probs_sel = router(tokens, gene_idx=gene_idx)
    print("Output shape:", probs_sel.shape)  # expected (C, len(gene_idx), m)
    print("Sum over factors (first sample):\n", probs_sel[0].sum(dim=-1))

    # -------------------------------
    # Case 4: combined (mask + selection)
    # -------------------------------
    print("\n[Case 4] Routing with both mask and gene selection:")
    probs_combined = router(tokens, gene_idx=gene_idx, mask_expr=mask_expr)
    print("Output shape:", probs_combined.shape)
    print("Masked-out values check:", (probs_combined[~mask_expr[:, gene_idx + 1]] == 0).all().item())
