import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .transformer import Transformer


class GumbelTopK(nn.Module):
    """
    Differentiable Top-K operator using Gumbel-Softmax on logits.

    Training mode:
        - Adds Gumbel noise to logits, softmax with temperature tau.
        - Approximates Top-K via a smooth mask (differentiable).

    Evaluation mode:
        - Hard Top-K selection on logits, normalized to sum=1.
    """
    def __init__(self, k: int = 30, tau: float = 1.0, softness: float = 1e-2):
        """
        Args:
            k: number of top elements to select
            tau: temperature for Gumbel noise / softmax (smaller=tighter)
            softness: smoothness factor for soft mask in training
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.softness = softness

    def _safe_topk(self, x: torch.Tensor, k: int, dim=-1):
        k = max(0, min(k, x.size(dim)))
        if k == 0:
            # return empty tensors of appropriate shape
            shape_values = list(x.shape)
            shape_values[dim] = 0
            values = x.new_empty(tuple(shape_values))
            shape_idx = list(x.shape)
            shape_idx[dim] = 0
            indices = x.new_empty(tuple(shape_idx), dtype=torch.long)
            return values, indices
        return torch.topk(x, k, dim=dim)

    def sample_gumbel(self, shape, device, dtype=torch.float32, eps=1e-20):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (C, S, m) unnormalized logits

        Returns:
            Tensor of same shape, with only top-k along last dim kept
            (soft in training, hard in eval with sum=1)
        """
        if logits.dim() < 1:
            raise ValueError("logits must be at least 1-D")

        device = logits.device
        C, S, m = logits.shape
        k_eff = min(self.k, m)
        if k_eff <= 0:
            return torch.zeros_like(logits)

        if self.training:
            # ---------- TRAINING: soft Top-K ----------
            gumbel = self.sample_gumbel(logits.shape, device=device, dtype=logits.dtype)
            perturbed = (logits + gumbel) / max(self.tau, 1e-12)
            probs = F.softmax(perturbed, dim=-1)

            # Compute smooth top-k mask
            topk_vals, _ = self._safe_topk(probs, k_eff, dim=-1)
            threshold = topk_vals[..., -1].unsqueeze(-1)
            smooth_mask = torch.sigmoid((probs - threshold) / max(self.softness, 1e-12))
            return probs * smooth_mask

        else:
            # ---------- EVAL: hard Top-K with normalization ----------
            topk_vals, topk_idx = self._safe_topk(logits, k_eff, dim=-1)
            mask = torch.zeros_like(logits)
            if topk_idx.numel() > 0:
                mask.scatter_(-1, topk_idx, 1.0)

            # Keep only top-k logits
            topk_logits = logits * mask

            # Normalize top-k logits to sum=1 along last dim
            sum_topk = topk_logits.sum(dim=-1, keepdim=True)
            sum_topk = sum_topk + 1e-12  # avoid divide by zero
            probs = topk_logits / sum_topk

            return probs


class Router(nn.Module):
    """
    Generic router that maps gene tokens to assignment probabilities over 
    a set of latent factors, optionally applying differentiable Top-K.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_factors: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.1,
        topk: Optional[int] = None,   # if specified, apply Top-K selection
        tau: float = 1.0   # temperature for GumbelTopK
    ):
        """
        Args:
            embed_dim: Dimension of token embeddings (E)
            num_factors: Number of latent factors (m)
            hidden_dim: Hidden dimension of the MLP
            dropout: Dropout probability
            topk: Optional, number of top factors to select per gene
            tau: Temperature for Gumbel-Softmax
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_factors = num_factors
        self.hidden_dim = hidden_dim

        # MLP mapping gene embeddings -> logits over factors
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors)
        )

        # Optional gating via differentiable top-k
        self.topk = topk
        if topk is not None:
            self.gating = GumbelTopK(k=topk, tau=tau)
        else:
            self.gating = None
    
    def forward(
        self,
        tokens: torch.Tensor,
        gene_idx: Optional[torch.Tensor] = None,
        mask_expr: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ):
        """
        Args:
            tokens: FloatTensor of shape (C, L=G+1, E)
                - C: batch size
                - L: total token count (1 condition + G genes)
                - E: embedding dimension
            gene_idx: LongTensor (S,) specifying which genes to route
            mask_expr: BoolTensor (C, L), True = valid token, False = masked
            temperature: Softmax temperature for logits

        Returns:
            probs: FloatTensor (C, S, m)
                Assignment probabilities over m factors for selected genes
        """
        if tokens.dim() != 3:
            raise ValueError("tokens must be shape (C, L, E)")
        C, L, E = tokens.shape
        G = L - 1
        device = tokens.device

        # Select gene tokens only
        gene_tokens = tokens[:, 1:, :]  # (C, G, E)

        # Handle mask_expr
        if mask_expr is not None:
            if mask_expr.shape != (C, L):
                raise ValueError(f"mask_expr must be shape (C, L={L})")
            mask_expr = mask_expr[:, 1:]  # remove condition token
        else:
            mask_expr = torch.ones((C, G), dtype=torch.bool, device=device)

        # Select specific genes if gene_idx is provided
        if gene_idx is not None:
            if gene_idx.dim() != 1 or gene_idx.max() >= G:
                raise ValueError(f"gene_idx must be 1-D tensor in [0, {G-1}]")
            sel_idx = gene_idx.to(device)
            gene_tokens = gene_tokens[:, sel_idx, :]  # (C, S, E)
            mask_expr = mask_expr[:, sel_idx]         # (C, S)

        # Compute logits over factors
        logits = self.mlp(gene_tokens)              # (C, S, m)
        logits = logits / temperature

        # Apply differentiable or hard Top-K if gating is enabled
        if self.gating is not None:
            probs = self.gating(logits)             # (C, S, m)
        else:
            probs = F.softmax(logits, dim=-1)       # (C, S, m)

        # Zero out invalid (masked) tokens
        probs = probs * mask_expr.unsqueeze(-1)      # (C, S, m)

        return probs


class InferenceModel(nn.Module):
    """
    Inference model for single-cell gene allocation.

    Architecture:
        - Backbone: Transformer over token embeddings
        - Two Routers: TF-router and TG-router mapping gene tokens to factor assignments
        - Final allocation: outer product of TF and TG allocations
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        num_factors: int = 256,
        topk: Optional[int] = 32,
        tau: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.backbone = Transformer(embed_dim, num_layers, num_heads, **kwargs)
        self.tfrouter = Router(embed_dim, num_factors, topk=topk, tau=tau)
        self.tgrouter = Router(embed_dim, num_factors, topk=topk, tau=tau)
    
    def forward(
        self,
        x: torch.Tensor,
        tf_idx: Optional[torch.Tensor] = None,
        tg_idx: Optional[torch.Tensor] = None,
        mask_expr: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Args:
            x: FloatTensor (C, L, E), token embeddings including condition + genes
            tf_idx: LongTensor (S_TF,), indices of TF genes to route
            tg_idx: LongTensor (S_TG,), indices of TG genes to route
            mask_expr: BoolTensor (C, L), True=valid token, False=masked
            **kwargs: extra arguments (e.g., temperature) for Routers

        Returns:
            allocation: FloatTensor (C, S_TF, S_TG), outer product of TF/TG allocations
        """
        # Pass through transformer backbone
        tokens = self.backbone(x)  # (C, L, E)

        # TF allocation
        tf_allocation = self.tfrouter(tokens, gene_idx=tf_idx, mask_expr=mask_expr, **kwargs)  # (C, S_TF, m)

        # TG allocation
        tg_allocation = self.tgrouter(tokens, gene_idx=tg_idx, mask_expr=mask_expr, **kwargs)  # (C, S_TG, m)

        # Final allocation: outer product over latent factors
        # shape: (C, S_TF, S_TG)
        allocation = torch.einsum('cfm,cgm->cfg', tf_allocation, tg_allocation)

        return allocation




if __name__ == "__main__":
    # -----------------------------
    # Configuration
    # -----------------------------
    batch_size = 4
    num_genes = 10           # G
    embed_dim = 16           # E
    num_factors = 8          # m
    num_layers = 2
    num_heads = 4
    topk = 3
    tau = 1.0

    # Randomly select some TF and TG gene indices
    tf_idx = torch.tensor([0, 2, 5], dtype=torch.long)
    tg_idx = torch.tensor([1, 3, 5, 7], dtype=torch.long)

    # -----------------------------
    # Create random embeddings and mask
    # -----------------------------
    # Include condition token, so L = 1 + G
    L = 1 + num_genes
    tokens = torch.randn(batch_size, L, embed_dim)

    # Random mask: True = valid gene, False = zero-expression gene
    mask_expr = torch.randint(0, 2, (batch_size, L), dtype=torch.bool)
    # Ensure first token (condition) is always valid
    mask_expr[:, 0] = True

    # -----------------------------
    # Instantiate model
    # -----------------------------
    model = InferenceModel(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_factors=num_factors,
        topk=topk,
        tau=tau
    )

    # -----------------------------
    # Forward pass
    # -----------------------------
    allocation = model(tokens, tf_idx=tf_idx, tg_idx=tg_idx, mask_expr=mask_expr)

    # -----------------------------
    # Print shapes and stats
    # -----------------------------
    print("Tokens shape:", tokens.shape)
    print("Mask shape:", mask_expr.shape)
    print("TF indices:", tf_idx)
    print("TG indices:", tg_idx)
    print("Allocation shape:", allocation.shape)  # Expected: (batch_size, len(tf_idx), len(tg_idx))