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
        topk: int = None,   # if specified, apply Top-K selection
        tau: float = 1.0,   # temperature for GumbelTopK
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


# class InferenceModel(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         num_factors: int = 128, 
#     ):
#         super().__init__()



if __name__ == "__main__":
    torch.manual_seed(42)

    # ---------- Hyperparameters ----------
    C = 2           # batch size
    G = 8           # number of genes
    E = 16          # embedding dim
    m = 5           # number of latent factors
    hidden_dim = 32
    topk = 3
    tau = 0.5

    # ---------- Create random tokens ----------
    # tokens: (C, L=G+1, E), first token = condition
    tokens = torch.randn(C, G + 1, E)

    # Random gene_idx selection (e.g., select all genes)
    gene_idx = torch.arange(G)

    # Random mask_expr (C, L), True = expressed, False = zero-expression
    mask_expr = torch.ones(C, G + 1, dtype=torch.bool)
    mask_expr[0, [2, 5]] = False  # example: mask out gene 2 and 5 in batch 0
    mask_expr[1, [3]] = False     # mask out gene 3 in batch 1

    # ---------- Instantiate Router ----------
    router = Router(embed_dim=E, num_factors=m, hidden_dim=hidden_dim, dropout=0.1, topk=topk, tau=tau)

    # ---------- TRAINING MODE (soft Gumbel-TopK) ----------
    router.train()
    probs_train = router(tokens, gene_idx=gene_idx, mask_expr=mask_expr)
    print("Training mode (soft Top-K) probabilities:")
    print(probs_train)
    print("Shape:", probs_train.shape)
    print("Sum over factors (should < 1 for masked genes):", probs_train.sum(dim=-1))

    # ---------- EVAL MODE (hard Top-K) ----------
    router.eval()
    probs_eval = router(tokens, gene_idx=gene_idx, mask_expr=mask_expr)
    print("\nEvaluation mode (hard Top-K) probabilities:")
    print(probs_eval)
    print("Shape:", probs_eval.shape)
    print("Sum over factors (masked genes should be 0):", probs_eval.sum(dim=-1))
