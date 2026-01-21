import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import TomoEmbedding
from .backbone import TomoEncoder
from ..load import load_tf_list


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
            logits: (C, L, m) unnormalized logits

        Returns:
            Tensor of same shape, with only top-k along last dim kept
            (soft in training, hard in eval with sum=1)
        """
        if logits.dim() < 1:
            raise ValueError("logits must be at least 1-D")

        device = logits.device
        _, _, m = logits.shape
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


class GeneRouter(nn.Module):
    """
    Gene router that maps gene tokens to assignment probabilities over 
    a set of latent factors, optionally applying differentiable Top-K.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_factors: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.1,
        topk: Optional[int] = None,   # if specified, apply Top-K selection
        tau: float = 1.0,   # temperature for GumbelTopK
        **kwargs
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
        x: torch.Tensor,
        gene_subset: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ):
        """
        Args:
            x: FloatTensor of shape (C, L, E)
            gene_subset: BoolTensor (C, L-1), True = selected token, False = unselected
            key_padding_mask: BoolTensor (C, L), True = valid token, False = masked
            temperature: Softmax temperature for logits

        Returns:
            probs: FloatTensor (C, S, m)
                Assignment probabilities over m factors for selected genes (S)
        """
        if x.dim() != 3:
            raise ValueError("x must be shape (C, L, E)")

        C, L, E = x.shape
        x = x[:, 1:, :]

        # -------------------------------
        # 1. gene_subset provided
        # -------------------------------
        if gene_subset is not None:
            if gene_subset.dtype != torch.bool or gene_subset.shape != (C, L-1):
                raise ValueError("gene_subset must be BoolTensor of shape (C, L-1)")

        # -------------------------------
        # 2. else use key_padding_mask
        # -------------------------------
        elif key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (C, L):
                raise ValueError("key_padding_mask must be BoolTensor (C, L)")

            gene_subset = ~key_padding_mask     # True = keep
            gene_subset = gene_subset[:, 1:]
        
        x = x[gene_subset].view(C, -1, E)   # (C, S, E)
        logits = self.mlp(x) / temperature  # (C, S, m)

        if self.gating is not None:
            probs = self.gating(logits)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs, gene_subset


class SFM(nn.Module):
    """
    Structure Foundation Model (SFM) for cell-specific GRN inference
    """
    def __init__(
        self,
        token_dict,
        embed_dim: int = 512,
        expr_num_bins: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        num_factors: int = 256,
        topk: int = 32,
        tf_list: Optional[list] = None,
        **kwargs
    ):
        super().__init__()
        assert (embed_dim % 2) == 0, "embed_dim must be even"

        self.token_dict = token_dict

        self.embedding = TomoEmbedding(
            token_dict, 
            D=embed_dim // 2,
            expr_num_bins=expr_num_bins,
            **kwargs
        )

        self.backbone = TomoEncoder(
            num_layers,
            embed_dim,
            num_heads,
            use_rotary=False,
            **kwargs
        )
    
        self.tfrouter = GeneRouter(
            embed_dim,
            num_factors,
            topk=topk
        )

        self.tgrouter = GeneRouter(
            embed_dim,
            num_factors,
            topk=topk
        )

        if tf_list is not None:
            self.tf_idx = self._tf2id(tf_list)
        else:
            self.tf_idx = None

        # apply initialization recursively to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _tf2id(self, tf_list):
        # Determine which dictionary to use
        use_gene_id = all(name.startswith("ENSG") for name in tf_list)

        symbol2id = dict(zip(self.token_dict["gene_symbol"], self.token_dict["token_index"]))
        id2id     = dict(zip(self.token_dict["gene_id"],     self.token_dict["token_index"]))

        lookup = id2id if use_gene_id else symbol2id

        # Collect token indices if present
        tf_idx = [lookup[name] for name in tf_list if name in lookup]

        if len(tf_idx) == 0:
            raise ValueError("None of the TFs are in token_dict. Please check tf_list!")

        return torch.tensor(tf_idx, dtype=torch.long)

    def update_tfs(self, tfs):
        tf_list = load_tf_list(tfs)
        self.tf_idx = self._tf2id(tf_list)

    def _query_gene_subset(self, tokens):
        gene_tokens = tokens["gene"]
        assert self.tf_idx is not None, "Please first specify tf_list when initialization!"
        self.tf_idx = self.tf_idx.to(gene_tokens.device)

        gene_subset = (gene_tokens.unsqueeze(-1) == self.tf_idx.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        return gene_subset

    def forward(self, tokens, return_factors=False, **kwargs):
        x, key_padding_mask = self.embedding(tokens)
        x = self.backbone(x, key_padding_mask, causal=False)

        if self.tf_idx is not None:
            gene_subset = self._query_gene_subset(tokens)
        else:
            gene_subset = None

        u, binary_tf = self.tfrouter(x, gene_subset, key_padding_mask, **kwargs)
        v, binary_tg = self.tgrouter(x, None, key_padding_mask, **kwargs)

        grn = torch.einsum('cfm,cgm->cfg', u, v)
        
        if return_factors:
            return grn, binary_tf, binary_tg, u, v
        else:
            return grn, binary_tf, binary_tg




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from ..tokenizer import TomeTokenizer

    adata = sc.read_h5ad("/data1021/xukaichen/data/CTA/pbmc.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")
    tf_dict = pd.read_csv("./resources/human_tfs.csv")
    tf_list = tf_dict["TF"].tolist()

    Ng = 2000
    Nc = 100
    tokenizer = TomeTokenizer(token_dict, simplify=True, max_length=Ng+1, n_top_genes=Ng)
    tokens = tokenizer(adata[:Nc, :].copy())

    model = SFM(token_dict, tf_list=tf_list)
    grn, binary_tf, binary_tg = model(tokens)
    print(binary_tf.shape)
