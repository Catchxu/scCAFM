import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


class FlashMHA(nn.Module):
    """
    Multi-head attention with Flash Attention.

    Args:
        embed_dim: total embedding dimension (2D)
        num_heads: number of attention heads
        use_rotary: whether to apply rotary positional encoding on query/key
    """
    def __init__(self, embed_dim, num_heads=8, use_rotary=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_rotary = use_rotary
        if use_rotary:
            # Precompute rotary frequencies for max L=2048, can be extended
            self.max_seq_len = 2048
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_rotary_pos_emb(self, q, k):
        """
        Apply rotary positional embedding to query and key.
        q, k: (C, num_heads, L, head_dim)
        Returns rotated q, k with same shape.
        """
        # ensure head_dim is even
        assert (self.head_dim % 2) == 0, "head_dim must be even for rotary"
        half_dim = self.head_dim // 2

        L = q.shape[2]
        device = q.device
        # freqs: (L, half_dim)
        freqs = torch.einsum("i,j->ij", torch.arange(L, device=device).float(), self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]   # (1,1,L, half_dim)
        sin = freqs.sin()[None, None, :, :]   # (1,1,L, half_dim)

        # split even / odd -> shape (..., half_dim)
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        # apply rotation on halves
        q_rot_even = q1 * cos - q2 * sin
        q_rot_odd  = q1 * sin + q2 * cos
        k_rot_even = k1 * cos - k2 * sin
        k_rot_odd  = k1 * sin + k2 * cos

        # interleave even/odd back to last dim
        q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
        k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)

        return q_rot, k_rot

    def forward(self, x: torch.Tensor, key_padding_mask=None, causal=False):
        """
        x: (C, L, 2D)
        key_padding_mask: (C, L) bool tensor, True for positions to mask
        causal: if True, apply causal mask
        """
        C, L, _ = x.shape

        # ---- Linear projections ----
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- Reshape to (C, num_heads, L, head_dim) ----
        q = q.view(C, L, self.num_heads, self.head_dim).transpose(1, 2)  # (C, num_heads, L, head_dim)
        k = k.view(C, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(C, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Apply rotary positional embedding if enabled ----
        if self.use_rotary:
            q, k = self.apply_rotary_pos_emb(q, k)

        # ---- Prepare mask ----
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (C, L) -> (C, num_heads, L, L)
            kp = key_padding_mask.unsqueeze(1)
            kp = kp.expand(-1, self.num_heads, -1)
            kp = kp.unsqueeze(2).expand(-1, -1, L, -1)
            attn_mask = kp  # bool mask

        # ---- Flash Attention ----
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
            # out: (C, num_heads, L, head_dim)

        # ---- Merge heads ----
        out = out.transpose(1, 2).reshape(C, L, self.embed_dim)  # (C, L, embed_dim)
        out = self.out_proj(out)

        return out




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from ..tokenizer import TomeTokenizer
    from .embedding import TomoEmbedding

    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    tokenizer = TomeTokenizer(token_dict, simplify=True)
    tokens = tokenizer(adata)

    D = 256
    embedding_layer = TomoEmbedding(token_dict, D=D)
    x, key_padding_mask = embedding_layer(tokens)

    model = FlashMHA(embed_dim=2*D)
    out = model.forward(x, key_padding_mask)

    print(out.shape)