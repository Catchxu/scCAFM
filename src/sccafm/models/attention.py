import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func


class MaskedMHA(nn.Module):
    """
    FlashAttention-backed multi-head attention (modern implementation).

    Improvements implemented:
    1. Fused QKV projection
    2. RoPE applied without transposes
    3. Precomputed RoPE cache
    4. Removed SDPA fallback
    5. Safe reshape instead of view
    6. Attention dropout
    8. Removed .any() GPU sync
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        use_rotary=False,
        max_seq_len=2048,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        # ---- fused qkv projection ----
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_rotary = use_rotary
        self.attn_dropout = attn_dropout
        self.softmax_scale = self.head_dim ** -0.5

        if use_rotary:
            self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, max_seq_len):
        half_dim = self.head_dim // 2

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim).float() / half_dim)
        )

        t = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def apply_rotary(self, q, k):
        """
        q,k: (B, L, H, Dh)
        """

        L = q.shape[1]

        cos = self.rope_cos[:L].unsqueeze(0).unsqueeze(2)
        sin = self.rope_sin[:L].unsqueeze(0).unsqueeze(2)

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)

        return q, k

    def forward(self, x, key_padding_mask=None, causal=False):
        """
        x: (B, L, D)
        key_padding_mask: (B, L) bool
        """

        B, L, _ = x.shape

        # ---- fused projection ----
        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ---- reshape to flash layout ----
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)

        # ---- rotary embedding ----
        if self.use_rotary:
            q, k = self.apply_rotary(q, k)

        # ---- attention ----
        if key_padding_mask is None:

            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        else:
            # assume contiguous valid tokens
            valid_mask = ~key_padding_mask

            seqlens = valid_mask.sum(dim=1, dtype=torch.int32)

            total = int(seqlens.sum().item())

            if total == 0:
                out = x.new_zeros(B, L, self.embed_dim)

            else:
                q_unpad = q[valid_mask]
                k_unpad = k[valid_mask]
                v_unpad = v[valid_mask]

                cu_seqlens = torch.zeros(
                    B + 1, device=x.device, dtype=torch.int32
                )
                cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

                max_seqlen = int(seqlens.max().item())

                out_unpad = flash_attn_varlen_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )

                out_padded = q.new_zeros(B, L, self.num_heads, self.head_dim)

                out_padded[valid_mask] = out_unpad

                out = out_padded

        # ---- merge heads ----
        out = out.reshape(B, L, self.embed_dim)

        out = self.out_proj(out)

        return out




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from ..tokenizer import TomeTokenizer
    from .embedding import TomoEmbedding

    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    tokenizer = TomeTokenizer(token_dict)
    tokens = tokenizer(adata)

    D = 256
    embedding_layer = TomoEmbedding(token_dict, D=D)
    x, key_padding_mask = embedding_layer(tokens)

    model = MaskedMHA(embed_dim=2*D)
    out = model.forward(x, key_padding_mask)

    print(out.shape)
