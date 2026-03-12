import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from flash_attn import flash_attn_func, flash_attn_varlen_func


class MaskedMHA(nn.Module):
    """
    Multi-head attention backed by FlashAttention.

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

        # ---- Reshape ----
        q_flash = q.view(C, L, self.num_heads, self.head_dim)  # (C, L, H, Dh)
        k_flash = k.view(C, L, self.num_heads, self.head_dim)
        v_flash = v.view(C, L, self.num_heads, self.head_dim)

        # ---- Apply rotary positional embedding if enabled ----
        if self.use_rotary:
            q_rot = q_flash.transpose(1, 2)  # (C, H, L, Dh)
            k_rot = k_flash.transpose(1, 2)
            q_rot, k_rot = self.apply_rotary_pos_emb(q_rot, k_rot)
            q_flash = q_rot.transpose(1, 2).contiguous()
            k_flash = k_rot.transpose(1, 2).contiguous()

        # ---- Attention ----
        if key_padding_mask is None or not key_padding_mask.any():
            out = flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
            )  # (C, L, H, Dh)
            out = out.reshape(C, L, self.embed_dim)
        else:
            # flash_attn_varlen_func assumes each sequence has contiguous valid tokens.
            # For non-contiguous masks, fallback to SDPA to preserve semantics.
            has_true_to_false = (~key_padding_mask[:, 1:] & key_padding_mask[:, :-1]).any()
            if has_true_to_false:
                q = q_flash.transpose(1, 2)  # (C, H, L, Dh)
                k = k_flash.transpose(1, 2)
                v = v_flash.transpose(1, 2)
                kp = key_padding_mask.unsqueeze(1)
                kp = kp.expand(-1, self.num_heads, -1)
                kp = kp.unsqueeze(2).expand(-1, -1, L, -1)
                out = scaled_dot_product_attention(q, k, v, attn_mask=kp, is_causal=causal)
                out = out.transpose(1, 2).reshape(C, L, self.embed_dim)
            else:
                valid_mask = ~key_padding_mask
                seqlens = valid_mask.sum(dim=1, dtype=torch.int32)  # (C,)
                total_tokens = int(seqlens.sum().item())
                if total_tokens == 0:
                    out = q_flash.new_zeros(C, L, self.embed_dim)
                else:
                    q_unpad = q_flash[valid_mask]  # (total, H, Dh)
                    k_unpad = k_flash[valid_mask]
                    v_unpad = v_flash[valid_mask]

                    cu_seqlens = torch.zeros(C + 1, device=x.device, dtype=torch.int32)
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
                        dropout_p=0.0,
                        softmax_scale=None,
                        causal=causal,
                        window_size=(-1, -1),
                        softcap=0.0,
                        alibi_slopes=None,
                        deterministic=False,
                        return_attn_probs=False,
                        block_table=None,
                    )  # (total, H, Dh)

                    out_padded = out_unpad.new_zeros(C, L, self.num_heads, self.head_dim)
                    out_padded[valid_mask] = out_unpad
                    out = out_padded.reshape(C, L, self.embed_dim)

        # ---- Merge heads ----
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
