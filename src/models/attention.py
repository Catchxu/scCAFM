import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.bert_padding import unpad_input, pad_input


class FlashMHA(nn.Module):
    """
    FlashAttention-backed multi-head self-attention with optional rotary embedding.

    Upgrades:
    1. Uses flash_attn_qkvpacked_func in the no-padding path
    2. Uses flash_attn.bert_padding unpad/pad utilities in the masked path

    Args:
        embed_dim: model dimension
        num_heads: number of attention heads
        use_rotary: whether to apply RoPE to Q/K
        attn_dropout: attention dropout probability
        qkv_bias: bias for fused qkv projection
        out_bias: bias for output projection
        rotary_base: RoPE base
        rotary_interleaved: False = GPT-NeoX style, True = GPT-J style
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        use_rotary: bool = False,
        attn_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rotary = use_rotary
        self.attn_dropout = attn_dropout
        self.softmax_scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_bias)

        if self.use_rotary:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"Rotary embedding requires even head_dim, got {self.head_dim}."
                )

            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                base=rotary_base,
                interleaved=rotary_interleaved,
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            key_padding_mask: (B, L) bool, where True means padded position
            causal: whether to use causal masking

        Returns:
            out: (B, L, D)
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, L, D), got {tuple(x.shape)}")

        B, L, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {D}."
            )

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must have shape {(B, L)}, got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)

        dropout_p = self.attn_dropout if self.training else 0.0

        # (B, L, 3D) -> (B, L, 3, H, Dh)
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)

        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv)

        # ------------------------------------------------------------
        # Path 1: no padding -> packed QKV FlashAttention
        # ------------------------------------------------------------
        if key_padding_mask is None:
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
            out = out.reshape(B, L, self.embed_dim)
            out = self.out_proj(out)
            return out

        # ------------------------------------------------------------
        # Path 2: padding present -> unpad -> varlen packed QKV -> repad
        # key_padding_mask: True = padded
        # attention mask expected by unpad_input: True = valid
        # ------------------------------------------------------------
        attention_mask = ~key_padding_mask  # (B, L), True = valid

        # qkv_unpad: (nnz, 3, H, Dh)
        # indices: positions of valid tokens in flattened (B, L)
        # cu_seqlens: cumulative lengths, shape (B + 1,)
        # max_seqlen: int
        unpad_outputs = unpad_input(qkv, attention_mask)
        if len(unpad_outputs) == 4:
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_outputs
        elif len(unpad_outputs) == 5:
            qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_outputs
        else:
            raise ValueError(
                f"Unexpected number of outputs from unpad_input: {len(unpad_outputs)}"
            )

        if qkv_unpad.shape[0] == 0:
            out = x.new_zeros(B, L, self.embed_dim)
            out = self.out_proj(out)
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return out

        out_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

        # out_unpad: (nnz, H, Dh) -> repad to (B, L, H, Dh)
        out = pad_input(out_unpad, indices, B, L)

        out = out.reshape(B, L, self.embed_dim)
        out = self.out_proj(out)
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out




if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the attention smoke test.")

    device = torch.device("cuda")
    torch.manual_seed(42)

    model = FlashMHA(
        embed_dim=64,
        num_heads=8,
        use_rotary=True,
        attn_dropout=0.0,
    ).to(device)
    model = model.half()
    model.eval()

    x = torch.randn(2, 16, 64, device=device, dtype=torch.float16)
    key_padding_mask = torch.zeros(2, 16, device=device, dtype=torch.bool)
    key_padding_mask[1, 12:] = True

    with torch.no_grad():
        out_no_padding = model(x, key_padding_mask=None, causal=False)
        out_with_padding = model(x, key_padding_mask=key_padding_mask, causal=False)

    print("device:", out_no_padding.device)
    print("dtype:", out_no_padding.dtype)
    print("no_padding_shape:", tuple(out_no_padding.shape))
    print("with_padding_shape:", tuple(out_with_padding.shape))
    print("masked_tail_norm:", out_with_padding[1, 12:].abs().sum().item())
