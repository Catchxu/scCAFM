from __future__ import annotations

import torch
import torch.nn as nn


class FlashAttentionBackend(nn.Module):
    """Shared interface for FlashAttention backend adapters."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax_scale = self.head_dim ** -0.5

    @staticmethod
    def normalize_attention_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(
            "Attention kernel returned an unsupported output type: "
            f"{type(output).__name__}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


def build_flash_attention_backend(
    attention_backend: str,
    *,
    embed_dim: int,
    num_heads: int,
) -> FlashAttentionBackend:
    """
    Build a FlashAttention backend adapter with a unified QKV-packed API.

    All backend adapters accept:
    - `qkv`: (B, L, 3, H, D)
    - `key_padding_mask`: optional (B, L), True where padded
    - `causal`: whether to use causal masking
    """

    normalized = str(attention_backend).lower()
    if normalized == "fa2":
        from .FA2 import FlashMHAFA2

        return FlashMHAFA2(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    if normalized == "fa4":
        from .FA4 import FlashMHAFA4

        return FlashMHAFA4(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    raise ValueError(
        f"`attention_backend` must be one of ['fa2', 'fa4'], got {attention_backend!r}."
    )


class FlashMHA(nn.Module):
    """
    FlashAttention-backed multi-head self-attention with optional rotary embedding.

    Shape convention:
    - `L`: full sequence length

    Supported backends:
    - `fa2`: FlashAttention 2 packed-QKV kernels
    - `fa4`: CuTe / FA4-style kernels

    Args:
        embed_dim: model dimension
        num_heads: number of attention heads
        attention_backend: one of `fa2` or `fa4`
        use_rotary: whether to apply RoPE to Q/K
        qkv_bias: bias for fused qkv projection
        out_bias: bias for output projection
        rotary_base: RoPE base
        rotary_interleaved: False = GPT-NeoX style, True = GPT-J style
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        attention_backend: str = "fa4",
        use_rotary: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_backend = str(attention_backend).lower()
        self.attention = build_flash_attention_backend(
            self.attention_backend,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )
        self.use_rotary = use_rotary

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_bias)

        if self.use_rotary:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"Rotary embedding requires even head_dim, got {self.head_dim}."
                )
            try:
                from flash_attn.layers.rotary import RotaryEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "`use_rotary=True` requires flash_attn.layers.rotary.RotaryEmbedding."
                ) from exc

            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                base=rotary_base,
                interleaved=rotary_interleaved,
            )
        else:
            self.rotary_emb = None

    def _normalize_key_padding_mask(
        self,
        key_padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        if key_padding_mask is None:
            return None
        if key_padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"key_padding_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(key_padding_mask.shape)}"
            )
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)
        if not key_padding_mask.any():
            return None
        return key_padding_mask

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        qkv = self.qkv_proj(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv)
        return qkv.contiguous()

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

        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {embed_dim}."
            )

        key_padding_mask = self._normalize_key_padding_mask(
            key_padding_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        qkv = self._project_qkv(x)
        out = self.attention(
            qkv,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        if key_padding_mask is not None:
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
        attention_backend="fa4",
        use_rotary=False,
    ).to(device)
    model = model.half()
    model.eval()

    print("backend_class:", type(model.attention).__name__)
    print("attention_backend:", model.attention_backend)

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
