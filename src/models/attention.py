import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.bert_padding import unpad_input, pad_input

try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
except ImportError:
    cute_flash_attn_func = None


class _FlashAttentionBackend(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax_scale = self.head_dim ** -0.5

    @staticmethod
    def _normalize_attention_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
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
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlashMHAFA2(_FlashAttentionBackend):
    """FA2 backend: packed QKV dense kernel, packed QKV varlen kernel for padding."""

    def _run_dense(self, qkv: torch.Tensor, *, causal: bool) -> torch.Tensor:
        return flash_attn_qkvpacked_func(
            qkv.contiguous(),
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

    def _run_varlen(
        self,
        qkv_unpad: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        causal: bool,
    ) -> torch.Tensor:
        return flash_attn_varlen_qkvpacked_func(
            qkv_unpad.contiguous(),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, L = qkv.shape[:2]
        if key_padding_mask is None:
            return self._run_dense(qkv, causal=causal)

        attention_mask = ~key_padding_mask
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
            return qkv.new_zeros(B, L, self.num_heads, self.head_dim)

        out_unpad = self._run_varlen(
            qkv_unpad,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            causal=causal,
        )
        return pad_input(out_unpad, indices, B, L)


class FlashMHAFA4(_FlashAttentionBackend):
    """FA4 backend: official CuTe `flash_attn_func(q, k, v, ...)` API."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        fa4_min_cc: int = 90,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self._fa4_available = cute_flash_attn_func is not None
        self._fa4_min_cc = int(fa4_min_cc)
        self._validate_support()

    def _validate_support(self) -> None:
        if not self._fa4_available:
            raise RuntimeError("`attention_backend='fa4'` requires flash_attn.cute kernels.")
        if not torch.cuda.is_available():
            raise RuntimeError("`attention_backend='fa4'` requires CUDA.")
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception as exc:
            raise RuntimeError("Could not determine CUDA device capability for FA4.") from exc
        cc = major * 10 + minor
        if cc < self._fa4_min_cc:
            raise RuntimeError(
                f"`attention_backend='fa4'` requires CUDA capability >= sm{self._fa4_min_cc}, "
                f"got sm{cc}."
            )

    def _run_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        return self._normalize_attention_output(
            cute_flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        )

    @staticmethod
    def _right_padding_lengths(
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        valid_lengths = (~key_padding_mask).sum(dim=1)
        if valid_lengths.numel() == 0:
            return None

        positions = torch.arange(
            key_padding_mask.shape[1],
            device=key_padding_mask.device,
        )
        expected_mask = positions.unsqueeze(0) >= valid_lengths.unsqueeze(1)
        if not torch.equal(key_padding_mask, expected_mask):
            return None
        return valid_lengths

    def _run_right_padded(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        B, L = key_padding_mask.shape
        valid_lengths = self._right_padding_lengths(key_padding_mask)
        if valid_lengths is None:
            raise RuntimeError(
                "`attention_backend='fa4'` supports padding masks only when valid "
                "tokens form a left-aligned prefix in each sequence."
            )

        out = q.new_zeros(B, L, self.num_heads, self.head_dim)
        for length in torch.unique(valid_lengths, sorted=True):
            trim_len = int(length.item())
            if trim_len == 0:
                continue

            batch_indices = torch.nonzero(
                valid_lengths == length,
                as_tuple=False,
            ).flatten()
            group_q = q.index_select(0, batch_indices)[:, :trim_len]
            group_k = k.index_select(0, batch_indices)[:, :trim_len]
            group_v = v.index_select(0, batch_indices)[:, :trim_len]
            group_out = self._run_dense(group_q, group_k, group_v, causal=causal)
            out[batch_indices, :trim_len] = group_out

        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            return self._run_dense(q, k, v, causal=causal)
        return self._run_right_padded(q, k, v, key_padding_mask, causal=causal)


class FlashMHA(nn.Module):
    """
    FlashAttention-backed multi-head self-attention with optional rotary embedding.

    Shape convention:
    - `L`: full sequence length

    Supported backends:
    - `fa2`: FlashAttention 2 packed-QKV kernels
    - `fa4`: CuTe / FA4-style kernels, explicit opt-in

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
        attention_backend: str = "fa2",
        use_rotary: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
        fa4_min_cc: int = 90,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_backend = str(attention_backend).lower()
        if self.attention_backend not in {"fa2", "fa4"}:
            raise ValueError(
                f"`attention_backend` must be one of ['fa2', 'fa4'], got {attention_backend!r}."
            )
        self.use_rotary = use_rotary
        self.attention = self._build_backend(fa4_min_cc=fa4_min_cc)

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

    def _build_backend(self, *, fa4_min_cc: int) -> _FlashAttentionBackend:
        if self.attention_backend == "fa4":
            return FlashMHAFA4(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                fa4_min_cc=fa4_min_cc,
            )
        return FlashMHAFA2(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

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
        B, L = x.shape[:2]
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv)
        return qkv.contiguous()

    def _run_attention(
        self,
        qkv: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if self.attention_backend == "fa4":
            q, k, v = qkv.unbind(dim=2)
            return self.attention(
                q,
                k,
                v,
                key_padding_mask=key_padding_mask,
                causal=causal,
            )
        return self.attention(
            qkv,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

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

        key_padding_mask = self._normalize_key_padding_mask(
            key_padding_mask,
            batch_size=B,
            seq_len=L,
        )

        qkv = self._project_qkv(x)
        out = self._run_attention(
            qkv,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

        out = out.reshape(B, L, self.embed_dim)
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
        attention_backend="fa2",
        use_rotary=True
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
