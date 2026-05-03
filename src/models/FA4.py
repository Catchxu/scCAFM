from __future__ import annotations

import torch

from .attention import FlashAttentionBackend

try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
except ImportError as exc:
    cute_flash_attn_func = None
    _FA4_IMPORT_ERROR = exc
else:
    _FA4_IMPORT_ERROR = None


FA4_MIN_CUDA_CAPABILITY = 90


class FlashMHAFA4(FlashAttentionBackend):
    """FA4 backend: official CuTe `flash_attn_func(q, k, v, ...)` API."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self._validate_support()

    def _validate_support(self) -> None:
        if cute_flash_attn_func is None:
            raise RuntimeError(
                "`attention_backend='fa4'` requires flash_attn.cute kernels."
            ) from _FA4_IMPORT_ERROR
        if not torch.cuda.is_available():
            raise RuntimeError("`attention_backend='fa4'` requires CUDA.")
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception as exc:
            raise RuntimeError("Could not determine CUDA device capability for FA4.") from exc
        cc = major * 10 + minor
        if cc < FA4_MIN_CUDA_CAPABILITY:
            raise RuntimeError(
                f"`attention_backend='fa4'` requires CUDA capability >= sm{FA4_MIN_CUDA_CAPABILITY}, "
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
        return self.normalize_attention_output(
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
        batch_size, seq_len = key_padding_mask.shape
        valid_lengths = self._right_padding_lengths(key_padding_mask)
        if valid_lengths is None:
            raise RuntimeError(
                "`attention_backend='fa4'` supports padding masks only when valid "
                "tokens form a left-aligned prefix in each sequence."
            )

        out = q.new_zeros(batch_size, seq_len, self.num_heads, self.head_dim)
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
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        q, k, v = qkv.unbind(dim=2)
        if key_padding_mask is None:
            return self._run_dense(q, k, v, causal=causal)
        return self._run_right_padded(q, k, v, key_padding_mask, causal=causal)
