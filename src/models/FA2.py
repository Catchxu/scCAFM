from __future__ import annotations

import torch

from .attention import FlashAttentionBackend

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError as exc:
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None
    pad_input = None
    unpad_input = None
    _FA2_IMPORT_ERROR = exc
else:
    _FA2_IMPORT_ERROR = None


class FlashMHAFA2(FlashAttentionBackend):
    """FA2 backend: packed QKV dense kernel, packed QKV varlen kernel for padding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        if (
            flash_attn_qkvpacked_func is None
            or flash_attn_varlen_qkvpacked_func is None
            or pad_input is None
            or unpad_input is None
        ):
            raise RuntimeError(
                "`attention_backend='fa2'` requires FlashAttention FA2 packed-QKV kernels."
            ) from _FA2_IMPORT_ERROR

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
        batch_size, seq_len = qkv.shape[:2]
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
            return qkv.new_zeros(batch_size, seq_len, self.num_heads, self.head_dim)

        out_unpad = self._run_varlen(
            qkv_unpad,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            causal=causal,
        )
        return pad_input(out_unpad, indices, batch_size, seq_len)
