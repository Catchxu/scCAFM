from __future__ import annotations

import torch

from .attention import FlashAttentionBackend

_FA2_IMPORT_ERROR = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError as exc:
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _FA2_IMPORT_ERROR = exc


class FlashMHAFA2(FlashAttentionBackend):
    """FA2 backend using `flash_attn_func` and `flash_attn_varlen_func`."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        if (
            flash_attn_func is None
            or flash_attn_varlen_func is None
            or pad_input is None
            or unpad_input is None
        ):
            raise RuntimeError(
                "`attention_backend='fa2'` requires FlashAttention FA2 q/k/v kernels."
            ) from _FA2_IMPORT_ERROR

    def _run_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        return self.normalize_attention_output(
            flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        )

    def _run_varlen(
        self,
        q_unpad: torch.Tensor,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> torch.Tensor:
        return self.normalize_attention_output(
            flash_attn_varlen_func(
                q_unpad.contiguous(),
                k_unpad.contiguous(),
                v_unpad.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        )

    @staticmethod
    def _unpad_tensor(
        tensor: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        unpad_outputs = unpad_input(tensor, attention_mask)
        if len(unpad_outputs) == 4:
            tensor_unpad, indices, cu_seqlens, max_seqlen = unpad_outputs
        elif len(unpad_outputs) == 5:
            tensor_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_outputs
        else:
            raise ValueError(
                f"Unexpected number of outputs from unpad_input: {len(unpad_outputs)}"
            )
        return tensor_unpad, indices, cu_seqlens, max_seqlen

    def forward(
        self,
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = qkv.shape[:2]
        q, k, v = qkv.unbind(dim=2)
        if key_padding_mask is None:
            return self._run_dense(q, k, v, causal=causal)

        attention_mask = ~key_padding_mask
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = self._unpad_tensor(
            q,
            attention_mask,
        )
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = self._unpad_tensor(
            k,
            attention_mask,
        )
        v_unpad, _, _, _ = self._unpad_tensor(
            v,
            attention_mask,
        )

        if q_unpad.shape[0] == 0:
            return qkv.new_zeros(batch_size, seq_len, self.num_heads, self.head_dim)

        if not torch.equal(indices_q, indices_k):
            raise RuntimeError("Q and K unpadding layouts differ for self-attention.")

        out_unpad = self._run_varlen(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )
        out = pad_input(out_unpad, indices_q, batch_size, seq_len)
        return out.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
