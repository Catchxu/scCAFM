from __future__ import annotations

import warnings

import torch

from .attention import FlashAttentionBackend

_FA4_IMPORT_ERROR = None
_CUTLASS_SCALAR_PTR_WARNING = "Use explicit `struct.scalar.ptr` for pointer instead."


def _suppress_cutlass_scalar_ptr_warning() -> None:
    previous_showwarning = warnings.showwarning
    if getattr(previous_showwarning, "_sccafm_suppresses_cutlass_scalar_ptr", False):
        return

    def showwarning(
        message,
        category,
        filename,
        lineno,
        file=None,
        line=None,
    ):
        if (
            issubclass(category, DeprecationWarning)
            and _CUTLASS_SCALAR_PTR_WARNING in str(message)
            and "nvidia_cutlass_dsl" in str(filename)
        ):
            return
        previous_showwarning(message, category, filename, lineno, file=file, line=line)

    showwarning._sccafm_suppresses_cutlass_scalar_ptr = True
    warnings.showwarning = showwarning


_suppress_cutlass_scalar_ptr_warning()

try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as cute_flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError as exc:
    cute_flash_attn_func = None
    cute_flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _FA4_IMPORT_ERROR = exc


FA4_MIN_CUDA_CAPABILITY = 90


class FlashMHAFA4(FlashAttentionBackend):
    """FA4 backend using CuTe `flash_attn_func` and `flash_attn_varlen_func`."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self._validate_support()

    def _validate_support(self) -> None:
        if (
            cute_flash_attn_func is None
            or cute_flash_attn_varlen_func is None
            or pad_input is None
            or unpad_input is None
        ):
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
            cute_flash_attn_varlen_func(
                q_unpad.contiguous(),
                k_unpad.contiguous(),
                v_unpad.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
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
