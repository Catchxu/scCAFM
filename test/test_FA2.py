import torch

from flash_attn import flash_attn_func
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input


def normalize_output(out):
    if isinstance(out, tuple):
        return out[0]
    return out


def print_env():
    import flash_attn

    print("=" * 80)
    print("FA2 test: flash_attn_func + flash_attn_varlen_func")
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))
    print("flash_attn path:", flash_attn.__file__)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("=" * 80)


def test_flash_attn_func():
    print("\n[FA2] Testing flash_attn_func")

    B, L, H, D = 2, 1024, 8, 64

    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=D ** -0.5,
        causal=False,
    )
    out = normalize_output(out)
    torch.cuda.synchronize()

    print("output shape:", tuple(out.shape))
    assert out.shape == (B, L, H, D)
    assert torch.isfinite(out).all()
    print("[FA2] flash_attn_func passed")


def test_flash_attn_varlen_func():
    print("\n[FA2] Testing flash_attn_varlen_func")

    B, L, H, D = 3, 16, 8, 64

    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)

    # True means PAD.
    key_padding_mask = torch.zeros(B, L, device="cuda", dtype=torch.bool)
    key_padding_mask[1, 12:] = True
    key_padding_mask[2, 7:] = True

    # True means valid token.
    attention_mask = ~key_padding_mask

    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        q,
        attention_mask,
    )[:4]
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
        k,
        attention_mask,
    )[:4]
    v_unpad, _, _, _ = unpad_input(
        v,
        attention_mask,
    )[:4]

    # Self-attention case: q/k/v share the same valid-token layout.
    assert torch.equal(indices_q, indices_k)
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    assert max_seqlen_q == max_seqlen_k

    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=D ** -0.5,
        causal=False,
    )
    out_unpad = normalize_output(out_unpad)

    out = pad_input(out_unpad, indices_q, B, L)
    out = out.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
    torch.cuda.synchronize()

    print("valid lengths:", attention_mask.sum(dim=1).tolist())
    print("output shape:", tuple(out.shape))
    print("padded tail norm:", out[key_padding_mask].abs().sum().item())

    assert out.shape == (B, L, H, D)
    assert torch.isfinite(out).all()
    assert out[key_padding_mask].abs().sum().item() == 0.0
    print("[FA2] flash_attn_varlen_func passed")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(42)

    print_env()
    test_flash_attn_func()
    test_flash_attn_varlen_func()

    print("\nFA2 flash_attn_func + flash_attn_varlen_func demo passed.")