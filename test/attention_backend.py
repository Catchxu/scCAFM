from __future__ import annotations

import argparse
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.models.attention import FlashMHA


def _print_env(backend: str) -> None:
    print("=" * 80)
    print(f"scCAFM attention backend smoke test: {backend}")
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    try:
        import flash_attn

        print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
        print("flash_attn path:", getattr(flash_attn, "__file__", None))
    except Exception as exc:
        print("flash_attn import:", f"{type(exc).__name__}: {exc}")

    if backend == "fa4":
        try:
            import flash_attn.cute

            print("flash_attn.cute path:", getattr(flash_attn.cute, "__file__", None))
        except Exception as exc:
            print("flash_attn.cute import:", f"{type(exc).__name__}: {exc}")

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("cuda available: false")
    print("=" * 80)


def _make_model(backend: str, embed_dim: int, num_heads: int) -> FlashMHA:
    model = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_backend=backend,
        use_rotary=False,
    )
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def _run_dense(backend: str, *, batch_size: int, seq_len: int, embed_dim: int, num_heads: int) -> None:
    print("\n[dense]")
    model = _make_model(backend, embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = model(x, key_padding_mask=None, causal=False)
    torch.cuda.synchronize()

    print("output shape:", tuple(out.shape))
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert torch.isfinite(out).all()


def _run_right_padded(
    backend: str,
    *,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
) -> None:
    print("\n[right-padded mask]")
    model = _make_model(backend, embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.bfloat16)
    key_padding_mask = torch.zeros(batch_size, seq_len, device="cuda", dtype=torch.bool)
    if batch_size > 1:
        key_padding_mask[1, seq_len * 3 // 4 :] = True
    if batch_size > 2:
        key_padding_mask[2, seq_len // 2 :] = True

    with torch.no_grad():
        out = model(x, key_padding_mask=key_padding_mask, causal=False)
    torch.cuda.synchronize()

    print("output shape:", tuple(out.shape))
    print("masked tail norm:", out[key_padding_mask].abs().sum().item())
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert torch.isfinite(out).all()
    assert out[key_padding_mask].abs().sum().item() == 0.0


def run_backend(backend: str) -> int:
    backend = backend.lower()
    if backend not in {"fa2", "fa4"}:
        raise ValueError(f"`backend` must be one of ['fa2', 'fa4'], got {backend!r}.")

    _print_env(backend)
    if not torch.cuda.is_available():
        print(f"\n{backend} smoke test skipped: CUDA is required.")
        return 1

    torch.manual_seed(42)
    _run_dense(backend, batch_size=2, seq_len=128, embed_dim=64, num_heads=8)
    _run_right_padded(backend, batch_size=3, seq_len=32, embed_dim=64, num_heads=8)
    print(f"\n{backend} smoke test passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test scCAFM FlashAttention backends.")
    parser.add_argument(
        "--backend",
        choices=("fa2", "fa4"),
        required=True,
        help="Attention backend to test through the scCAFM FlashMHA wrapper.",
    )
    args = parser.parse_args()
    return run_backend(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
