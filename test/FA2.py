try:
    from attention_backend import run_backend
except ImportError:
    from .attention_backend import run_backend


if __name__ == "__main__":
    raise SystemExit(run_backend("fa2"))
