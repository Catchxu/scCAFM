"""Public scCAFM package API."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("scCAFM")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "ScPreprocessor",
    "cell_fate",
    "grn",
    "load_vocab_json",
    "load_yaml_config",
    "resolve_model_assets",
]


def __getattr__(name: str):
    if name in {"cell_fate", "grn"}:
        return getattr(import_module(".inference", __name__), name)
    if name == "ScPreprocessor":
        return getattr(import_module(".data", __name__), name)
    if name in {"load_vocab_json", "resolve_model_assets"}:
        return getattr(import_module(".assets", __name__), name)
    if name == "load_yaml_config":
        return getattr(import_module(".config", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
