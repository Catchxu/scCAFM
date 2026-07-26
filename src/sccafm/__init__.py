"""Public scCAFM package API."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("scCAFM")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "ChIPSeqEvaluation",
    "ChIPSeqReference",
    "CellSpecificGRNs",
    "GRNInferencer",
    "PooledGRN",
    "ScPreprocessor",
    "evaluate_chipseq_grn",
    "load_vocab_json",
    "load_yaml_config",
    "prepare_chipseq_reference",
    "resolve_model_assets",
    "write_cell_specific_grns_csv",
    "write_pooled_grn_csv",
]


def __getattr__(name: str):
    if name == "ScPreprocessor":
        return getattr(import_module(".data", __name__), name)
    if name in {"load_vocab_json", "resolve_model_assets"}:
        return getattr(import_module(".assets", __name__), name)
    if name == "load_yaml_config":
        return getattr(import_module(".config", __name__), name)
    if name in {
        "ChIPSeqEvaluation",
        "ChIPSeqReference",
        "CellSpecificGRNs",
        "GRNInferencer",
        "PooledGRN",
        "evaluate_chipseq_grn",
        "prepare_chipseq_reference",
        "write_cell_specific_grns_csv",
        "write_pooled_grn_csv",
    }:
        return getattr(import_module(".tasks", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
