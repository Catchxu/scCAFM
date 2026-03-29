from .metrics import (
    binary_auprc,
    binary_auroc,
    early_precision_ratio,
    summarize_binary_metrics,
)

__all__ = [
    "EvaluationGRNCache",
    "build_evaluation_grn_cache",
    "build_reference_grn",
    "evaluate_cell_specific_grns",
    "prepare_evaluation_paths",
    "run_evaluation",
    "binary_auprc",
    "binary_auroc",
    "early_precision_ratio",
    "summarize_binary_metrics",
]


def __getattr__(name: str):
    if name in {
        "EvaluationGRNCache",
        "build_evaluation_grn_cache",
        "build_reference_grn",
        "evaluate_cell_specific_grns",
        "prepare_evaluation_paths",
        "run_evaluation",
    }:
        from . import grn as _grn

        return getattr(_grn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
