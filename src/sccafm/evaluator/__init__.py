from .metrics import (
    binary_auprc,
    binary_auroc,
    early_precision_ratio,
    median_similarity_distribution,
    summarize_binary_metrics,
)

__all__ = [
    "CellFateMedianSimilarityResult",
    "CellFateSimilarityResult",
    "EvaluationGRNCache",
    "EvaluationPairSpec",
    "build_evaluation_grn_cache",
    "build_reference_grn",
    "evaluate_cell_specific_grns",
    "evaluate_median_similarity",
    "prepare_evaluation_paths",
    "run_evaluation",
    "select_target_up_degs",
    "binary_auprc",
    "binary_auroc",
    "early_precision_ratio",
    "median_similarity_distribution",
    "summarize_binary_metrics",
]


def __getattr__(name: str):
    if name in {
        "CellFateMedianSimilarityResult",
        "CellFateSimilarityResult",
        "evaluate_median_similarity",
        "select_target_up_degs",
    }:
        from . import cell_fate as _cell_fate

        return getattr(_cell_fate, name)
    if name in {
        "EvaluationGRNCache",
        "EvaluationPairSpec",
        "build_evaluation_grn_cache",
        "build_reference_grn",
        "evaluate_cell_specific_grns",
        "prepare_evaluation_paths",
        "run_evaluation",
    }:
        from . import grn as _grn

        return getattr(_grn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
