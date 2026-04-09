__all__ = [
    "BasicTokenizer",
    "CondTokenizer",
    "ExprTokenizer",
    "GeneTokenizer",
    "ScTokenizer",
    "ScBatchCollator",
    "ScDataset",
    "ScPreprocessor",
    "preprocess_adata",
    "PreprocessedScDataset",
    "PretrainingAssets",
    "PretrainingDataBundle",
    "build_pretraining_assets",
    "build_data_bundle_for_path",
    "estimate_total_training_steps",
    "resolve_train_paths",
]


def __getattr__(name: str):
    if name in {
        "PreprocessedScDataset",
        "PretrainingAssets",
        "PretrainingDataBundle",
        "build_data_bundle_for_path",
        "build_pretraining_assets",
        "estimate_total_training_steps",
        "resolve_train_paths",
    }:
        from . import builders as _builders

        return getattr(_builders, name)

    if name == "ScBatchCollator":
        from . import collator as _collator

        return getattr(_collator, name)

    if name == "ScDataset":
        from . import dataset as _dataset

        return getattr(_dataset, name)

    if name in {"ScPreprocessor", "preprocess_adata"}:
        from . import preprocess as _preprocess

        return getattr(_preprocess, name)

    if name in {
        "BasicTokenizer",
        "CondTokenizer",
        "ExprTokenizer",
        "GeneTokenizer",
        "ScTokenizer",
    }:
        from . import tokenizer as _tokenizer

        return getattr(_tokenizer, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
