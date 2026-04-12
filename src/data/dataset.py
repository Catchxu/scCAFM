from __future__ import annotations

from anndata import AnnData
from torch.utils.data import Dataset

from typing import Any, Optional

from .tokenizer import ScTokenizer, ScTokenizerOutput


class ScDataset(Dataset):
    """
    PyTorch dataset for tokenized single-cell data.

    The dataset tokenizes the whole `AnnData` object once during initialization
    and returns one cell per item, including the TF-aware `non_tf_mask`.
    """

    def __init__(
        self,
        adata: AnnData,
        tokenizer: ScTokenizer,
        gene_key: Optional[str] = None,
        preprocessor: Any | None = None,
    ) -> None:
        if not isinstance(adata, AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got {type(adata).__name__}.")
        if not isinstance(tokenizer, ScTokenizer):
            raise TypeError(
                f"`tokenizer` must be a ScTokenizer object, got {type(tokenizer).__name__}."
            )

        self.raw_n_obs = int(adata.n_obs)
        self.raw_n_vars = int(adata.n_vars)
        processed = preprocessor(adata) if preprocessor is not None else adata
        self.processed_n_obs = int(processed.n_obs)
        self.processed_n_vars = int(processed.n_vars)

        self.adata = processed
        self.tokenizer = tokenizer
        self.gene_key = gene_key

        self.tokenized: ScTokenizerOutput = self.tokenizer(
            processed,
            gene_key=gene_key,
        )

        # After tokenization, the training pipeline only reads from the cached
        # token tensors. Releasing the AnnData object here avoids keeping a full
        # CPU-side copy of the processed matrix alive for the whole file.
        if getattr(self.adata, "file", None) is not None:
            self.adata.file.close()
        self.adata = None

    def __len__(self) -> int:
        return self.processed_n_obs

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(f"`index` out of range: {index}")

        return {
            "input_ids": self.tokenized.input_ids[index],
            "expression_values": self.tokenized.expression_values[index],
            "condition_ids": self.tokenized.condition_ids[index],
            "padding_mask": (
                None
                if self.tokenized.padding_mask is None
                else self.tokenized.padding_mask[index]
            ),
            "non_tf_mask": self.tokenized.non_tf_mask[index],
            "gene_name_type": self.tokenized.gene_name_type,
        }

    def close(self) -> None:
        adata = getattr(self, "adata", None)
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        self.adata = None
        self.tokenized = None


class PreprocessedScDataset(ScDataset):
    def __init__(
        self,
        adata: AnnData,
        tokenizer: ScTokenizer,
        gene_key: Optional[str] = None,
        preprocessor: Any | None = None,
    ) -> None:
        super().__init__(
            adata=adata,
            tokenizer=tokenizer,
            gene_key=gene_key,
            preprocessor=preprocessor,
        )




if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    from ..assets import load_vocab_json
    from ..assets import resolve_model_assets
    from .collator import ScBatchCollator

    root_dir = Path(__file__).resolve().parents[2]
    assets = resolve_model_assets(root_dir / "assets")
    token_dict = load_vocab_json(assets.vocab)
    human_tfs = pd.read_csv(assets.human_tfs)
    mouse_tfs = pd.read_csv(assets.mouse_tfs)

    obs = pd.DataFrame(
        {
            "platform": ["10X", "smart-seq"],
            "species": ["human", "mouse"],
            "tissue": ["lung", "brain"],
            "disease": ["healthy", "tumor"],
        },
        index=["cell_0", "cell_1"],
    )
    var_names = pd.Index(["AATF", "TSPAN6"])
    adata = AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=var_names),
    )

    tokenizer = ScTokenizer(
        token_dict=token_dict,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        platform_key="platform",
        species_key="species",
        tissue_key="tissue",
        disease_key="disease",
        max_length=5,
    )
    dataset = ScDataset(adata=adata, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=ScBatchCollator(),
    )
    batch = next(iter(dataloader))

    print("len:", len(dataset))
    print("sample_keys:", sorted(dataset[0].keys()))
    print("input_ids:")
    print(dataset[0]["input_ids"])
    print("expression_values:")
    print(dataset[0]["expression_values"])
    print("condition_ids:")
    print(dataset[0]["condition_ids"])
    print("padding_mask:")
    print(dataset[0]["padding_mask"])
    print("non_tf_mask:")
    print(dataset[0]["non_tf_mask"])
    print("gene_name_type:")
    print(dataset[0]["gene_name_type"])
    print("batch_keys:", batch.keys())
    print("batch_input_ids_shape:", tuple(batch["input_ids"].shape))
    print(
        "batch_padding_mask_shape:",
        None if batch["padding_mask"] is None else tuple(batch["padding_mask"].shape),
    )
    print("batch_non_tf_mask_shape:", tuple(batch["non_tf_mask"].shape))
