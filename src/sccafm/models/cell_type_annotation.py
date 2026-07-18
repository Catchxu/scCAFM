from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .efm import EFM, EFMOutput


@dataclass(slots=True)
class CellTypeAnnotationModelOutput:
    """Classifier logits together with the underlying EFM output."""

    logits: torch.Tensor
    efm_output: EFMOutput


class EFMCellTypeClassifier(nn.Module):
    """Fine-tunable EFM followed by a linear multiclass annotation head."""

    def __init__(self, efm: EFM, num_classes: int) -> None:
        super().__init__()
        if not isinstance(efm, nn.Module):
            raise TypeError("`efm` must be a torch module.")
        if int(num_classes) <= 1:
            raise ValueError(f"`num_classes` must be at least 2, got {num_classes}.")
        embed_dim = getattr(efm, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("`efm` must expose a positive `embed_dim` attribute.")
        if int(embed_dim) <= 0:
            raise ValueError(f"`efm.embed_dim` must be positive, got {embed_dim}.")

        self.efm = efm
        self.classifier = nn.Linear(int(embed_dim), int(num_classes))

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
    ) -> CellTypeAnnotationModelOutput:
        efm_output = self.efm(tokens)
        return CellTypeAnnotationModelOutput(
            logits=self.classifier(efm_output.cell_embedding),
            efm_output=efm_output,
        )
