from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class CellTypeAnnotationResult:
    """Held-out multiclass annotation metrics and label metadata."""

    accuracy: float
    macro_f1: float
    test_cell_count: int
    class_names: list[str]

    def to_summary(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "test_cell_count": self.test_cell_count,
            "class_names": list(self.class_names),
        }


def evaluate_cell_type_annotation(
    true_label_ids: Sequence[int] | np.ndarray,
    predicted_label_ids: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str],
) -> CellTypeAnnotationResult:
    """Compute accuracy and all-class macro-F1 for cell-type annotation."""

    y_true = np.asarray(true_label_ids, dtype=np.int64)
    y_pred = np.asarray(predicted_label_ids, dtype=np.int64)
    labels = [str(name) for name in class_names]
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("True and predicted label IDs must both be one-dimensional.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "True and predicted label IDs must have matching shapes, got "
            f"{tuple(y_true.shape)} and {tuple(y_pred.shape)}."
        )
    if y_true.size == 0:
        raise ValueError("At least one labelled cell is required for evaluation.")
    if len(labels) < 2:
        raise ValueError("`class_names` must contain at least two cell types.")

    class_ids = np.arange(len(labels), dtype=np.int64)
    if np.any(y_true < 0) or np.any(y_true >= len(labels)):
        raise ValueError("True label IDs are outside the configured class range.")
    if np.any(y_pred < 0) or np.any(y_pred >= len(labels)):
        raise ValueError("Predicted label IDs are outside the configured class range.")

    return CellTypeAnnotationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        test_cell_count=int(y_true.size),
        class_names=labels,
    )
