from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sccafm.assets import load_json, load_model_state_dict
from sccafm.evaluator.cell_type_annotation import evaluate_cell_type_annotation
from sccafm.models import EFMCellTypeClassifier, GeneOrderState
from sccafm.trainer.efm_cell_type_annotation import (
    _ordered_tokens,
    _save_checkpoint,
    balanced_class_weights,
    encode_and_split_labels,
)


def test_encode_and_split_labels_is_deterministic_and_complete() -> None:
    labels = pd.Series(["B", "A", "C", "A", "B", "C", "A", "B", "C"])
    first = encode_and_split_labels(labels, train_fraction=0.7, split_seed=42)
    second = encode_and_split_labels(labels, train_fraction=0.7, split_seed=42)

    class_names, label_ids, train_indices, test_indices = first
    assert class_names == ["A", "B", "C"]
    assert np.array_equal(train_indices, second[2])
    assert np.array_equal(test_indices, second[3])
    assert set(label_ids[train_indices]) == {0, 1, 2}
    assert set(label_ids[test_indices]) == {0, 1, 2}


def test_encode_and_split_labels_rejects_missing_and_rare_labels() -> None:
    with pytest.raises(ValueError, match="missing"):
        encode_and_split_labels(pd.Series(["A", None, "B", "A", "B", "B"]))
    with pytest.raises(ValueError, match="Label counts"):
        encode_and_split_labels(pd.Series(["A", "A", "B", "B", "C"]))


def test_balanced_class_weights_are_inverse_frequency_and_mean_one() -> None:
    weights = balanced_class_weights(np.array([0, 0, 0, 0, 1, 2, 2]), num_classes=3)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[1] > weights[2] > weights[0]


def test_annotation_metrics_include_all_classes() -> None:
    result = evaluate_cell_type_annotation(
        [0, 1, 2, 0],
        [0, 1, 1, 0],
        class_names=["A", "B", "C"],
    )
    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx((1.0 + (2.0 / 3.0) + 0.0) / 3.0)
    assert result.test_cell_count == 4


class _FrozenSFM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, tokens, **_kwargs):
        batch_size, sequence_length = tokens["input_ids"].shape
        positions = torch.arange(sequence_length - 1, -1, -1).repeat(batch_size, 1)
        order = GeneOrderState(
            positions=positions,
            active_lengths=torch.full((batch_size,), sequence_length, dtype=torch.long),
            cycle_break_counts=torch.zeros(batch_size, dtype=torch.long),
        )
        return SimpleNamespace(foundations={"sfm": SimpleNamespace(gene_order=order)})


class _TinyEFM(torch.nn.Module):
    embed_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens):
        value = tokens["expression_values"].mean(dim=1, keepdim=True) * self.scale
        return SimpleNamespace(cell_embedding=value.repeat(1, self.embed_dim))


def _tiny_tokens() -> dict[str, torch.Tensor | None]:
    return {
        "input_ids": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
        "expression_values": torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
        "condition_ids": torch.zeros((2, 4), dtype=torch.long),
        "non_tf_mask": torch.zeros((2, 2), dtype=torch.bool),
        "padding_mask": None,
    }


def test_frozen_sfm_has_no_gradients_while_efm_and_head_update() -> None:
    frozen_sfm = _FrozenSFM()
    model = EFMCellTypeClassifier(_TinyEFM(), num_classes=2)
    run = SimpleNamespace(
        device=torch.device("cpu"),
        config={"runtime": {"precision": {"autocast_dtype": "fp32"}}},
        sfm=frozen_sfm,
    )
    ordered_tokens = _ordered_tokens(run, _tiny_tokens())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before_efm = model.efm.scale.detach().clone()
    before_head = model.classifier.weight.detach().clone()

    loss = torch.nn.CrossEntropyLoss()(model(ordered_tokens).logits, torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()

    assert frozen_sfm.weight.grad is None
    assert torch.equal(frozen_sfm.weight.detach(), torch.tensor(1.0))
    assert not torch.equal(model.efm.scale.detach(), before_efm)
    assert not torch.equal(model.classifier.weight.detach(), before_head)


def test_checkpoint_and_label_map_round_trip(tmp_path) -> None:
    model = EFMCellTypeClassifier(_TinyEFM(), num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    run = SimpleNamespace(model=model, class_names=["A", "B"])

    _save_checkpoint(
        run=run,
        output_dir=tmp_path,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=2,
    )

    classifier_path = tmp_path / "checkpoints/models/cell_type_annotation/classifier.safetensors"
    labels_path = tmp_path / "checkpoints/models/cell_type_annotation/labels.json"
    assert load_json(labels_path) == {"class_names": ["A", "B"]}
    loaded = load_model_state_dict(classifier_path)
    assert set(loaded) == {"weight", "bias"}
    assert torch.allclose(loaded["weight"], model.classifier.weight.detach().cpu())
