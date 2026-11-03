"""Shared EFM freezing policy for supervised downstream tasks."""

from ..models.efm import EFM


def configure_efm_freezing(model: EFM, freeze_layers: int) -> None:
    """Freeze leading blocks and input embeddings; all blocks freezes the EFM."""
    num_layers = model.backbone.num_layers
    if not 0 <= freeze_layers <= num_layers:
        raise ValueError(f"freeze_layers must be between 0 and {num_layers}.")
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    model.embedding.requires_grad_(freeze_layers == 0)
    for layer in model.backbone.layers[freeze_layers:]:
        layer.requires_grad_(True)
    model.backbone.final_norm.requires_grad_(freeze_layers < num_layers)


def set_efm_training_mode(model: EFM, freeze_layers: int) -> None:
    """Disable dropout in frozen modules while trainable blocks learn."""
    model.train()
    if freeze_layers > 0:
        model.embedding.eval()
    for layer in model.backbone.layers[:freeze_layers]:
        layer.eval()
    if freeze_layers == model.backbone.num_layers:
        model.eval()
