from __future__ import annotations

import torch


class ScBatchCollator:
    """
    Collate per-cell single-cell samples into a batch dictionary.

    The returned batch matches the tensor-oriented inputs expected by the model
    stack, while preserving optional `padding_mask`.
    """

    def __call__(
        self,
        batch: list[dict[str, object]],
    ) -> dict[str, object]:
        if len(batch) == 0:
            raise ValueError("`batch` must contain at least one sample.")

        input_ids = [sample["input_ids"] for sample in batch]
        expression_values = [sample["expression_values"] for sample in batch]
        condition_ids = [sample["condition_ids"] for sample in batch]
        non_tf_mask = [sample["non_tf_mask"] for sample in batch]
        padding_masks = [sample["padding_mask"] for sample in batch]
        gene_name_types = [sample["gene_name_type"] for sample in batch]

        collated = {
            "input_ids": torch.stack(input_ids, dim=0),
            "expression_values": torch.stack(expression_values, dim=0),
            "condition_ids": torch.stack(condition_ids, dim=0),
            "non_tf_mask": torch.stack(non_tf_mask, dim=0),
        }

        if all(mask is None for mask in padding_masks):
            collated["padding_mask"] = None
        elif any(mask is None for mask in padding_masks):
            raise ValueError(
                "Cannot collate a mixed batch where some samples have `padding_mask` "
                "and others do not."
            )
        else:
            collated["padding_mask"] = torch.stack(padding_masks, dim=0)

        if all(gene_name_type == gene_name_types[0] for gene_name_type in gene_name_types):
            collated["gene_name_type"] = gene_name_types[0]
        else:
            collated["gene_name_type"] = gene_name_types

        return collated
