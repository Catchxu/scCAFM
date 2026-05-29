from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GeneOrderState:
    """
    Per-cell gene ordering produced from an SFM GRN tensor.

    Shape convention:
    - `C`: cell count
    - `G`: gene-token sequence length
    """

    positions: torch.LongTensor
    active_lengths: torch.LongTensor
    cycle_break_counts: torch.LongTensor


def _validate_grn(grn: torch.Tensor) -> None:
    if not torch.is_tensor(grn):
        raise TypeError("`grn` must be a torch.Tensor.")
    if grn.ndim != 3 or grn.shape[1] != grn.shape[2]:
        raise ValueError(f"`grn` must have shape (C, G, G), got {tuple(grn.shape)}.")


def _active_gene_mask(
    grn: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.BoolTensor:
    batch_size, seq_len, _ = grn.shape
    if padding_mask is None:
        return torch.ones((batch_size, seq_len), device=grn.device, dtype=torch.bool)
    if not torch.is_tensor(padding_mask):
        raise TypeError("`padding_mask` must be a torch.Tensor or None.")
    if padding_mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"`padding_mask` must have shape {(batch_size, seq_len)}, "
            f"got {tuple(padding_mask.shape)}."
        )
    return ~padding_mask.to(device=grn.device, dtype=torch.bool)


def _argmax_with_position_tie_break(
    candidates: torch.Tensor,
    scores: torch.Tensor,
) -> int:
    if candidates.numel() == 0:
        raise ValueError("Cannot select from an empty candidate set.")
    candidate_scores = scores.index_select(dim=0, index=candidates)
    max_score = candidate_scores.max()
    tied = candidates[candidate_scores == max_score]
    return int(tied.min().item())


def _order_one_cell(
    weights: torch.Tensor,
    active_positions: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, int]:
    seq_len = int(weights.shape[0])
    if active_positions.numel() == 0:
        return active_positions, 0

    work = weights.to(dtype=torch.float32).clone()
    diagonal = torch.arange(seq_len, device=work.device)
    work[diagonal, diagonal] = 0.0
    work = work.clamp_min(0.0)

    active = torch.zeros(seq_len, device=work.device, dtype=torch.bool)
    active[active_positions] = True
    remaining = active.clone()

    masked_work = work * active.unsqueeze(0).to(work.dtype) * active.unsqueeze(1).to(work.dtype)
    in_degree = masked_work.sum(dim=0)
    out_degree = masked_work.sum(dim=1)

    ordered: list[int] = []
    cycle_breaks = 0
    while bool(remaining.any()):
        remaining_positions = remaining.nonzero(as_tuple=True)[0]
        zero_in_positions = remaining_positions[in_degree.index_select(0, remaining_positions) <= 0.0]

        if zero_in_positions.numel() > 0:
            chosen = _argmax_with_position_tie_break(
                zero_in_positions,
                out_degree,
            )
        else:
            cycle_breaks += 1
            cycle_scores = out_degree / (in_degree + float(eps))
            chosen = _argmax_with_position_tie_break(
                remaining_positions,
                cycle_scores,
            )

        ordered.append(chosen)
        remaining[chosen] = False

        if bool(remaining.any()):
            in_degree = in_degree - work[chosen]
            out_degree = out_degree - work[:, chosen]
            in_degree = in_degree.clamp_min(0.0)
            out_degree = out_degree.clamp_min(0.0)
            in_degree[~remaining] = 0.0
            out_degree[~remaining] = 0.0

    return torch.tensor(ordered, device=weights.device, dtype=torch.long), cycle_breaks


def order_genes_from_grn(
    grn: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> GeneOrderState:
    """
    Build a deterministic weighted causal priority order from an SFM GRN tensor.

    The returned `positions` tensor is a full gather index over the current
    gene-token sequence: ordered active gene positions first, followed by
    inactive/padded positions in their original order.
    """

    _validate_grn(grn)
    if eps <= 0.0:
        raise ValueError(f"`eps` must be positive, got {eps}.")

    active_mask = _active_gene_mask(grn, padding_mask)
    batch_size, seq_len, _ = grn.shape
    all_positions = torch.arange(seq_len, device=grn.device, dtype=torch.long)
    full_positions = torch.empty((batch_size, seq_len), device=grn.device, dtype=torch.long)
    active_lengths = torch.empty(batch_size, device=grn.device, dtype=torch.long)
    cycle_break_counts = torch.empty(batch_size, device=grn.device, dtype=torch.long)

    for cell_idx in range(batch_size):
        active_positions = active_mask[cell_idx].nonzero(as_tuple=True)[0]
        ordered_active, cycle_breaks = _order_one_cell(
            grn[cell_idx],
            active_positions,
            eps=eps,
        )
        inactive_positions = all_positions[~active_mask[cell_idx]]
        full_positions[cell_idx] = torch.cat((ordered_active, inactive_positions), dim=0)
        active_lengths[cell_idx] = int(active_positions.numel())
        cycle_break_counts[cell_idx] = int(cycle_breaks)

    return GeneOrderState(
        positions=full_positions,
        active_lengths=active_lengths,
        cycle_break_counts=cycle_break_counts,
    )
