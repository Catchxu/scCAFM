import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class FactorState:
    """
    Container for factorized GRN state shared across training/evaluation.
    Shapes:
      - binary_tf: (C, L)
      - binary_tg: (C, L)
      - u: (C, TF, M)
      - v: (C, TG_sel, M)
    """
    binary_tf: torch.Tensor
    binary_tg: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    u_score: Optional[torch.Tensor] = None
    v_score: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.binary_tf.ndim != 2 or self.binary_tg.ndim != 2:
            raise ValueError(
                f"binary_tf/binary_tg must be 2-D, got {tuple(self.binary_tf.shape)} and {tuple(self.binary_tg.shape)}"
            )
        if self.u.ndim != 3 or self.v.ndim != 3:
            raise ValueError(f"u/v must be 3-D, got {tuple(self.u.shape)} and {tuple(self.v.shape)}")
        c = self.u.shape[0]
        if self.binary_tf.shape[0] != c or self.binary_tg.shape[0] != c or self.v.shape[0] != c:
            raise ValueError(
                "Batch size mismatch across factor tensors: "
                f"u={self.u.shape[0]}, v={self.v.shape[0]}, "
                f"binary_tf={self.binary_tf.shape[0]}, binary_tg={self.binary_tg.shape[0]}"
            )

        tf_counts = self.binary_tf.bool().sum(dim=1)
        tg_counts = self.binary_tg.bool().sum(dim=1)
        if not torch.all(tf_counts == self.u.shape[1]):
            raise ValueError(
                f"u TF dim ({self.u.shape[1]}) does not match binary_tf true-counts per sample: {tf_counts.tolist()}"
            )
        if not torch.all(tg_counts == self.v.shape[1]):
            raise ValueError(
                f"v TG dim ({self.v.shape[1]}) does not match binary_tg true-counts per sample: {tg_counts.tolist()}"
            )
        if self.u_score is not None:
            if self.u_score.shape != self.u.shape:
                raise ValueError(
                    f"u_score shape mismatch: expected {tuple(self.u.shape)}, got {tuple(self.u_score.shape)}"
                )
        if self.v_score is not None:
            if self.v_score.shape != self.v.shape:
                raise ValueError(
                    f"v_score shape mismatch: expected {tuple(self.v.shape)}, got {tuple(self.v_score.shape)}"
                )


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor):
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu and sigma should have the same shape, got {mu.shape} and {sigma.shape}!"
        )
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(rms + self.eps)
        out = x_norm * self.weight.float()
        return out.to(dtype=x.dtype)


def expand_grn(grn: torch.Tensor, binary_tf: torch.Tensor, binary_tg: torch.Tensor):
    """
    Expands the predicted GRN from [C, TF, TG] to [C, TG, TG] based on gate indices.
    
    Args:
        grn: Predicted regulatory logits of shape (C, TF, TG)
        binary_tf: Binary gate/mask for TFs of shape (C, TG)
        binary_tg: Binary gate/mask for Target Genes of shape (C, TG)
    """
    if grn.ndim != 3:
        raise ValueError(f"grn must be (C, TF, TG_sel), got {tuple(grn.shape)}")
    if binary_tf.ndim != 2 or binary_tg.ndim != 2:
        raise ValueError(
            f"binary_tf and binary_tg must be 2-D, got {tuple(binary_tf.shape)} and {tuple(binary_tg.shape)}"
        )

    C, TF, TG_sel = grn.shape
    if binary_tf.shape[0] != C or binary_tg.shape[0] != C:
        raise ValueError(
            f"Batch size mismatch: grn C={C}, binary_tf C={binary_tf.shape[0]}, binary_tg C={binary_tg.shape[0]}"
        )
    if binary_tf.shape[1] != binary_tg.shape[1]:
        raise ValueError(
            f"binary_tf/binary_tg length mismatch: {binary_tf.shape[1]} vs {binary_tg.shape[1]}"
        )

    binary_tf = binary_tf.bool()
    binary_tg = binary_tg.bool()

    tf_counts = binary_tf.sum(dim=1)
    tg_counts = binary_tg.sum(dim=1)
    if not torch.all(tf_counts == TF):
        raise ValueError(
            f"grn TF dim ({TF}) does not match binary_tf true-counts per sample: {tf_counts.tolist()}"
        )
    if not torch.all(tg_counts == TG_sel):
        raise ValueError(
            f"grn TG dim ({TG_sel}) does not match binary_tg true-counts per sample: {tg_counts.tolist()}"
        )

    device = grn.device
    dtype = grn.dtype

    TG = binary_tg.shape[1]
    grn_full = torch.zeros((C, TG, TG), device=device, dtype=dtype)

    # Assign grn[c] into grn_full[c] for each sample
    for c in range(C):
        tf_pos = binary_tf[c].nonzero(as_tuple=True)[0]
        tg_pos = binary_tg[c].nonzero(as_tuple=True)[0]
        grn_full[c][tf_pos[:, None], tg_pos] = grn[c]

    return grn_full


def expand_u(u: torch.Tensor, binary_tf: torch.Tensor):
    """
    Expands u from [C, TF, M] to [C, TG, M] based on the TF indices.
    
    Args:
        u: Factor matrix for TFs of shape (C, TF, M)
        binary_tf: Binary gate/mask for TFs of shape (C, TG)
    """
    if u.ndim != 3:
        raise ValueError(f"u must be (C, TF, M), got {tuple(u.shape)}")
    if binary_tf.ndim != 2:
        raise ValueError(f"binary_tf must be 2-D (C, TG), got {tuple(binary_tf.shape)}")

    C, TF, M = u.shape
    if binary_tf.shape[0] != C:
        raise ValueError(
            f"Batch size mismatch: u C={C}, binary_tf C={binary_tf.shape[0]}"
        )

    binary_tf = binary_tf.bool()
    tf_counts = binary_tf.sum(dim=1)
    if not torch.all(tf_counts == TF):
        raise ValueError(
            f"u TF dim ({TF}) does not match binary_tf true-counts per sample: {tf_counts.tolist()}"
        )

    TG = binary_tf.shape[1] # The total number of genes in the sequence
    device = u.device
    dtype = u.dtype

    # Initialize the full matrix with zeros
    u_full = torch.zeros((C, TG, M), device=device, dtype=dtype)

    # Assign u[c] into u_full[c] for each sample
    for c in range(C):
        tf_pos = binary_tf[c].nonzero(as_tuple=True)[0]
        # We slice u_full at the TF positions and assign the dense factors
        u_full[c, tf_pos, :] = u[c]

    return u_full
