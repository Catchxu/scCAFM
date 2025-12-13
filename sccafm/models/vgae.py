import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()

        # Project scalar gene expression into an E-dimensional embedding
        self.expr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Predict mean and (log) scale of the latent variable z
        self.z_proj = nn.Linear(hidden_dim, 2)
    
    def _check_shape(
            self, 
            grn: torch.Tensor, 
            expr_tf: torch.Tensor, 
            expr_tg: torch.Tensor
    ):
        if grn.dim() != 3:
            raise ValueError(f"grn must be (C, TF, TG), got {grn.shape}!")

        C, TF, TG = grn.shape

        if expr_tf.ndim != 3:
            raise ValueError(f"expr_tf must be (C, TF, E), got {expr_tf.shape}!")
        if expr_tg.ndim != 3:
            raise ValueError(f"expr_tg must be (C, TG, E), got {expr_tg.shape}!")

        if expr_tf.shape[0] != C or expr_tf.shape[1] != TF:
            raise ValueError(
                f"expr_tf first two dims must be (C, TF)=({C}, {TF}), "
                f"got {expr_tf.shape[:2]}"
            )
        if expr_tg.shape[0] != C or expr_tg.shape[1] != TG:
            raise ValueError(
                f"expr_tg first two dims must be (C, TG)=({C}, {TG}), "
                f"got {expr_tg.shape[:2]}"
            )

        if expr_tf.shape[2] != expr_tg.shape[2]:
            raise ValueError(
                f"Embedding dim mismatch: "
                f"expr_tf E={expr_tf.shape[2]}, expr_tg E={expr_tg.shape[2]}"
            )
    
    def forward(self, tokens, grn, binary_tf, binary_tg):
        expr = tokens["expr"]
        expr = expr.unsqueeze(-1)
        expr = self.expr_proj(expr)
        C, _, E = expr.shape
    
        # Select TF and TG embeddings using boolean masks
        # Assumes a fixed number of TFs / TGs per cell
        expr_tf = expr[binary_tf].view(C, -1, E)
        expr_tg = expr[binary_tg].view(C, -1, E)

        self._check_shape(grn, expr_tf, expr_tg)

        # Predict TG embeddings from TF embeddings via the GRN
        # Sum over TF dimension: (C, TF, TG) x (C, TF, E) -> (C, TG, E)
        pred_tg = torch.einsum("cfg,cfe->cge", grn, expr_tf)

        # Encode the residual TG signal into a latent distribution
        out = self.z_proj(expr_tg - pred_tg)

        # Split into mean and scale (parameterized in log-space for stability)
        mu_z, log_sigma_z = out.unbind(dim=-1)
        sigma_z = F.softplus(log_sigma_z) + 1e-6

        return mu_z, sigma_z

