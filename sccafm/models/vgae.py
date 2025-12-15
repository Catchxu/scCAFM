import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


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


class ExprModeling(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.z_proj = nn.Linear(1, hidden_dim)

        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def _expand_grn(self, grn: torch.Tensor, binary_tf, binary_tg):
        """
        Expand sparse GRN (TF x TG) into full (TG x TG) matrix for each condition c.
        """
        C, _, TG = grn.shape
        device = grn.device
        dtype = grn.dtype

        grn_full = torch.zeros((C, TG, TG), device=device, dtype=dtype)

        for c in range(C):
            tf_idx = binary_tf[c]   # indices or boolean mask
            tg_idx = binary_tg[c]

            # Insert TF→TG edges into full TG×TG matrix
            # Shape alignment:
            # grn[c]            -> (TF, TG)
            # grn_full[c][tf_idx][:, tg_idx] -> (TF, TG)
            grn_full[c][tf_idx][:, tg_idx] = grn[c]

        return grn_full

    def _check_shape(self, z, grn_full):
        if z.dim() != 2:
            raise ValueError(f"latent z must be (C, TG), got {z.shape}!")
        
        C, TG = z.shape

        if grn_full.shape != (C, TG, TG):
            raise ValueError(
                f"grn_full must be a expanded GRN (C, TG, TG), got {grn_full.shape}!"
            )

    def forward(self, z: torch.Tensor, grn, binary_tf, binary_tg):
        grn_full = self._expand_grn(grn, binary_tf, binary_tg)
        self._check_shape(z, grn_full)

        z = z.unsqueeze(-1)
        z = self.z_proj(z)

        C, TG, _ = grn_full.shape
        I = torch.eye(TG, device=grn_full.device, dtype=grn_full.dtype)
        I = I.expand(C, TG, TG)

        M = I - grn_full

        # Solves: M @ X = Z  →  X = M^{-1} Z
        out = torch.linalg.solve(M, z)
        out = self.h_proj(out)

        mu_h, log_sigma_h = out.unbind(dim=-1)
        sigma_h = F.softplus(log_sigma_h) + 1e-6

        return mu_h, sigma_h


class DropModeling(nn.Module):
    def __init__(self):
        super().__init__()