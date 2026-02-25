import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import reparameterize, expand_grn, expand_u


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
    
    def _check_shape(self, expr_tf: torch.Tensor, expr_tg: torch.Tensor):
        if expr_tf.ndim != 3:
            raise ValueError(f"expr_tf must be (C, TF, E), got {expr_tf.shape}!")
        if expr_tg.ndim != 3:
            raise ValueError(f"expr_tg must be (C, TG, E), got {expr_tg.shape}!")

        C, TF, _ = expr_tf.shape
        TG = expr_tg.shape[1]
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

    def _predict_tg_from_factors(self, expr_tf: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        # grn ~= U @ V^T, then pred_tg = grn^T @ expr_tf = V @ (U^T @ expr_tf)
        tmp = torch.einsum("cfm,cfe->cme", u, expr_tf)      # (C, M, E)
        pred_tg = torch.einsum("cgm,cme->cge", v, tmp)      # (C, TG, E)
        return pred_tg
    
    def forward(self, x, grn, binary_tf, binary_tg, u=None, v=None):
        x = x.unsqueeze(-1)
        x = self.expr_proj(x)
        C, _, E = x.shape
    
        # Select TF and TG embeddings using boolean masks
        # Assumes a fixed number of TFs / TGs per cell
        expr_tf = x[binary_tf].view(C, -1, E)
        expr_tg = x[binary_tg].view(C, -1, E)

        self._check_shape(expr_tf, expr_tg)

        if u is not None and v is not None:
            pred_tg = self._predict_tg_from_factors(expr_tf, u, v)
        else:
            if grn is None:
                raise ValueError("Either `grn` or (`u`, `v`) must be provided.")
            if grn.dim() != 3:
                raise ValueError(f"grn must be (C, TF, TG), got {grn.shape}!")
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
    def __init__(self, hidden_dim=64, dropout=0.1, fp_steps: int = 3, fp_damping: float = 0.5):
        super().__init__()
        self.z_proj = nn.Linear(1, hidden_dim)
        self.fp_steps = fp_steps
        self.fp_damping = fp_damping
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def _check_shape(self, z, grn_full):
        if z.dim() != 2:
            raise ValueError(f"latent z must be (C, TG), got {z.shape}!")
        
        C, TG = z.shape

        if grn_full.shape != (C, TG, TG):
            raise ValueError(
                f"grn_full must be a expanded GRN (C, TG, TG), got {grn_full.shape}!"
            )

    def _apply_a_with_factors(self, h: torch.Tensor, u_full: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # A @ h where A ~= U_full @ V^T.
        # h: (C, S, H), u_full: (C, S, M), v: (C, S, M)
        vt_h = torch.bmm(v.transpose(1, 2), h)   # (C, M, H)
        return torch.bmm(u_full, vt_h)           # (C, S, H)

    def _solve_with_factors(self, z_proj: torch.Tensor, u: torch.Tensor, v: torch.Tensor, binary_tf: torch.Tensor):
        # Fixed-point iteration for h = z + gamma * A h.
        # This avoids dense SxS matrix construction and solve.
        u_full = expand_u(u, binary_tf)          # (C, S, M)
        h = z_proj
        for _ in range(max(1, self.fp_steps)):
            ah = self._apply_a_with_factors(h, u_full, v)
            h = z_proj + self.fp_damping * ah
        return h

    def _solve_dense_fallback(self, z_proj: torch.Tensor, grn, binary_tf, binary_tg):
        grn_full = expand_grn(grn, binary_tf, binary_tg)
        self._check_shape(z_proj.squeeze(-1), grn_full)  # only checks C,TG dimensions

        C, TG, _ = grn_full.shape
        I = torch.eye(TG, device=grn_full.device, dtype=grn_full.dtype).expand(C, TG, TG)
        M = I - grn_full

        # NOTE: batched LU solve may be unsupported for lower precision dtypes.
        solve_dtype = z_proj.dtype
        out = torch.linalg.solve(M.float(), z_proj.float()).to(solve_dtype)
        return out

    def forward(self, z: torch.Tensor, grn, binary_tf, binary_tg, u=None, v=None):
        z = z.unsqueeze(-1)
        z = self.z_proj(z)  # (C, S, H)

        if u is not None and v is not None:
            out = self._solve_with_factors(z, u, v, binary_tf)
        else:
            out = self._solve_dense_fallback(z, grn, binary_tf, binary_tg)
        out = self.h_proj(out)

        mu_h, log_sigma_h = out.unbind(dim=-1)
        sigma_h = F.softplus(log_sigma_h) + 1e-6

        return mu_h, sigma_h


class DropModeling(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.drop_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h: torch.Tensor):
        h = h.unsqueeze(-1)
        logits = self.drop_proj(h)
        return torch.sigmoid(logits.squeeze(-1))


class VariationalDecoder(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.expr_model = ExprModeling(hidden_dim, dropout)
        self.dropout_model = DropModeling(hidden_dim, dropout)
    
    def forward(
            self, 
            mu_z: torch.Tensor, 
            sigma_z: torch.Tensor,
            grn: torch.Tensor, 
            binary_tf, binary_tg,
            u=None, v=None
    ):
        # Sample latent regulator state z ~ q(z)
        z = reparameterize(mu_z, sigma_z)   # (C, TG)

        # Predict expression distribution p(h | z)
        mu_h, sigma_h = self.expr_model(z, grn, binary_tf, binary_tg, u=u, v=v)
        # Shapes: (C, TG), (C, TG)

        # Sample true expression before dropout
        h = reparameterize(mu_h, sigma_h)   # (C, TG)

        # Predict dropout probabilities p(w = 1 | h)
        p_drop = self.dropout_model(h)      # (C, TG)

        return mu_h, sigma_h, p_drop


class ELBOLoss(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = VariationalEncoder(hidden_dim, dropout)
        self.decoder = VariationalDecoder(hidden_dim, dropout)

    def zinormal_loglik(self, x, mu, sigma, p_drop, eps=1e-8):
        """
        Zero-inflated Gaussian log-likelihood (stable broadcasting)
        """
        normal = torch.distributions.Normal(mu, sigma)
        
        # Compute log-probability of zero for all genes
        log_prob_zero = normal.log_prob(torch.zeros_like(mu))  # (C, TG)

        # log-likelihood (broadcast-safe)
        log_prob = torch.where(
            x == 0,
            torch.log(p_drop + (1 - p_drop) * torch.exp(log_prob_zero) + eps),
            torch.log(1 - p_drop + eps) + normal.log_prob(x)
        )

        return log_prob.sum(dim=-1)

    def kl_normal(self, mu, sigma):
        return 0.5 * torch.sum(
            mu.pow(2) + sigma.pow(2) - 1 - torch.log(sigma.pow(2)),
            dim=-1
        )
    
    def forward(self, tokens, grn, binary_tf, binary_tg, u=None, v=None):
        x = tokens["expr"]
        mu_z, sigma_z = self.encoder(x, grn, binary_tf, binary_tg, u=u, v=v)
        mu_h, sigma_h, p_drop = self.decoder(mu_z, sigma_z, grn, binary_tf, binary_tg, u=u, v=v)

        # Align expression targets with decoder outputs:
        # decoder operates on TG-selected (non-pad) genes, not full padded length.
        C = x.shape[0]
        x = x[binary_tg].view(C, -1)

        # Likelihood term
        log_px = self.zinormal_loglik(x, mu_h, sigma_h, p_drop).mean()

        # KL term
        kl_z = self.kl_normal(mu_z, sigma_z).mean()

        # Negative ELBO (for minimization)
        loss = -log_px + kl_z
        return loss




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd

    from .sfm import SFM
    from ..tokenizer import TomeTokenizer


    adata = sc.read_h5ad("/data1021/xukaichen/data/CTA/pbmc.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")
    tf_dict = pd.read_csv("./resources/human_tfs.csv")
    tf_list = tf_dict["TF"].tolist()

    Ng = 2000
    Nc = 100
    tokenizer = TomeTokenizer(token_dict, simplify=True, max_length=Ng+1, n_top_genes=Ng)
    tokens = tokenizer(adata[:Nc, :].copy())

    model = SFM(token_dict, tf_list=tf_list)
    grn, binary_tf, binary_tg = model(tokens)

    loss_fn = ELBOLoss()
    loss = loss_fn(tokens, grn, binary_tf, binary_tg)
    print("ELBO loss:", loss.item())
