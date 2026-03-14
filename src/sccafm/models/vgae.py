import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FactorState, reparameterize, expand_u


class VariationalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
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

    def _predict_tg_from_factors(
        self,
        expr_tf: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        u_score: torch.Tensor,
        v_score: torch.Tensor,
    ):
        # Structure from u/v (support), strength from free scores.
        u_eff = u * u_score
        v_eff = v * v_score
        tmp = torch.einsum("cfm,cfe->cme", u_eff, expr_tf)      # (C, M, E)
        pred_tg = torch.einsum("cgm,cme->cge", v_eff, tmp)      # (C, TG, E)
        return pred_tg

    def _select_fixed_count(self, x: torch.Tensor, mask: torch.Tensor, name: str):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.ndim != 2:
            raise ValueError(f"{name} mask must be (C, L), got {tuple(mask.shape)}")
        if x.ndim != 3 or x.shape[:2] != mask.shape:
            raise ValueError(
                f"Shape mismatch for {name}: x={tuple(x.shape)}, mask={tuple(mask.shape)}; expected x=(C,L,E), mask=(C,L)"
            )

        counts = mask.sum(dim=1)
        if not torch.all(counts == counts[0]):
            raise ValueError(
                f"{name} selected counts must be identical across batch for dense batching, got {counts.tolist()}"
            )

        C, _, E = x.shape
        S = int(counts[0].item())
        return x[mask].view(C, S, E)
    
    def forward(self, x, binary_tf, binary_tg, u, v, u_score, v_score):
        x = x.unsqueeze(-1)
        x = self.expr_proj(x)
        C, _, _ = x.shape
    
        # Select TF and TG embeddings using boolean masks
        # Assumes a fixed number of TFs / TGs per cell
        expr_tf = self._select_fixed_count(x, binary_tf, "binary_tf")
        expr_tg = self._select_fixed_count(x, binary_tg, "binary_tg")

        self._check_shape(expr_tf, expr_tg)

        if u.shape[:2] != expr_tf.shape[:2]:
            raise ValueError(
                f"u shape {tuple(u.shape)} incompatible with selected TF embeddings {tuple(expr_tf.shape)}"
            )
        if v.shape[0] != C or v.shape[1] != expr_tg.shape[1]:
            raise ValueError(
                f"v shape {tuple(v.shape)} incompatible with selected TG embeddings {tuple(expr_tg.shape)}"
            )
        pred_tg = self._predict_tg_from_factors(expr_tf, u, v, u_score, v_score)

        # Encode the residual TG signal into a latent distribution
        out = self.z_proj(expr_tg - pred_tg)

        # Split into mean and scale (parameterized in log-space for stability)
        mu_z, log_sigma_z = out.unbind(dim=-1)
        sigma_z = F.softplus(log_sigma_z) + 1e-6

        return mu_z, sigma_z


class ExprModeling(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, fp_steps: int = 3, fp_damping: float = 0.5):
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

    def _apply_a_with_factors(self, h: torch.Tensor, u_full: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # A^T @ h where A ~= U_full @ V^T and A[src, tgt] = edge src->tgt.
        # This keeps SEM-style target update aligned with source->target convention.
        # h: (C, S, H), u_full: (C, S, M), v: (C, S, M)
        ut_h = torch.bmm(u_full.transpose(1, 2), h)  # (C, M, H)
        return torch.bmm(v, ut_h)                    # (C, S, H)

    def _masked_select_fixed_count(self, x: torch.Tensor, mask: torch.Tensor, name: str):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if x.ndim != 2 or mask.ndim != 2 or x.shape != mask.shape:
            raise ValueError(
                f"{name}: expected x/mask both (C, L), got {tuple(x.shape)} and {tuple(mask.shape)}"
            )
        counts = mask.sum(dim=1)
        if not torch.all(counts == counts[0]):
            raise ValueError(
                f"{name}: selected counts must be identical across batch for dense batching, got {counts.tolist()}"
            )
        C = x.shape[0]
        S = int(counts[0].item())
        return x[mask].view(C, S)

    def _solve_with_factors(
        self,
        z_proj: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        u_score: torch.Tensor,
        v_score: torch.Tensor,
        binary_tf: torch.Tensor,
        binary_tg: torch.Tensor,
    ):
        # Fixed-point iteration for h = z + gamma * A h.
        # This avoids dense SxS matrix construction and solve.
        binary_tf_sel = self._masked_select_fixed_count(binary_tf.to(u.dtype), binary_tg, "binary_tf/binary_tg")
        u_full = expand_u(u * u_score, binary_tf_sel)      # (C, S, M), S matches selected TG space
        v_eff = v * v_score
        h = z_proj
        for _ in range(max(1, self.fp_steps)):
            ah = self._apply_a_with_factors(h, u_full, v_eff)
            h = z_proj + self.fp_damping * ah
        return h

    def forward(self, z: torch.Tensor, binary_tf, binary_tg, u, v, u_score, v_score):
        z = z.unsqueeze(-1)
        z = self.z_proj(z)  # (C, S, H)
        out = self._solve_with_factors(z, u, v, u_score, v_score, binary_tf, binary_tg)
        out = self.h_proj(out)

        mu_h, log_sigma_h = out.unbind(dim=-1)
        sigma_h = F.softplus(log_sigma_h) + 1e-6

        return mu_h, sigma_h


class DropModeling(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
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
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.expr_model = ExprModeling(hidden_dim, dropout)
        self.dropout_model = DropModeling(hidden_dim, dropout)
    
    def forward(
        self, 
        mu_z: torch.Tensor, 
        sigma_z: torch.Tensor,
        binary_tf, binary_tg,
        u, v, u_score, v_score
    ):
        # Sample latent regulator state z ~ q(z)
        z = reparameterize(mu_z, sigma_z)   # (C, TG)

        # Predict expression distribution p(h | z)
        mu_h, sigma_h = self.expr_model(
            z,
            binary_tf,
            binary_tg,
            u=u,
            v=v,
            u_score=u_score,
            v_score=v_score,
        )
        # Shapes: (C, TG), (C, TG)

        # Sample true expression before dropout
        h = reparameterize(mu_h, sigma_h)   # (C, TG)

        # Predict dropout probabilities p(w = 1 | h)
        p_drop = self.dropout_model(h)      # (C, TG)

        return mu_h, sigma_h, p_drop


class ELBOLoss(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, recon_reduction: str = "sum"):
        super().__init__()
        self.encoder = VariationalEncoder(hidden_dim, dropout)
        self.decoder = VariationalDecoder(hidden_dim, dropout)
        if recon_reduction not in {"mean", "sum"}:
            raise ValueError(f"recon_reduction must be 'mean' or 'sum', got {recon_reduction}")
        self.recon_reduction = recon_reduction

    def zinormal_loglik(self, x, mu, sigma, p_drop, eps=1e-8):
        """
        Zero-inflated Gaussian log-likelihood (stable broadcasting)
        """
        x = x.to(mu.dtype)
        sigma = sigma.clamp_min(1e-5)
        p_drop = p_drop.clamp(min=1e-5, max=1.0 - 1e-5)
        normal = torch.distributions.Normal(mu, sigma)
        
        # Compute log-probability of zero for all genes
        log_prob_zero = normal.log_prob(torch.zeros_like(mu))  # (C, TG)

        # log-likelihood (broadcast-safe)
        log_prob = torch.where(
            x == 0,
            torch.log(p_drop + (1 - p_drop) * torch.exp(log_prob_zero) + eps),
            torch.log(1 - p_drop + eps) + normal.log_prob(x)
        )

        if self.recon_reduction == "mean":
            # Normalize reconstruction scale by selected TG count.
            return log_prob.mean(dim=-1)
        return log_prob.sum(dim=-1)

    def kl_normal(self, mu, sigma):
        sigma = sigma.clamp_min(1e-5)
        return 0.5 * torch.sum(
            mu.pow(2) + sigma.pow(2) - 1 - torch.log(sigma.pow(2)),
            dim=-1
        )

    def _masked_select_fixed_count(self, x: torch.Tensor, mask: torch.Tensor, name: str):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if x.ndim != 2 or mask.ndim != 2 or x.shape != mask.shape:
            raise ValueError(
                f"{name}: expected x/mask both (C, L), got {tuple(x.shape)} and {tuple(mask.shape)}"
            )
        counts = mask.sum(dim=1)
        if not torch.all(counts == counts[0]):
            raise ValueError(
                f"{name}: selected counts must be identical across batch for dense batching, got {counts.tolist()}"
            )
        C = x.shape[0]
        S = int(counts[0].item())
        return x[mask].view(C, S)
    
    def forward(self, tokens, factors: FactorState = None):
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()
        binary_tf = factors.binary_tf
        binary_tg = factors.binary_tg
        u = factors.u
        v = factors.v
        u_score = factors.u_score if factors.u_score is not None else torch.ones_like(u)
        v_score = factors.v_score if factors.v_score is not None else torch.ones_like(v)
        x = tokens["expr"]
        mu_z, sigma_z = self.encoder(
            x,
            binary_tf,
            binary_tg,
            u=u,
            v=v,
            u_score=u_score,
            v_score=v_score,
        )
        mu_h, sigma_h, p_drop = self.decoder(
            mu_z,
            sigma_z,
            binary_tf,
            binary_tg,
            u=u,
            v=v,
            u_score=u_score,
            v_score=v_score,
        )

        # Align expression targets with decoder outputs:
        # decoder operates on TG-selected (non-pad) genes, not full padded length.
        x = self._masked_select_fixed_count(x, binary_tg, "tokens['expr']/binary_tg")

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
    tokenizer = TomeTokenizer(token_dict, max_length=Ng+1, n_top_genes=Ng)
    tokens = tokenizer(adata[:Nc, :].copy())

    model = SFM(token_dict, tf_list=tf_list)
    _, factors = model(tokens, return_factors=True, compute_grn=False)

    loss_fn = ELBOLoss()
    loss = loss_fn(tokens, factors=factors)
    print("ELBO loss:", loss.item())
