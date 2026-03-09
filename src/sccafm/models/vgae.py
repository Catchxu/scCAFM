import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FactorState, reparameterize, expand_u


class StructureAwareGraphAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (C, S, H), got {tuple(x.shape)}")
        if edge_weight.ndim != 3:
            raise ValueError(
                f"edge_weight must be (C, S, S), got {tuple(edge_weight.shape)}"
            )
        if edge_weight.shape[0] != x.shape[0] or edge_weight.shape[1] != x.shape[1] or edge_weight.shape[2] != x.shape[1]:
            raise ValueError(
                f"Shape mismatch between x={tuple(x.shape)} and edge_weight={tuple(edge_weight.shape)}"
            )

        c, s, _ = x.shape
        q = self.q_proj(x).view(c, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(c, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(c, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        structure = edge_weight.unsqueeze(1)
        scores = scores + torch.log(structure.clamp_min(1e-8))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(c, s, self.hidden_dim)
        return self.out_proj(out)


class GATBlock(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = StructureAwareGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), edge_weight)
        x = x + self.ffn(self.norm2(x))
        return x


class GraphStructureMixin:
    def _masked_select_fixed_count(self, x: torch.Tensor, mask: torch.Tensor, name: str):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.ndim != 2:
            raise ValueError(f"{name} mask must be (C, L), got {tuple(mask.shape)}")
        if x.ndim == 2:
            if x.shape != mask.shape:
                raise ValueError(
                    f"{name}: expected x/mask both (C, L), got {tuple(x.shape)} and {tuple(mask.shape)}"
                )
            c = x.shape[0]
            counts = mask.sum(dim=1)
            if not torch.all(counts == counts[0]):
                raise ValueError(
                    f"{name}: selected counts must be identical across batch for dense batching, got {counts.tolist()}"
                )
            s = int(counts[0].item())
            return x[mask].view(c, s)

        if x.ndim == 3 and x.shape[:2] == mask.shape:
            c, _, e = x.shape
            counts = mask.sum(dim=1)
            if not torch.all(counts == counts[0]):
                raise ValueError(
                    f"{name}: selected counts must be identical across batch for dense batching, got {counts.tolist()}"
                )
            s = int(counts[0].item())
            return x[mask].view(c, s, e)

        raise ValueError(
            f"{name}: unsupported x/mask shapes {tuple(x.shape)} and {tuple(mask.shape)}"
        )

    def _build_selected_structure(
        self,
        binary_tf: torch.Tensor,
        binary_tg: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        binary_tf_sel = self._masked_select_fixed_count(
            binary_tf.float(), binary_tg, "binary_tf/binary_tg"
        )
        u_full = expand_u(u, binary_tf_sel)
        edge_weight = torch.bmm(u_full, v.transpose(1, 2))

        s = edge_weight.shape[-1]
        eye = torch.eye(s, device=edge_weight.device, dtype=edge_weight.dtype).unsqueeze(0)
        edge_weight = edge_weight + eye
        edge_weight = edge_weight / edge_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return edge_weight


class VariationalEncoder(GraphStructureMixin, nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2):
        super().__init__()

        self.expr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GATBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)

        self.z_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x, binary_tf, binary_tg, u, v):
        x = self.expr_proj(x.unsqueeze(-1))
        x = self._masked_select_fixed_count(x, binary_tg, "binary_tg")
        edge_weight = self._build_selected_structure(binary_tf, binary_tg, u, v)

        for layer in self.layers:
            x = layer(x, edge_weight)
        out = self.z_proj(self.norm(x))

        mu_z, log_sigma_z = out.unbind(dim=-1)
        sigma_z = F.softplus(log_sigma_z) + 1e-6

        return mu_z, sigma_z


class ExprModeling(GraphStructureMixin, nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2):
        super().__init__()
        self.z_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [GATBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor, binary_tf, binary_tg, u, v):
        x = self.z_proj(z.unsqueeze(-1))
        edge_weight = self._build_selected_structure(binary_tf, binary_tg, u, v)
        for layer in self.layers:
            x = layer(x, edge_weight)
        out = self.h_proj(self.norm(x))

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
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2):
        super().__init__()
        self.expr_model = ExprModeling(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.dropout_model = DropModeling(hidden_dim, dropout)
    
    def forward(
            self, 
            mu_z: torch.Tensor, 
            sigma_z: torch.Tensor,
            binary_tf, binary_tg,
            u, v
    ):
        # Sample latent regulator state z ~ q(z)
        z = reparameterize(mu_z, sigma_z)   # (C, TG)

        # Predict expression distribution p(h | z)
        mu_h, sigma_h = self.expr_model(z, binary_tf, binary_tg, u=u, v=v)
        # Shapes: (C, TG), (C, TG)

        # Sample true expression before dropout
        h = reparameterize(mu_h, sigma_h)   # (C, TG)

        # Predict dropout probabilities p(w = 1 | h)
        p_drop = self.dropout_model(h)      # (C, TG)

        return mu_h, sigma_h, p_drop


class ELBOLoss(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        dropout=0.1,
        recon_reduction: str = "sum",
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = VariationalEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = VariationalDecoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        if recon_reduction not in {"mean", "sum"}:
            raise ValueError(f"recon_reduction must be 'mean' or 'sum', got {recon_reduction}")
        self.recon_reduction = recon_reduction

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

        if self.recon_reduction == "mean":
            # Normalize reconstruction scale by selected TG count.
            return log_prob.mean(dim=-1)
        return log_prob.sum(dim=-1)

    def kl_normal(self, mu, sigma):
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
        c = x.shape[0]
        s = int(counts[0].item())
        return x[mask].view(c, s)
    
    def forward(self, tokens, factors: FactorState = None):
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()
        binary_tf = factors.binary_tf
        binary_tg = factors.binary_tg
        u = factors.u
        v = factors.v
        x = tokens["expr"]
        mu_z, sigma_z = self.encoder(x, binary_tf, binary_tg, u=u, v=v)
        mu_h, sigma_h, p_drop = self.decoder(mu_z, sigma_z, binary_tf, binary_tg, u=u, v=v)

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
