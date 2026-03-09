import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FactorState, RMSNorm, reparameterize


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

    def _prepare_selected_state(
        self,
        x: torch.Tensor,
        binary_tf: torch.Tensor,
        binary_tg: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ):
        x_tg = self._masked_select_fixed_count(x, binary_tg, "binary_tg")
        binary_tf_sel = self._masked_select_fixed_count(
            binary_tf.float(), binary_tg, "binary_tf/binary_tg"
        ).bool()

        tf_counts = binary_tf_sel.sum(dim=1)
        if not torch.all(tf_counts == u.shape[1]):
            raise ValueError(
                f"u TF dim ({u.shape[1]}) does not match selected binary_tf true-counts: {tf_counts.tolist()}"
            )
        if v.shape[0] != x_tg.shape[0] or v.shape[1] != x_tg.shape[1]:
            raise ValueError(
                f"v shape {tuple(v.shape)} incompatible with selected TG embeddings {tuple(x_tg.shape)}"
            )

        return x_tg, binary_tf_sel


class StructureAwareSparseAttention(nn.Module):
    """
    Sparse GAT over TF->TG edges.
    Queries are all selected TG nodes, while keys/values are TF-selected nodes only.
    """

    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1, struct_scale=1.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.struct_scale = float(struct_scale)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_tg: torch.Tensor,
        x_tf: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if x_tg.ndim != 3 or x_tf.ndim != 3:
            raise ValueError(
                f"x_tg/x_tf must be 3-D, got {tuple(x_tg.shape)} and {tuple(x_tf.shape)}"
            )

        c, s, _ = x_tg.shape
        tf = x_tf.shape[1]
        if tf == 0:
            return torch.zeros_like(x_tg)
        if u.shape[:2] != (c, tf):
            raise ValueError(
                f"u shape {tuple(u.shape)} incompatible with TF state {tuple(x_tf.shape)}"
            )
        if v.shape[:2] != (c, s):
            raise ValueError(
                f"v shape {tuple(v.shape)} incompatible with TG state {tuple(x_tg.shape)}"
            )

        q = self.q_proj(x_tg).view(c, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_tf).view(c, tf, self.num_heads, self.head_dim).transpose(1, 2)
        vv = self.v_proj(x_tf).view(c, tf, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention logits in fp32 for numerical stability under AMP/bf16.
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        struct_scores = torch.einsum("ctm,csm->cst", u.float(), v.float()).clamp_min(1e-6)
        struct_scores = struct_scores / struct_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = scores + self.struct_scale * torch.log(struct_scores).unsqueeze(1)

        attn = F.softmax(scores, dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vv)
        out = out.transpose(1, 2).reshape(c, s, self.hidden_dim)
        return self.out_proj(out)


class SparseGATBlock(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1, struct_scale=1.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = StructureAwareSparseAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            struct_scale=struct_scale,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x_tg: torch.Tensor,
        x_tf: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        x_tg = x_tg + self.attn(self.norm1(x_tg), x_tf, u, v)
        x_tg = x_tg + self.ffn(self.norm2(x_tg))
        return x_tg


class VariationalEncoder(GraphStructureMixin, nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2, struct_scale=1.0):
        super().__init__()
        self.expr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                SparseGATBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    struct_scale=struct_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x, binary_tf, binary_tg, u, v):
        x = self.expr_proj(x.unsqueeze(-1))
        x_tg, binary_tf_sel = self._prepare_selected_state(x, binary_tf, binary_tg, u, v)

        for layer in self.layers:
            x_tf = self._masked_select_fixed_count(x_tg, binary_tf_sel, "x_tg/binary_tf_sel")
            x_tg = layer(x_tg, x_tf, u, v)

        out = self.z_proj(self.norm(x_tg))
        mu_z, log_sigma_z = out.unbind(dim=-1)
        sigma_z = F.softplus(log_sigma_z) + 1e-6
        return mu_z, sigma_z


class ExprModeling(GraphStructureMixin, nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2, struct_scale=1.0):
        super().__init__()
        self.z_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [
                SparseGATBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    struct_scale=struct_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_dim)
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor, binary_tf, binary_tg, u, v):
        x_tg = self.z_proj(z.unsqueeze(-1))
        # z is already in TG-selected space from encoder output (C, S_sel).
        # So we should not mask by binary_tg again here.
        binary_tf_sel = self._masked_select_fixed_count(
            binary_tf.float(), binary_tg, "binary_tf/binary_tg"
        ).bool()
        tf_counts = binary_tf_sel.sum(dim=1)
        if not torch.all(tf_counts == u.shape[1]):
            raise ValueError(
                f"u TF dim ({u.shape[1]}) does not match selected binary_tf true-counts: {tf_counts.tolist()}"
            )
        if v.shape[0] != x_tg.shape[0] or v.shape[1] != x_tg.shape[1]:
            raise ValueError(
                f"v shape {tuple(v.shape)} incompatible with TG-selected decoder state {tuple(x_tg.shape)}"
            )

        for layer in self.layers:
            x_tf = self._masked_select_fixed_count(x_tg, binary_tf_sel, "x_tg/binary_tf_sel")
            x_tg = layer(x_tg, x_tf, u, v)

        out = self.h_proj(self.norm(x_tg))
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor):
        h = h.unsqueeze(-1)
        logits = self.drop_proj(h)
        return torch.sigmoid(logits.squeeze(-1))


class VariationalDecoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, num_heads=4, num_layers=2, struct_scale=1.0):
        super().__init__()
        self.expr_model = ExprModeling(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
            struct_scale=struct_scale,
        )
        self.dropout_model = DropModeling(hidden_dim, dropout)

    def forward(self, mu_z: torch.Tensor, sigma_z: torch.Tensor, binary_tf, binary_tg, u, v):
        z = reparameterize(mu_z, sigma_z)
        mu_h, sigma_h = self.expr_model(z, binary_tf, binary_tg, u=u, v=v)
        h = reparameterize(mu_h, sigma_h)
        p_drop = self.dropout_model(h)
        return mu_h, sigma_h, p_drop


class ELBOLoss(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        dropout=0.1,
        recon_reduction: str = "sum",
        num_heads: int = 4,
        num_layers: int = 2,
        struct_scale: float = 1.0,
    ):
        super().__init__()
        self.encoder = VariationalEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
            struct_scale=struct_scale,
        )
        self.decoder = VariationalDecoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
            struct_scale=struct_scale,
        )
        if recon_reduction not in {"mean", "sum"}:
            raise ValueError(f"recon_reduction must be 'mean' or 'sum', got {recon_reduction}")
        self.recon_reduction = recon_reduction

    def zinormal_loglik(self, x, mu, sigma, p_drop, eps=1e-8):
        sigma = sigma.float().clamp_min(1e-5)
        p_drop = p_drop.float().clamp(min=1e-5, max=1.0 - 1e-5)
        x = x.float()
        mu = mu.float()
        normal = torch.distributions.Normal(mu, sigma)
        log_prob_zero = normal.log_prob(torch.zeros_like(mu))
        log_prob = torch.where(
            x == 0,
            torch.log(p_drop + (1 - p_drop) * torch.exp(log_prob_zero) + eps),
            torch.log(1 - p_drop + eps) + normal.log_prob(x),
        )
        if self.recon_reduction == "mean":
            return log_prob.mean(dim=-1)
        return log_prob.sum(dim=-1)

    def kl_normal(self, mu, sigma):
        mu = mu.float()
        sigma = sigma.float().clamp_min(1e-5)
        return 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - 1 - torch.log(sigma.pow(2)), dim=-1)

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
        x = self._masked_select_fixed_count(x, binary_tg, "tokens['expr']/binary_tg")

        log_px = self.zinormal_loglik(x, mu_h, sigma_h, p_drop).mean()
        kl_z = self.kl_normal(mu_z, sigma_z).mean()
        return -log_px + kl_z


if __name__ == "__main__":
    import pandas as pd
    import scanpy as sc

    from .sfm import SFM
    from ..tokenizer import TomeTokenizer

    adata = sc.read_h5ad("/data1021/xukaichen/data/CTA/pbmc.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")
    tf_dict = pd.read_csv("./resources/human_tfs.csv")
    tf_list = tf_dict["TF"].tolist()

    ng = 2000
    nc = 32
    tokenizer = TomeTokenizer(token_dict, max_length=ng + 1, n_top_genes=ng)
    tokens = tokenizer(adata[:nc, :].copy())

    model = SFM(token_dict, tf_list=tf_list)
    _, factors = model(tokens, return_factors=True, compute_grn=False)

    loss_fn = ELBOLoss()
    loss = loss_fn(tokens, factors=factors)
    print("ELBO loss:", float(loss.item()))
