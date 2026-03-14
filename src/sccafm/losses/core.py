import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..tokenizer import TomeTokenizer
from ..models.utils import FactorState, expand_u
from .vgae import ELBOLoss


class PriorLoss(nn.Module):
    def __init__(
        self,
        tome_tokenizer: TomeTokenizer,
        true_grn_df: Optional[pd.DataFrame] = None,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        neg_sample_ratio: Optional[float] = None,
    ):
        super().__init__()
        gt = tome_tokenizer.gene_tokenizer
        self.symbol2id = gt.symbol2id
        self.id2id = gt.id2id
        self.pad_index = gt.pad_index
        token_max_1 = 1
        if len(self.symbol2id) > 0:
            token_max_1 = max(token_max_1, max(int(v) for v in self.symbol2id.values()) + 1)
        if len(self.id2id) > 0:
            token_max_1 = max(token_max_1, max(int(v) for v in self.id2id.values()) + 1)
        self._pair_key_base = int(token_max_1)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.neg_sample_ratio = None if neg_sample_ratio is None else float(neg_sample_ratio)
        self._cached_src_ids = None
        self._cached_tgt_ids = None
        self._cached_prior_ref = None
        if true_grn_df is not None:
            self._prepare_prior_cache(true_grn_df)

    def _map_gene_to_token(self, name: str) -> Optional[int]:
        s = str(name)
        if s.startswith("ENSG") and s in self.id2id:
            return int(self.id2id[s])
        if s in self.symbol2id:
            return int(self.symbol2id[s])
        if s in self.id2id:
            return int(self.id2id[s])
        return None

    def _prepare_prior_cache(self, true_grn: pd.DataFrame):
        if not {"Gene1", "Gene2"}.issubset(true_grn.columns):
            raise ValueError("true_grn_df must contain columns: Gene1, Gene2")

        src_ids = []
        tgt_ids = []
        for g1, g2 in zip(true_grn["Gene1"].tolist(), true_grn["Gene2"].tolist()):
            s = self._map_gene_to_token(g1)
            t = self._map_gene_to_token(g2)
            if s is None or t is None:
                continue
            src_ids.append(s)
            tgt_ids.append(t)

        if len(src_ids) == 0:
            self._cached_src_ids = torch.empty(0, dtype=torch.long)
            self._cached_tgt_ids = torch.empty(0, dtype=torch.long)
        else:
            pairs = torch.tensor(list(zip(src_ids, tgt_ids)), dtype=torch.long)
            pairs = torch.unique(pairs, dim=0)
            self._cached_src_ids = pairs[:, 0].contiguous()
            self._cached_tgt_ids = pairs[:, 1].contiguous()

        self._cached_prior_ref = true_grn

    def _filter_prior_pairs_by_used_genes(self, gene_tokens: torch.Tensor):
        device = gene_tokens.device
        if self._cached_src_ids is None or self._cached_tgt_ids is None:
            raise RuntimeError("Prior cache is empty. Call with true_grn_df at least once.")
        if self._cached_src_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        used_ids = torch.unique(gene_tokens)
        src_ids = self._cached_src_ids.to(device=device)
        tgt_ids = self._cached_tgt_ids.to(device=device)
        keep = torch.isin(src_ids, used_ids) & torch.isin(tgt_ids, used_ids)
        return src_ids[keep], tgt_ids[keep]

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

    def forward(self, tokens, factors: FactorState = None, true_grn_df: Optional[pd.DataFrame] = None):
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()
        binary_tf = factors.binary_tf
        binary_tg = factors.binary_tg
        u = factors.u
        v = factors.v
        u_score = factors.u_score if factors.u_score is not None else torch.ones_like(u)
        v_score = factors.v_score if factors.v_score is not None else torch.ones_like(v)
        gene_tokens = tokens["gene"]

        gene_tokens = self._masked_select_fixed_count(gene_tokens, binary_tg, "gene_tokens/binary_tg")
        model_tf_mask = self._masked_select_fixed_count(binary_tf.to(u.dtype), binary_tg, "binary_tf/binary_tg")

        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)

        filt_src_ids, filt_tgt_ids = self._filter_prior_pairs_by_used_genes(gene_tokens)
        if filt_src_ids.numel() == 0:
            return u.new_zeros(())

        u_eff = u * u_score
        v_eff = v * v_score
        tf_mask = model_tf_mask.bool()
        u_full = expand_u(u_eff, tf_mask)
        src_unique = torch.unique(filt_src_ids)
        tgt_unique = torch.unique(filt_tgt_ids)

        edge_raw = torch.bmm(u_full, v_eff.transpose(1, 2))
        edge_logit = torch.log(edge_raw.abs() + 1e-8)

        src_assoc = torch.isin(gene_tokens, src_unique) & tf_mask
        tgt_assoc = torch.isin(gene_tokens, tgt_unique)
        supervise_mask = torch.einsum("bi,bj->bij", src_assoc, tgt_assoc).bool()

        C, S = gene_tokens.shape
        P = int(filt_src_ids.numel())
        target = torch.zeros_like(edge_logit)

        sorted_tokens, sort_idx = torch.sort(gene_tokens, dim=1)
        src_vals = filt_src_ids.view(1, -1).expand(C, -1).contiguous()
        tgt_vals = filt_tgt_ids.view(1, -1).expand(C, -1).contiguous()

        src_loc = torch.searchsorted(sorted_tokens, src_vals)
        src_in = src_loc < S
        src_loc_safe = src_loc.clamp(max=max(S - 1, 0))
        src_hit = src_in & (sorted_tokens.gather(1, src_loc_safe) == src_vals)
        src_pos = sort_idx.gather(1, src_loc_safe)

        tgt_loc = torch.searchsorted(sorted_tokens, tgt_vals)
        tgt_in = tgt_loc < S
        tgt_loc_safe = tgt_loc.clamp(max=max(S - 1, 0))
        tgt_hit = tgt_in & (sorted_tokens.gather(1, tgt_loc_safe) == tgt_vals)
        tgt_pos = sort_idx.gather(1, tgt_loc_safe)

        src_tf_valid = tf_mask.gather(1, src_pos)
        pair_valid = src_hit & tgt_hit & src_tf_valid
        if pair_valid.any():
            b_idx = torch.arange(C, device=gene_tokens.device).view(C, 1).expand(C, P)
            target[b_idx[pair_valid], src_pos[pair_valid], tgt_pos[pair_valid]] = 1.0

        loss = F.binary_cross_entropy_with_logits(edge_logit, target, reduction="none")
        weight = target * self.pos_weight + (1.0 - target) * self.neg_weight

        if self.neg_sample_ratio is not None and self.neg_sample_ratio > 0:
            sampled_mask = torch.zeros_like(supervise_mask, dtype=torch.bool)
            for b in range(C):
                mask_b = supervise_mask[b]
                pos_mask = (target[b] > 0.5) & mask_b
                neg_mask = (~(target[b] > 0.5)) & mask_b
                pos_count = int(pos_mask.sum().item())
                neg_count = int(neg_mask.sum().item())
                if pos_count > 0 and neg_count > 0:
                    sample_k = min(neg_count, int(pos_count * self.neg_sample_ratio))
                    if sample_k > 0:
                        neg_idx = torch.nonzero(neg_mask, as_tuple=False)
                        perm = torch.randperm(neg_idx.shape[0], device=neg_idx.device)[:sample_k]
                        chosen = neg_idx[perm]
                        sampled_neg = torch.zeros_like(neg_mask)
                        sampled_neg[chosen[:, 0], chosen[:, 1]] = True
                        sampled_mask[b] = pos_mask | sampled_neg
                    else:
                        sampled_mask[b] = pos_mask
                else:
                    sampled_mask[b] = mask_b
            supervise_mask_f = sampled_mask.to(loss.dtype)
        else:
            supervise_mask_f = supervise_mask.to(loss.dtype)

        numer = (loss * weight * supervise_mask_f).sum()
        denom = (weight * supervise_mask_f).sum()
        return numer / (denom + 1e-8)


class DAGLoss(nn.Module):
    def __init__(self, alpha: float = 0.0, rho: float = 0.1, rho_max: float = 1e6, update_period: int = 100):
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self.rho_max = rho_max
        self.prev_h_val = float("inf")
        self.update_period = update_period
        self.step_counter = 0
        self.accumulated_h = 0.0

    def _compute_dag_constraint(self, adj):
        device = adj.device
        batch_size, M, _ = adj.shape
        adj_sq = adj * adj
        eye = torch.eye(M, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        matrix_poly = eye + adj_sq / M
        res = torch.matrix_power(matrix_poly, M)
        h = torch.diagonal(res, dim1=-2, dim2=-1).sum(-1) - M
        return h

    def _auto_update_params(self):
        avg_h = self.accumulated_h / self.update_period
        if avg_h > 0.25 * self.prev_h_val:
            self.rho = min(self.rho * 10.0, self.rho_max)
        else:
            self.alpha += self.rho * avg_h
        self.prev_h_val = avg_h
        self.accumulated_h = 0.0
        self.step_counter = 0

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

    def forward(self, factors: FactorState):
        factors.validate()
        u = factors.u
        v = factors.v
        binary_tf = factors.binary_tf
        binary_tg = factors.binary_tg
        if binary_tg is not None:
            binary_tf = self._masked_select_fixed_count(binary_tf.to(u.dtype), binary_tg, "binary_tf/binary_tg")
        u_full = expand_u(u, binary_tf)
        adj_factor = torch.bmm(v.transpose(1, 2), u_full)

        dag_h_batch = self._compute_dag_constraint(adj_factor).mean()
        if self.training:
            self.step_counter += 1
            self.accumulated_h += dag_h_batch.item()
            if self.step_counter >= self.update_period:
                self._auto_update_params()

        loss = self.alpha * dag_h_batch + (self.rho / 2) * (dag_h_batch ** 2)
        return loss


class SFMLoss(nn.Module):
    def __init__(
        self,
        use_prior: bool = True,
        use_dag: bool = True,
        tome_tokenizer: Optional[TomeTokenizer] = None,
        true_grn_df: Optional[pd.DataFrame] = None,
        num_epochs: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.use_prior = use_prior
        self.use_dag = use_dag
        self.T = num_epochs
        self.current_epoch = 0

        self.elbo_criterion = ELBOLoss(
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout=kwargs.get("dropout", 0.1),
            recon_reduction=kwargs.get("recon_reduction", "mean"),
        )

        if self.use_prior:
            if tome_tokenizer is None or true_grn_df is None:
                raise ValueError(
                    "Both 'tome_tokenizer' and 'true_grn_df' must be provided if use_prior=True"
                )
            self.prior_criterion = PriorLoss(
                tome_tokenizer,
                true_grn_df=true_grn_df,
                pos_weight=kwargs.get("prior_pos_weight", 1.0),
                neg_weight=kwargs.get("prior_neg_weight", 1.0),
                neg_sample_ratio=kwargs.get("prior_neg_sample_ratio", None),
            )
            self.true_grn_df = true_grn_df

        if self.use_dag:
            self.dag_criterion = DAGLoss(
                alpha=kwargs.get("alpha", 0.0),
                rho=kwargs.get("rho", 0.01),
                rho_max=kwargs.get("rho_max", 1e6),
                update_period=kwargs.get("update_period", 100)
            )

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_prior_weight(self) -> float:
        if self.T is None or self.T == 0:
            return 1.0
        t_eff = min(self.current_epoch, self.T)
        return 0.5 + 0.5 * np.cos(np.pi * t_eff / self.T)

    def forward(self, tokens, factors: FactorState = None):
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()

        total_loss = 0.0
        loss_dict = {}

        loss_elbo = self.elbo_criterion(tokens, factors=factors)
        total_loss += loss_elbo
        loss_dict["elbo"] = loss_elbo.item()

        if self.use_prior:
            w_p = self.get_prior_weight()
            loss_p = self.prior_criterion(tokens, factors=factors, true_grn_df=self.true_grn_df)
            total_loss += w_p * loss_p
            loss_dict["prior"] = loss_p.item()

        if self.use_dag:
            loss_dag = self.dag_criterion(factors)
            total_loss += loss_dag
            loss_dict["dag"] = loss_dag.item()

        return total_loss, loss_dict
