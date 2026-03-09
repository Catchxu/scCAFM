import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

from .tokenizer import TomeTokenizer
from .models.vgae import ELBOLoss
from .models.utils import FactorState, expand_u


class PriorLoss(nn.Module):
    def __init__(
        self,
        tome_tokenizer: TomeTokenizer,
        true_grn_df: Optional[pd.DataFrame] = None,
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
        self._cached_src_ids = None
        self._cached_tgt_ids = None
        self._cached_prior_ref = None
        if true_grn_df is not None:
            self._prepare_prior_cache(true_grn_df)

    def _binary_cross_entropy_prob(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        prob = prob.float().clamp(eps, 1.0 - eps)
        target = target.float()
        return -(target * torch.log(prob) + (1.0 - target) * torch.log1p(-prob))

    def _map_gene_to_token(self, name: str) -> Optional[int]:
        """
        Robust mapping: try ENSG/id map first when applicable, then symbol map fallback.
        """
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

    def _build_gt_matrix(self, gene_tokens: torch.Tensor):
        """
        Args:
            gene_tokens: (B, L)
        """
        C, S = gene_tokens.shape
        device = gene_tokens.device

        if self._cached_src_ids is None or self._cached_tgt_ids is None:
            raise RuntimeError("Prior cache is empty. Call with true_grn_df at least once.")
        if self._cached_src_ids.numel() == 0:
            return torch.zeros((C, S, S), device=device)

        src_ids = self._cached_src_ids.to(device=device)
        tgt_ids = self._cached_tgt_ids.to(device=device)
        target = torch.zeros((C, S, S), device=device)

        # Build one sample at a time to avoid allocating (B, S, P) tensors.
        for b in range(C):
            src_matches = (gene_tokens[b].unsqueeze(-1) == src_ids.view(1, -1)).float()  # (S, P)
            tgt_matches = (gene_tokens[b].unsqueeze(-1) == tgt_ids.view(1, -1)).float()  # (S, P)
            target[b] = (src_matches @ tgt_matches.transpose(0, 1) > 0).float()

        return target

    def _filter_prior_pairs_by_used_genes(self, gene_tokens: torch.Tensor):
        """
        Explicitly filter cached prior edges to genes used in this batch.

        Args:
            gene_tokens: (B, S) selected gene token ids
        Returns:
            filtered_src_ids, filtered_tgt_ids: 1D tensors
        """
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

    def _build_gt_matrix_from_pairs(
        self,
        gene_tokens: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ):
        """
        Build batch target matrix using already-filtered prior pairs.
        """
        C, S = gene_tokens.shape
        device = gene_tokens.device
        if src_ids.numel() == 0:
            return torch.zeros((C, S, S), device=device)

        target = torch.zeros((C, S, S), device=device)
        for b in range(C):
            src_matches = (gene_tokens[b].unsqueeze(-1) == src_ids.view(1, -1)).float()
            tgt_matches = (gene_tokens[b].unsqueeze(-1) == tgt_ids.view(1, -1)).float()
            target[b] = (src_matches @ tgt_matches.transpose(0, 1) > 0).float()
        return target

    def _build_assoc_mask_from_pairs(
        self,
        gene_tokens: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ):
        """
        Supervise only source/target genes that appear in filtered prior pairs.
        """
        C, S = gene_tokens.shape
        device = gene_tokens.device
        if src_ids.numel() == 0:
            return torch.zeros((C, S, S), dtype=torch.bool, device=device)

        src_unique = torch.unique(src_ids)
        tgt_unique = torch.unique(tgt_ids)
        src_assoc = (gene_tokens.unsqueeze(-1) == src_unique.view(1, 1, -1)).any(dim=-1)
        tgt_assoc = (gene_tokens.unsqueeze(-1) == tgt_unique.view(1, 1, -1)).any(dim=-1)
        return torch.einsum("bi,bj->bij", src_assoc, tgt_assoc).bool()

    def _factorized_logits(self, u: torch.Tensor, v: torch.Tensor, model_tf_mask: torch.Tensor):
        # Build full-source factors in selected TG space and compute dense scores lazily.
        # scores: (C, S, S) with S = selected TG count
        u_full = expand_u(u, model_tf_mask.bool())
        return torch.bmm(u_full, v.transpose(1, 2))

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

    def forward(
        self,
        tokens,
        factors: FactorState = None,
        true_grn_df: Optional[pd.DataFrame] = None,
    ):
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()
        binary_tf = factors.binary_tf
        binary_tg = factors.binary_tg
        u = factors.u
        v = factors.v
        gene_tokens = tokens["gene"]

        # 1. Project everything to the same TG-selected space used by GRN prediction.
        gene_tokens = self._masked_select_fixed_count(gene_tokens, binary_tg, "gene_tokens/binary_tg")
        model_tf_mask = self._masked_select_fixed_count(binary_tf.float(), binary_tg, "binary_tf/binary_tg")

        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)

        # 2. Explicitly filter prior by used genes.
        filt_src_ids, filt_tgt_ids = self._filter_prior_pairs_by_used_genes(gene_tokens)
        if filt_src_ids.numel() == 0:
            return gene_tokens.new_zeros((), dtype=torch.float32)

        # 3. Build dense source factors once, then supervise only relevant submatrices.
        u_full = expand_u(u, model_tf_mask.bool())  # (C, S, M)
        src_unique = torch.unique(filt_src_ids)
        tgt_unique = torch.unique(filt_tgt_ids)

        total_loss_sum = gene_tokens.new_zeros((), dtype=torch.float32)
        total_supervised = gene_tokens.new_zeros((), dtype=torch.float32)

        for b in range(gene_tokens.shape[0]):
            tokens_b = gene_tokens[b]
            tf_mask_b = model_tf_mask[b].bool()
            pos_map = torch.full(
                (self._pair_key_base,),
                -1,
                dtype=torch.long,
                device=tokens_b.device,
            )
            pos_map[tokens_b] = torch.arange(tokens_b.shape[0], device=tokens_b.device)

            src_ids_b = src_unique[pos_map[src_unique] >= 0]
            tgt_ids_b = tgt_unique[pos_map[tgt_unique] >= 0]
            if src_ids_b.numel() == 0 or tgt_ids_b.numel() == 0:
                continue
            src_pos = pos_map[src_ids_b]
            tgt_pos = pos_map[tgt_ids_b]
            src_tf_keep = tf_mask_b[src_pos]
            src_ids_b = src_ids_b[src_tf_keep]
            src_pos = src_pos[src_tf_keep]
            if src_pos.numel() == 0:
                continue

            edge_prob_sub = torch.matmul(u_full[b, src_pos, :], v[b, tgt_pos, :].transpose(0, 1))
            target_sub = torch.zeros_like(edge_prob_sub, dtype=torch.float32)

            src_local_map = torch.full(
                (self._pair_key_base,),
                -1,
                dtype=torch.long,
                device=tokens_b.device,
            )
            tgt_local_map = torch.full(
                (self._pair_key_base,),
                -1,
                dtype=torch.long,
                device=tokens_b.device,
            )
            src_local_map[src_ids_b] = torch.arange(src_ids_b.numel(), device=tokens_b.device)
            tgt_local_map[tgt_ids_b] = torch.arange(tgt_ids_b.numel(), device=tokens_b.device)

            src_local = src_local_map[filt_src_ids]
            tgt_local = tgt_local_map[filt_tgt_ids]
            pos_keep = (src_local >= 0) & (tgt_local >= 0)
            if pos_keep.any():
                target_sub[src_local[pos_keep], tgt_local[pos_keep]] = 1.0

            loss_sub = self._binary_cross_entropy_prob(edge_prob_sub, target_sub)
            total_loss_sum = total_loss_sum + loss_sub.sum()
            total_supervised = total_supervised + target_sub.new_tensor(target_sub.numel(), dtype=torch.float32)

        return total_loss_sum / (total_supervised + 1e-8)


class DAGLoss(nn.Module):
    def __init__(
            self, 
            alpha: float = 0.0, 
            rho: float = 0.1, 
            rho_max: float = 1e6, 
            update_period: int = 100
    ):
        """
        Args:
            alpha: Initial Lagrange multiplier.
            rho: Initial penalty parameter.
            rho_max: Upper bound for rho.
            update_period: How many steps to wait before updating alpha/rho.
        """
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self.rho_max = rho_max
        self.prev_h_val = float('inf')

        self.update_period = update_period
        self.step_counter = 0
        self.accumulated_h = 0.0

    def _compute_dag_constraint(self, adj):
        device = adj.device
        batch_size, M, _ = adj.shape
        adj_sq = adj * adj
        eye = torch.eye(M, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # (I + A^2/M)^M approximation
        matrix_poly = eye + adj_sq / M
        res = torch.matrix_power(matrix_poly, M)
        h = torch.diagonal(res, dim1=-2, dim2=-1).sum(-1) - M
        return h

    def _auto_update_params(self):
        """
        Internal logic to update alpha and rho based on accumulated h(A).
        """
        avg_h = self.accumulated_h / self.update_period
        
        # ALM update rules
        if avg_h > 0.25 * self.prev_h_val:
            self.rho = min(self.rho * 10.0, self.rho_max)
        else:
            self.alpha += self.rho * avg_h
            
        self.prev_h_val = avg_h
        
        # Reset accumulators
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
            binary_tf = self._masked_select_fixed_count(binary_tf.float(), binary_tg, "binary_tf/binary_tg")
        u_full = expand_u(u, binary_tf)
        adj_factor = torch.bmm(v.transpose(1, 2), u_full)
        
        # Compute current batch acyclicity violation
        dag_h_batch = self._compute_dag_constraint(adj_factor).mean()
        
        # Store for the scheduled update
        if self.training:
            self.step_counter += 1
            self.accumulated_h += dag_h_batch.item()
            
            # Trigger update when period is reached
            if self.step_counter >= self.update_period:
                self._auto_update_params()
        
        # Compute Augmented Lagrangian loss
        loss = self.alpha * dag_h_batch + (self.rho / 2) * (dag_h_batch ** 2)
        
        return loss


class SFMLoss(nn.Module):
    def __init__(
        self, 
        use_prior: bool = True, 
        use_dag: bool = True,
        tome_tokenizer: Optional[TomeTokenizer] = None, # Mandatory if use_prior
        true_grn_df: Optional[pd.DataFrame] = None, # Mandatory if use_prior
        num_epochs: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.use_prior = use_prior
        self.use_dag = use_dag
        self.T = num_epochs
        self.current_epoch = 0
        
        # 1. ELBO Loss
        self.elbo_criterion = ELBOLoss(
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout=kwargs.get("dropout", 0.1),
            recon_reduction=kwargs.get("recon_reduction", "mean"),
        )
        
        # 2. Prior Loss Setup
        if self.use_prior:
            if tome_tokenizer is None or true_grn_df is None:
                raise ValueError(
                    "Both 'tome_tokenizer' and 'true_grn_df' must be provided if use_prior=True"
                )
            else:
                self.prior_criterion = PriorLoss(
                    tome_tokenizer,
                    true_grn_df=true_grn_df,
                )
                self.true_grn_df = true_grn_df # Store prior knowledge internally
            
        # 3. DAG Loss Setup
        if self.use_dag:
            self.dag_criterion = DAGLoss(
                alpha=kwargs.get("alpha", 0.0),
                rho=kwargs.get("rho", 0.01),
                rho_max=kwargs.get("rho_max", 1e6),
                update_period=kwargs.get("update_period", 100)
            )

    def update_epoch(self, epoch: int):
        """Update the internal epoch counter."""
        self.current_epoch = epoch

    def get_prior_weight(self) -> float:
        """Calculates 0.5 + 0.5 * cos(pi * current_epoch / T)"""
        if self.T is None or self.T == 0:
            return 1.0 # Default to full prior if T not set
        # Clip current_epoch to T
        t_eff = min(self.current_epoch, self.T)
        return 0.5 + 0.5 * np.cos(np.pi * t_eff / self.T)

    def forward(self, tokens, factors: FactorState = None):
        """
        Calculates total loss based on ELBO, Prior, and DAG constraints.
        """
        if factors is None:
            raise ValueError("factors must be provided.")
        factors.validate()

        total_loss = 0.0
        loss_dict = {}

        # 1. ELBO Loss
        loss_elbo = self.elbo_criterion(
            tokens,
            factors=factors,
        )
        total_loss += loss_elbo
        loss_dict["elbo"] = loss_elbo.item()

        # 2. Prior Loss
        if self.use_prior:
            w_p = self.get_prior_weight()
            # Use self.true_grn_df stored during init
            loss_p = self.prior_criterion(
                tokens,
                factors=factors,
                true_grn_df=self.true_grn_df,
            )
            total_loss += w_p * loss_p
            loss_dict["prior"] = loss_p.item()

        # 3. DAG Loss
        if self.use_dag:
            loss_d = self.dag_criterion(factors)
            total_loss += loss_d
            loss_dict["dag"] = loss_d.item()

        return total_loss, loss_dict




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    import torch
    from .models import SFM 

    # 1. Setup Data and Tokenizer
    adata = sc.read_h5ad("/data1021/xukaichen/data/CTA/pbmc.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")
    tf_dict = pd.read_csv("./resources/human-tfs.csv")
    tf_list = tf_dict["TF"].tolist()

    Ng = 1000
    Nc = 50
    tokenizer = TomeTokenizer(token_dict, max_length=Ng+1, n_top_genes=Ng)
    tokens = tokenizer(adata[:Nc, :].copy())

    # 2. Initialize SFM Model
    model = SFM(token_dict, tf_list=tf_list)
    
    # 3. Model Forward Pass
    # For training, keep factorized state only (no dense GRN).
    _, factors = model(tokens, return_factors=True, compute_grn=False)

    # 4. Setup SFMLoss
    # Loading the Prior DataFrame
    true_grn_df = pd.read_csv("./resources/OmniPath.csv")
    
    # Define training parameters for the loss integration
    num_epochs = 100
    steps_per_epoch = 10 # Usually len(dataloader)
    
    criterion = SFMLoss(
        use_prior=True,
        use_dag=True,
        tome_tokenizer=tokenizer,
        true_grn_df=true_grn_df,
        num_epochs=num_epochs,
    )

    # 5. Simulate an Epoch update (e.g., Epoch 25)
    test_epoch = 25
    criterion.update_epoch(test_epoch)

    # 6. Compute Integrated Loss
    total_loss, loss_dict = criterion(
        tokens=tokens,
        factors=factors,
    )

    # 7. Verification Output
    print(f"--- SFMLoss Verification (Epoch {test_epoch}) ---")
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Breakdown:")
    for key, value in loss_dict.items():
            print(f"  - {key}: {value:.6f}")

    # 8. Check Backpropagation
    total_loss.backward()
    print("\nGradient Check:")
    has_grad_tf = next(model.tfrouter.parameters()).grad is not None
    has_grad_tg = next(model.tgrouter.parameters()).grad is not None
    print(f"  - Gradient flow to TF Router: {has_grad_tf}")
    print(f"  - Gradient flow to TG Router: {has_grad_tg}")
