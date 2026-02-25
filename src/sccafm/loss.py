import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

from .tokenizer import TomeTokenizer
from .models import ELBOLoss, expand_grn, expand_u


class PriorLoss(nn.Module):
    def __init__(
        self,
        tome_tokenizer: TomeTokenizer,
        true_grn_df: Optional[pd.DataFrame] = None,
        pos_weight: float = 20.0,
        neg_weight: float = 1.0,
        neg_sample_ratio: Optional[float] = None,
    ):
        super().__init__()
        gt = tome_tokenizer.gene_tokenizer
        self.symbol2id = gt.symbol2id
        self.id2id = gt.id2id
        self.pad_index = gt.pad_index
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.neg_sample_ratio = None if neg_sample_ratio is None else float(neg_sample_ratio)
        self._cached_src_ids = None
        self._cached_tgt_ids = None
        self._cached_prior_ref = None
        
        # Using reduction='none' to apply custom masking later
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        if true_grn_df is not None:
            self._prepare_prior_cache(true_grn_df)

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

    def _factorized_logits(self, u: torch.Tensor, v: torch.Tensor, model_tf_mask: torch.Tensor):
        # Build full-source factors in selected TG space and compute dense logits lazily.
        # logits: (C, S, S) with S = selected TG count
        u_full = expand_u(u, model_tf_mask.bool())
        return torch.bmm(u_full, v.transpose(1, 2))

    def forward(
        self,
        tokens,
        grn,
        binary_tf,
        binary_tg,
        true_grn_df: Optional[pd.DataFrame] = None,
        u: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        gene_tokens = tokens["gene"]
        C = gene_tokens.shape[0]

        # 1. Project everything to the same TG-selected space used by GRN prediction.
        gene_tokens = gene_tokens[binary_tg].view(C, -1)
        model_tf_mask = binary_tf[binary_tg].view(C, -1).float()

        if true_grn_df is not None and true_grn_df is not self._cached_prior_ref:
            self._prepare_prior_cache(true_grn_df)

        # 2. Map ground truth to sequence positions in the selected space.
        target = self._build_gt_matrix(gene_tokens)
        
        # 3. Compute predicted logits in selected-space dimensions (B, S, S)
        if u is not None and v is not None:
            grn_full = self._factorized_logits(u, v, model_tf_mask)
        else:
            if grn is None:
                raise ValueError("Either `grn` or (`u`, `v`) must be provided for PriorLoss.")
            grn_full = expand_grn(grn, binary_tf, binary_tg)
        
        # 4. Construct 2D mask (Source x Target):
        # source gene must be an active TF in selected space.
        all_targets = torch.ones_like(model_tf_mask)
        valid_mask = torch.einsum('bi,bj->bij', model_tf_mask, all_targets)

        # 5. Compute pixel-wise loss
        loss = self.criterion(grn_full, target)
        weight_mask = target * self.pos_weight + (1.0 - target) * self.neg_weight
        loss = loss * weight_mask
        
        # 6. Optional negative sampling to avoid overwhelming positive supervision.
        supervise_mask = valid_mask
        if self.neg_sample_ratio is not None and self.neg_sample_ratio > 0:
            pos_mask = (target > 0.5) & (valid_mask > 0)
            neg_mask = (~(target > 0.5)) & (valid_mask > 0)
            pos_count = int(pos_mask.sum().item())
            neg_count = int(neg_mask.sum().item())

            if pos_count > 0 and neg_count > 0:
                sample_k = min(neg_count, int(pos_count * self.neg_sample_ratio))
                if sample_k > 0:
                    neg_idx = torch.nonzero(neg_mask, as_tuple=False)
                    perm = torch.randperm(neg_idx.shape[0], device=neg_idx.device)[:sample_k]
                    chosen = neg_idx[perm]
                    sampled_neg = torch.zeros_like(neg_mask)
                    sampled_neg[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                    supervise_mask = (pos_mask | sampled_neg).float()
        
        # 7. Normalize by supervised entries to prevent loss inflation.
        total_loss = (loss * supervise_mask).sum() / (supervise_mask.sum() + 1e-8)
        
        return total_loss


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

    def forward(self, u, v, binary_tf, binary_tg=None):
        if binary_tg is not None:
            C = binary_tf.shape[0]
            binary_tf = binary_tf[binary_tg].view(C, -1)
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
            hidden_dim=kwargs.get("hidden_dim", 64),
            dropout=kwargs.get("dropout", 0.1)
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
                    pos_weight=kwargs.get("prior_pos_weight", 10.0),
                    neg_weight=kwargs.get("prior_neg_weight", 1.0),
                    neg_sample_ratio=kwargs.get("prior_neg_sample_ratio", None),
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

    def forward(
        self, tokens, grn, binary_tf, binary_tg, 
        u=None, v=None
    ):
        """
        Calculates total loss based on ELBO, Prior, and DAG constraints.
        """
        total_loss = 0.0
        loss_dict = {}

        # 1. ELBO Loss
        loss_elbo = self.elbo_criterion(tokens, grn, binary_tf, binary_tg, u=u, v=v)
        total_loss += loss_elbo
        loss_dict["elbo"] = loss_elbo.item()

        # 2. Prior Loss
        if self.use_prior:
            w_p = self.get_prior_weight()
            # Use self.true_grn_df stored during init
            loss_p = self.prior_criterion(
                tokens, grn, binary_tf, binary_tg, self.true_grn_df, u=u, v=v
            )
            total_loss += w_p * loss_p
            loss_dict["prior"] = loss_p.item()

        # 3. DAG Loss
        if self.use_dag:
            # Note: Ensure u/v are not None if use_dag is True
            loss_d = self.dag_criterion(u, v, binary_tf, binary_tg)
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
    tokenizer = TomeTokenizer(token_dict, simplify=True, max_length=Ng+1, n_top_genes=Ng)
    tokens = tokenizer(adata[:Nc, :].copy())

    # 2. Initialize SFM Model
    model = SFM(token_dict, tf_list=tf_list)
    
    # 3. Model Forward Pass
    # Ensure your SFM.forward returns: grn, binary_tf, binary_tg, u, v
    grn, binary_tf, binary_tg, u, v = model(tokens, return_factors=True)

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
        num_epochs=num_epochs,
    )

    # 5. Simulate an Epoch update (e.g., Epoch 25)
    test_epoch = 25
    criterion.update_epoch(test_epoch)

    # 6. Compute Integrated Loss
    total_loss, loss_dict = criterion(
        tokens=tokens,
        grn=grn,
        binary_tf=binary_tf,
        binary_tg=binary_tg,
        u=u,
        v=v,
        true_grn_df=true_grn_df
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
