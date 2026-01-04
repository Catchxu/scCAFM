import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

from .tokenizer import TomeTokenizer
from .models import ELBOLoss, expand_grn, expand_u


class PriorLoss(nn.Module):
    def __init__(self, tome_tokenizer: TomeTokenizer):
        super().__init__()
        gt = tome_tokenizer.gene_tokenizer
        self.symbol2id = gt.symbol2id
        self.id2id = gt.id2id
        self.pad_index = gt.pad_index
        
        # Using reduction='none' to apply custom masking later
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def get_gt_matrix(self, true_grn: pd.DataFrame, gene_tokens: torch.Tensor):
        """
        Args:
            true_grn: DataFrame with ["Gene1", "Gene2"]
            gene_tokens: (B, L)
        """
        device = gene_tokens.device
        
        # 1. Identify which lookup to use
        sample_gene = str(true_grn["Gene1"].iloc[0])
        lookup = self.id2id if sample_gene.startswith("ENSG") else self.symbol2id
        
        # 2. Filter DataFrame to ensure BOTH genes exist in our vocabulary
        # This is the crucial step to keep dimensions P equal
        valid_prior = true_grn[
            true_grn["Gene1"].isin(lookup.keys()) & 
            true_grn["Gene2"].isin(lookup.keys())
        ].copy()

        if len(valid_prior) == 0:
            # Return empty matrix if no overlaps found
            return torch.zeros((gene_tokens.shape[0], gene_tokens.shape[1], gene_tokens.shape[1]), device=device)

        # 3. Map to IDs and convert to tensors
        src_ids = torch.tensor(valid_prior["Gene1"].map(lookup).values, dtype=torch.long, device=device)
        tgt_ids = torch.tensor(valid_prior["Gene2"].map(lookup).values, dtype=torch.long, device=device)

        # 4. Vectorized Position Search
        # src_matches: (B, L, P), tgt_matches: (B, L, P)
        # P is now guaranteed to be len(valid_prior)
        src_matches = (gene_tokens.unsqueeze(-1) == src_ids.view(1, 1, -1))
        tgt_matches = (gene_tokens.unsqueeze(-1) == tgt_ids.view(1, 1, -1))

        # 5. Construct Matrix (B, L, L)
        # 'bsp, btp -> bst' (Batch, Source_pos, Target_pos)
        gt_matrix = torch.einsum('bsp,btp->bst', src_matches.float(), tgt_matches.float())
        
        return (gt_matrix > 0).float()

    def forward(self, tokens, grn, binary_tf, binary_tg, true_grn_df):
        gene_tokens = tokens["gene"]
        pad_mask = tokens["pad"]

        # 1. Align with model's internal routing: only genes selected as TFs should contribute to loss
        model_tf_mask = binary_tf.squeeze(-1).float() 

        # 2. Map ground truth to sequence positions
        target = self.get_gt_matrix(true_grn_df, gene_tokens)
        
        # 3. Expand predicted GRN to full sequence dimensions (B, L, L)
        grn_full = expand_grn(grn, binary_tf, binary_tg)
        
        # 4. Construct 2D mask (Source x Target)
        # Loss is only valid where: 1) Gene is not padding, AND 2) Source is an active TF
        is_not_pad = (~pad_mask).float()
        valid_mask = torch.einsum('bi,bj->bij', model_tf_mask * is_not_pad, is_not_pad)

        # 5. Compute pixel-wise loss
        loss = self.criterion(grn_full, target)
        weight_mask = target * 10.0 + (1.0 - target) * 1.0
        loss = loss * weight_mask
        
        # 6. Normalize by valid entries to prevent loss inflation from padded/inactive regions
        total_loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
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

    def forward(self, u, v, binary_tf):
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
        tome_tokenizer: Optional[TomeTokenizer] = None, 
        num_epochs: Optional[int] = None,
        **kwargs
    ):
        """
        Unified loss for SFM.
        
        Args:
            use_prior: Whether to include PriorLoss.
            use_dag: Whether to include DAGLoss.
            tome_tokenizer: Instance of TomeTokenizer.
            num_epochs: Total training epochs (T).
            **kwargs: Arguments for sub-losses (e.g., hidden_dim, rho, alpha).
        """
        super().__init__()
        self.use_prior = use_prior
        self.use_dag = use_dag
        self.T = num_epochs
        self.current_epoch = 0
        
        # 1. ELBO Loss (Core reconstruction)
        self.elbo_criterion = ELBOLoss(
            hidden_dim=kwargs.get("hidden_dim", 64),
            dropout=kwargs.get("dropout", 0.1)
        )
        
        # 2. Prior Loss (Knowledge-based)
        if self.use_prior:
            if tome_tokenizer is None:
                raise ValueError(
                    "tome_tokenizer should be provided as use_prior=True"
                )
            else:
                self.prior_criterion = PriorLoss(tome_tokenizer)
            
        # 3. DAG Loss (Structure constraint)
        if self.use_dag:
            self.dag_criterion = DAGLoss(
                alpha=kwargs.get("alpha", 0.0),
                rho=kwargs.get("rho", 0.01), # Started at 0.01 to ensure initial gradient
                rho_max=kwargs.get("rho_max", 1e6),
                update_period=kwargs.get("update_period", 100)
            )

    def update_epoch(self, epoch):
        """Update the internal epoch counter to calculate dynamic weights."""
        self.current_epoch = epoch

    def get_prior_weight(self):
        """Calculates 0.5 + 0.5 * cos(pi * t / T)"""
        if self.T is None:
            raise ValueError(
                "num_epochs should be provided as use_prior=True"
            )      
        else:
            return 0.5 + 0.5 * np.cos(np.pi * self.current_epoch / self.T)

    def forward(
            self, tokens, grn, binary_tf, binary_tg, 
            u=None, v=None, true_grn_df=None
    ):
        """
        Args:
            tokens: Dictionary containing "expr", "gene", "pad".
            grn: Pred GRN [C, TF, TG].
            binary_tf/tg: Gates from SFM.
            u, v: Factors from SFM for DAG loss.
            true_grn_df: pd.DataFrame for PriorLoss.
        """
        total_loss = 0.0
        loss_dict = {}

        # 1. ELBO Loss
        loss_elbo = self.elbo_criterion(tokens, grn, binary_tf, binary_tg)
        total_loss += loss_elbo
        loss_dict["elbo"] = loss_elbo.item()

        # 2. Prior Loss with dynamic weight
        if self.use_prior and true_grn_df is not None:
            w_p = self.get_prior_weight()
            loss_p = self.prior_criterion(tokens, grn, binary_tf, binary_tg, true_grn_df)
            total_loss += w_p * loss_p
            loss_dict["prior"] = loss_p.item()

        # 3. DAG Loss
        if self.use_dag:
            loss_d = self.dag_criterion(u, v, binary_tf)
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
    tokenizer = TomeTokenizer(token_dict, simplify=True, max_length=Ng+1)
    tokens = tokenizer(adata[:Nc, :].copy(), n_top_genes=Ng)

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