import torch
import torch.nn as nn
import pandas as pd

from .tokenizer import TomeTokenizer
from .models import expand_grn, expand_u


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

    def forward(self, tokens, grn, binary_tf, binary_tg, true_grn_df, tf_idx=None):
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
            rho: float = 0.0, 
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

    def forward(self, u, binary_tf, v, binary_tg):
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
    def __init__(self, lamb, num_epochs):
        super().__init__()




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    import torch
    from .tokenizer import TomeTokenizer
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

    # 2. Model Inference with return_factors=True
    model = SFM(token_dict, tf_list=tf_list)
    
    # We now request u and v to compute DAG loss
    grn, u, binary_tf, v, binary_tg = model(tokens, return_factors=True)

    # 3. Verify PriorLoss
    criterion_prior = PriorLoss(tokenizer)
    true_grn_df = pd.read_csv("./resources/OmniPath.csv")
    
    # Note: expand_grn is passed as a function handle for internal expansion
    prior_loss = criterion_prior(
        tokens, grn, binary_tf, binary_tg, true_grn_df
    )

    # 4. Verify DAGLoss
    # Initialize with default alpha and rho
    criterion_dag = DAGLoss()
    
    dag_loss = criterion_dag(
        u,          # [C, TF, M]
        binary_tf,  # [C, TG] - After squeezing if necessary
        v,          # [C, TG, M]
        binary_tg   # [C, TG]
    )

    # 5. Output Results
    print(f"Prior Loss: {prior_loss.item():.6f}")
    print(f"DAG Loss:   {dag_loss.item():.6f}")