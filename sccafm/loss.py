import torch
import torch.nn as nn
import pandas as pd

from .tokenizer import TomeTokenizer


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

    def forward(self, pred_grn, true_grn_df, tokens, tf_idx=None):
        """
        Args:
            pred_grn: (B, L_tf, L_tg) from SFM
            true_grn_df: pd.DataFrame
            tokens: dict from TomoTokenizer
            tf_idx: The self.tf_idx tensor from your SFM model
        """
        gene_tokens = tokens["gene"]
        pad_mask = tokens["pad"] # (B, L)
        device = gene_tokens.device

        # 1. Identify which tokens in the current sequence are TFs
        # This mirrors the SFM._query_gene_subset logic
        if tf_idx is not None:
            tf_idx = tf_idx.to(device)
            # (B, L)
            gene_is_tf = (gene_tokens.unsqueeze(-1) == tf_idx.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        else:
            gene_is_tf = torch.ones_like(gene_tokens, dtype=torch.bool)

        # 2. Generate target matrix (Full L x L)
        target = self.get_gt_matrix(true_grn_df, gene_tokens)
        
        # 3. Slice target to match the dimensions of pred_grn [B, TF, TG]
        # We take only the rows that correspond to TFs
        # Note: Since different cells have different TFs in sequence, we use masking 
        # instead of hard slicing if pred_grn is fixed size (B, L, L)
        
        # Create 2D mask: [Valid TF in sequence] x [Valid Gene in sequence]
        is_not_pad = (~pad_mask).float()
        tf_multiplier = gene_is_tf.float()
        
        # This mask is (B, L, L) - 1.0 only if row is a TF and both are not pad
        valid_mask = torch.einsum('bi,bj->bij', tf_multiplier * is_not_pad, is_not_pad)

        # 4. Calculate Loss
        # If SFM returns (B, L, L), we apply mask to both.
        # If SFM returns (B, num_tfs, L), you would slice target here.
        print(pred_grn.shape)
        print(target.shape)
        loss = self.criterion(pred_grn, target)
        
        # 5. Final masked mean
        total_loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        return total_loss


class SFMLoss(nn.Module):
    def __init__(self, lamb, num_epochs):
        super().__init__()




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    from .tokenizer import TomeTokenizer
    from .models import SFM

    adata = sc.read_h5ad("/data1021/xukaichen/data/CTA/pbmc.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")
    tf_dict = pd.read_csv("./resources/human-tfs.csv")
    tf_list = tf_dict["TF"].tolist()

    Ng = 1000
    Nc = 50
    tokenizer = TomeTokenizer(token_dict, simplify=True, max_length=Ng+1)
    tokens = tokenizer(adata[:Nc, :].copy(), n_top_genes=Ng)

    model = SFM(token_dict, tf_list=tf_list)
    grn, _, _ = model(tokens)

    criterion = PriorLoss(tokenizer)
    true_grn_df = pd.read_csv("./resources/OmniPath.csv")
    loss = criterion(grn, true_grn_df, tokens, tf_idx=model.tf_idx)