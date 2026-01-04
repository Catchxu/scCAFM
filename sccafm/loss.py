import torch
import torch.nn as nn
import pandas as pd

from .tokenizer import TomeTokenizer
from .models import expand_grn


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
        device = gene_tokens.device

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
    grn, binary_tf, binary_tg = model(tokens)

    criterion = PriorLoss(tokenizer)
    true_grn_df = pd.read_csv("./resources/OmniPath.csv")
    loss = criterion(
        tokens, grn, binary_tf, binary_tg, true_grn_df, tf_idx=model.tf_idx
    )

    print("prior loss:", loss.item())