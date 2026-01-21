import os
import gc
import scanpy as sc
from tqdm import tqdm
from typing import Union, List

import torch
from torch.utils.data import DataLoader

from .models import SFM
from .loss import SFMLoss
from .tokenizer import TomeTokenizer, TomeDataset, tome_collate_fn


def sfm_trainer(
    model: SFM,
    adata_files: Union[str, List[str]], 
    tokenizer: TomeTokenizer, 
    criterion: SFMLoss,
    learning_rate: float,
    weight_decay: float,
    epochs_per_file=1,
    batch_size=32,
    device="cuda",
    checkpoint_dir="./checkpoints",
    resume=True
):
    """
    Complete training pipeline for SFM model.
    """
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "sfm_latest.pt")
    
    start_file_idx = 0
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 1. Resume Logic
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if 'dag_state' in ckpt and hasattr(criterion, 'dag_criterion') and ckpt['dag_state'] is not None:
            criterion.dag_criterion.alpha = ckpt['dag_state']['alpha'].to(device)
            criterion.dag_criterion.rho = ckpt['dag_state']['rho'].to(device)
            criterion.dag_criterion.prev_h_val = ckpt['dag_state']['prev_h_val']
        
        start_file_idx = ckpt['file_idx'] + 1
        print(f"Resuming from File Index: {start_file_idx}")

    # 2. Global T Setup
    total_global_epochs = len(adata_files) * epochs_per_file
    criterion.T = total_global_epochs

    # 3. Main File Loop
    for file_idx in range(start_file_idx, len(adata_files)):
        file_path = adata_files[file_idx]
        print(f"\n[File {file_idx+1}/{len(adata_files)}] Loading: {file_path}")
        
        adata = sc.read_h5ad(file_path)
        with torch.no_grad():
            tokens_dict = tokenizer(adata, preprocess=True)
        
        dataset = TomeDataset(tokens_dict)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=tome_collate_fn, num_workers=0, pin_memory=True
        )
        
        model.train()
        for epoch in range(epochs_per_file):
            # Calculate global epoch for cosine schedule
            global_epoch_idx = (file_idx * epochs_per_file) + epoch
            criterion.update_epoch(global_epoch_idx)
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_per_file}")
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Model Forward
                grn, b_tf, b_tg, u, v = model(batch, return_factors=True)
                
                # Loss Forward (true_grn_df is now internal to criterion)
                total_loss, loss_dict = criterion(
                    tokens=batch, grn=grn, binary_tf=b_tf, binary_tg=b_tg, u=u, v=v
                )
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                display_metrics = {"loss": f"{total_loss.item():.3f}"}
                for k, v in loss_dict.items():
                    display_metrics[k] = "{v:.3f}"

                pbar.set_postfix(display_metrics)

        # 4. Save Checkpoint after each file
        torch.save({
            'file_idx': file_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dag_state': {
                'alpha': criterion.dag_criterion.alpha,
                'rho': criterion.dag_criterion.rho,
                'prev_h_val': criterion.dag_criterion.prev_h_val
            } if hasattr(criterion, 'dag_criterion') else None,
        }, checkpoint_path)

        # 5. Cleanup
        del adata, tokens_dict, dataset, loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining workflow completed.")