import os
import gc
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from typing import Union, List, Optional

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .models import SFM
from .loss import SFMLoss
from .tokenizer import TomeTokenizer, TomeDataset, tome_collate_fn


def _unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


def _setup_distributed(device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    initialized_here = False
    rank = 0
    local_rank = 0

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            initialized_here = True
        rank = dist.get_rank()
    return is_distributed, initialized_here, rank, local_rank, world_size


def sfm_trainer(
    model: SFM,
    adata_files: Union[str, List[str]], 
    tokenizer: TomeTokenizer, 
    criterion: SFMLoss,
    human_tfs: Optional[pd.DataFrame] = None,
    mouse_tfs: Optional[pd.DataFrame] = None,
    species_key: str = "species",
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    epochs_per_file: int = 1,
    batch_size: int = 32,
    device="cuda",
    checkpoint_dir="./checkpoints",
    resume=True
):
    """
    Complete training pipeline for SFM model.
    """
    if isinstance(adata_files, str):
        adata_files = [adata_files]

    is_distributed, initialized_here, rank, local_rank, world_size = _setup_distributed(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else device
    rank0 = rank == 0

    model.to(device)
    criterion.to(device)

    if is_distributed:
        ddp_kwargs = {}
        if device.startswith("cuda"):
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        model = DDP(model, **ddp_kwargs)
        criterion = DDP(criterion, **ddp_kwargs)

    model_core = _unwrap_ddp(model)
    criterion_core = _unwrap_ddp(criterion)

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
        if rank0:
            print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_core.load_state_dict(ckpt['model_state_dict'])
        if 'criterion_state_dict' in ckpt:
            criterion_core.load_state_dict(ckpt['criterion_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if 'dag_state' in ckpt and hasattr(criterion_core, 'dag_criterion') and ckpt['dag_state'] is not None:
            alpha = ckpt['dag_state']['alpha']
            rho = ckpt['dag_state']['rho']
            criterion_core.dag_criterion.alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
            criterion_core.dag_criterion.rho = rho.item() if isinstance(rho, torch.Tensor) else float(rho)
            criterion_core.dag_criterion.prev_h_val = ckpt['dag_state']['prev_h_val']

        start_file_idx = ckpt['file_idx'] + 1
        if rank0:
            print(f"Resuming from File Index: {start_file_idx}")

    # 2. Global T Setup
    total_global_epochs = len(adata_files) * epochs_per_file
    criterion_core.T = total_global_epochs

    # 3. Main File Loop
    model.train()
    criterion.train()
    for file_idx in range(start_file_idx, len(adata_files)):
        file_path = adata_files[file_idx]
        if rank0:
            print(f"\n[File {file_idx+1}/{len(adata_files)}] Loading: {file_path}")

        adata = sc.read_h5ad(file_path)
        with torch.no_grad():
            tokens_dict = tokenizer(adata, preprocess=True)

        dataset = TomeDataset(tokens_dict)
        sampler = None
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=tome_collate_fn,
            num_workers=0,
            pin_memory=True
        )

        try:
            species = adata.uns[species_key]
        except Exception:
            species = "human"  # default species is human

        if species == "human":
            model_core.update_tfs(human_tfs)
        elif species == "mouse":
            model_core.update_tfs(mouse_tfs)
        else:
            raise ValueError(
                f"{species} isn't supported!"
            )

        for epoch in range(epochs_per_file):
            # Calculate global epoch for cosine schedule
            global_epoch_idx = (file_idx * epochs_per_file) + epoch
            criterion_core.update_epoch(global_epoch_idx)
            if sampler is not None:
                sampler.set_epoch(global_epoch_idx)

            iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_per_file}") if rank0 else loader
            for batch in iterator:
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

                if rank0:
                    display_metrics = {"loss": f"{total_loss.item():.3f}"}
                    for k, v in loss_dict.items():
                        display_metrics[k] = f"{v:.3f}"
                    iterator.set_postfix(display_metrics)

        # 4. Save Checkpoint after each file (rank 0 only)
        if rank0:
            torch.save({
                'file_idx': file_idx,
                'model_state_dict': model_core.state_dict(),
                'criterion_state_dict': criterion_core.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dag_state': {
                    'alpha': criterion_core.dag_criterion.alpha,
                    'rho': criterion_core.dag_criterion.rho,
                    'prev_h_val': criterion_core.dag_criterion.prev_h_val
                } if hasattr(criterion_core, 'dag_criterion') else None,
            }, checkpoint_path)
        if is_distributed:
            dist.barrier()

        # 5. Cleanup
        del adata, tokens_dict, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank0:
        print("\nTraining workflow completed.")
    if initialized_here:
        dist.destroy_process_group()
