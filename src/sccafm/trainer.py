import os
import gc
import logging
import signal
from contextlib import nullcontext
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


def _setup_logger(
    rank0: bool,
    checkpoint_dir: str,
    log_dir: Optional[str],
    log_name: str,
    log_overwrite: bool = True,
):
    if not rank0:
        return None
    if log_dir is None:
        log_dir = checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("sccafm.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w" if log_overwrite else "a")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _resolve_species_for_tf(
    adata,
    species_obs_key: Optional[str],
):
    if species_obs_key is not None and species_obs_key in adata.obs:
        vals = (
            adata.obs[species_obs_key]
            .astype(str)
            .str.lower()
            .replace("nan", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            raise ValueError(
                f"Found multiple species values in adata.obs['{species_obs_key}']: {vals[:10]}. "
                "Training currently expects one species per dataset file."
            )

    species_uns = adata.uns.get("species", None)
    if species_uns is not None:
        return str(species_uns).lower()

    return "human"


def sfm_trainer(
    model: SFM,
    adata_files: Union[str, List[str]], 
    tokenizer: TomeTokenizer, 
    criterion: SFMLoss,
    human_tfs: Optional[pd.DataFrame] = None,
    mouse_tfs: Optional[pd.DataFrame] = None,
    species_key: Optional[str] = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    epochs_per_file: int = 1,
    batch_size: int = 32,
    device="cuda",
    checkpoint_dir="./checkpoints",
    resume=True,
    log_dir: Optional[str] = None,
    log_name: str = "pretrain.log",
    log_interval: int = 100,
    use_tqdm: bool = True,
    tqdm_mininterval: float = 1.0,
    log_overwrite: bool = True,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
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
    logger = _setup_logger(rank0, checkpoint_dir, log_dir, log_name, log_overwrite=log_overwrite)

    amp_dtype = amp_dtype.lower()
    if amp_dtype not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}. Use 'bf16' or 'fp16'.")
    amp_enabled = bool(use_amp and device.startswith("cuda"))
    if amp_enabled and amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "use_amp=True with amp_dtype='bf16' requires CUDA bf16 support on this GPU/runtime."
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == "fp16")

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

    prev_handlers = {}
    if logger is not None:
        def _log_signal(signum, _frame):
            logger.error("Received termination signal %s. Training will stop.", signum)
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _log_signal)

    global_step = 0
    current_file_idx = start_file_idx - 1
    current_epoch = -1
    try:
        # 1. Resume Logic
        if resume and os.path.exists(checkpoint_path):
            if rank0:
                print(f"Loading checkpoint from {checkpoint_path}...")
                if logger:
                    logger.info("Loading checkpoint from %s", checkpoint_path)
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
                if logger:
                    logger.info("Resuming from file index %d", start_file_idx)

        # 2. Global T Setup
        total_global_epochs = len(adata_files) * epochs_per_file
        criterion_core.T = total_global_epochs

        # Prefer train.species_key when provided, otherwise reuse tokenizer species key.
        tokenizer_species_key = None
        try:
            tokenizer_species_key = tokenizer.cond_tokenizer.condition_keys[1]
        except Exception:
            tokenizer_species_key = None
        species_obs_key = species_key if species_key is not None else tokenizer_species_key

        # 3. Main File Loop
        model.train()
        criterion.train()
        if rank0 and logger:
            logger.info(
                "Training start | files=%d | epochs_per_file=%d | batch_size=%d | device=%s | ddp=%s | amp=%s | amp_dtype=%s",
                len(adata_files), epochs_per_file, batch_size, device, is_distributed, amp_enabled, amp_dtype
            )
        for file_idx in range(start_file_idx, len(adata_files)):
            current_file_idx = file_idx
            file_path = adata_files[file_idx]
            if rank0:
                print(f"\n[File {file_idx+1}/{len(adata_files)}] Loading: {file_path}")
                if logger:
                    logger.info("[File %d/%d] Loading: %s", file_idx + 1, len(adata_files), file_path)

            adata = sc.read_h5ad(file_path)
            raw_cells, raw_genes = int(adata.n_obs), int(adata.n_vars)
            with torch.no_grad():
                tokens_dict = tokenizer(adata, preprocess=True)
            post_cells = int(tokens_dict["gene"].shape[0])
            post_genes = int((~tokens_dict["pad"][0].bool()).sum().item()) if post_cells > 0 else 0
            if rank0 and logger:
                logger.info(
                    "Preprocess result | file=%d | raw_cells=%d raw_genes=%d -> post_cells=%d post_genes=%d",
                    file_idx + 1,
                    raw_cells,
                    raw_genes,
                    post_cells,
                    post_genes,
                )

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

            species = _resolve_species_for_tf(adata, species_obs_key)
            if species == "human":
                model_core.update_tfs(human_tfs)
            elif species == "mouse":
                model_core.update_tfs(mouse_tfs)
            else:
                raise ValueError(f"{species} isn't supported!")

            for epoch in range(epochs_per_file):
                current_epoch = epoch
                global_epoch_idx = (file_idx * epochs_per_file) + epoch
                criterion_core.update_epoch(global_epoch_idx)
                if sampler is not None:
                    sampler.set_epoch(global_epoch_idx)

                if rank0 and logger:
                    logger.info(
                        "[File %d/%d] Epoch %d/%d start",
                        file_idx + 1,
                        len(adata_files),
                        epoch + 1,
                        epochs_per_file,
                    )

                iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_per_file}", mininterval=tqdm_mininterval) if (rank0 and use_tqdm) else loader
                for batch in iterator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    autocast_ctx = (
                        torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16,
                        )
                        if amp_enabled
                        else nullcontext()
                    )
                    with autocast_ctx:
                        grn, b_tf, b_tg, u, v = model(batch, return_factors=True, compute_grn=False)
                        total_loss, loss_dict = criterion(tokens=batch, grn=grn, binary_tf=b_tf, binary_tg=b_tg, u=u, v=v)

                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        optimizer.step()
                    global_step += 1

                    if rank0:
                        metric_keys = [k for k in ("elbo", "prior", "dag") if k in loss_dict]
                        display_metrics = {k: f"{loss_dict[k]:.3f}" for k in metric_keys}
                        if use_tqdm:
                            iterator.set_postfix(display_metrics)
                        if logger and log_interval > 0 and global_step % log_interval == 0:
                            metric_text = " ".join([f"{k}={loss_dict[k]:.6f}" for k in metric_keys])
                            logger.info(
                                "step=%d file=%d epoch=%d %s",
                                global_step, file_idx + 1, epoch + 1, metric_text
                            )

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
                if logger:
                    logger.info("Checkpoint saved: %s", checkpoint_path)
            if is_distributed:
                dist.barrier()

            del adata, tokens_dict, dataset, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if rank0:
            print("\nTraining workflow completed.")
            if logger:
                logger.info("Training workflow completed.")
    except BaseException:
        if logger:
            logger.exception(
                "Training aborted | file_idx=%d epoch=%d global_step=%d",
                current_file_idx + 1,
                current_epoch + 1,
                global_step,
            )
        raise
    finally:
        for sig, handler in prev_handlers.items():
            signal.signal(sig, handler)
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
