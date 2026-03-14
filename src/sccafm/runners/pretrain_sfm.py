import argparse
import os
import subprocess
import sys
from pathlib import Path

from sccafm.load import load_cfg, load_resources
from sccafm.builder import build_model, build_loss, build_tokenizer
from sccafm.runtime import find_free_port, print_model_size
from sccafm.trainer import sfm_trainer


def _as_float(v, key):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError as e:
            raise ValueError(f"`{key}` must be a float, got: {v}") from e
    raise ValueError(f"`{key}` must be a float, got type: {type(v).__name__}")


def _as_int(v, key):
    if isinstance(v, bool):
        raise ValueError(f"`{key}` must be an int, got bool: {v}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as e:
            raise ValueError(f"`{key}` must be an int, got: {v}") from e
    raise ValueError(f"`{key}` must be an int, got type: {type(v).__name__}")


def _as_bool(v, key):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"`{key}` must be a bool, got: {v}")


def _as_str(v, key):
    if isinstance(v, str):
        return v
    raise ValueError(f"`{key}` must be a string, got: {v}")


def _normalize_devices(v, key):
    if v is None:
        return None
    if isinstance(v, int):
        if v < 0:
            raise ValueError(f"`{key}` GPU ids must be non-negative, got: {v}")
        return [v]
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return None
        if s == "cpu":
            return []
        if s.startswith("cuda:"):
            s = s.split(":", 1)[1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        try:
            out = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"`{key}` must be GPU ids like [0,1] or '0,1', got: {v}") from e
        if any(x < 0 for x in out):
            raise ValueError(f"`{key}` GPU ids must be non-negative, got: {v}")
        return out
    if isinstance(v, list):
        if not all(isinstance(x, int) and x >= 0 for x in v):
            raise ValueError(f"`{key}` must be a list of non-negative ints, got: {v}")
        return list(v)
    raise ValueError(f"`{key}` must be an int, string, list of ints, or null, got: {type(v).__name__}")


def _normalize_pretrain_cfg(train_cfg, loss_cfg):
    train_cfg["learning_rate"] = _as_float(train_cfg.get("learning_rate", 1e-5), "train.learning_rate")
    train_cfg["weight_decay"] = _as_float(train_cfg.get("weight_decay", 1e-2), "train.weight_decay")
    train_cfg["epochs_per_file"] = _as_int(train_cfg.get("epochs_per_file", 1), "train.epochs_per_file")
    train_cfg["batch_size"] = _as_int(train_cfg.get("batch_size", 32), "train.batch_size")
    train_cfg["grad_accum_steps"] = _as_int(train_cfg.get("grad_accum_steps", 1), "train.grad_accum_steps")
    train_cfg["warmup_steps"] = _as_int(train_cfg.get("warmup_steps", 1000), "train.warmup_steps")
    devices = train_cfg.get("devices", None)
    if devices is None and "device" in train_cfg:
        devices = train_cfg["device"]
    train_cfg["devices"] = _normalize_devices(devices, "train.devices")
    train_cfg["resume"] = _as_bool(train_cfg.get("resume", True), "train.resume")
    train_cfg["log_interval"] = _as_int(train_cfg.get("log_interval", 100), "train.log_interval")
    train_cfg["use_tqdm"] = _as_bool(train_cfg.get("use_tqdm", True), "train.use_tqdm")
    train_cfg["tqdm_mininterval"] = _as_float(train_cfg.get("tqdm_mininterval", 1.0), "train.tqdm_mininterval")
    train_cfg["log_overwrite"] = _as_bool(train_cfg.get("log_overwrite", True), "train.log_overwrite")
    train_cfg["use_amp"] = _as_bool(train_cfg.get("use_amp", False), "train.use_amp")
    train_cfg["amp_dtype"] = _as_str(train_cfg.get("amp_dtype", "bf16"), "train.amp_dtype").lower()

    loss_kwargs = loss_cfg.get("kwargs", {})
    if "alpha" in loss_kwargs:
        loss_kwargs["alpha"] = _as_float(loss_kwargs["alpha"], "loss.kwargs.alpha")
    if "rho" in loss_kwargs:
        loss_kwargs["rho"] = _as_float(loss_kwargs["rho"], "loss.kwargs.rho")
    if "rho_max" in loss_kwargs:
        loss_kwargs["rho_max"] = _as_float(loss_kwargs["rho_max"], "loss.kwargs.rho_max")
    if "update_period" in loss_kwargs:
        loss_kwargs["update_period"] = _as_int(loss_kwargs["update_period"], "loss.kwargs.update_period")


def _expand_adata_entry(entry: str):
    p = Path(entry).expanduser()
    if p.is_dir():
        files = sorted(str(f.resolve()) for f in p.rglob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found in directory: {p}")
        return files
    if p.is_file():
        return [str(p.resolve())]
    raise FileNotFoundError(f"`datasets.adata_files` path not found: {entry}")


def _resolve_adata_files(raw):
    if isinstance(raw, str):
        return _expand_adata_entry(raw)
    if isinstance(raw, list):
        out = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(
                    "`datasets.adata_files` list items must be string paths."
                )
            out.extend(_expand_adata_entry(item))
        return out
    raise ValueError(
        "`datasets.adata_files` must be a string path or a list of string paths."
    )

def main():
    parser = argparse.ArgumentParser(description="Pretrain scCAFM-SFM model")
    parser.add_argument("--config", type=str, default="meta.yaml", help="Meta config YAML path")
    parser.add_argument("--override", nargs="*", default=[], help="Optional overrides: key=value")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths without training.")
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Master port for torchrun rendezvous. If unset, auto-pick a free local port.",
    )
    args = parser.parse_args()

    # Load the configs
    meta_cfg = load_cfg(args.config)

    model_cfg = load_cfg(meta_cfg["model_config"])
    tokenizer_cfg = load_cfg(meta_cfg["tokenizer_config"])
    pretrain_cfg = load_cfg(meta_cfg["pretrain_sfm_config"])

    data_cfg = pretrain_cfg["datasets"]
    loss_cfg = pretrain_cfg["loss"]
    train_cfg = pretrain_cfg["train"]
    cfg_root = {"datasets": data_cfg, "loss": loss_cfg, "train": train_cfg}

    for item in args.override:
        key, value = item.split("=", 1)
        keys = key.split(".")

        # Prefer full rooted paths (e.g., train.resume=true).
        d = cfg_root
        try:
            for k in keys[:-1]:
                d = d[k]
            old_val = d[keys[-1]]
            if isinstance(old_val, bool):
                value = value.lower() == "true"
            elif isinstance(old_val, int):
                value = int(value)
            elif isinstance(old_val, float):
                value = float(value)
            d[keys[-1]] = value
            continue
        except Exception:
            pass

        # Backward compatible fallback: allow train-local keys (e.g., resume=true).
        d = train_cfg
        for k in keys[:-1]:
            d = d[k]
        old_val = d[keys[-1]]
        if isinstance(old_val, bool):
            value = value.lower() == "true"
        elif isinstance(old_val, int):
            value = int(value)
        elif isinstance(old_val, float):
            value = float(value)
        d[keys[-1]] = value

    _normalize_pretrain_cfg(train_cfg, loss_cfg)

    adata_files = _resolve_adata_files(data_cfg["adata_files"])
    visible_devices = train_cfg.pop("devices", None)
    runtime_device = "cpu" if visible_devices == [] else "cuda"
    nproc_per_node = 1 if visible_devices in (None, []) else max(1, len(visible_devices))
    if visible_devices not in (None, []):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible_devices)

    in_torchrun = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if nproc_per_node > 1 and not in_torchrun:
        master_port = args.master_port
        if master_port is None:
            env_port = os.environ.get("MASTER_PORT") or os.environ.get("TORCHRUN_MASTER_PORT")
            master_port = int(env_port) if env_port else find_free_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            "sccafm.runners.pretrain_sfm",
            "--config",
            args.config,
        ]
        if args.override:
            cmd.extend(["--override", *args.override])
        print(f"Resolved dataset files: {len(adata_files)}")
        if visible_devices is not None:
            print(f"Visible devices: {visible_devices} | nproc_per_node={nproc_per_node}")
        print("Command:", " ".join(cmd))
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
        return

    if args.dry_run:
        print(f"Resolved dataset files: {len(adata_files)}")
        if visible_devices is not None:
            print(f"Visible devices: {visible_devices} | nproc_per_node={nproc_per_node}")
        print(
            "Command:",
            " ".join(
                [sys.executable, "-m", "sccafm.runners.pretrain_sfm", "--config", args.config]
                + (["--override", *args.override] if args.override else [])
            ),
        )
        return

    token_dict = load_resources(data_cfg["token_dict"])
    human_tfs = load_resources(data_cfg["human_tfs"])
    mouse_tfs = load_resources(data_cfg["mouse_tfs"])
    true_grn_df = load_resources(data_cfg["true_grn"])

    model = build_model(model_cfg["SFM"], token_dict=token_dict)
    print_model_size(model, prefix="SFM size")
    tokenizer = build_tokenizer(tokenizer_cfg["Tome"], token_dict=token_dict)
    loss = build_loss(loss_cfg, tome_tokenizer=tokenizer, true_grn_df=true_grn_df)

    sfm_trainer(
        model=model,
        adata_files=adata_files,
        tokenizer=tokenizer,
        criterion=loss,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        device=runtime_device,
        **train_cfg
    )




if __name__ == "__main__":
    main()
