import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from sccafm.builder import build_model, build_tokenizer
from sccafm.evaluators import evaluate_grn
from sccafm.load import load_cfg, load_resources


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


def _as_metric_list(v, key):
    if isinstance(v, str):
        out = [v.strip().lower()]
    elif isinstance(v, list):
        if not all(isinstance(x, str) for x in v):
            raise ValueError(f"`{key}` list items must be strings, got: {v}")
        out = [x.strip().lower() for x in v]
    else:
        raise ValueError(f"`{key}` must be a string or list of strings, got: {v}")

    out = [x for x in out if x]
    if not out:
        raise ValueError(f"`{key}` must contain at least one metric.")
    return list(dict.fromkeys(out))


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
                raise ValueError("`datasets.adata_files` list items must be string paths.")
            out.extend(_expand_adata_entry(item))
        return out
    raise ValueError("`datasets.adata_files` must be a string path or a list of string paths.")


def _normalize_eval_cfg(eval_cfg):
    eval_cfg["batch_size"] = _as_int(eval_cfg.get("batch_size", 32), "eval.batch_size")
    eval_cfg["log_interval"] = _as_int(eval_cfg.get("log_interval", 100), "eval.log_interval")
    eval_cfg["metric"] = _as_metric_list(eval_cfg.get("metric", "auprc"), "eval.metric")
    eval_cfg["device"] = _as_str(eval_cfg.get("device", "cuda"), "eval.device")
    eval_cfg["output_dir"] = _as_str(eval_cfg.get("output_dir", "./eval/grn"), "eval.output_dir")
    eval_cfg["log_name"] = _as_str(eval_cfg.get("log_name", "grn_eval.log"), "eval.log_name")
    eval_cfg["use_amp"] = _as_bool(eval_cfg.get("use_amp", False), "eval.use_amp")
    eval_cfg["amp_dtype"] = _as_str(eval_cfg.get("amp_dtype", "bf16"), "eval.amp_dtype").lower()
    eval_cfg["preprocess"] = _as_bool(eval_cfg.get("preprocess", True), "eval.preprocess")
    eval_cfg["log_overwrite"] = _as_bool(eval_cfg.get("log_overwrite", True), "eval.log_overwrite")


def _normalize_tokenizer_eval_cfg(tokenizer_cfg):
    tokenizer_cfg["gene_key"] = tokenizer_cfg.get("gene_key", None)
    tokenizer_cfg["platform_key"] = tokenizer_cfg.get("platform_key", None)
    tokenizer_cfg["species_key"] = tokenizer_cfg.get("species_key", None)
    tokenizer_cfg["tissue_key"] = tokenizer_cfg.get("tissue_key", None)
    tokenizer_cfg["disease_key"] = tokenizer_cfg.get("disease_key", None)


def _load_checkpoint(model, checkpoint_path: str, map_location: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Evaluate cell-specific GRN against GT GRN")
    parser.add_argument("--config", type=str, default="meta.yaml", help="Meta config YAML path")
    parser.add_argument("--override", nargs="*", default=[], help="Optional overrides: key=value")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths without evaluation.")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of processes for DDP evaluation via torchrun (default: 1).",
    )
    args = parser.parse_args()

    meta_cfg = load_cfg(args.config)

    model_cfg = load_cfg(meta_cfg["model_config"])
    tokenizer_cfg = load_cfg(meta_cfg["tokenizer_config"])
    eval_cfg = load_cfg(meta_cfg["eval_grn_config"])

    data_cfg = eval_cfg["datasets"]
    eval_tokenizer_cfg = eval_cfg.get("tokenizer", {})
    eval_task_cfg = eval_cfg["eval"]

    cfg_root = {"datasets": data_cfg, "tokenizer": eval_tokenizer_cfg, "eval": eval_task_cfg}
    for item in args.override:
        key, value = item.split("=", 1)
        keys = key.split(".")

        d = cfg_root
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

    _normalize_eval_cfg(eval_task_cfg)
    _normalize_tokenizer_eval_cfg(eval_tokenizer_cfg)

    adata_files = _resolve_adata_files(data_cfg["adata_files"])

    in_torchrun = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.nproc_per_node > 1 and not in_torchrun:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            "-m",
            "sccafm.runners.eval_grn",
            "--config",
            args.config,
        ]
        if args.override:
            cmd.extend(["--override", *args.override])
        print(f"Resolved dataset files: {len(adata_files)}")
        print("Command:", " ".join(cmd))
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
        return

    if args.dry_run:
        print(f"Resolved dataset files: {len(adata_files)}")
        print("Command:", " ".join([sys.executable, "-m", "sccafm.runners.eval_grn", "--config", args.config]))
        return

    token_dict = load_resources(data_cfg["token_dict"])
    human_tfs = load_resources(data_cfg["human_tfs"])
    mouse_tfs = load_resources(data_cfg["mouse_tfs"])
    eval_grn_key = "eval_grn" if "eval_grn" in data_cfg else "true_grn"
    eval_grn_df = load_resources(data_cfg[eval_grn_key])

    model = build_model(model_cfg["SFM"], token_dict=token_dict)
    tokenizer = build_tokenizer(tokenizer_cfg["Tome"], token_dict=token_dict)

    checkpoint_path = eval_task_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        missing, unexpected = _load_checkpoint(model, str(ckpt_path), map_location="cpu")
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    summary, _ = evaluate_grn(
        model=model,
        adata_files=adata_files,
        tokenizer=tokenizer,
        eval_grn_df=eval_grn_df,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        batch_size=eval_task_cfg["batch_size"],
        device=eval_task_cfg["device"],
        output_dir=eval_task_cfg["output_dir"],
        log_dir=eval_task_cfg.get("log_dir"),
        log_name=eval_task_cfg["log_name"],
        log_interval=eval_task_cfg["log_interval"],
        metric=eval_task_cfg["metric"],
        preprocess=eval_task_cfg["preprocess"],
        gene_key=eval_tokenizer_cfg["gene_key"],
        platform_key=eval_tokenizer_cfg["platform_key"],
        cond_species_key=eval_tokenizer_cfg["species_key"],
        tissue_key=eval_tokenizer_cfg["tissue_key"],
        disease_key=eval_tokenizer_cfg["disease_key"],
        use_amp=eval_task_cfg["use_amp"],
        amp_dtype=eval_task_cfg["amp_dtype"],
        log_overwrite=eval_task_cfg["log_overwrite"],
    )

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("GRN evaluation finished.")
        metrics = eval_task_cfg["metric"]
        for m in metrics:
            print(
                f"{m} mean={summary[f'{m}_mean']:.6f} "
                f"std={summary[f'{m}_std']:.6f} valid_cells={summary[f'{m}_valid_cells']}"
            )


if __name__ == "__main__":
    main()
