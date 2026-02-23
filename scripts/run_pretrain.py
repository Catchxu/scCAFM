#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _resolve_cfg_path(raw: str, project_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    cfg_candidate = project_root / "configs" / raw
    if cfg_candidate.is_file():
        return cfg_candidate.resolve()

    raise FileNotFoundError(f"Cannot find config file: {raw}")


def _resolve_resource_path(raw: str, project_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    res_candidate = project_root / "resources" / raw
    if res_candidate.is_file():
        return res_candidate.resolve()

    raise FileNotFoundError(f"Cannot find resource file: {raw}")


def _as_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("`datasets.adata_files` must be a string path or a list of string paths.")


def _expand_adata_entry(raw: str):
    p = Path(raw).expanduser()
    if p.is_file():
        return [p.resolve()]
    if p.is_dir():
        files = sorted(p.rglob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found in directory: {p}")
        return [f.resolve() for f in files]
    raise FileNotFoundError(f"Missing dataset path: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scCAFM SFM pretraining with config checks.")
    parser.add_argument(
        "--meta-config",
        default="meta.yaml",
        help="Meta config path or filename under configs/ (default: meta.yaml).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional passthrough overrides, e.g. train.batch_size=16 train.device=cuda",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print command without launching training.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPU processes for DDP launch via torchrun (default: 1).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    meta_cfg_path = _resolve_cfg_path(args.meta_config, project_root)

    with meta_cfg_path.open("r", encoding="utf-8") as f:
        meta_cfg = yaml.safe_load(f) or {}

    pretrain_cfg_path = _resolve_cfg_path(meta_cfg["pretrain_sfm_config"], project_root)
    with pretrain_cfg_path.open("r", encoding="utf-8") as f:
        pretrain_cfg = yaml.safe_load(f) or {}

    datasets = pretrain_cfg.get("datasets", {})
    adata_files = _as_list(datasets.get("adata_files"))
    expanded_adata_files = []
    for adata_file in adata_files:
        expanded_adata_files.extend(_expand_adata_entry(adata_file))
    if not expanded_adata_files:
        raise FileNotFoundError(
            f"No dataset files resolved from `datasets.adata_files` in {pretrain_cfg_path}"
        )

    for key in ("token_dict", "human_tfs", "mouse_tfs", "true_grn"):
        _resolve_resource_path(datasets[key], project_root)

    env = os.environ.copy()
    extra_pythonpath = os.pathsep.join([str(project_root / "src"), str(project_root)])
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra_pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra_pythonpath

    in_torchrun = int(env.get("WORLD_SIZE", "1")) > 1
    if args.nproc_per_node > 1 and not in_torchrun:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            "-m",
            "sccafm.runner.pretrain_sfm",
            "--config",
            args.meta_config,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "sccafm.runner.pretrain_sfm",
            "--config",
            args.meta_config,
        ]
    if args.override:
        cmd.extend(["--override", *args.override])

    print(f"Meta config: {meta_cfg_path}")
    print(f"Pretrain config: {pretrain_cfg_path}")
    print(f"Resolved dataset files: {len(expanded_adata_files)}")
    print("Command:", " ".join(cmd))
    print(f"PYTHONPATH={env['PYTHONPATH']}")

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True, env=env, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
