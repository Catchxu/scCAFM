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
    for adata_file in adata_files:
        resolved_adata = Path(adata_file).expanduser()
        if not resolved_adata.is_file():
            raise FileNotFoundError(
                f"Missing dataset file: {adata_file}\n"
                f"Update `datasets.adata_files` in {pretrain_cfg_path}"
            )

    for key in ("token_dict", "human_tfs", "mouse_tfs", "true_grn"):
        _resolve_resource_path(datasets[key], project_root)

    env = os.environ.copy()
    extra_pythonpath = os.pathsep.join([str(project_root / "src"), str(project_root)])
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra_pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra_pythonpath

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
    print("Command:", " ".join(cmd))
    print(f"PYTHONPATH={env['PYTHONPATH']}")

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True, env=env, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
