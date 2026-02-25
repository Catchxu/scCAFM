#!/usr/bin/env python3
"""
Compatibility wrapper.

This script keeps the old command entrypoint:
    python3 scripts/run_pretrain.py ...
and forwards options to the canonical CLI:
    python -m sccafm.runners.pretrain_sfm ...
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility launcher for scCAFM pretraining.")
    parser.add_argument(
        "--meta-config",
        default="meta.yaml",
        help="Meta config path (forwarded to --config).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional overrides: key=value",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths without training.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of DDP processes for torchrun.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    extra_pythonpath = os.pathsep.join([str(project_root / "src"), str(project_root)])
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra_pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra_pythonpath

    cmd = [
        sys.executable,
        "-m",
        "sccafm.runners.pretrain_sfm",
        "--config",
        args.meta_config,
        "--nproc-per-node",
        str(args.nproc_per_node),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.override:
        cmd.extend(["--override", *args.override])

    subprocess.run(cmd, check=True, env=env, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
