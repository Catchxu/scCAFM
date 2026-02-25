#!/usr/bin/env python3
"""
Compatibility wrapper for GRN evaluation.

Entrypoint:
    python3 scripts/run_eval_grn.py ...
Forwards to:
    python -m sccafm.runner.eval_grn ...
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility launcher for scCAFM GRN evaluation.")
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
        help="Validate config and paths without evaluation.",
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
        "sccafm.runner.eval_grn",
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
