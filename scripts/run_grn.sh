#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_pretrain.py --nproc-per-node 4
python3 scripts/run_eval_grn.py --nproc-per-node 4
