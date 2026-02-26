# scCAFM command cookbook
# Purpose:
#   Keep frequently used training/evaluation commands in one place.
# Usage:
#   Run from project root: e.g., cd /data1021/xukaichen/scCAFM
#   Use foreground commands for debugging and nohup commands for long jobs.
# Notes:
#   - Redirected nohup command below suppresses terminal output.
#   - For persistent logs, redirect to a file instead of /dev/null.

# model training
python3 scripts/run_pretrain.py --nproc-per-node 4
nohup python3 scripts/run_pretrain.py --nproc-per-node 4 > /dev/null 2>&1 &

# evaluation
python3 scripts/run_eval_grn.py --nproc-per-node 4
nohup python3 scripts/run_eval_grn.py --nproc-per-node 4 > /dev/null 2>&1 &

# pipeline
nohup bash scripts/run_grn.sh > /dev/null 2>&1 &