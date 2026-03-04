# scCAFM command cookbook
# Purpose:
#   Keep frequently used training/evaluation commands in one place.
# Usage:
#   Run from project root: e.g., cd /data1021/xukaichen/scCAFM
#   Use foreground commands for test and nohup commands for long jobs.
# Notes:
#   - Redirected nohup command below suppresses terminal output.
#   - For persistent logs, redirect to a file instead of /dev/null.

# model training
setsid nohup python3 scripts/run_pretrain.py --nproc-per-node 4 > /dev/null 2>&1 < /dev/null &

# evaluation
setsid nohup python3 scripts/run_eval_grn.py --nproc-per-node 4 > /dev/null 2>&1 < /dev/null &

# pipeline
setsid nohup bash scripts/run_grn.sh > /dev/null 2>&1 < /dev/null &