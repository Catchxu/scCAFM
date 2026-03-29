#!/usr/bin/env bash

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/sfm.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_grn.yaml}"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m src.evaluator.grn \
  --model-config "$MODEL_CONFIG" \
  --eval-config "$EVAL_CONFIG"
