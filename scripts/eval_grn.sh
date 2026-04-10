#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EVAL_GRN_CONFIG="${EVAL_GRN_CONFIG:-configs/eval_grn.yaml}"

ARGS=(
  --eval-grn-config "${EVAL_GRN_CONFIG}"
)

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC_PER_NODE}" -m src.evaluator.grn "${ARGS[@]}"
fi

exec python -m src.evaluator.grn "${ARGS[@]}"
