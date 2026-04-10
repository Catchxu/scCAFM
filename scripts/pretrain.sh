#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain.yaml}"

ARGS=(
  --pretrain-config "${PRETRAIN_CONFIG}"
)

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC_PER_NODE}" -m src.trainer.pretrain "${ARGS[@]}"
fi

exec python -m src.trainer.pretrain "${ARGS[@]}"
