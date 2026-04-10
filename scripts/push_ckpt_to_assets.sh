#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/checkpoints/models}"
ASSETS_MODELS_DIR="${ASSETS_MODELS_DIR:-${ROOT_DIR}/assets/models}"

required_files=(
  "sfm_config.json"
  "sfm_model.safetensors"
  "cond_dict.json"
  "vocab.json"
  "vocab.safetensors"
)

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint model package not found: ${CKPT_DIR}" >&2
  exit 1
fi

for file_name in "${required_files[@]}"; do
  if [[ ! -f "${CKPT_DIR}/${file_name}" ]]; then
    echo "Missing required checkpoint file: ${CKPT_DIR}/${file_name}" >&2
    exit 1
  fi
done

mkdir -p "${ASSETS_MODELS_DIR}"

for file_name in "${required_files[@]}"; do
  cp -f "${CKPT_DIR}/${file_name}" "${ASSETS_MODELS_DIR}/${file_name}"
done

echo "Promoted checkpoint package from ${CKPT_DIR} to ${ASSETS_MODELS_DIR}"
