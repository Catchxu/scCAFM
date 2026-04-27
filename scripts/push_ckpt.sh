#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/checkpoints/models}"
ASSETS_DIR="${ASSETS_DIR:-${ROOT_DIR}/assets}"
ASSETS_MODELS_DIR="${ASSETS_MODELS_DIR:-${ASSETS_DIR}/models}"
PUSH_HF="${PUSH_HF:-1}"
HF_REPO_ID="${HF_REPO_ID:-kaichenxu/scCAFM}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_EXCLUDE="${HF_EXCLUDE:-.cache/*}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload scCAFM model assets}"

usage() {
  cat <<EOF
Usage: bash scripts/push_ckpt.sh [--push-hf 0|1] [--hf-repo-id REPO_ID] [--commit-message MESSAGE]

Promote checkpoint model files into assets/models and upload assets to Hugging Face.

Options:
  --push-hf 0|1             Upload assets to Hugging Face after promotion.
  --hf-repo-id REPO_ID      Hugging Face repo ID to upload to.
  --commit-message MESSAGE  Hugging Face upload commit message.
  -h, --help                Show this help message.

Environment overrides:
  CKPT_DIR, ASSETS_DIR, ASSETS_MODELS_DIR, HF_REPO_TYPE, HF_EXCLUDE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push-hf)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --push-hf" >&2
        usage >&2
        exit 1
      fi
      if [[ "$2" != "0" && "$2" != "1" ]]; then
        echo "Invalid value for --push-hf: $2. Expected 0 or 1." >&2
        usage >&2
        exit 1
      fi
      PUSH_HF="$2"
      shift 2
      ;;
    --hf-repo-id)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --hf-repo-id" >&2
        usage >&2
        exit 1
      fi
      HF_REPO_ID="$2"
      shift 2
      ;;
    --commit-message)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --commit-message" >&2
        usage >&2
        exit 1
      fi
      HF_COMMIT_MESSAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

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

if [[ "${PUSH_HF}" == "1" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "Hugging Face CLI not found. Install it or set PUSH_HF=0 to skip upload." >&2
    exit 1
  fi

  echo "Uploading ${ASSETS_DIR} to Hugging Face repo ${HF_REPO_ID}"
  hf upload "${HF_REPO_ID}" "${ASSETS_DIR}" \
    --type "${HF_REPO_TYPE}" \
    --exclude "${HF_EXCLUDE}" \
    --commit-message "${HF_COMMIT_MESSAGE}"
fi
