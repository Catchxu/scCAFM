#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/checkpoints}"
ASSETS_DIR="${ASSETS_DIR:-${ROOT_DIR}/assets}"
PUSH_HF="${PUSH_HF:-1}"
HF_REPO_ID="${HF_REPO_ID:-kaichenxu/scCAFM}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_EXCLUDE="${HF_EXCLUDE:-.cache/*}"
HF_DELETE_LEGACY="${HF_DELETE_LEGACY:-1}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload scCAFM model assets}"

usage() {
  cat <<EOF
Usage: bash scripts/push_ckpt.sh [--push-hf 0|1] [--hf-repo-id REPO_ID] [--commit-message MESSAGE]

Promote a structured checkpoint package into assets and upload assets to Hugging Face.

Options:
  --push-hf 0|1             Upload assets to Hugging Face after promotion.
  --hf-repo-id REPO_ID      Hugging Face repo ID to upload to.
  --commit-message MESSAGE  Hugging Face upload commit message.
  -h, --help                Show this help message.

Environment overrides:
  CKPT_DIR, ASSETS_DIR, HF_REPO_TYPE, HF_EXCLUDE, HF_DELETE_LEGACY
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
  "release.json"
  "tokenizer/cond_dict.json"
  "tokenizer/vocab.json"
  "tokenizer/vocab.safetensors"
  "models/sfm/sfm_config.json"
  "models/sfm/sfm_model.safetensors"
  "models/efm/efm_config.json"
)
required_asset_resources=(
  "resources/human_tfs.csv"
  "resources/mouse_tfs.csv"
  "resources/OmniPath.csv"
  "resources/homologous.csv"
)
optional_efm_files=(
  "models/efm/efm_model.safetensors"
)
legacy_hf_files=(
  "human_tfs.csv"
  "mouse_tfs.csv"
  "OmniPath.csv"
  "homologous.csv"
  "cond_dict.csv"
  "cond_dict.json"
  "sfm_config.json"
  "sfm_model.safetensors"
  "efm_config.json"
  "efm_model.safetensors"
  "vocab.json"
  "vocab.safetensors"
  "models/cond_dict.json"
  "models/sfm_config.json"
  "models/sfm_model.safetensors"
  "models/efm_config.json"
  "models/efm_model.safetensors"
  "models/vocab.json"
  "models/vocab.safetensors"
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

for file_name in "${required_asset_resources[@]}"; do
  if [[ ! -f "${ASSETS_DIR}/${file_name}" ]]; then
    echo "Missing shared asset resource: ${ASSETS_DIR}/${file_name}" >&2
    exit 1
  fi
done

for file_name in "${required_files[@]}"; do
  mkdir -p "$(dirname "${ASSETS_DIR}/${file_name}")"
  cp -f "${CKPT_DIR}/${file_name}" "${ASSETS_DIR}/${file_name}"
done

efm_file_count=0
for file_name in "${optional_efm_files[@]}"; do
  if [[ -f "${CKPT_DIR}/${file_name}" ]]; then
    efm_file_count=$((efm_file_count + 1))
  fi
done
if [[ "${efm_file_count}" -eq "${#optional_efm_files[@]}" ]]; then
  for file_name in "${optional_efm_files[@]}"; do
    mkdir -p "$(dirname "${ASSETS_DIR}/${file_name}")"
    cp -f "${CKPT_DIR}/${file_name}" "${ASSETS_DIR}/${file_name}"
  done
elif [[ "${efm_file_count}" -eq 0 ]]; then
  rm -f "${ASSETS_DIR}/models/efm/efm_model.safetensors"
else
  echo "Incomplete EFM checkpoint package under ${CKPT_DIR}/models/efm" >&2
  exit 1
fi

echo "Promoted structured checkpoint package from ${CKPT_DIR} to ${ASSETS_DIR}"

if [[ "${PUSH_HF}" == "1" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "Hugging Face CLI not found. Install it or set PUSH_HF=0 to skip upload." >&2
    exit 1
  fi

  echo "Uploading ${ASSETS_DIR} to Hugging Face repo ${HF_REPO_ID}"
  hf_args=(
    upload
    "${HF_REPO_ID}"
    "${ASSETS_DIR}"
    --type "${HF_REPO_TYPE}"
    --exclude "${HF_EXCLUDE}"
    --commit-message "${HF_COMMIT_MESSAGE}"
  )
  if [[ "${HF_DELETE_LEGACY}" == "1" ]]; then
    for file_name in "${legacy_hf_files[@]}"; do
      hf_args+=(--delete "${file_name}")
    done
  fi
  hf "${hf_args[@]}"
fi
