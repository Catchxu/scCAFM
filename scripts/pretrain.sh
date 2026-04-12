#!/usr/bin/env bash
#SBATCH --job-name=pretrain
#SBATCH --account=general
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kxu248@emory.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python3 or python is required to run pretraining." >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${ROOT_DIR}/configs/pretrain.yaml}"

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrain-config)
      PRETRAIN_CONFIG="$(resolve_path "$2")"
      shift 2
      ;;
    -h|--help)
      exec "${PYTHON_BIN}" -u -m src.trainer.pretrain --help
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  --pretrain-config "${PRETRAIN_CONFIG}"
  "${EXTRA_ARGS[@]}"
)

if [[ ! -f "${PRETRAIN_CONFIG}" ]]; then
  echo "Error: pretrain config file not found: ${PRETRAIN_CONFIG}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC_PER_NODE}" -m src.trainer.pretrain "${ARGS[@]}"
fi

exec "${PYTHON_BIN}" -u -m src.trainer.pretrain "${ARGS[@]}"
