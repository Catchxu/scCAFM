#!/usr/bin/env bash
#SBATCH --job-name=efm-pretrain
#SBATCH --account=general
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=120:00:00
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
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python3 or python is required to run EFM pretraining." >&2
  exit 1
fi

for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      exec "${PYTHON_BIN}" -u -m sccafm.trainer.efm_pretrain --help
      ;;
  esac
done

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_GPU_LIST[@]}"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    if [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
      NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
    else
      IFS=',' read -ra _GPU_LIST <<< "${SLURM_GPUS_ON_NODE}"
      NPROC_PER_NODE="${#_GPU_LIST[@]}"
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')" && [[ -n "${GPU_COUNT}" && "${GPU_COUNT}" -gt 0 ]]; then
      NPROC_PER_NODE="${GPU_COUNT}"
    else
      NPROC_PER_NODE=1
    fi
  else
    NPROC_PER_NODE=1
  fi
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_VISIBLE_GPU_LIST[@]}"
  if [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
    echo "Warning: NPROC_PER_NODE=${NPROC_PER_NODE} exceeds CUDA_VISIBLE_DEVICES count (${VISIBLE_GPU_COUNT}). Clamping to ${VISIBLE_GPU_COUNT}." >&2
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
  fi
fi

EFM_PRETRAIN_CONFIG="${EFM_PRETRAIN_CONFIG:-${PRETRAIN_CONFIG:-${ROOT_DIR}/configs/efm_pretrain.yaml}}"
EXTRA_ARGS=()

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --efm-pretrain-config|--pretrain-config)
      EFM_PRETRAIN_CONFIG="$(resolve_path "$2")"
      shift 2
      ;;
    -h|--help)
      exec "${PYTHON_BIN}" -u -m sccafm.trainer.efm_pretrain --help
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  --efm-pretrain-config "${EFM_PRETRAIN_CONFIG}"
  "${EXTRA_ARGS[@]}"
)

if [[ ! -f "${EFM_PRETRAIN_CONFIG}" ]]; then
  echo "Error: EFM pretrain config file not found: ${EFM_PRETRAIN_CONFIG}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC_PER_NODE}" -m sccafm.trainer.efm_pretrain "${ARGS[@]}"
fi

exec "${PYTHON_BIN}" -u -m sccafm.trainer.efm_pretrain "${ARGS[@]}"
