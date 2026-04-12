#!/usr/bin/env bash
#SBATCH --job-name=eval-grn
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
  echo "Error: python3 or python is required to run evaluation." >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EVAL_GRN_CONFIG="${EVAL_GRN_CONFIG:-${ROOT_DIR}/configs/eval_grn.yaml}"

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
    --eval-grn-config)
      EVAL_GRN_CONFIG="$(resolve_path "$2")"
      shift 2
      ;;
    -h|--help)
      exec "${PYTHON_BIN}" -u -m src.evaluator.grn --help
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  --eval-grn-config "${EVAL_GRN_CONFIG}"
  "${EXTRA_ARGS[@]}"
)

if [[ ! -f "${EVAL_GRN_CONFIG}" ]]; then
  echo "Error: eval config file not found: ${EVAL_GRN_CONFIG}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC_PER_NODE}" -m src.evaluator.grn "${ARGS[@]}"
fi

exec "${PYTHON_BIN}" -u -m src.evaluator.grn "${ARGS[@]}"
