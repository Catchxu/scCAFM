#!/usr/bin/env bash
#SBATCH --job-name=efm-cell-type-annotation
#SBATCH --account=general
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python3 or python is required." >&2
  exit 1
fi

CONFIG_PATH="${EFM_CELL_TYPE_ANNOTATION_CONFIG:-${ROOT_DIR}/configs/efm_cell_type_annotation.yaml}"
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
    --efm-cell-type-annotation-config|--config)
      CONFIG_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    -h|--help)
      exec "${PYTHON_BIN}" -u -m sccafm.trainer.efm_cell_type_annotation --help
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: annotation config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" -u -m sccafm.trainer.efm_cell_type_annotation \
  --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
