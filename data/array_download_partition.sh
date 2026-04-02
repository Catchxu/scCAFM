#!/usr/bin/env bash
#SBATCH --job-name=Download
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kxu248@emory.edu


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

INDEX_PATH="/scratch/kxu248/data/Atlas"  # "path/to/index"
QUERY_PATH="${REPO_ROOT}/data/query_list.txt"
DATA_PATH="/scratch/kxu248/data/Atlas"  # "path/to/data"
ORGANISMS=("Homo sapiens" "Mus musculus")
TOKEN_DICT_PATH="${REPO_ROOT}/resources/token_dict.csv"
HOMOLOGY_PATH="${REPO_ROOT}/resources/homologous.csv"

query_name="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${QUERY_PATH}")"

if [[ -z "${query_name}" ]]; then
  echo "Error: no query found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

for ORGANISM in "${ORGANISMS[@]}"; do
  echo "downloading ${query_name} for ${ORGANISM}"
  "${REPO_ROOT}/data/download_partition.sh" \
    "${query_name}" \
    "${INDEX_PATH}" \
    "${DATA_PATH}" \
    "${ORGANISM}" \
    "${TOKEN_DICT_PATH}" \
    "${HOMOLOGY_PATH}"
done
