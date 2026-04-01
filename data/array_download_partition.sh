#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INDEX_PATH="path/to/index"
QUERY_PATH="path/to/query"
DATA_PATH="path/to/data"
ORGANISMS=("Homo sapiens" "Mus musculus")
TOKEN_DICT_PATH=""

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

for ORGANISM in "${ORGANISMS[@]}"; do
    echo "downloading ${query_name} for ${ORGANISM}"
    "${SCRIPT_DIR}/download_partition.sh" ${query_name} ${INDEX_PATH} ${DATA_PATH} "${ORGANISM}" "${TOKEN_DICT_PATH}"
done
