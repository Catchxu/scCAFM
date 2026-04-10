#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUERY="${1:?Usage: download_partition.sh QUERY INDEX_DIR OUTPUT_DIR [ORGANISM] [TOKEN_DICT_PATH] [HOMOLOGY_PATH] [PARTITION_IDX]}"
INDEX_DIR="${2:?Usage: download_partition.sh QUERY INDEX_DIR OUTPUT_DIR [ORGANISM] [TOKEN_DICT_PATH] [HOMOLOGY_PATH] [PARTITION_IDX]}"
OUTPUT_DIR="${3:?Usage: download_partition.sh QUERY INDEX_DIR OUTPUT_DIR [ORGANISM] [TOKEN_DICT_PATH] [HOMOLOGY_PATH] [PARTITION_IDX]}"
ORGANISM="${4:-Homo sapiens}"
TOKEN_DICT_PATH="${5:-}"
HOMOLOGY_PATH="${6:-${ROOT_DIR}/assets/homologous.csv}"
PARTITION_IDX="${7:-}"

MAX_PARTITION_SIZE="${MAX_PARTITION_SIZE:-200000}"

ORGANISM_DIR=$(PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from data_config import organism_output_name; import sys; print(organism_output_name(sys.argv[1]))' "${ORGANISM}")
IDX_FILE="${INDEX_DIR}/${ORGANISM_DIR}/${QUERY}.idx"
total_num=$(wc -l "${IDX_FILE}" | awk '{ print $1 }')
if [ "${total_num}" -eq 0 ]; then
    exit 0
fi
total_partition=$(( (total_num + MAX_PARTITION_SIZE - 1) / MAX_PARTITION_SIZE - 1 ))

if [ -n "${PARTITION_IDX}" ]; then
    partition_start="${PARTITION_IDX}"
    partition_end="${PARTITION_IDX}"
else
    partition_start=0
    partition_end="${total_partition}"
fi

for i in $(seq "${partition_start}" "${partition_end}"); do
    echo "downloading partition ${i}/${total_partition} for ${QUERY}"
    if [ -n "${TOKEN_DICT_PATH}" ]; then
        python3 "${SCRIPT_DIR}/download_partition.py" \
            --query-name "${QUERY}" \
            --index-dir "${INDEX_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --partition-idx "${i}" \
            --max-partition-size "${MAX_PARTITION_SIZE}" \
            --organism "${ORGANISM}" \
            --homology-path "${HOMOLOGY_PATH}" \
            --token-dict-path "${TOKEN_DICT_PATH}"
    else
        python3 "${SCRIPT_DIR}/download_partition.py" \
            --query-name "${QUERY}" \
            --index-dir "${INDEX_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --partition-idx "${i}" \
            --max-partition-size "${MAX_PARTITION_SIZE}" \
            --organism "${ORGANISM}" \
            --homology-path "${HOMOLOGY_PATH}"
    fi
done
