#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INDEX_DIR="${1:?Usage: demo_download.sh INDEX_DIR OUTPUT_DIR [QUERY] [TOKEN_DICT_PATH] [HOMOLOGY_PATH]}"
OUTPUT_DIR="${2:?Usage: demo_download.sh INDEX_DIR OUTPUT_DIR [QUERY] [TOKEN_DICT_PATH] [HOMOLOGY_PATH]}"
QUERY="${3:-brain}"
TOKEN_DICT_PATH="${4:-${ROOT_DIR}/assets/models/vocab.json}"
HOMOLOGY_PATH="${5:-${ROOT_DIR}/assets/homologous.csv}"
DEMO_MAX_PARTITION_SIZE="${DEMO_MAX_PARTITION_SIZE:-5000}"

for ORGANISM in "Homo sapiens" "Mus musculus"; do
    echo "building demo index for ${QUERY} (${ORGANISM})"
    python3 "${SCRIPT_DIR}/build_soma_idx.py" \
        --query-name "${QUERY}" \
        --output-dir "${INDEX_DIR}" \
        --organism "${ORGANISM}"

    echo "downloading demo partition 0 for ${QUERY} (${ORGANISM})"
    MAX_PARTITION_SIZE="${DEMO_MAX_PARTITION_SIZE}" "${SCRIPT_DIR}/download_partition.sh" \
        "${QUERY}" \
        "${INDEX_DIR}" \
        "${OUTPUT_DIR}" \
        "${ORGANISM}" \
        "${TOKEN_DICT_PATH}" \
        "${HOMOLOGY_PATH}" \
        0
done

echo "demo download finished"
