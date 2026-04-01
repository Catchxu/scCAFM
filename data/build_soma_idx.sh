#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_DIR="${1:?Usage: build_soma_idx.sh OUTPUT_DIR [QUERY_LIST] [ORGANISM]}"
QUERY_LIST="${2:-${SCRIPT_DIR}/query_list.txt}"
ORGANISM="${3:-Homo sapiens}"

while IFS= read -r QUERY || [ -n "${QUERY}" ]; do
    if [ -z "${QUERY}" ] || [[ "${QUERY}" == \#* ]]; then
        continue
    fi
    echo "building index for ${QUERY} (${ORGANISM})"
    python3 "${SCRIPT_DIR}/build_soma_idx.py" \
        --query-name "${QUERY}" \
        --output-dir "${OUTPUT_DIR}" \
        --organism "${ORGANISM}"
done < "${QUERY_LIST}"
