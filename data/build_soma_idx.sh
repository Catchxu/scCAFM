#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# output directory for the index 
OUTPUT_DIR=$1
QUERY_LIST=$2
ORGANISM=${3:-"Homo sapiens"}

while read QUERY; do
    echo "building index for ${QUERY}"
    python3 "${SCRIPT_DIR}/build_soma_idx.py" --query-name ${QUERY} --output-dir ${OUTPUT_DIR} --organism "${ORGANISM}"
done < ${QUERY_LIST}
