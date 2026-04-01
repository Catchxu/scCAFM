#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUERY=$1
INDEX_DIR=$2
OUTPUT_DIR=$3
ORGANISM=${4:-"Homo sapiens"}
TOKEN_DICT_PATH=${5:-""}

MAX_PARTITION_SIZE=200000

ORGANISM_DIR=$(PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from data_config import organism_output_name; import sys; print(organism_output_name(sys.argv[1]))' "${ORGANISM}")
total_num=`wc -l ${INDEX_DIR}/${ORGANISM_DIR}/${QUERY}.idx | awk '{ print $1 }'`
total_partition=$(($total_num / $MAX_PARTITION_SIZE))
# echo $total_num
# echo $total_partition"

for i in $(seq 0 $total_partition)
do
    echo "downloading partition ${i}/${total_partition} for ${QUERY}"
    if [ -n "${TOKEN_DICT_PATH}" ]; then
        python3 "${SCRIPT_DIR}/download_partition.py" \
            --query-name ${QUERY} \
            --index-dir ${INDEX_DIR} \
            --output-dir ${OUTPUT_DIR} \
            --partition-idx ${i} \
            --max-partition-size ${MAX_PARTITION_SIZE} \
            --organism "${ORGANISM}" \
            --token-dict-path "${TOKEN_DICT_PATH}"
    else
        python3 "${SCRIPT_DIR}/download_partition.py" \
            --query-name ${QUERY} \
            --index-dir ${INDEX_DIR} \
            --output-dir ${OUTPUT_DIR} \
            --partition-idx ${i} \
            --max-partition-size ${MAX_PARTITION_SIZE} \
            --organism "${ORGANISM}"
    fi
done
