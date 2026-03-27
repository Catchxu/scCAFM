#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MODEL_NAME="${MODEL_NAME:-/data1021/xukaichen/data/LLM/pubmedbert-base-embeddings}"
TOKEN_DICT="${TOKEN_DICT:-$ROOT_DIR/resources/token_dict.csv}"
INPUT_CSV="${INPUT_CSV:-$ROOT_DIR/resources/gene_go_terms.csv}"
OUTPUT_CKPT="${OUTPUT_CKPT:-$ROOT_DIR/checkpoints/gene_embeddings.pt}"

python "$ROOT_DIR/data/build_gene_go_terms.py"

python "$ROOT_DIR/data/build_gene_embeddings.py" \
  --model-name "$MODEL_NAME" \
  --token-dict "$TOKEN_DICT" \
  --input-csv "$INPUT_CSV" \
  --output-ckpt "$OUTPUT_CKPT" \
  --batch-size "$BATCH_SIZE" \
  --gpus "$GPUS"
