# Data Pipeline Guide

This directory contains two separate workflows:

1. Build the pretraining cell corpus from Cellxgene Census.
2. Build initial gene embeddings from GO annotations with PubMedBERT.


## Directory Overview

### Cell-corpus scripts
* `build_soma_idx.py` and `build_soma_idx.sh`: build SOMA index files for query groups.
* `download_partition.py` and related shell scripts: download partitioned `.h5ad` files.
* `run_download_all.py`: one-command entrypoint for index building and downloads.
* `check_and_redownload_h5ad.py`: integrity check and auto-repair for downloaded `.h5ad` files.

### Gene-embedding scripts
* `build_gene_go_terms.py`: query Ensembl BioMart and export GO annotations to `resources/gene_go_terms.csv`.
* `build_gene_embeddings.py`: build gene embeddings from GO-term text with PubMedBERT.
* `build_gene_embeddings.sh`: run both gene-related steps together.


## Workflow A: Build the Cell Corpus

### What this workflow does
* Query cells from Cellxgene Census using predefined filters.
* Build SOMA index files for each query group.
* Download filtered single-cell data as partitioned `.h5ad` files.

### Requirements
* A Python environment with dependencies used by the scripts in `data/`.
* Internet access to Cellxgene Census.
* A SLURM environment if you want to use the provided array-job scripts.

### Main configuration files
* `data/data_config.py`: query settings and filter definitions.
* `data/query_list.txt`: query groups to process.
* `data/cancer_list.txt`: cancer-related filtering settings.

### Recommended order
1. Customize the query configuration if needed.
2. Build SOMA index files.
3. Download `.h5ad` partitions.
4. Optionally run integrity check and redownload broken files.

### One-command run
```bash
python3 data/run_download_all.py \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```

Useful flags:
* `--resume`: skip partitions that already exist.
* `--skip-index`: reuse existing `.idx` files.
* `--max-partition-size 200000`: control partition size.

### Step 1: Build SOMA index files
```bash
INDEX_PATH="/path/to/index"
QUERY_PATH="data/query_list.txt"

bash data/build_soma_idx.sh "$INDEX_PATH" "$QUERY_PATH"
```

Notes:
* Reuse the same `INDEX_PATH` in the download step.
* Make sure `QUERY_PATH` matches your configured query list.

### Step 2: Download `.h5ad` partitions
Before submitting jobs, configure these variables in `data/array_download_partition.sh`:
* `DATA_PATH`
* `INDEX_PATH`
* `QUERY_PATH`

Then submit:
```bash
sbatch data/array_download_partition.sh
```

### Step 3: Integrity check and auto-repair
```bash
python3 data/check_and_redownload_h5ad.py \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```

### Output
This workflow produces a directory tree of partitioned `.h5ad` files for downstream preprocessing and model training.


## Workflow B: Build Initial Gene Embeddings

### What this workflow does
1. Read `resources/token_dict.csv`.
2. Query Ensembl BioMart for GO annotations.
3. Save filtered GO annotations to `resources/gene_go_terms.csv`.
4. Convert each GO annotation into text:

```text
GO term: {go_term_name}. Domain: {go_domain}.
```

5. Run PubMedBERT to embed each GO text.
6. Mean-pool GO-text embeddings into one embedding per gene.
7. Use `GO term: unknown. Domain: unknown.` for genes without GO terms.
8. Save the final checkpoint to `checkpoints/gene_embeddings.pt`.

### Requirements
* Internet access for `build_gene_go_terms.py`.
* CUDA GPUs for `build_gene_embeddings.py`.
* A PubMedBERT checkpoint available at `/data1021/xukaichen/data/LLM/pubmedbert-base-embeddings`, or another compatible path.

### Recommended order
1. Build GO-term annotations.
2. Build embeddings from those annotations.

### One-command run
```bash
bash data/build_gene_embeddings.sh
```

### Common environment variables for the shell script
* `GPUS`: comma-separated GPU ids, such as `0` or `0,1,2,3`.
* `BATCH_SIZE`: embedding batch size. Default is `1024`.
* `MODEL_NAME`: PubMedBERT model path.
* `TOKEN_DICT`: token dictionary path. Default is `resources/token_dict.csv`.
* `INPUT_CSV`: GO-term CSV path. Default is `resources/gene_go_terms.csv`.
* `OUTPUT_CKPT`: output checkpoint path. Default is `checkpoints/gene_embeddings.pt`.

Example:
```bash
GPUS=0,1,2,3 BATCH_SIZE=1024 bash data/build_gene_embeddings.sh
```

### Run each step separately
Build GO annotations:
```bash
python data/build_gene_go_terms.py
```

Build gene embeddings:
```bash
python data/build_gene_embeddings.py \
  --gpus 0,1,2,3 \
  --batch-size 1024
```

### Output
The final checkpoint contains:
* `gene_symbols`: a list of gene symbols.
* `embeddings`: a `torch.Tensor` of shape `(num_genes, 768)`.


## Practical Notes

### Network-dependent steps
* Cell-corpus querying depends on Cellxgene Census availability.
* GO-term export depends on Ensembl BioMart availability.

### GPU-dependent steps
* `build_gene_embeddings.py` requires CUDA.
* Multi-GPU inference is supported through `--gpus`.

### Related files outside `data/`
* `resources/token_dict.csv`: gene vocabulary source.
* `resources/gene_go_terms.csv`: GO annotations used for embedding generation.
* `checkpoints/gene_embeddings.pt`: saved initial gene embeddings.
