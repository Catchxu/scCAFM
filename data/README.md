# Building a pretraining cell corpus from Cellxgene Census
This folder contains scripts for building large-scale single-cell datasets from Cellxgene Census for scCAFM pretraining. The workflow is designed for SLURM-based clusters and assumes internet access during Census queries and downloads.


## What this pipeline does
* Queries cells from Cellxgene Census using predefined tissue-level filters.
* Builds SOMA index files for selected query groups.
* Downloads filtered data in partitioned `.h5ad` chunks.


## Prerequisites
* A SLURM environment (`sbatch`) for array jobs.
* Python environment with dependencies used by scripts in `data/`.
* Network access from compute nodes to Cellxgene Census.


## Data preparation workflow
1. (Optional) Customize query configuration.
2. Build SOMA index files from query groups.
3. Download partitioned `.h5ad` data.


## One-command run
Run the full workflow (build index + download all partitions) in one command:

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


## Integrity check and auto-repair
Check all expected partitions, delete broken `.h5ad`, and redownload missing/broken files:

```bash
python3 data/check_and_redownload_h5ad.py \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```


## (1) Optional: customize query configuration
Update these files if you want custom data coverage:
* `data/data_config.py`
* `data/query_list.txt`
* `data/cancer_list.txt` (for cancer filtering settings)


Key settings in `data/data_config.py`:
* `MAJOR_TISSUE_LIST`: high-level tissue groups used in querying.
* `VERSION`: Cellxgene Census release version.
* `DISEASE`-related filters: controls healthy/cancer subsets.


## (2) Build SOMA index files
Create index files (SOMA IDs) for each query group:

```bash
INDEX_PATH="path/to/index"
QUERY_PATH="path/to/query_list.txt"

bash data/build_soma_idx.sh "$INDEX_PATH" "$QUERY_PATH"
```

Notes:
* `INDEX_PATH` should be reused in the next step.
* `QUERY_PATH` should be consistent with your configured query list.


## (3) Download `.h5ad` partitions
Configure paths in `data/array_download_partition.sh`:
* `DATA_PATH`
* `INDEX_PATH`
* `QUERY_PATH`

Then submit:

```bash
sbatch data/array_download_partition.sh
```

Each task downloads partitioned `.h5ad` chunks. Partition size is controlled by `MAX_PARTITION_SIZE` in `data/download_partition.sh`.


## Output
The result of this pipeline is a directory tree of downloaded `.h5ad` files organized by query groups, ready for downstream preprocessing and model training.
