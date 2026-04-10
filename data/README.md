# Data Pipeline Guide

This directory contains:

1. Cell download scripts for building pretraining partitions from Cellxgene Census.
2. Gene embedding scripts for rebuilding `checkpoints/models/vocab.safetensors` from GO annotations.


## Cell Download

### What it does
* Build SOMA index files for each query in `data/query_list.txt`.
* Download partitioned `.h5ad` files from Cellxgene Census.
* Support both `Homo sapiens` and `Mus musculus`.
* Save partitions into species-specific folders.
* Add `obs["species"]` to each partition.
* Optionally keep only genes found in `assets/models/vocab.json`.
* For mouse downloads, remap `feature_id` and `feature_name` through `assets/homologous.csv` so the exported files align with the human token vocabulary while preserving original mouse feature columns as `source_feature_id` and `source_feature_name`.
* Drop `soma_joinid` from both `obs` and `var`, and drop all `obs` columns ending with `_term_id` before saving.

### Preferred workflow
The supported workflow is shell-based:
* `data/demo_download.sh` for a quick smoke test
* `data/build_soma_idx.sh` to build index files
* `data/download_partition.sh` to download one query for one organism
* `data/array_download_partition.sh` to run the full SLURM array

### Demo download
```bash
bash data/demo_download.sh /path/to/index /path/to/data
```

Equivalent positional form:
```bash
bash data/demo_download.sh /path/to/index /path/to/data brain
```

This builds the index and downloads only `partition_0.h5ad` for:
* `human/brain`
* `mouse/brain`

The demo uses a smaller partition size by default:
```bash
DEMO_MAX_PARTITION_SIZE=5000 bash data/demo_download.sh /path/to/index /path/to/data brain
```

### Build index files
Build all index files for one organism:
```bash
bash data/build_soma_idx.sh /path/to/index data/query_list.txt "Homo sapiens"
```

Build mouse indexes:
```bash
bash data/build_soma_idx.sh /path/to/index data/query_list.txt "Mus musculus"
```

### Download partitions
Download all partitions for one query and one organism:
```bash
bash data/download_partition.sh \
  brain \
  /path/to/index \
  /path/to/data \
  "Homo sapiens" \
  assets/models/vocab.json \
  assets/homologous.csv
```

Download only partition `0`:
```bash
bash data/download_partition.sh \
  brain \
  /path/to/index \
  /path/to/data \
  "Mus musculus" \
  assets/models/vocab.json \
  assets/homologous.csv \
  0
```

### Output layout
```text
/path/to/index/human/<query>.idx
/path/to/index/mouse/<query>.idx
/path/to/data/human/<query>/partition_0.h5ad
/path/to/data/mouse/<query>/partition_0.h5ad
```

### SLURM download
Set these variables in `data/array_download_partition.sh`:
* `INDEX_PATH`
* `QUERY_PATH`
* `DATA_PATH`
* `TOKEN_DICT_PATH`
* `HOMOLOGY_PATH`

Then run:
```bash
sbatch data/array_download_partition.sh
```

### Integrity check
```bash
python3 data/check_and_redownload_h5ad.py \
  --organism "Homo sapiens" \
  --token-dict-path assets/models/vocab.json \
  --homology-path assets/homologous.csv \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```


## Gene Embeddings

### What it does
1. Read `assets/models/vocab.json`.
2. Query Ensembl BioMart for GO annotations.
3. Save GO annotations to `checkpoints/gene_go_terms.csv`.
4. Encode GO text with PubMedBERT.
5. Save the final gene embedding asset to `checkpoints/models/vocab.safetensors`.

### One-command run
```bash
bash data/build_gene_embeddings.sh
```

Example:
```bash
GPUS=0,1,2,3 BATCH_SIZE=1024 bash data/build_gene_embeddings.sh
```

### Step-by-step
```bash
python data/build_gene_go_terms.py
python data/build_gene_embeddings.py --gpus 0,1,2,3 --batch-size 1024
```

### Output
* `checkpoints/gene_go_terms.csv`
* `checkpoints/models/vocab.safetensors`
