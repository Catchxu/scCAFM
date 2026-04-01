# Data Pipeline Guide

This directory contains:

1. Cell download scripts for building pretraining partitions from Cellxgene Census.
2. Gene embedding scripts for building initial gene embeddings from GO annotations.


## Cell Download

### What it does
* Build SOMA index files for each query in `data/query_list.txt`.
* Download partitioned `.h5ad` files from Cellxgene Census.
* Support both `Homo sapiens` and `Mus musculus`.
* Save partitions into species-specific folders.
* Add `obs["species"]` to each partition.
* Optionally keep only genes found in `resources/token_dict.csv`.

### One-command run
```bash
python3 data/run_download_all.py \
  --organism "Homo sapiens" \
  --token-dict-path resources/token_dict.csv \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```

Useful flags:
* `--organism "Homo sapiens"` or `--organism "Mus musculus"`
* `--token-dict-path resources/token_dict.csv`
* `--resume`
* `--skip-index`
* `--max-partition-size 200000`

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
* `ORGANISM`
* `TOKEN_DICT_PATH`

Then run:
```bash
sbatch data/array_download_partition.sh
```

### Integrity check
```bash
python3 data/check_and_redownload_h5ad.py \
  --organism "Homo sapiens" \
  --token-dict-path resources/token_dict.csv \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```


## Gene Embeddings

### What it does
1. Read `resources/token_dict.csv`.
2. Query Ensembl BioMart for GO annotations.
3. Save GO annotations to `resources/gene_go_terms.csv`.
4. Encode GO text with PubMedBERT.
5. Save the final checkpoint to `checkpoints/gene_embeddings.pt`.

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
* `resources/gene_go_terms.csv`
* `checkpoints/gene_embeddings.pt`
