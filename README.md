# Building a causality-aware single-cell RNA-Seq foundation model via context-specific causal regulation modeling

scCAFM is a causality-aware foundation model designed for large-scale single-cell transcriptomic analysis. Unlike existing single-cell foundation models that mainly learn associative gene relationships or operate only at the dataset‐ or cell-type level, scCAFM enables cell-specific causal inference at atlas scale while simultaneously learning transferable gene and cell embeddings enriched with causal semantics. By jointly modeling gene regulatory structure and context-dependent embeddings, scCAFM provides a powerful foundation for studying heterogeneous cellular states, developmental trajectories, disease progression, and perturbation responses.

<br/>
<div align=center>
<img src="/resources/Fig1.png" width="70%">
</div>
<br/>


## Key Features
**Structure Foundation Module (SFM)**
* Efficient, context-aware causal GRN inference in a latent factor space.
* Uses a Mixture-of-Experts (MoE) architecture so different latent experts capture distinct regulatory contexts; this enables per-cell GRN specialization without learning a full causal model per cell.
* Outputs: per-cell directed edges with causal confidence, context assignment, and compact latent summaries.

**Embedding Foundation Module (EFM)**
* Learns gene and cell embeddings guided by the SFM-inferred causal structure (e.g., contrastive/cause-aware objectives).
* Embeddings are transferable: they improve downstream supervised and unsupervised tasks (drug sensitivity, perturbation response prediction, trajectory/lineage inference).


## Installation
scCAFM is a Python package for causal modeling of single-cell RNA-seq data. It requires Python 3.10–3.14 (Python 3.13.9 recommended).

First, you can download this repository and install it locally:
```bash
git clone https://github.com/Catchxu/scCAFM.git
cd scCAFM
pip install .
```
where the `resources/` and `configs/` directories are included automatically in the package, so you don’t need to copy them manually.

If you encounter the conflicts of dependencies while using scCAFM, you can report the errors at [Issues](https://github.com/Catchxu/scCAFM/issues). In this case, we also recommend that you can try installing a strict and reproducible environment which is verified that there are no conflicts:
```bash
pip install .[server]
```
where exact versions of dependencies are specified.


## Data Download
You can build and download Cellxgene-based pretraining data with one command:

```bash
python3 data/run_download_all.py \
  --query-list data/query_list.txt \
  --index-dir /path/to/index \
  --output-dir /path/to/data
```

For complete data pipeline details (workflow, options, and integrity check), see [Data Download Guide](data/README.md).


## Training Commands
Detailed end-to-end tutorials will be provided separately. This section lists the core training commands.

Taking pretraining SFM task as an example, you can:

Validate config and required files:
```bash
python3 scripts/run_pretrain.py --dry-run
```

Pretrain SFM on a single GPU:
```bash
python3 scripts/run_pretrain.py
```

Pretrain SFM on multiple GPUs (DDP):
```bash
python3 scripts/run_pretrain.py --nproc-per-node 4
```

Notes:
- Set dataset path in `configs/pretrain_sfm.yaml` (`datasets.adata_files`) first.
- In DDP, `batch_size` is per GPU process.
- Checkpoints are saved by rank 0 in `train.checkpoint_dir`.
