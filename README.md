# Building a causality-aware single-cell RNA-seq foundation model via context-specific causal regulation modeling

scCAFM is a causality-aware foundation model designed for large-scale single-cell transcriptomic analysis. Unlike existing single-cell foundation models that mainly learn associative gene relationships or operate only at the dataset‐ or cell-type level, scCAFM enables cell-specific causal inference at atlas scale while simultaneously learning transferable gene and cell embeddings enriched with causal semantics. By jointly modeling gene regulatory structure and context-dependent embeddings, scCAFM provides a powerful foundation for studying heterogeneous cellular states, developmental trajectories, disease progression, and perturbation responses.

<br/>
<div align=center>
<img src="/docs/Fig1.png" width="70%">
</div>
<br/>


## Key features
**Structure foundation module (SFM)**
* Efficient, context-aware causal GRN inference in a latent factor space.
* Uses a Mixture-of-Experts (MoE) architecture so different latent experts capture distinct regulatory contexts; this enables per-cell GRN specialization without learning a full causal model per cell.
* Outputs: per-cell directed edges with causal confidence, context assignment, and compact latent summaries.

**Embedding foundation module (EFM)**
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
where the `assets/` and `configs/` directories are included automatically in the package, so you don’t need to copy them manually.

If you encounter the conflicts of dependencies while using scCAFM, you can report the errors at [Issues](https://github.com/Catchxu/scCAFM/issues). In this case, we also recommend that you can try installing a strict and reproducible environment which is verified that there are no conflicts:
```bash
pip install .[server]
```
where exact versions of dependencies are specified.

For better attention efficiency, we strongly recommend installing FlashAttention for your specific hardware/software environment by following the official installation instructions in the FlashAttention repository: https://github.com/Dao-AILab/flash-attention. After installation, please run the upstream validation tests from the same repository to confirm everything is working correctly.


## Hugging Face assets
scCAFM now uses an asset package that is compatible with both a local directory and a Hugging Face model repository. The canonical asset files are:

* `sfm_config.json`
* `sfm_model.safetensors`
* `efm_config.json`
* `efm_model.safetensors`
* `vocab.json`
* `vocab.safetensors`
* supporting CSV files such as `cond_dict.csv`, `human_tfs.csv`, `mouse_tfs.csv`, `OmniPath.csv` and `homologous.csv`

The canonical Hugging Face model ID for this project is:

* `kaichenxu/scCAFM`

Runtime code resolves these files through a single `model_source` setting:

* If `model_source` is a local directory, scCAFM loads assets from that directory directly.
* Otherwise, scCAFM treats `model_source` as a Hugging Face model ID and downloads the asset package before loading.

Project configs now default to:

```yaml
model_source: kaichenxu/scCAFM
```

For local development, you can still point to the bundled assets directory instead:

```yaml
model_source: assets
```

This makes it easy to work with either the published HF repo or the bundled local `assets/` directory, both of which follow the same flat file layout.


## Data download
The data pipeline supports both `Homo sapiens` and `Mus musculus`, writes species-specific folders, adds a `species` column to each downloaded partition, and can keep only genes found in `assets/vocab.json`. The supported workflow is now shell-based, including a small demo download before the full SLURM run.

For complete data pipeline details, see [Data Download Guide](data/README.md).
