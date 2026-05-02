# Building a causality-aware single-cell RNA-seq foundation model via context-specific causal regulation modeling

scCAFM is a causality-aware foundation model designed for large-scale single-cell transcriptomic analysis. Unlike existing single-cell foundation models that mainly learn associative gene relationships or operate only at the dataset‐ or cell-type level, scCAFM enables cell-specific causal inference at atlas scale while simultaneously learning transferable gene and cell embeddings enriched with causal semantics. By jointly modeling gene regulatory structure and context-dependent embeddings, scCAFM provides a powerful foundation for studying heterogeneous cellular states, developmental trajectories, disease progression, and perturbation responses.

<br/>
<div align=center>
<img src="/docs/Fig1.png" width="70%">
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
scCAFM is a Python package for causal modeling of single-cell RNA-seq data. It requires Python 3.10–3.14 (Python 3.12.9 recommended).

First, you can download this repository and install it locally:
```bash
git clone https://github.com/Catchxu/scCAFM.git
cd scCAFM
pip install .
```
The package includes the runtime code and `configs/`, but model assets are not bundled with the install. By default, scCAFM resolves model assets from the Hugging Face model repo `kaichenxu/scCAFM` through `model_source`.

If you want a local asset copy instead of on-demand HF resolution, you can download it manually:
```bash
hf download kaichenxu/scCAFM --local-dir assets
```

If you encounter dependency conflicts while using scCAFM, please report them at [Issues](https://github.com/Catchxu/scCAFM/issues). For this repository's current Python 3.12 environment, we also provide an exact pinned extra in `pyproject.toml`:
```bash
pip install .[py312]
```

Please note that GPU-specific packages such as FlashAttention still depend on your CUDA, PyTorch, compiler, and GPU stack.


## FlashAttention
scCAFM uses FlashAttention in the transformer attention path when `flash-attn` is installed in your environment.

Current behavior:
* The runtime prefers the newer CuTe / FA4-style kernels when available.
* If the CuTe / FA4 path raises an error, scCAFM automatically falls back to the FA2 kernels.
* Attention dropout is disabled in this path, so FA4 and FA2 use the same effective behavior.

We recommend installing FlashAttention for your specific hardware and software environment by following the official repository instructions:

* https://github.com/Dao-AILab/flash-attention

After installation, you can validate the local attention backend with:

```bash
python -m src.models.attention
```


## Hugging Face Assets
scCAFM now uses an asset package that is compatible with both a local directory and a [Hugging Face](https://huggingface.co/kaichenxu/scCAFM) model repository. The canonical asset files are:

* `models/sfm_config.json`
* `models/sfm_model.safetensors`
* `models/cond_dict.json`
* `models/vocab.json`
* root-level CSV sidecars such as `human_tfs.csv`, `mouse_tfs.csv`, `OmniPath.csv`, and `homologous.csv`

Runtime code resolves these files through a single `model_source` setting:

* If `model_source` is a local directory, scCAFM loads assets from that directory directly.
* Otherwise, scCAFM treats `model_source` as our Hugging Face model ID, `kaichenxu/scCAFM`, and downloads the asset package before loading.

Project configs now default to:

```yaml
model_source: kaichenxu/scCAFM
```

For local development, you can still point to a locally downloaded `assets/` directory instead:

```yaml
model_source: assets
```

This makes it easy to work with either the published HF repo or a locally downloaded `assets/` directory. In the current package layout, model files and JSON dictionaries live under `models/`, while supporting CSV files stay at the asset root.


## Data Download
The data pipeline supports both `Homo sapiens` and `Mus musculus`, writes species-specific folders, adds a `species` column to each downloaded partition, and can keep only genes found in `assets/models/vocab.json`. The supported workflow is now shell-based, including a small demo download before the full SLURM run.

For complete data pipeline details, see [Data Download Guide](data/README.md).
