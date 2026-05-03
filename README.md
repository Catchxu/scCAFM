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

Install the package from a local clone:
```bash
git clone https://github.com/Catchxu/scCAFM.git
cd scCAFM
pip install .
```

The package install includes the runtime code and `configs/`, but **not model assets** (such as pretrained model files and external prior knowledge). scCAFM resolves assets on the Hugging Face [model repo](https://huggingface.co/kaichenxu/scCAFM) (ID: `kaichenxu/scCAFM`). We recommand to download a local copy as:
```bash
cd scCAFM
hf download kaichenxu/scCAFM --local-dir assets
```

If you encounter dependency conflicts while using scCAFM, please report them at [Issues](https://github.com/Catchxu/scCAFM/issues). For this repository's current Python 3.12 environment, we also provide an exact pinned extra in `pyproject.toml`:
```bash
pip install .[py312]
```
Please note that GPU-specific packages such as FlashAttention still depend on your CUDA, PyTorch, compiler, and GPU stack.


## FlashAttention
scCAFM are developed with the latest FlashAttention-4 (FA4) to enhance the compuational performance, where FA4 is optimized and only available for Blackwell GPUs (e.g. B200). If your hardware or FlashAttention build only supports FA2, change config files under `configs` as:
```yaml
runtime:
  attention_backend: fa2
```

If you haven't installed any FA, please install suitable FA according to your specific hardware and software environment. You can follow the [official repository instructions](https://github.com/Dao-AILab/flash-attention) to install it.

Before training on a new machine, it can be helpful to run the backend smoke test:
```bash
python test/FA4.py # or test/FA2.py
```
The smoke tests use scCAFM's `FlashMHA` wrapper, so they check the same FA2/FA4 path used by training and evaluation.


## Data Download
The data pipeline supports both `Homo sapiens` and `Mus musculus`, writes species-specific folders, adds a `species` column to each downloaded partition, and can keep only genes found in `assets/models/vocab.json`. The supported workflow is now shell-based, including a small demo download before the full SLURM run.

For complete data pipeline details, see [Data Download Guide](data/README.md).
