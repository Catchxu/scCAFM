# Building a causality-aware single-cell RNA-Seq foundation model via context-specific causal regulation modeling

scCAFM is a causality-aware foundation model designed for large-scale single-cell transcriptomic analysis. Unlike existing single-cell foundation models that mainly learn associative gene relationships or operate only at the dataset‐ or cell-type level, scCAFM enables cell-specific causal inference at atlas scale while simultaneously learning transferable gene and cell embeddings enriched with causal semantics. By jointly modeling gene regulatory structure and context-dependent embeddings, scCAFM provides a powerful foundation for studying heterogeneous cellular states, developmental trajectories, disease progression, and perturbation responses.

<br/>
<div align=center>
<img src="/img/framework.png" width="70%">
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
scCAFM is developed as a Python package. You will need to install Python, and the recommended version is Python 3.13.9.

You can download this repository and install it locally:
```bash
git clone https://github.com/Catchxu/scCAFM.git
cd scCAFM
pip install .
```
where the files in `resources/` and `configs/` have been included.
