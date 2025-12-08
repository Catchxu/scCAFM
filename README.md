# Building a causality-aware single-cell RNA-Seq foundation model via context-specific causal regulation modeling

scCAFM is a causality-aware foundation model designed for large-scale single-cell transcriptomic analysis. Unlike existing single-cell foundation models that mainly learn associative gene relationships or operate only at the dataset‐ or cell-type level, scCAFM enables cell-specific causal inference at atlas scale while simultaneously learning transferable gene and cell embeddings enriched with causal semantics. By jointly modeling gene regulatory structure and context-dependent embeddings, scCAFM provides a powerful foundation for studying heterogeneous cellular states, developmental trajectories, disease progression, and perturbation responses.

<br/>
<div align=center>
<img src="/img/Fig1.png" width="70%">
</div>
<br/>

At its core, scCAFM couples two synergistic components:
1. **Structure Foundation Module (SFM)**
Performs efficient context-specific causal gene regulatory network (GRN) inference using a Mixture-of-Experts mediated latent factorization, enabling scalable estimation of directed cell-specific GRNs.

2. **Embedding Foundation Module (EFM)**
Learns semantic gene and cell representations whose embedding geometry is explicitly shaped by causal structure inferred via SFM. This yields representations that are biologically meaningful, transferable across datasets, and predictive in downstream applications.