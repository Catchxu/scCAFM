<h1 align="center">Building a causality-aware single-cell RNA-seq foundation model via context-specific causal regulation modeling</h1>

<p align="center">
  <strong>A causality-aware foundation model for single-cell transcriptomics</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/kaichenxu/scCAFM">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-FFD21E">
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">
    <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10–3.14-3776AB?logo=python&logoColor=white">
</p>

**scCAFM** learns context-specific gene-regulatory structure together with transferable gene and cell representations from single-cell RNA sequencing data. It combines a **Structure Foundation Module (SFM)** for regulatory modeling with an **Embedding Foundation Module (EFM)** guided by the structure learned by SFM.

<p align="center">
  <img src="docs/Fig1.png" width="85%" alt="Overview of the scCAFM framework">
</p>

## What scCAFM provides

- **Cell-specific gene-regulatory networks:** infer a TF-to-target network for every cell while preserving cellular heterogeneity.
- **Pooled gene-regulatory networks:** summarize cell-specific networks into one network for a population of cells.
- **Structure-guided representations:** learn gene and cell embeddings informed by context-specific regulatory structure.
- **Human and mouse support:** use shared vocabularies, transcription-factor catalogues, and cross-species resources.
- **Memory-aware result generation:** stream cell-specific networks and optionally retain edges by score threshold or top-k selection.

The model is designed for research in gene regulation, cellular heterogeneity, perturbation response, developmental biology, and related single-cell applications.

## Install scCAFM

scCAFM supports Python 3.10–3.14. The reproducible environment described below uses Python 3.12 and includes the dependencies needed for the package and tutorials. The tutorials require one CUDA-capable NVIDIA GPU; GPU memory requirements vary with the number of genes and the inference batch size.

### Review the tested configuration

The current release has been tested with the following configuration. Other operating systems and software combinations have not been formally tested.

| Component | Tested version |
|---|---|
| Operating system | Ubuntu 24.04.4 LTS |
| Python | 3.12.13 |
| NVIDIA GPU | GeForce RTX 5090, 32 GB |
| NVIDIA driver | 580.173.02 |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| FlashAttention | 2.8.3 |

The pinned Python 3.12 environment uses these package versions:

| Package | Version |
|---|---:|
| AnnData | 0.12.10 |
| Hugging Face Hub | 1.24.0 |
| Hatchling | 1.31.0 |
| IPython | 9.12.0 |
| ipykernel | 7.2.0 |
| JupyterLab | 4.6.2 |
| matplotlib | 3.9.1 |
| NumPy | 2.4.3 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |
| safetensors | 0.7.0 |
| Scanpy | 1.12 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.1 |
| tqdm | 4.67.3 |

PyTorch and FlashAttention are listed separately in the tested configuration because they depend on the CUDA platform.

### Install the tested Python environment

Create and activate a Python 3.12 environment:

```bash
conda create -n sccafm python=3.12.13
conda activate sccafm
```

Clone the repository:

```bash
git clone https://github.com/Catchxu/scCAFM.git
cd scCAFM
```

Install the tested CUDA-enabled PyTorch build, followed by scCAFM and the remaining pinned dependencies:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install hatchling==1.31.0
pip install ".[py312]" --no-build-isolation
```

Installing scCAFM itself with `pip install .` in an environment where its dependencies are already available typically takes less than five minutes. This estimate excludes CUDA, PyTorch and FlashAttention setup, as well as model and data downloads, which depend on the system and network connection.

### Download the model and tutorial data

Download the pretrained model and shared resources from the [scCAFM model repository](https://huggingface.co/kaichenxu/scCAFM):

```bash
pip install -U huggingface_hub
hf download kaichenxu/scCAFM --local-dir assets
```

Download the prepared demonstration datasets from the [scCAFM tutorial-data repository](https://huggingface.co/datasets/kaichenxu/scCAFM-data):

```bash
hf download kaichenxu/scCAFM-data \
  --repo-type dataset \
  --local-dir tutorial_data
```

The complete tutorial-data collection is approximately 914 MB. Its directory layout already matches the paths used by the notebooks. The `assets/` and `tutorial_data/` directories are intentionally not tracked by Git.

## Explore the tutorials

We provide a series of tutorials to help users get started with scCAFM and apply it to common gene-regulatory-network tasks.

| Tutorial | What it demonstrates |
|---|---|
| [Inferring pooled GRNs from homogeneous cell populations with ChIP-seq-based benchmarking](docs/chipseq_grn_recovery.ipynb) | Preprocess hESC and mESC data, infer pooled GRNs, and compare them with ChIP-seq reference networks |
| [Inferring cell-specific GRNs from heterogeneous cell populations](docs/cell_specific_grns.ipynb) | Preprocess mouse-pancreas data, generate cell-specific GRNs, and inspect representative edges |
| [Inferring pooled GRNs from homogeneous cell populations with Perturb-seq-based validation](docs/perturbseq_edge_validation.ipynb) | Infer a pooled K562 GRN and validate highly ranked edges with Perturb-seq |

## Choose an attention backend

scCAFM supports FlashAttention-4 (FA4) and FlashAttention-2 (FA2). FA4 is intended for compatible Blackwell GPUs such as the B200; use FA2 when your hardware or software stack does not support FA4. Follow the [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention) for your CUDA and PyTorch environment.

Validate the selected backend before running a large job:

```bash
PYTHONPATH=. python test/test_FA4.py
# Or, for FA2:
PYTHONPATH=. python test/test_FA2.py
```

## Find your way around the repository

| Path | Contents |
|---|---|
| `src/sccafm/` | Public package, model implementations, preprocessing, GRN tasks, and training code |
| `docs/` | Task-oriented notebooks for GRN inference and validation |
| `configs/` | Model and training configurations |
| `data/` | Data acquisition and preparation utilities |
| `test/` and `tests/` | Backend checks and automated tests |
| `assets/` | Ignored local directory for pretrained weights and shared resources |

For dataset acquisition and vocabulary-aware preparation, see the [data pipeline guide](data/README.md). For checkpoint contents, intended use, and model limitations, see the [Hugging Face model card](https://huggingface.co/kaichenxu/scCAFM).

## Use scCAFM responsibly

scCAFM is a research model and is not intended for clinical diagnosis or treatment decisions. Predicted regulatory relationships are computational hypotheses and should be validated with suitable experimental or independent evidence. Results may vary across tissues, technologies, species, preprocessing choices, and biological contexts.

## Get support

Questions, bug reports, and feature requests are welcome through [GitHub Issues](https://github.com/Catchxu/scCAFM/issues).

## License

scCAFM is released under the [GNU General Public License v3.0](LICENSE).
