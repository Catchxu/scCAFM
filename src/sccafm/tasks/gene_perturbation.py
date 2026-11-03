"""Prediction of post-perturbation gene expression from unperturbed cells."""

from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch.utils.data import DataLoader

from ..assets import resolve_model_assets
from ..data import ScTokenizer
from ..models.efm import EFM, reorder_gene_aligned_tokens
from ..models.perturbation import GEARSPerturbationHead
from ..models.sfm import SFM
from ._efm import configure_efm_freezing, set_efm_training_mode
from .grn import GRNInferencer
from .perturbation_data import PerturbationGraphs


@dataclass(frozen=True)
class GenePerturbationPredictions:
    """Condition means; n_control_cells counts sampled controls per condition."""

    mean_expression: pd.DataFrame
    n_control_cells: int

    def to_expression_table(self) -> pd.DataFrame:
        return self.mean_expression.copy()


@dataclass(frozen=True)
class GenePerturbationEvaluation:
    """Condition-level metrics; group summaries weight conditions equally."""

    conditions: pd.DataFrame

    def to_condition_table(self) -> pd.DataFrame:
        return self.conditions.copy()

    def to_group_table(self) -> pd.DataFrame:
        metrics = ["mse", "mse_de20", "pearson_delta", "pearson_delta_de20"]
        groups = self.conditions.groupby("evaluation_group", observed=True)
        summary = groups[metrics].mean()
        summary.insert(0, "n_conditions", groups.size())
        summary.insert(1, "n_cells", groups["n_cells"].sum())
        summary["n_pearson_defined"] = groups["pearson_delta"].count()
        summary["n_pearson_de20_defined"] = groups["pearson_delta_de20"].count()
        overall = self.conditions[metrics].mean().to_frame().T
        overall.index = pd.Index(["overall"], name="evaluation_group")
        overall.insert(0, "n_conditions", len(self.conditions))
        overall.insert(1, "n_cells", self.conditions["n_cells"].sum())
        overall["n_pearson_defined"] = self.conditions["pearson_delta"].count()
        overall["n_pearson_de20_defined"] = self.conditions["pearson_delta_de20"].count()
        return pd.concat([overall, summary])


def rank_perturbation_response_genes(
    observed: AnnData, *, condition_key: str = "condition", control_label: str = "ctrl",
) -> dict[str, list[str]]:
    """Read condition-wise source or newly computed DE rankings.

    The caller attaches source rankings or computes them on normalized expression.
    Gene IDs are filtered to the selected expression axis before taking DE20.
    Validation rankings select checkpoints; test rankings are used only to score.
    """
    present = set(observed.var_names)
    rankings = observed.uns["rank_genes_groups_cov_all"]
    return {
        condition: [
            str(g) for g in rankings[cells.obs["condition_name"].iloc[0]]
            if str(g) in present
        ][:20]
        for condition, cells in (
            (str(condition), observed[observed.obs[condition_key].astype(str).eq(str(condition))])
            for condition in observed.obs[condition_key].astype(str).unique()
            if str(condition) != control_label
        )
    }


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    # A constant predicted delta has no defined Pearson coefficient.
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_gene_perturbation(
    predictions: GenePerturbationPredictions,
    observed: AnnData,
    *,
    de_genes: dict[str, list[str]],
    control_mean: np.ndarray,
    condition_key: str = "condition",
) -> GenePerturbationEvaluation:
    """Compare condition means on a fixed gene panel and control reference.

    control_mean is the unperturbed reference mean on the prediction gene axis,
    normally the mean of the common training-control input pool.
    de_genes contains DE gene IDs selected on the fitted gene panel. NaN Pearson values remain
    explicitly undefined; they are not silently replaced with zero.
    """
    truth = observed[:, predictions.mean_expression.columns]
    rows = []
    for condition, predicted in predictions.mean_expression.iterrows():
        cells = truth[truth.obs[condition_key].eq(condition)]
        target = np.asarray(cells.X.mean(axis=0)).ravel()
        de = truth.var_names.get_indexer(de_genes[condition])
        if np.any(de < 0):
            raise ValueError("Evaluation DE genes must belong to the fitted gene panel.")
        estimate = predicted.to_numpy()
        target_delta = target - control_mean
        predicted_delta = estimate - control_mean
        rows.append({
            "condition": condition,
            "evaluation_group": cells.obs["evaluation_group"].iloc[0],
            "n_cells": cells.n_obs,
            "mse": float(np.mean((estimate - target) ** 2)),
            "mse_de20": float(np.mean((estimate[de] - target[de]) ** 2)),
            "pearson_delta": _pearson(predicted_delta, target_delta),
            "pearson_delta_de20": _pearson(predicted_delta[de], target_delta[de]),
        })
    return GenePerturbationEvaluation(pd.DataFrame(rows).set_index("condition"))


class GenePerturbationPredictor:
    """Train a GEARS expression decoder on cell-specific EFM gene states.

    fit() receives separate training and validation cells. predict() receives unperturbed cells and
    intervention identities, with no perturbed outcome expression. The SFM stays
    frozen. freeze_layers controls the leading EFM blocks, exactly as in the cell
    type annotation API. use_gene_context=False supplies a GEARS baseline using
    learned gene embeddings on the same panel, graphs, split and training loss.
    """

    def __init__(
        self, *, model: EFM, ordering_model: SFM, tokenizer: ScTokenizer,
        tokenizer_config: dict, vocabulary_signature: str,
        device: str | torch.device, autocast_dtype: torch.dtype | None,
        hidden_size: int, use_gene_context: bool,
    ) -> None:
        self.model = model
        self.ordering_model = ordering_model
        self.ordering_model.requires_grad_(False)
        self.ordering_model.eval()
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.vocabulary_signature = vocabulary_signature
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.hidden_size = hidden_size
        self.use_gene_context = use_gene_context
        self.attention_mode = "cell_global"
        self.head = None

    @classmethod
    def from_pretrained(
        cls, model_source: str | Path, *,
        device: str | torch.device = "cuda",
        attention_backend: str | None = None,
        max_length: int = 2048,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        hidden_size: int = 64,
        use_gene_context: bool = True,
        platform_key: str | None = None,
        species_key: str | None = "species",
        tissue_key: str | None = None,
        disease_key: str | None = None,
    ) -> GenePerturbationPredictor:
        assets = resolve_model_assets(
            model_source, require_efm_config=True, require_efm_weights=True,
            require_cond_dict=True, require_resources=True,
        )
        model = EFM.from_pretrained(
            model_source, device=device, attention_backend=attention_backend
        )
        config = dict(
            max_length=max_length, platform_key=platform_key, species_key=species_key,
            tissue_key=tissue_key, disease_key=disease_key,
        )
        ordering = GRNInferencer.from_pretrained(
            model_source, device=device, attention_backend=attention_backend,
            autocast_dtype=autocast_dtype, **config,
        )
        return cls(
            model=model, ordering_model=ordering.model, tokenizer=ordering.tokenizer,
            tokenizer_config=config, vocabulary_signature=hashlib.sha256(
                assets.vocab.read_bytes() + assets.cond_dict.read_bytes()
            ).hexdigest(),
            device=device, autocast_dtype=autocast_dtype,
            hidden_size=hidden_size, use_gene_context=use_gene_context,
        )

    def _autocast(self):
        if self.device.type == "cuda" and self.autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def _new_head(self) -> GEARSPerturbationHead:
        return GEARSPerturbationHead(
            num_genes=len(self.gene_names_),
            num_perturbations=len(self.perturbation_genes_),
            context_dim=self.model.embed_dim,
            hidden_size=self.hidden_size,
            ontology_edges=self.graphs_.ontology_edges,
            ontology_weights=self.graphs_.ontology_weights,
            use_gene_context=self.use_gene_context,
        ).to(self.device)

    def _prepare_controls(self, controls: AnnData, batch_size: int, *, verbose: bool = False):
        tokens = self.tokenizer(controls, gene_key=None)
        values = {
            "input_ids": tokens.input_ids,
            "expression_values": tokens.expression_values,
            "condition_ids": tokens.condition_ids,
            "padding_mask": tokens.padding_mask,
            "non_tf_mask": tokens.non_tf_mask,
        }
        positions = []
        with torch.no_grad():
            loader = DataLoader(range(controls.n_obs), batch_size=batch_size)
            for step, indices in enumerate(loader, 1):
                batch = self._token_batch(values, indices)
                with self._autocast():
                    _, order = self.ordering_model(
                        batch, return_factors=False, compute_grn=False, compute_order=True
                    )
                positions.append(order.positions.cpu())
                if verbose and step % 20 == 0:
                    print(f"SFM ordering controls: batch {step}/{len(loader)}", flush=True)
        positions = torch.cat(positions)
        return reorder_gene_aligned_tokens(values, positions), positions.argsort(1)

    def _token_batch(self, values, indices):
        return {
            key: None if value is None else value[indices].to(self.device)
            for key, value in values.items()
        }

    def _cache_frozen_prefix(self, tokens, batch_size: int, *, verbose: bool):
        # Only deterministic, frozen modules enter this CPU cache. Its full
        # precision matches the residual stream; trainable layers are never cached.
        self.model.eval()
        n_controls, n_tokens = tokens["input_ids"].shape
        cached = torch.empty(
            n_controls, n_tokens + 1, self.model.embed_dim,
            dtype=self.model.embedding.gene_embedding.weight.dtype,
        )
        loader = DataLoader(range(n_controls), batch_size=batch_size)
        with torch.no_grad():
            for step, indices in enumerate(loader, 1):
                batch = self._token_batch(tokens, indices)
                with self._autocast():
                    embedded = self.model.embedding(**batch)
                    states = embedded.embeddings
                    for layer in self.model.backbone.layers[:self.freeze_layers_]:
                        states = layer(
                            states, key_padding_mask=embedded.key_padding_mask,
                            causal=True, global_cell_token=True,
                        )
                cached[indices] = states.cpu()
                if verbose and step % 50 == 0:
                    print(f"Frozen EFM prefix: batch {step}/{len(loader)}", flush=True)
        return cached

    def _contexts(self, tokens, inverse, indices, frozen_prefix=None):
        if frozen_prefix is None:
            states = self.model.encode_gene_context(
                self._token_batch(tokens, indices), attention_mode=self.attention_mode
            )
        else:
            states = frozen_prefix[indices].to(self.device)
            padding = tokens["padding_mask"]
            if padding is not None:
                padding = torch.cat([
                    torch.zeros(len(indices), 1, dtype=torch.bool, device=self.device),
                    padding[indices].to(self.device),
                ], dim=1)
            for layer in self.model.backbone.layers[self.freeze_layers_:]:
                states = layer(
                    states, key_padding_mask=padding, causal=True, global_cell_token=True
                )
            states = self.model.backbone.final_norm(states)
            if padding is not None:
                states = states.masked_fill(padding.unsqueeze(-1), 0.0)
            states = states[:, 1:]
        restored = states.gather(
            1, inverse[indices].to(self.device).unsqueeze(-1).expand_as(states)
        )
        return restored[:, :len(self.gene_names_)]

    def _cache_control_contexts(self, tokens, inverse, batch_size: int, *, verbose: bool):
        """Cache canonical gene states per control cell, never a pooled gene table."""
        self.model.eval()
        n_controls = len(tokens["input_ids"])
        features = torch.empty(
            n_controls, len(self.gene_names_), self.model.embed_dim, dtype=torch.float32
        )
        loader = DataLoader(range(n_controls), batch_size=batch_size)
        with torch.no_grad():
            for step, indices in enumerate(loader, 1):
                with self._autocast():
                    states = self._contexts(tokens, inverse, indices)
                features[indices] = states.float().cpu()
                if verbose and step % 50 == 0:
                    print(f"Cell-specific gene contexts: batch {step}/{len(loader)}", flush=True)
        return features

    def _perturbation_masks(self, conditions):
        lookup = {g: i for i, g in enumerate(self.perturbation_genes_)}
        masks = torch.zeros(len(conditions), len(lookup), dtype=torch.float32)
        for i, condition in enumerate(conditions):
            if condition != self.control_label_:
                for gene in condition.split("+"):
                    if gene != self.control_label_:
                        masks[i, lookup[gene]] = 1
        return masks

    def _capture_state(self):
        modules = {"head": self.head}
        if self.use_gene_context:
            modules.update(embedding=self.model.embedding, backbone=self.model.backbone)
        return {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()
        }

    def _restore_state(self, state):
        self.head.load_state_dict(state["head"], strict=True)
        if self.use_gene_context:
            self.model.embedding.load_state_dict(state["embedding"], strict=True)
            self.model.backbone.load_state_dict(state["backbone"], strict=True)

    def _metadata(self):
        return {
            "format": "sccafm-gene-perturbation-v2",
            "tokenizer_config": self.tokenizer_config,
            "vocabulary_signature": self.vocabulary_signature,
            "hidden_size": self.hidden_size,
            "use_gene_context": self.use_gene_context,
            "attention_mode": self.attention_mode,
            "gene_names": self.gene_names_,
            "perturbation_genes": self.perturbation_genes_,
            "graphs": self.graphs_.state_dict(),
            "control_label": self.control_label_,
            "condition_key": self.condition_key_,
            "control_mean": self.control_mean_,
            "freeze_layers": self.freeze_layers_,
            "history": self.history_,
            "best_epoch": self.best_epoch_,
            "best_validation_mse_de": self.best_validation_mse_de_,
            "selection": "Lowest validation condition-mean DE20 MSE",
        }

    def save_checkpoint(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save task weights and their gene/graph axes, without altering assets."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**self._metadata(), "state": self._capture_state()}, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, model_source: str | Path,
        device: str | torch.device = "cuda",
        attention_backend: str | None = None,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ) -> GenePerturbationPredictor:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["format"] != "sccafm-gene-perturbation-v2":
            raise ValueError("Not a scCAFM gene-perturbation checkpoint.")
        predictor = cls.from_pretrained(
            model_source, device=device, attention_backend=attention_backend,
            autocast_dtype=autocast_dtype, hidden_size=saved["hidden_size"],
            use_gene_context=saved["use_gene_context"], **saved["tokenizer_config"],
        )
        if predictor.vocabulary_signature != saved["vocabulary_signature"]:
            raise ValueError("Checkpoint and pretrained vocabulary do not match.")
        predictor.gene_names_ = tuple(saved["gene_names"])
        predictor.perturbation_genes_ = tuple(saved["perturbation_genes"])
        predictor.graphs_ = PerturbationGraphs(**saved["graphs"])
        predictor.control_label_ = saved["control_label"]
        predictor.condition_key_ = saved["condition_key"]
        predictor.control_mean_ = saved["control_mean"]
        predictor.freeze_layers_ = saved["freeze_layers"]
        predictor.history_ = saved["history"]
        predictor.best_epoch_ = saved["best_epoch"]
        predictor.best_validation_mse_de_ = saved["best_validation_mse_de"]
        predictor.attention_mode = saved["attention_mode"]
        predictor.head = predictor._new_head()
        predictor._restore_state(saved["state"])
        configure_efm_freezing(predictor.model, predictor.freeze_layers_)
        predictor.model.eval()
        predictor.head.eval()
        return predictor

    def fit(
        self, train_adata: AnnData, *, validation_adata: AnnData,
        graphs: PerturbationGraphs,
        condition_key: str = "condition", control_label: str = "ctrl",
        control_id_key: str = "control_cell_id",
        freeze_layers: int = 12,
        epochs: int = 20, batch_size: int = 32,
        evaluation_batch_size: int = 128, ordering_batch_size: int = 32,
        learning_rate: float = 1e-3, backbone_learning_rate: float = 1e-5,
        weight_decay: float = 5e-4, direction_weight: float = 0.1,
        lr_step_size: int = 1, lr_gamma: float = 0.5,
        early_stopping_patience: int | None = None,
        random_state: int = 0, verbose: bool = True,
        cache_frozen_prefix: bool = False,
        training_checkpoint: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> GenePerturbationPredictor:
        """Train with the reference GEARS loss, Adam and validation selection.

        Controls and targets use the fixed control_cell_id pairs prepared once
        before fitting. StepLR uses lr_step_size and lr_gamma; its defaults halve
        Adam's learning rate each epoch. Gradient values are clipped to [-1, 1].
        Training drops its incomplete final batch.
        The head runs in float32, including its BatchNorm statistics.

        freeze_layers counts frozen leading EFM blocks as in cell annotation.
        Freezing every block caches deterministic, cell-specific per-gene states
        in CPU memory. Partial/full fine-tuning keeps the trainable EFM path live.
        SFM stays frozen. EFM inference uses the configured autocast precision.

        Validation uses the preprocessing DE ranking on the fitted gene panel.
        The returned model is the epoch with the lowest condition-mean DE20 MSE.
        early_stopping_patience optionally stops after that many consecutive
        epochs without validation improvement. Recovery retains the latest
        optimizer/scheduler/RNG and best model states.
        """
        if epochs < 1 or batch_size < 2:
            raise ValueError("epochs must be positive and GEARS batch_size at least two.")
        condition_fields = {
            self.tokenizer_config[key]
            for key in ("platform_key", "species_key", "tissue_key", "disease_key")
        }
        if condition_key in condition_fields:
            raise ValueError("Intervention labels must not be EFM condition metadata.")
        if not self.use_gene_context and freeze_layers != self.model.backbone.num_layers:
            raise ValueError("The learned-embedding GEARS baseline freezes every EFM block.")
        self.condition_key_ = condition_key
        self.control_label_ = control_label
        self.gene_names_ = tuple(train_adata.var_names)
        if self.gene_names_ != tuple(graphs.gene_names):
            raise ValueError("Graph and training-expression gene axes must match.")
        validation = validation_adata[:, list(self.gene_names_)]
        train_conditions = set(train_adata.obs[condition_key]) - {control_label}
        if train_conditions.intersection(validation.obs[condition_key]):
            raise ValueError("Training and validation perturbation conditions must be disjoint.")
        self.perturbation_genes_ = tuple(graphs.perturbation_genes)
        self.graphs_ = graphs
        self.freeze_layers_ = freeze_layers
        configure_efm_freezing(self.model, freeze_layers)
        torch.manual_seed(random_state)
        self.head = self._new_head()
        controls = train_adata[train_adata.obs[condition_key].eq(control_label)]
        control_x = torch.tensor(controls.X.toarray(), dtype=torch.float32)
        self.control_mean_ = control_x.mean(0)
        ctrl_mean = self.control_mean_.to(self.device)

        def prepare_dataset(adata):
            names = tuple(adata.obs[condition_key].astype(str).unique())
            lookup = {name: i for i, name in enumerate(names)}
            ids = torch.tensor([lookup[p] for p in adata.obs[condition_key].astype(str)])
            paired = controls.obs_names.get_indexer(adata.obs[control_id_key].astype(str))
            if np.any(paired < 0):
                raise ValueError("Every fixed control pair must identify a training control.")
            de = rank_perturbation_response_genes(
                adata, condition_key=condition_key, control_label=control_label
            )
            return {
                "names": names, "ids": ids,
                "x": torch.tensor(adata.X.toarray(), dtype=torch.float32),
                "paired": torch.tensor(paired),
                "masks": self._perturbation_masks(names).to(self.device),
                "de": {name: adata.var_names.get_indexer(genes) for name, genes in de.items()},
                "cells": tuple(adata.obs_names),
            }

        train = prepare_dataset(train_adata)
        valid = prepare_dataset(validation)
        active = torch.stack([
            (train["x"][train["ids"] == i] != 0).any(0)
            if name != control_label
            else torch.ones(len(self.gene_names_), dtype=torch.bool)
            for i, name in enumerate(train["names"])
        ]).to(self.device)
        groups = [{"params": self.head.parameters(), "lr": learning_rate}]
        if self.use_gene_context and freeze_layers < self.model.backbone.num_layers:
            groups.append({
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": backbone_learning_rate,
            })
        optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        signature = hashlib.sha256()
        for dataset, adata in ((train, train_adata), (valid, validation)):
            for value in (dataset["x"], dataset["ids"], dataset["paired"]):
                signature.update(value.numpy().tobytes())
            signature.update("\n".join(dataset["cells"]).encode())
            signature.update("\n".join(dataset["names"]).encode())
            for field in sorted(condition_fields - {None}):
                signature.update(pd.util.hash_pandas_object(
                    adata.obs[field], index=False
                ).to_numpy().tobytes())
            for name in dataset["de"]:
                signature.update(name.encode())
                signature.update(dataset["de"][name].tobytes())
        for value in (graphs.ontology_edges, graphs.ontology_weights):
            signature.update(value.numpy().tobytes())
        spec = {
            "protocol": "reference-gears-v2",
            "input_sha256": signature.hexdigest(),
            "gene_names": self.gene_names_,
            "perturbation_genes": self.perturbation_genes_,
            "vocabulary_signature": self.vocabulary_signature,
            "tokenizer_config": self.tokenizer_config,
            "hidden_size": self.hidden_size, "use_gene_context": self.use_gene_context,
            "freeze_layers": freeze_layers, "batch_size": batch_size,
            "evaluation_batch_size": evaluation_batch_size,
            "ordering_batch_size": ordering_batch_size,
            "learning_rate": learning_rate, "backbone_learning_rate": backbone_learning_rate,
            "weight_decay": weight_decay, "direction_weight": direction_weight,
            "lr_step_size": lr_step_size, "lr_gamma": lr_gamma,
            "early_stopping_patience": early_stopping_patience,
            "random_state": random_state, "autocast_dtype": str(self.autocast_dtype),
        }
        path = Path(training_checkpoint) if training_checkpoint is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and (resume_from is None or Path(resume_from).resolve() != path.resolve()):
                raise FileExistsError(path)
        self.history_ = []
        self.best_epoch_ = 0
        self.best_validation_mse_de_ = float("inf")
        first_epoch = 0
        best_state = None
        tokens, inverse = None, None
        if resume_from is not None:
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            if saved["training_spec"] != spec:
                raise ValueError("Recovery inputs, DE rankings or training settings differ.")
            self._restore_state(saved["state"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            tokens, inverse = saved["ordered_tokens"], saved["inverse_order"]
            first_epoch = saved["completed_epochs"]
            self.history_ = saved["history"]
            self.best_epoch_ = saved["best_epoch"]
            self.best_validation_mse_de_ = saved["best_validation_mse_de"]
            best_state = saved["best_state"]
            torch.set_rng_state(saved["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"], self.device)
        features, prefix = None, None
        # Feature preparation must not alter the training sampler/dropout RNG.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.use_gene_context:
                if tokens is None:
                    tokens, inverse = self._prepare_controls(
                        controls, ordering_batch_size, verbose=verbose
                    )
                if freeze_layers == self.model.backbone.num_layers:
                    features = self._cache_control_contexts(tokens, inverse, batch_size, verbose=verbose)
                elif cache_frozen_prefix:
                    if freeze_layers == 0:
                        raise ValueError("A trainable EFM prefix cannot be cached.")
                    prefix = self._cache_frozen_prefix(tokens, batch_size, verbose=verbose)

        def predict_batch(dataset, indices):
            paired = dataset["paired"][indices]
            x = control_x[paired].to(self.device)
            context = None
            if self.use_gene_context:
                if features is not None:
                    context = features[paired].to(self.device)
                else:
                    with self._autocast():
                        context = self._contexts(tokens, inverse, paired, prefix)
                    context = context.float()
            ids = dataset["ids"][indices].to(self.device)
            return self.head(x, dataset["masks"][ids], context), ids

        def metrics(loader, dataset):
            self.model.eval()
            self.head.eval()
            predictions, targets, condition_ids = [], [], []
            with torch.no_grad():
                for indices in loader:
                    prediction, ids = predict_batch(dataset, indices)
                    predictions.append(prediction.cpu().numpy())
                    targets.append(dataset["x"][indices].numpy())
                    condition_ids.append(ids.cpu().numpy())
            prediction = np.concatenate(predictions)
            truth = np.concatenate(targets)
            ids = np.concatenate(condition_ids)
            overall, differential = [], []
            for i in np.unique(ids):
                rows = ids == i
                error = prediction[rows].mean(0) - truth[rows].mean(0)
                overall.append(np.mean(error ** 2))
                name = dataset["names"][i]
                if name != control_label:
                    differential.append(np.mean(error[dataset["de"][name]] ** 2))
            return float(np.mean(overall)), float(np.mean(differential))

        train_loader = DataLoader(
            range(len(train["x"])), batch_size=batch_size, shuffle=True, drop_last=True
        )
        validation_loader = DataLoader(
            range(len(valid["x"])), batch_size=evaluation_batch_size, shuffle=True
        )
        for epoch in range(first_epoch, epochs):
            if early_stopping_patience is not None and (
                epoch - self.best_epoch_ >= early_stopping_patience
            ):
                if verbose:
                    print(f"Early stopping after {epoch} epochs; best epoch={self.best_epoch_}.", flush=True)
                break
            started = time.monotonic()
            set_efm_training_mode(self.model, freeze_layers)
            self.head.train()
            running = torch.zeros((), device=self.device)
            used = 0
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            for step, indices in enumerate(train_loader, 1):
                optimizer.zero_grad(set_to_none=True)
                prediction, ids = predict_batch(train, indices)
                truth = train["x"][indices].to(self.device)
                retain = active[ids]
                per_cell = (
                    ((prediction - truth).pow(4) + direction_weight * (
                        torch.sign(truth - ctrl_mean) - torch.sign(prediction - ctrl_mean)
                    ).square()) * retain
                ).sum(1) / retain.sum(1)
                _, group = ids.unique(return_inverse=True)
                count = torch.bincount(group)
                grouped = torch.zeros(len(count), device=self.device)
                grouped.index_add_(0, group, per_cell)
                loss = (grouped / count).mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(
                    [p for group_ in optimizer.param_groups for p in group_["params"]], 1.0
                )
                optimizer.step()
                running += loss.detach() * len(indices)
                used += len(indices)
                if verbose and step % 250 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: batch {step}/{len(train_loader)}", flush=True)
            scheduler.step()
            train_mse, train_de = metrics(train_loader, train)
            val_mse, val_de = metrics(validation_loader, valid)
            row = {
                "epoch": epoch + 1, "loss": float(running / used),
                "train_mse": train_mse, "train_mse_de20": train_de,
                "validation_mse": val_mse, "validation_mse_de20": val_de,
                "learning_rate": learning_rates[0],
                "seconds": time.monotonic() - started,
            }
            if not np.isfinite(row["loss"]) or not np.isfinite(val_de):
                raise FloatingPointError(f"Non-finite loss or validation score at epoch {epoch + 1}.")
            self.history_.append(row)
            if val_de < self.best_validation_mse_de_:
                self.best_validation_mse_de_ = val_de
                self.best_epoch_ = epoch + 1
                best_state = self._capture_state()
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: loss={row['loss']:.6f}, "
                    f"validation DE20 MSE={val_de:.6f}, best epoch={self.best_epoch_}, "
                    f"{row['seconds']:.1f}s", flush=True,
                )
            if path is not None:
                recovery = {
                    **self._metadata(), "state": self._capture_state(),
                    "best_state": best_state, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "training_spec": spec,
                    "ordered_tokens": tokens, "inverse_order": inverse,
                    "completed_epochs": epoch + 1,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                }
                temporary = path.with_suffix(path.suffix + ".pending")
                torch.save(recovery, temporary)
                temporary.replace(path)
        self._restore_state(best_state)
        self.model.eval()
        self.head.eval()
        return self

    def predict(
        self, control_adata: AnnData, *,
        perturbations: list[str] | list[list[str]],
        batch_size: int = 300, ordering_batch_size: int = 32,
        n_control_samples: int = 300, random_state: int = 1,
        verbose: bool = False,
    ) -> GenePerturbationPredictions:
        """Average 300 sampled control predictions per intervention by default.

        Sampling is with replacement from the supplied control pool, matching
        GEARS.predict(). Raw condition strings retain their labels in the output;
        lists of gene names produce a '+'-joined query label. No response cells
        are accepted as prediction inputs.
        """
        if not control_adata.obs[self.condition_key_].eq(self.control_label_).all():
            raise ValueError("predict() requires unperturbed control cells only.")
        controls = control_adata[:, list(self.gene_names_)]
        conditions = [p if isinstance(p, str) else "+".join(p) for p in perturbations]
        if len(set(conditions)) != len(conditions):
            raise ValueError("Prediction conditions must be distinct.")
        masks = self._perturbation_masks(conditions).to(self.device)
        generator = np.random.RandomState(random_state)
        samples = generator.randint(0, controls.n_obs, size=(len(conditions), n_control_samples))
        used, inverse_sample = np.unique(samples, return_inverse=True)
        sample_rows = torch.tensor(inverse_sample.reshape(samples.shape))
        selected = controls[used]
        control_x = torch.tensor(selected.X.toarray(), dtype=torch.float32)
        features = None
        self.model.eval()
        self.head.eval()
        if self.use_gene_context:
            tokens, inverse = self._prepare_controls(selected, ordering_batch_size, verbose=verbose)
            features = self._cache_control_contexts(tokens, inverse, 32, verbose=verbose)
        results = []
        with torch.no_grad():
            for i, condition in enumerate(conditions):
                outputs = []
                for indices in sample_rows[i].split(batch_size):
                    context = features[indices].to(self.device) if self.use_gene_context else None
                    prediction = self.head(
                        control_x[indices].to(self.device),
                        masks[i].expand(len(indices), -1), context,
                    )
                    outputs.append(prediction.cpu().numpy())
                results.append(np.concatenate(outputs).mean(axis=0))
                if verbose and (i + 1) % 20 == 0:
                    print(f"Predicted {i + 1}/{len(conditions)} interventions", flush=True)
        return GenePerturbationPredictions(
            pd.DataFrame(
                np.stack(results),
                index=pd.Index(conditions, name="condition"),
                columns=pd.Index(self.gene_names_, name="gene_id"),
            ),
            n_control_samples,
        )
