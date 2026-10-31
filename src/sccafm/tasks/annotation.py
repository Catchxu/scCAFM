"""Cell-type annotation by supervised fine-tuning of EFM cell states."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from ..assets import resolve_model_assets
from ..data import ScTokenizer
from ..data.tokenizer import ScTokenizerOutput
from ..models.efm import EFM, reorder_gene_aligned_tokens
from ..models.sfm import SFM
from .grn import GRNInferencer


def _cell_labels(adata: AnnData, label_key: str) -> np.ndarray:
    labels = adata.obs[label_key]
    if labels.isna().any():
        raise ValueError("Cell-type labels must not contain missing values.")
    return labels.astype(str).to_numpy()


@dataclass(frozen=True, slots=True)
class CellTypePredictions:
    """Predictions and EFM cell-state embeddings, in input cell order.

    Probabilities and embeddings are CPU float32 tensors. The maximum softmax
    probability is a model confidence score, not a calibrated probability of
    biological correctness or an unknown-cell detector.
    """

    cell_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    probabilities: torch.Tensor
    cell_embeddings: torch.Tensor

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def predicted_labels(self) -> tuple[str, ...]:
        return tuple(
            self.class_names[index]
            for index in self.probabilities.argmax(dim=1).tolist()
        )

    def to_cell_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "cell_id": self.cell_ids,
            "predicted_cell_type": self.predicted_labels,
            "confidence": self.probabilities.max(dim=1).values.numpy(),
        })

    def to_probability_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.probabilities.numpy(),
            index=pd.Index(self.cell_ids, name="cell_id"),
            columns=self.class_names,
        )


@dataclass(frozen=True, slots=True)
class CellTypeEvaluation:
    """Closed-set annotation metrics aligned by cell ID."""

    n_cells: int
    n_classes: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion: pd.DataFrame
    _class_table: pd.DataFrame

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_cells": self.n_cells,
            "n_classes": self.n_classes,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }

    def to_class_table(self) -> pd.DataFrame:
        return self._class_table.copy()


def evaluate_cell_type_annotation(
    predictions: CellTypePredictions,
    adata: AnnData,
    *,
    label_key: str = "cell_type",
) -> CellTypeEvaluation:
    """Evaluate predictions against labels; unseen test classes are not discarded."""
    truth = _cell_labels(adata[list(predictions.cell_ids)], label_key)
    unknown = set(truth).difference(predictions.class_names)
    if unknown:
        raise ValueError(
            f"Closed-set evaluation contains unseen cell types: {sorted(unknown)}."
        )
    labels = predictions.predicted_labels
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, labels, labels=list(predictions.class_names), zero_division=0
    )
    matrix = confusion_matrix(
        truth, labels, labels=list(predictions.class_names)
    )
    return CellTypeEvaluation(
        n_cells=predictions.n_cells,
        n_classes=len(predictions.class_names),
        accuracy=float(accuracy_score(truth, labels)),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        confusion=pd.DataFrame(
            matrix,
            index=pd.Index(predictions.class_names, name="True cell type"),
            columns=pd.Index(predictions.class_names, name="Predicted cell type"),
        ),
        _class_table=pd.DataFrame({
            "cell_type": predictions.class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }),
    )


class CellTypeAnnotator:
    """Annotate cells by supervised fine-tuning of global EFM cell states.

    Inputs are preprocessed AnnData objects using the pretrained gene vocabulary.
    Fit preprocessing and gene selection on training data only, then apply the
    same expression transformation and selected genes to query cells.
    Labels are used only to train/evaluate the classifier, never as condition
    tokens. Predictions are closed-set over classes observed during fit().
    """

    def __init__(
        self,
        *,
        model: EFM,
        ordering_model: SFM,
        tokenizer: ScTokenizer,
        tokenizer_config: dict,
        vocabulary_signature: str,
        device: str | torch.device,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.1,
    ) -> None:
        self.model = model
        self.model.eval()
        self.ordering_model = ordering_model
        self.ordering_model.requires_grad_(False)
        self.ordering_model.eval()
        self.attention_mode = "cell_global"
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.vocabulary_signature = vocabulary_signature
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.classifier_config = {
            "hidden_dim": classifier_hidden_dim,
            "dropout": classifier_dropout,
        }
        self.classifier: nn.Module | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_source: str | Path,
        *,
        device: str | torch.device = "cuda",
        attention_backend: str | None = None,
        max_length: int = 2048,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        platform_key: str | None = None,
        species_key: str | None = "species",
        tissue_key: str | None = None,
        disease_key: str | None = None,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.1,
    ) -> CellTypeAnnotator:
        assets = resolve_model_assets(
            model_source,
            require_efm_config=True,
            require_efm_weights=True,
            require_cond_dict=True,
            require_resources=True,
        )
        model = EFM.from_pretrained(
            model_source, device=device, attention_backend=attention_backend
        )
        tokenizer_config = {
            "max_length": max_length,
            "platform_key": platform_key,
            "species_key": species_key,
            "tissue_key": tissue_key,
            "disease_key": disease_key,
        }
        ordering = GRNInferencer.from_pretrained(
            model_source,
            device=device,
            attention_backend=attention_backend,
            autocast_dtype=autocast_dtype,
            **tokenizer_config,
        )
        signature = hashlib.sha256(
            assets.vocab.read_bytes() + assets.cond_dict.read_bytes()
        ).hexdigest()
        return cls(
            model=model,
            ordering_model=ordering.model,
            tokenizer=ordering.tokenizer,
            tokenizer_config=tokenizer_config,
            vocabulary_signature=signature,
            device=device,
            autocast_dtype=autocast_dtype,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
        )

    def _new_classifier(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.model.embed_dim, self.classifier_config["hidden_dim"]),
            nn.SiLU(),
            nn.Dropout(self.classifier_config["dropout"]),
            nn.Linear(self.classifier_config["hidden_dim"], len(self.class_names_)),
        ).to(self.device)

    def _autocast_context(self):
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _token_batch(
        self, tokenized: ScTokenizerOutput, indices: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        values = {
            "input_ids": tokenized.input_ids,
            "expression_values": tokenized.expression_values,
            "condition_ids": tokenized.condition_ids,
            "padding_mask": tokenized.padding_mask,
            "non_tf_mask": tokenized.non_tf_mask,
        }
        return {
            name: None if value is None else value[indices].to(self.device)
            for name, value in values.items()
        }

    def _prepare_tokens(
        self, adata: AnnData, gene_key: str | None, batch_size: int,
        *, verbose: bool = False,
    ) -> ScTokenizerOutput:
        """Tokenize expression and apply pretrained SFM gene order."""
        tokenized = self.tokenizer(adata, gene_key=gene_key)
        return self._order_tokens(tokenized, batch_size, verbose=verbose)

    def _order_tokens(
        self, tokenized: ScTokenizerOutput, batch_size: int,
        *, verbose: bool = False,
    ) -> ScTokenizerOutput:
        """Compute SFM order for tokenized cells."""
        positions = []
        self.ordering_model.eval()
        loader = DataLoader(range(len(tokenized.input_ids)), batch_size=batch_size)
        with torch.no_grad():
            for batch_number, indices in enumerate(loader, start=1):
                with self._autocast_context():
                    _, gene_order = self.ordering_model(
                        self._token_batch(tokenized, indices),
                        return_factors=False,
                        compute_grn=False,
                        compute_order=True,
                    )
                positions.append(gene_order.positions.cpu())
                if verbose and batch_number % 200 == 0:
                    print(
                        f"SFM gene ordering: batch {batch_number}/{len(loader)}",
                        flush=True,
                    )
        values = {
            "input_ids": tokenized.input_ids,
            "expression_values": tokenized.expression_values,
            "condition_ids": tokenized.condition_ids,
            "padding_mask": tokenized.padding_mask,
            "non_tf_mask": tokenized.non_tf_mask,
        }
        ordered = reorder_gene_aligned_tokens(values, torch.cat(positions))
        return ScTokenizerOutput(
            **ordered, gene_name_type=tokenized.gene_name_type
        )

    def _encode(
        self, tokenized: ScTokenizerOutput, batch_size: int,
        *, verbose: bool = False,
    ) -> torch.Tensor:
        self.model.eval()
        features = []
        loader = DataLoader(range(len(tokenized.input_ids)), batch_size=batch_size)
        with torch.no_grad():
            for batch_number, indices in enumerate(loader, start=1):
                with self._autocast_context():
                    state = self.model.encode_cell_state(
                        self._token_batch(tokenized, indices),
                        attention_mode=self.attention_mode,
                    )
                features.append(state.float().cpu())
                if verbose and batch_number % 200 == 0:
                    print(
                        f"EFM cell-state encoding: batch {batch_number}/{len(loader)}",
                        flush=True,
                    )
        return torch.cat(features)

    def _align_features(self, adata: AnnData) -> AnnData:
        genes = (
            adata.var_names if self.gene_key_ is None
            else pd.Index(adata.var[self.gene_key_].astype(str))
        )
        positions = genes.get_indexer(self.gene_names_)
        if np.any(positions < 0):
            missing = np.asarray(self.gene_names_)[positions < 0].tolist()
            raise ValueError(f"Query data are missing training genes: {missing}.")
        return adata[:, positions]

    def _state_modules(self) -> dict[str, nn.Module]:
        return {
            "embedding": self.model.embedding,
            "backbone": self.model.backbone,
            "classifier": self.classifier,
        }

    def _capture_state(self) -> dict:
        return {
            "modules": {
                name: {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                for name, module in self._state_modules().items()
            },
        }

    def _restore_state(self, state: dict) -> None:
        for name, module in self._state_modules().items():
            module.load_state_dict(state["modules"][name], strict=True)

    def fit(
        self,
        train_adata: AnnData,
        *,
        label_key: str = "cell_type",
        gene_key: str | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        ordering_batch_size: int = 64,
        learning_rate: float = 1e-3,
        backbone_learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        class_weight: str | None = None,
        random_state: int = 42,
        verbose: bool = True,
        training_checkpoint: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> CellTypeAnnotator:
        """Supervise the EFM encoder and classifier for a fixed number of epochs.

        SFT updates the complete EFM embedding module, transformer backbone and
        classifier. SFM supplies pretrained gene orders in evaluation mode.
        EFM expression-prediction heads are unused by this task. SFM inference
        uses ordering_batch_size independently of the SFT batch_size.

        training_checkpoint saves the latest completed epoch, optimizer, cached
        gene order and RNG states. resume_from continues that state with the same
        inputs and training settings. Recovery never selects weights by a score.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        condition_keys = {
            self.tokenizer_config[key]
            for key in ("platform_key", "species_key", "tissue_key", "disease_key")
        }
        if label_key in condition_keys:
            raise ValueError("The target label must not be used as condition metadata.")
        torch.manual_seed(random_state)
        train_labels = _cell_labels(train_adata, label_key)
        self.class_names_ = tuple(sorted(set(train_labels)))
        class_to_id = {
            name: index for index, name in enumerate(self.class_names_)
        }
        train_y = torch.tensor([class_to_id[x] for x in train_labels])
        self.gene_key_ = gene_key
        self.gene_names_ = tuple(
            train_adata.var_names.astype(str) if gene_key is None
            else train_adata.var[gene_key].astype(str)
        )
        self.classifier = self._new_classifier()
        self.model.requires_grad_(False)
        self.model.embedding.requires_grad_(True)
        self.model.backbone.requires_grad_(True)

        progress_path = Path(training_checkpoint) if training_checkpoint is not None else None
        resume_state = None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            if progress_path.exists() and (
                resume_from is None or Path(resume_from).resolve() != progress_path.resolve()
            ):
                raise FileExistsError(progress_path)
        if training_checkpoint is not None or resume_from is not None:
            raw_tokens = self.tokenizer(train_adata, gene_key=gene_key)
            digest = hashlib.sha256()
            for value in (
                raw_tokens.input_ids, raw_tokens.expression_values,
                raw_tokens.condition_ids, raw_tokens.padding_mask, raw_tokens.non_tf_mask,
            ):
                if value is not None:
                    digest.update(value.numpy().tobytes())
            training_spec = {
                "input_sha256": digest.hexdigest(),
                "label_sha256": hashlib.sha256(train_y.numpy().tobytes()).hexdigest(),
                "cell_ids": list(train_adata.obs_names),
                "gene_names": list(self.gene_names_),
                "class_names": list(self.class_names_),
                "tokenizer_config": self.tokenizer_config,
                "classifier_config": self.classifier_config,
                "vocabulary_signature": self.vocabulary_signature,
                "batch_size": batch_size, "ordering_batch_size": ordering_batch_size,
                "learning_rate": learning_rate,
                "backbone_learning_rate": backbone_learning_rate,
                "weight_decay": weight_decay, "class_weight": class_weight,
                "random_state": random_state, "autocast_dtype": str(self.autocast_dtype),
            }
            if resume_from is not None:
                resume_state = torch.load(resume_from, map_location="cpu", weights_only=True)
                if resume_state["training_spec"] != training_spec:
                    raise ValueError("Resume requires the same training inputs and settings.")
                if resume_state["epoch"] > epochs:
                    raise ValueError("The recovery checkpoint exceeds the requested epoch count.")
                self._restore_state(resume_state["state"])
                train_tokens = ScTokenizerOutput(**resume_state["tokens"])
            else:
                if verbose:
                    print("Computing SFM gene orders for training cells.")
                train_tokens = self._order_tokens(
                    raw_tokens, ordering_batch_size, verbose=verbose
                )
            del raw_tokens
        else:
            if verbose:
                print("Computing SFM gene orders for training cells.")
            train_tokens = self._prepare_tokens(
                train_adata, gene_key, ordering_batch_size, verbose=verbose
            )
        weights = None
        if class_weight == "balanced":
            counts = torch.bincount(train_y, minlength=len(self.class_names_))
            weights = (
                len(train_y) / (len(self.class_names_) * counts.float())
            ).to(self.device)
        elif class_weight is not None:
            raise ValueError("class_weight must be None or 'balanced'.")
        criterion = nn.CrossEntropyLoss(weight=weights)
        groups = [
            {"params": self.classifier.parameters(), "lr": learning_rate},
            {
                "params": [
                    *self.model.embedding.parameters(),
                    *self.model.backbone.parameters(),
                ],
                "lr": backbone_learning_rate,
            },
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.autocast_dtype == torch.float16,
        )
        parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        generator = torch.Generator().manual_seed(random_state)
        loader = DataLoader(
            range(train_adata.n_obs),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        history = []
        first_epoch = 1
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            scaler.load_state_dict(resume_state["scaler"])
            generator.set_state(resume_state["generator_rng"])
            torch.set_rng_state(resume_state["torch_rng"])
            torch.cuda.set_rng_state(resume_state["cuda_rng"], self.device)
            history = resume_state["history"]
            first_epoch = resume_state["epoch"] + 1
            if verbose:
                print(f"Resuming after epoch {resume_state['epoch']}.", flush=True)
            del resume_state
        for epoch in range(first_epoch, epochs + 1):
            self.model.train()
            self.classifier.train()
            total_loss = 0.0
            for batch_number, indices in enumerate(loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    state = self.model.encode_cell_state(
                        self._token_batch(train_tokens, indices),
                        attention_mode=self.attention_mode,
                    )
                    logits = self.classifier(state)
                    loss = criterion(logits, train_y[indices].to(self.device))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(indices)
                if verbose and batch_number % 200 == 0:
                    print(
                        f"Epoch {epoch:02d}/{epochs}, "
                        f"batch {batch_number}/{len(loader)}: loss={loss.item():.4f}",
                        flush=True,
                    )

            history.append({
                "epoch": epoch,
                "train_loss": total_loss / train_adata.n_obs,
            })
            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs}: "
                    f"train loss={history[-1]['train_loss']:.4f}",
                    flush=True,
                )
            if progress_path is not None:
                tokens = {
                    "input_ids": train_tokens.input_ids,
                    "expression_values": train_tokens.expression_values,
                    "condition_ids": train_tokens.condition_ids,
                    "padding_mask": train_tokens.padding_mask,
                    "non_tf_mask": train_tokens.non_tf_mask,
                    "gene_name_type": train_tokens.gene_name_type,
                }
                temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
                with temporary.open("xb") as handle:
                    torch.save({
                        "epoch": epoch, "training_spec": training_spec,
                        "state": self._capture_state(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "history": history, "tokens": tokens,
                        "generator_rng": generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(self.device),
                    }, handle)
                temporary.replace(progress_path)
                if verbose:
                    print(f"Saved recovery state after epoch {epoch}.", flush=True)
        self.epochs_ = epochs
        self.history_ = pd.DataFrame(history)
        self.model.eval()
        self.classifier.eval()
        return self

    def predict(
        self, adata: AnnData, *, batch_size: int = 64,
        ordering_batch_size: int = 64, verbose: bool = False,
    ) -> CellTypePredictions:
        """Predict known cell types without reading or requiring query labels."""
        if self.classifier is None:
            raise RuntimeError("Fit the annotator or load a checkpoint before predict().")
        prepared = self._align_features(adata)
        tokenized = self._prepare_tokens(
            prepared, self.gene_key_, ordering_batch_size, verbose=verbose
        )
        features = self._encode(tokenized, batch_size, verbose=verbose)
        self.classifier.eval()
        probabilities = []
        with torch.no_grad():
            for indices in DataLoader(range(adata.n_obs), batch_size=batch_size):
                with self._autocast_context():
                    logits = self.classifier(features[indices].to(self.device))
                probabilities.append(logits.float().softmax(dim=1).cpu())
        return CellTypePredictions(
            cell_ids=tuple(adata.obs_names.astype(str)),
            class_names=self.class_names_,
            probabilities=torch.cat(probabilities),
            cell_embeddings=features,
        )

    def save_checkpoint(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save task weights, vocabulary identity, label mapping and feature schema."""
        if self.classifier is None:
            raise RuntimeError("Fit the annotator before saving a checkpoint.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing results unless explicitly overridden.
        with path.open("wb" if overwrite else "xb") as handle:
            torch.save({
                "format_version": 4,
                "attention_mode": self.attention_mode,
                "tokenizer_config": self.tokenizer_config,
                "vocabulary_signature": self.vocabulary_signature,
                "classifier_config": self.classifier_config,
                "class_names": list(self.class_names_),
                "gene_names": list(self.gene_names_),
                "gene_key": self.gene_key_,
                "epochs": self.epochs_,
                "history": self.history_.to_dict(orient="records"),
                "state": self._capture_state(),
            }, handle)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        model_source: str | Path,
        device: str | torch.device = "cuda",
        attention_backend: str | None = None,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ) -> CellTypeAnnotator:
        """Restore an annotator using its original pretrained SFM and resources."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format_version"] != 4:
            raise ValueError("Unsupported cell-type annotation checkpoint version.")
        if payload["attention_mode"] != "cell_global":
            raise ValueError("The checkpoint must use global-cell annotation attention.")
        annotator = cls.from_pretrained(
            model_source,
            device=device,
            attention_backend=attention_backend,
            autocast_dtype=autocast_dtype,
            classifier_hidden_dim=payload["classifier_config"]["hidden_dim"],
            classifier_dropout=payload["classifier_config"]["dropout"],
            **payload["tokenizer_config"],
        )
        if payload["vocabulary_signature"] != annotator.vocabulary_signature:
            raise ValueError("The checkpoint and model source use different vocabularies.")
        annotator.class_names_ = tuple(payload["class_names"])
        annotator.gene_names_ = tuple(payload["gene_names"])
        annotator.gene_key_ = payload["gene_key"]
        annotator.epochs_ = payload["epochs"]
        annotator.history_ = pd.DataFrame(payload["history"])
        annotator.classifier = annotator._new_classifier()
        annotator._restore_state(payload["state"])
        annotator.model.eval()
        annotator.classifier.eval()
        return annotator
