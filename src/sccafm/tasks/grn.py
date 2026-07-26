from __future__ import annotations

import math
import numbers

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

from ..assets import (
    load_model_state_dict,
    load_sfm_config,
    load_table_json,
    load_vocab_json,
    resolve_model_assets,
)
from ..data.collator import ScBatchCollator
from ..data.tokenizer import ScTokenizer, ScTokenizerOutput
from ..models.sfm import FactorState, SFM


_SFM_STATE_PREFIX = "foundation_modules.sfm."


def _validate_edge_filter(
    *,
    score_threshold: float | None,
    top_k_edges: int | None,
) -> tuple[float | None, int | None]:
    if score_threshold is not None and top_k_edges is not None:
        raise ValueError(
            "`score_threshold` and `top_k_edges` are mutually exclusive."
        )

    normalized_threshold: float | None = None
    if score_threshold is not None:
        if isinstance(score_threshold, bool) or not isinstance(
            score_threshold, numbers.Real
        ):
            raise TypeError("`score_threshold` must be a finite real number.")
        normalized_threshold = float(score_threshold)
        if not math.isfinite(normalized_threshold):
            raise ValueError("`score_threshold` must be finite.")

    normalized_top_k: int | None = None
    if top_k_edges is not None:
        if isinstance(top_k_edges, bool) or not isinstance(
            top_k_edges, numbers.Integral
        ):
            raise TypeError("`top_k_edges` must be a positive integer.")
        normalized_top_k = int(top_k_edges)
        if normalized_top_k <= 0:
            raise ValueError("`top_k_edges` must be a positive integer.")

    return normalized_threshold, normalized_top_k


def _deterministic_topk_mask(
    flattened_scores: torch.Tensor,
    top_k_edges: int,
) -> torch.BoolTensor:
    if flattened_scores.ndim != 2:
        raise ValueError(
            "`flattened_scores` must have shape (rows, edges), got "
            f"{tuple(flattened_scores.shape)}."
        )

    row_count, edge_count = flattened_scores.shape
    if edge_count == 0:
        raise ValueError("Cannot select edges from an empty score tensor.")
    if top_k_edges >= edge_count:
        return torch.ones_like(flattened_scores, dtype=torch.bool)

    selected = torch.zeros_like(flattened_scores, dtype=torch.bool)
    for row_index in range(row_count):
        row = flattened_scores[row_index]
        cutoff = torch.topk(
            row,
            k=top_k_edges,
            largest=True,
            sorted=False,
        ).values.min()
        greater = row > cutoff
        selected[row_index] = greater
        remaining = top_k_edges - int(greater.sum().item())
        if remaining > 0:
            tied_indices = torch.nonzero(row == cutoff, as_tuple=False).flatten()
            selected[row_index, tied_indices[:remaining]] = True
    return selected


def _apply_edge_filter(
    scores: torch.Tensor,
    *,
    score_threshold: float | None,
    top_k_edges: int | None,
) -> tuple[torch.Tensor, torch.BoolTensor | None]:
    score_threshold, top_k_edges = _validate_edge_filter(
        score_threshold=score_threshold,
        top_k_edges=top_k_edges,
    )
    if scores.ndim not in {2, 3}:
        raise ValueError(
            "`scores` must have shape (TFs, genes) or (cells, TFs, genes), "
            f"got {tuple(scores.shape)}."
        )
    if not torch.isfinite(scores).all():
        raise ValueError("GRN reconstruction produced non-finite scores.")
    if score_threshold is None and top_k_edges is None:
        return scores, None

    if score_threshold is not None:
        selected = scores >= score_threshold
    else:
        edge_count = int(scores.shape[-2] * scores.shape[-1])
        flattened = scores.reshape(-1, edge_count)
        selected = _deterministic_topk_mask(
            flattened_scores=flattened,
            top_k_edges=int(top_k_edges),
        ).reshape_as(scores)

    return scores.masked_fill(~selected, 0.0), selected


def _normalize_species(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"human", "homo sapiens", "hs"}:
        return "human"
    if normalized in {"mouse", "mus musculus", "mm"}:
        return "mouse"
    return normalized


def _extract_sfm_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prefixed = {
        name[len(_SFM_STATE_PREFIX) :]: value
        for name, value in state_dict.items()
        if name.startswith(_SFM_STATE_PREFIX)
    }
    if prefixed:
        return prefixed
    return dict(state_dict)


def _edge_table(
    scores: torch.Tensor,
    *,
    source_genes: tuple[str, ...],
    target_genes: tuple[str, ...],
    selected_mask: torch.BoolTensor | None,
    cell_id: str | None = None,
) -> pd.DataFrame:
    if scores.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional GRN matrix, got {tuple(scores.shape)}."
        )
    expected_shape = (len(source_genes), len(target_genes))
    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            f"Score shape {tuple(scores.shape)} does not match labels {expected_shape}."
        )
    if selected_mask is not None and tuple(selected_mask.shape) != expected_shape:
        raise ValueError("Selected-edge mask does not match the GRN score shape.")

    flat_scores = scores.detach().cpu().float().reshape(-1)
    if selected_mask is None:
        flat_indices = torch.arange(flat_scores.numel(), dtype=torch.long)
    else:
        flat_indices = torch.nonzero(
            selected_mask.detach().cpu().reshape(-1),
            as_tuple=False,
        ).flatten()

    target_count = len(target_genes)
    source_indices = torch.div(
        flat_indices,
        target_count,
        rounding_mode="floor",
    ).numpy()
    target_indices = torch.remainder(flat_indices, target_count).numpy()
    source_array = np.asarray(source_genes, dtype=object)
    target_array = np.asarray(target_genes, dtype=object)
    values = flat_scores.index_select(0, flat_indices).numpy()

    payload: dict[str, Any] = {
        "Gene1": source_array[source_indices],
        "Gene2": target_array[target_indices],
        "score": values,
    }
    if cell_id is not None:
        payload = {
            "cell_id": np.repeat(str(cell_id), len(flat_indices)),
            **payload,
        }
    return pd.DataFrame(payload)


class _TokenizedCellDataset(Dataset):
    def __init__(self, tokenized: ScTokenizerOutput) -> None:
        self.tokenized = tokenized

    def __len__(self) -> int:
        return int(self.tokenized.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.tokenized.input_ids[index],
            "expression_values": self.tokenized.expression_values[index],
            "condition_ids": self.tokenized.condition_ids[index],
            "padding_mask": (
                None
                if self.tokenized.padding_mask is None
                else self.tokenized.padding_mask[index]
            ),
            "non_tf_mask": self.tokenized.non_tf_mask[index],
            "gene_name_type": self.tokenized.gene_name_type,
        }


@dataclass(slots=True)
class _PreparedGRNInput:
    dataset: _TokenizedCellDataset
    cell_ids: tuple[str, ...]
    source_genes: tuple[str, ...]
    target_genes: tuple[str, ...]
    source_positions: torch.LongTensor
    batch_size: int
    num_workers: int


class CellSpecificGRNs:
    """Lazy, reusable cell-specific GRN collection."""

    def __init__(
        self,
        *,
        inferencer: GRNInferencer,
        prepared: _PreparedGRNInput,
        score_threshold: float | None,
        top_k_edges: int | None,
    ) -> None:
        self._inferencer = inferencer
        self._prepared = prepared
        self.cell_ids = prepared.cell_ids
        self.source_genes = prepared.source_genes
        self.target_genes = prepared.target_genes
        self.score_threshold = score_threshold
        self.top_k_edges = top_k_edges

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.cell_ids),
            len(self.source_genes),
            len(self.target_genes),
        )

    def _iter_filtered_batches(
        self,
    ) -> Iterator[
        tuple[tuple[str, ...], torch.FloatTensor, torch.BoolTensor | None]
    ]:
        yield from self._inferencer._iter_filtered_cell_batches(
            prepared=self._prepared,
            score_threshold=self.score_threshold,
            top_k_edges=self.top_k_edges,
        )

    def iter_batches(
        self,
    ) -> Iterator[tuple[tuple[str, ...], torch.FloatTensor]]:
        for cell_ids, scores, _ in self._iter_filtered_batches():
            yield cell_ids, scores

    def __iter__(
        self,
    ) -> Iterator[tuple[tuple[str, ...], torch.FloatTensor]]:
        return self.iter_batches()

    def iter_edge_tables(self) -> Iterator[pd.DataFrame]:
        for cell_ids, scores, selected_mask in self._iter_filtered_batches():
            for cell_index, cell_id in enumerate(cell_ids):
                cell_mask = (
                    None
                    if selected_mask is None
                    else selected_mask[cell_index]
                )
                yield _edge_table(
                    scores[cell_index],
                    source_genes=self.source_genes,
                    target_genes=self.target_genes,
                    selected_mask=cell_mask,
                    cell_id=cell_id,
                )


@dataclass(frozen=True, slots=True)
class PooledGRN:
    scores: torch.FloatTensor
    cell_ids: tuple[str, ...]
    source_genes: tuple[str, ...]
    target_genes: tuple[str, ...]
    score_threshold: float | None = None
    top_k_edges: int | None = None
    _selected_mask: torch.BoolTensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        expected_shape = (len(self.source_genes), len(self.target_genes))
        if self.scores.ndim != 2 or tuple(self.scores.shape) != expected_shape:
            raise ValueError(
                f"Pooled score shape {tuple(self.scores.shape)} does not match "
                f"labels {expected_shape}."
            )
        if self.scores.device.type != "cpu" or self.scores.dtype != torch.float32:
            raise ValueError("`PooledGRN.scores` must be a CPU float32 tensor.")
        if len(self.cell_ids) == 0:
            raise ValueError("`PooledGRN.cell_ids` must not be empty.")
        if self._selected_mask is not None:
            if tuple(self._selected_mask.shape) != expected_shape:
                raise ValueError("Selected-edge mask does not match pooled scores.")
            if self._selected_mask.device.type != "cpu":
                raise ValueError("Selected-edge mask must be on CPU.")
        _validate_edge_filter(
            score_threshold=self.score_threshold,
            top_k_edges=self.top_k_edges,
        )

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.scores.shape)

    def to_edge_table(self) -> pd.DataFrame:
        return _edge_table(
            self.scores,
            source_genes=self.source_genes,
            target_genes=self.target_genes,
            selected_mask=self._selected_mask,
        )


class GRNInferencer:
    """Generate cell-specific and pooled GRNs from preprocessed AnnData."""

    def __init__(
        self,
        *,
        model: SFM,
        tokenizer: ScTokenizer,
        device: torch.device,
        max_length: int,
        autocast_dtype: torch.dtype | None,
        species_key: str | None,
    ) -> None:
        if max_length <= 1:
            raise ValueError("`max_length` must be greater than one.")
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.autocast_dtype = autocast_dtype
        self.species_key = species_key

    @classmethod
    def from_pretrained(
        cls,
        model_source: str | Path,
        *,
        device: str | torch.device = "cuda",
        attention_backend: str | None = None,
        max_length: int = 4096,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        platform_key: str | None = None,
        species_key: str | None = "species",
        tissue_key: str | None = None,
        disease_key: str | None = None,
    ) -> GRNInferencer:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if max_length <= 1:
            raise ValueError("`max_length` must be greater than one.")

        assets = resolve_model_assets(
            model_source,
            require_model_weights=True,
            require_cond_dict=True,
            require_resources=True,
        )
        token_dict = load_vocab_json(assets.vocab)
        cond_dict = load_table_json(assets.cond_dict)
        human_tfs = pd.read_csv(assets.human_tfs)
        mouse_tfs = pd.read_csv(assets.mouse_tfs)
        if cond_dict.empty or "token_index" not in cond_dict.columns:
            raise ValueError("Condition vocabulary must contain token indices.")
        cond_vocab_size = int(cond_dict["token_index"].max()) + 1

        tokenizer = ScTokenizer(
            token_dict=token_dict,
            cond_dict=cond_dict,
            human_tfs=human_tfs,
            mouse_tfs=mouse_tfs,
            max_length=max_length,
            platform_key=platform_key,
            species_key=species_key,
            tissue_key=tissue_key,
            disease_key=disease_key,
        )

        config = load_sfm_config(assets.sfm_config)
        sfm_kwargs = dict(config["sfm"])
        sfm_kwargs.pop("cond_vocab_size", None)
        sfm_kwargs.pop("gene_embedding_ckpt", None)
        if attention_backend is not None:
            sfm_kwargs["attention_backend"] = attention_backend
        model = SFM(
            token_dict=token_dict,
            cond_vocab_size=cond_vocab_size,
            gene_embedding_ckpt=str(assets.vocab_tensors),
            **sfm_kwargs,
        )
        state_dict = _extract_sfm_state_dict(
            load_model_state_dict(assets.sfm_model)
        )
        model.load_state_dict(state_dict, strict=True)
        model.requires_grad_(False)
        model.eval()
        model.to(resolved_device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            max_length=max_length,
            autocast_dtype=autocast_dtype,
            species_key=species_key,
        )

    @staticmethod
    def _validate_loader_options(batch_size: int, num_workers: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
            raise TypeError("`batch_size` must be a positive integer.")
        if int(batch_size) <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if isinstance(num_workers, bool) or not isinstance(
            num_workers, numbers.Integral
        ):
            raise TypeError("`num_workers` must be a non-negative integer.")
        if int(num_workers) < 0:
            raise ValueError("`num_workers` must be a non-negative integer.")

    def _validate_species(self, adata: AnnData) -> str:
        if self.species_key is None:
            return "human"
        if self.species_key not in adata.obs.columns:
            raise KeyError(
                f"`species_key={self.species_key}` not found in `adata.obs`."
            )
        species = {
            _normalize_species(value)
            for value in adata.obs[self.species_key].tolist()
        }
        unsupported = species.difference({"human", "mouse"})
        if unsupported:
            raise ValueError(
                f"Unsupported species values: {sorted(unsupported)}."
            )
        if len(species) != 1:
            raise ValueError(
                "GRN inference requires all cells in one AnnData to share one species."
            )
        return next(iter(species))

    @staticmethod
    def _validate_expression(adata: AnnData) -> None:
        expression = adata.X
        if sp.issparse(expression):
            if not np.issubdtype(expression.dtype, np.number) or np.issubdtype(
                expression.dtype, np.complexfloating
            ):
                raise TypeError("`adata.X` must contain real numeric values.")
            values = np.asarray(expression.data)
        else:
            values = np.asarray(expression)
            if values.ndim != 2:
                raise ValueError("`adata.X` must be two-dimensional.")
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
                values.dtype, np.complexfloating
            ):
                raise TypeError("`adata.X` must contain real numeric values.")
        if values.size > 0 and not np.isfinite(values).all():
            raise ValueError("`adata.X` must contain only finite values.")
        if values.size > 0 and np.any(values < 0):
            raise ValueError("`adata.X` must contain non-negative values.")

    def _prepare_input(
        self,
        adata: AnnData,
        *,
        batch_size: int,
        gene_key: str | None,
        num_workers: int,
    ) -> _PreparedGRNInput:
        if not isinstance(adata, AnnData):
            raise TypeError(
                f"`adata` must be an AnnData, got {type(adata).__name__}."
            )
        self._validate_loader_options(batch_size, num_workers)
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            raise ValueError("`adata` must contain at least one cell and one gene.")
        if adata.n_vars > self.max_length - 1:
            raise ValueError(
                f"Input contains {adata.n_vars} genes but at most "
                f"{self.max_length - 1} are supported."
            )
        self._validate_expression(adata)
        self._validate_species(adata)

        if gene_key is None:
            gene_labels = tuple(str(value) for value in adata.var_names.tolist())
        else:
            if gene_key not in adata.var.columns:
                raise KeyError(f"`gene_key={gene_key}` not found in `adata.var`.")
            gene_labels = tuple(
                str(value) for value in adata.var[gene_key].tolist()
            )
        if pd.Index(gene_labels).has_duplicates:
            raise ValueError("Input gene identifiers must be unique.")

        token_ids = self.tokenizer.gene_tokenizer.encode_gene_list(gene_labels)
        pad_index = self.tokenizer.gene_tokenizer.pad_index
        unsupported_positions = torch.nonzero(
            token_ids == pad_index,
            as_tuple=False,
        ).flatten()
        if unsupported_positions.numel() > 0:
            examples = [gene_labels[index] for index in unsupported_positions[:5]]
            raise ValueError(
                f"{unsupported_positions.numel()} input genes are absent from the "
                f"model vocabulary; examples: {examples}."
            )
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise ValueError(
                "Multiple input genes map to the same model vocabulary token."
            )

        cell_ids = tuple(str(value) for value in adata.obs_names.tolist())
        if pd.Index(cell_ids).has_duplicates:
            raise ValueError("Input cell identifiers must be unique.")
        tokenized = self.tokenizer(adata, gene_key=gene_key)
        active_non_tf_mask = tokenized.non_tf_mask[:, : adata.n_vars]
        reference_mask = active_non_tf_mask[0].expand_as(active_non_tf_mask)
        if not torch.equal(active_non_tf_mask, reference_mask):
            raise ValueError("TF source masks differ across cells.")
        source_positions = torch.nonzero(
            ~active_non_tf_mask[0],
            as_tuple=False,
        ).flatten()
        if source_positions.numel() == 0:
            raise ValueError("No valid transcription factors were found in the input.")
        source_genes = tuple(
            gene_labels[int(index)] for index in source_positions
        )

        return _PreparedGRNInput(
            dataset=_TokenizedCellDataset(tokenized),
            cell_ids=cell_ids,
            source_genes=source_genes,
            target_genes=gene_labels,
            source_positions=source_positions,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
        )

    def _autocast_context(self):
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
        )

    def _iter_factor_batches(
        self,
        prepared: _PreparedGRNInput,
    ) -> Iterator[tuple[tuple[str, ...], torch.Tensor, torch.Tensor]]:
        loader = DataLoader(
            prepared.dataset,
            batch_size=prepared.batch_size,
            shuffle=False,
            num_workers=prepared.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=ScBatchCollator(),
        )
        source_positions = prepared.source_positions.to(self.device)
        gene_count = len(prepared.target_genes)
        offset = 0
        for batch in loader:
            tokens: dict[str, torch.Tensor | None] = {}
            for name in (
                "input_ids",
                "expression_values",
                "condition_ids",
                "padding_mask",
                "non_tf_mask",
            ):
                value = batch[name]
                tokens[name] = (
                    None
                    if value is None
                    else value.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                )
            with torch.inference_mode(), self._autocast_context():
                output = self.model(
                    tokens,
                    return_factors=True,
                    compute_grn=False,
                    compute_order=False,
                )
            if not (isinstance(output, tuple) and len(output) == 2):
                raise TypeError("SFM must return `(grn, FactorState)`.")
            _, factors = output
            if not isinstance(factors, FactorState):
                raise TypeError("SFM did not return a FactorState.")
            if factors.u.shape != factors.v.shape:
                raise ValueError("SFM factor tensors must share the same shape.")
            if factors.u.ndim != 3 or factors.u.shape[1] < gene_count:
                raise ValueError("SFM factors do not cover all input genes.")

            current_batch_size = int(factors.u.shape[0])
            cell_ids = prepared.cell_ids[offset : offset + current_batch_size]
            offset += current_batch_size
            yield (
                cell_ids,
                factors.u.index_select(1, source_positions),
                factors.v[:, :gene_count],
            )
        if offset != len(prepared.cell_ids):
            raise RuntimeError("Inference loader did not visit every input cell.")

    def _iter_filtered_cell_batches(
        self,
        *,
        prepared: _PreparedGRNInput,
        score_threshold: float | None,
        top_k_edges: int | None,
    ) -> Iterator[
        tuple[tuple[str, ...], torch.FloatTensor, torch.BoolTensor | None]
    ]:
        for cell_ids, u_tf, v in self._iter_factor_batches(prepared):
            raw_scores = torch.einsum(
                "btm,bgm->btg",
                u_tf.float(),
                v.float(),
            )
            filtered_scores, selected_mask = _apply_edge_filter(
                raw_scores,
                score_threshold=score_threshold,
                top_k_edges=top_k_edges,
            )
            yield (
                cell_ids,
                filtered_scores.detach().cpu().float(),
                None
                if selected_mask is None
                else selected_mask.detach().cpu(),
            )

    def infer_cell_specific(
        self,
        adata: AnnData,
        *,
        batch_size: int = 8,
        gene_key: str | None = None,
        num_workers: int = 0,
        score_threshold: float | None = None,
        top_k_edges: int | None = None,
    ) -> CellSpecificGRNs:
        score_threshold, top_k_edges = _validate_edge_filter(
            score_threshold=score_threshold,
            top_k_edges=top_k_edges,
        )
        prepared = self._prepare_input(
            adata,
            batch_size=batch_size,
            gene_key=gene_key,
            num_workers=num_workers,
        )
        return CellSpecificGRNs(
            inferencer=self,
            prepared=prepared,
            score_threshold=score_threshold,
            top_k_edges=top_k_edges,
        )

    def infer_pooled(
        self,
        adata: AnnData,
        *,
        batch_size: int = 8,
        gene_key: str | None = None,
        num_workers: int = 0,
        score_threshold: float | None = None,
        top_k_edges: int | None = None,
    ) -> PooledGRN:
        score_threshold, top_k_edges = _validate_edge_filter(
            score_threshold=score_threshold,
            top_k_edges=top_k_edges,
        )
        prepared = self._prepare_input(
            adata,
            batch_size=batch_size,
            gene_key=gene_key,
            num_workers=num_workers,
        )

        score_sum = torch.zeros(
            (len(prepared.source_genes), len(prepared.target_genes)),
            dtype=torch.float32,
            device=self.device,
        )
        cell_count = 0
        for _, u_tf, v in self._iter_factor_batches(prepared):
            score_sum.add_(
                torch.einsum(
                    "btm,bgm->tg",
                    u_tf.float(),
                    v.float(),
                )
            )
            cell_count += int(u_tf.shape[0])
        if cell_count != len(prepared.cell_ids):
            raise RuntimeError("Pooled inference did not include every input cell.")

        raw_pooled_scores = score_sum / float(cell_count)
        filtered_scores, selected_mask = _apply_edge_filter(
            raw_pooled_scores,
            score_threshold=score_threshold,
            top_k_edges=top_k_edges,
        )
        return PooledGRN(
            scores=filtered_scores.detach().cpu().float(),
            cell_ids=prepared.cell_ids,
            source_genes=prepared.source_genes,
            target_genes=prepared.target_genes,
            score_threshold=score_threshold,
            top_k_edges=top_k_edges,
            _selected_mask=(
                None
                if selected_mask is None
                else selected_mask.detach().cpu()
            ),
        )


__all__ = [
    "CellSpecificGRNs",
    "GRNInferencer",
    "PooledGRN",
]
