from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from ..assets import load_vocab_json, resolve_model_assets
from .grn import PooledGRN


@dataclass(frozen=True, slots=True)
class ChIPSeqReference:
    """A ChIP-seq reference mapped to the scCAFM gene vocabulary."""

    species: str
    supported_tfs: tuple[str, ...]
    catalog_tfs: tuple[str, ...]
    raw_edge_count: int
    mapped_edge_count: int
    unmapped_edge_count: int
    self_loop_edge_count: int
    duplicate_edge_count: int
    _pair_keys: torch.LongTensor = field(repr=False, compare=False)
    _supported_tf_token_ids: torch.LongTensor = field(repr=False, compare=False)
    _pair_key_base: int = field(repr=False, compare=False)
    _symbol_to_index: dict[str, int] = field(repr=False, compare=False)
    _ensembl_to_index: dict[str, int] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ChIPSeqEvaluation:
    """Summary metrics for one pooled GRN evaluated against ChIP-seq."""

    auprc: float
    early_precision: float
    n_candidates: int
    n_positive_edges: int
    positive_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "auprc": self.auprc,
            "early_precision": self.early_precision,
            "n_candidates": self.n_candidates,
            "n_positive_edges": self.n_positive_edges,
            "positive_rate": self.positive_rate,
        }


def _normalize_species(species: str) -> str:
    normalized = str(species).strip().lower()
    aliases = {
        "human": "human",
        "homo sapiens": "human",
        "hs": "human",
        "mouse": "mouse",
        "mus musculus": "mouse",
        "mm": "mouse",
    }
    if normalized not in aliases:
        raise ValueError(
            "`species` must identify human or mouse, "
            f"got {species!r}."
        )
    return aliases[normalized]


def _normalize_symbol(value: object) -> str:
    return str(value).strip().upper()


def _normalize_ensembl(value: object) -> str:
    normalized = str(value).strip().upper()
    if normalized.startswith("ENS"):
        normalized = normalized.split(".", 1)[0]
    return normalized


def _build_token_maps(
    token_dict: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    required_columns = {"token_index", "gene_symbol", "gene_id"}
    missing = required_columns.difference(token_dict.columns)
    if missing:
        raise ValueError(
            "Model vocabulary is missing required columns: "
            f"{sorted(missing)}."
        )

    symbol_to_index: dict[str, int] = {}
    ensembl_to_index: dict[str, int] = {}
    index_to_name: dict[int, str] = {}
    for _, row in token_dict.iterrows():
        if pd.isna(row["token_index"]):
            continue
        token_index = int(row["token_index"])
        symbol = row.get("gene_symbol")
        gene_id = row.get("gene_id")

        if pd.notna(symbol) and str(symbol).strip():
            normalized_symbol = _normalize_symbol(symbol)
            if not normalized_symbol.startswith("<"):
                symbol_to_index[normalized_symbol] = token_index
                index_to_name.setdefault(token_index, str(symbol).strip())
        if pd.notna(gene_id) and str(gene_id).strip():
            normalized_ensembl = _normalize_ensembl(gene_id)
            if not normalized_ensembl.startswith("<"):
                ensembl_to_index[normalized_ensembl] = token_index
                index_to_name.setdefault(token_index, str(gene_id).strip())

    return symbol_to_index, ensembl_to_index, index_to_name


def _map_gene(
    gene: object,
    *,
    symbol_to_index: dict[str, int],
    ensembl_to_index: dict[str, int],
) -> int | None:
    symbol_key = _normalize_symbol(gene)
    ensembl_key = _normalize_ensembl(gene)
    return symbol_to_index.get(symbol_key, ensembl_to_index.get(ensembl_key))


def _read_reference(reference: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(reference, pd.DataFrame):
        dataframe = reference.copy()
    elif isinstance(reference, (str, Path)):
        path = Path(reference).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ChIP-seq reference not found: {path}")
        dataframe = pd.read_csv(path)
    else:
        raise TypeError(
            "`reference` must be a CSV path or pandas DataFrame, got "
            f"{type(reference).__name__}."
        )

    required_columns = {"Gene1", "Gene2"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "ChIP-seq reference must contain columns `Gene1` and `Gene2`."
        )
    if dataframe.empty:
        raise ValueError("ChIP-seq reference must contain at least one edge.")
    return dataframe


def prepare_chipseq_reference(
    reference: str | Path | pd.DataFrame,
    *,
    model_source: str | Path,
    species: str,
) -> ChIPSeqReference:
    """Map a ChIP-seq edge list and identify its supported source TFs."""

    normalized_species = _normalize_species(species)
    dataframe = _read_reference(reference)
    assets = resolve_model_assets(
        model_source,
        require_model_weights=False,
        require_cond_dict=False,
        require_resources=True,
    )
    token_dict = load_vocab_json(assets.vocab)
    symbol_to_index, ensembl_to_index, index_to_name = _build_token_maps(
        token_dict
    )

    tf_path = assets.human_tfs if normalized_species == "human" else assets.mouse_tfs
    tf_dataframe = pd.read_csv(tf_path)
    if "TF" not in tf_dataframe.columns:
        raise ValueError(f"TF catalogue must contain a `TF` column: {tf_path}")
    catalog_tfs = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in tf_dataframe["TF"].dropna().tolist()
            if str(value).strip()
        )
    )
    catalog_token_ids = {
        token_index
        for gene in catalog_tfs
        if (
            token_index := _map_gene(
                gene,
                symbol_to_index=symbol_to_index,
                ensembl_to_index=ensembl_to_index,
            )
        )
        is not None
    }

    pair_key_base = max(int(token_dict["token_index"].max()) + 1, 1)
    mapped_pair_keys: list[int] = []
    source_token_ids: set[int] = set()
    unmapped_edge_count = 0
    self_loop_edge_count = 0
    for source_gene, target_gene in zip(
        dataframe["Gene1"].tolist(),
        dataframe["Gene2"].tolist(),
    ):
        source_token = _map_gene(
            source_gene,
            symbol_to_index=symbol_to_index,
            ensembl_to_index=ensembl_to_index,
        )
        target_token = _map_gene(
            target_gene,
            symbol_to_index=symbol_to_index,
            ensembl_to_index=ensembl_to_index,
        )
        if source_token is None or target_token is None:
            unmapped_edge_count += 1
            continue
        if source_token == target_token:
            self_loop_edge_count += 1
            continue
        mapped_pair_keys.append(source_token * pair_key_base + target_token)
        source_token_ids.add(source_token)

    unique_pair_keys = sorted(set(mapped_pair_keys))
    duplicate_edge_count = len(mapped_pair_keys) - len(unique_pair_keys)
    supported_tf_token_ids = sorted(source_token_ids.intersection(catalog_token_ids))
    supported_tfs = tuple(
        index_to_name[token_index]
        for token_index in supported_tf_token_ids
        if token_index in index_to_name
    )
    if not unique_pair_keys:
        raise ValueError(
            "ChIP-seq reference contains no non-self edges recognized by the "
            "model vocabulary."
        )
    if not supported_tfs:
        raise ValueError(
            "ChIP-seq reference contains no source TF recognized by the "
            f"{normalized_species} TF catalogue."
        )

    return ChIPSeqReference(
        species=normalized_species,
        supported_tfs=supported_tfs,
        catalog_tfs=catalog_tfs,
        raw_edge_count=int(len(dataframe)),
        mapped_edge_count=len(unique_pair_keys),
        unmapped_edge_count=unmapped_edge_count,
        self_loop_edge_count=self_loop_edge_count,
        duplicate_edge_count=duplicate_edge_count,
        _pair_keys=torch.tensor(unique_pair_keys, dtype=torch.long),
        _supported_tf_token_ids=torch.tensor(
            supported_tf_token_ids,
            dtype=torch.long,
        ),
        _pair_key_base=pair_key_base,
        _symbol_to_index=symbol_to_index,
        _ensembl_to_index=ensembl_to_index,
    )


def _validate_binary_inputs(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.detach().to(dtype=torch.float64, device="cpu")
    labels = labels.detach().to(dtype=torch.float64, device="cpu")
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Evaluation scores and labels must be aligned vectors.")
    if not torch.isfinite(scores).all():
        raise ValueError("Pooled GRN scores must contain only finite values.")
    return scores, labels


def _binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    positive_count = int(labels.sum().item())
    if positive_count <= 0:
        raise ValueError("ChIP-seq evaluation requires at least one positive edge.")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    group_positives = torch.tensor(
        [float(group.sum().item()) for group in sorted_labels.split(counts.tolist())],
        dtype=torch.float64,
    )
    cumulative_positives = torch.cumsum(group_positives, dim=0)
    cumulative_count = torch.cumsum(counts.to(torch.float64), dim=0)
    precision = cumulative_positives / cumulative_count
    recall_increment = group_positives / float(positive_count)
    return float(torch.sum(recall_increment * precision).item())


def _early_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _validate_binary_inputs(scores, labels)
    positive_count = int(labels.sum().item())
    if positive_count <= 0:
        raise ValueError("ChIP-seq evaluation requires at least one positive edge.")

    k = min(positive_count, int(labels.numel()))
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    group_positives = torch.tensor(
        [float(group.sum().item()) for group in sorted_labels.split(counts.tolist())],
        dtype=torch.float64,
    )
    group_counts = counts.to(torch.float64)
    group_ends = torch.cumsum(group_counts, dim=0)
    group_starts = group_ends - group_counts
    k_float = float(k)

    fully_selected = group_ends <= k_float
    selected_positives = torch.sum(group_positives[fully_selected])
    boundary = torch.nonzero(
        (group_starts < k_float) & (group_ends > k_float),
        as_tuple=False,
    )
    if boundary.numel() > 0:
        index = int(boundary[0].item())
        remaining = k_float - float(group_starts[index].item())
        selected_positives = selected_positives + group_positives[index] * (
            remaining / group_counts[index]
        )
    return float((selected_positives / k_float).item())


def evaluate_chipseq_grn(
    grn: PooledGRN,
    reference: ChIPSeqReference,
) -> ChIPSeqEvaluation:
    """Evaluate one unfiltered pooled GRN against a prepared ChIP-seq reference."""

    if not isinstance(grn, PooledGRN):
        raise TypeError(f"`grn` must be a PooledGRN, got {type(grn).__name__}.")
    if not isinstance(reference, ChIPSeqReference):
        raise TypeError(
            "`reference` must be a ChIPSeqReference, got "
            f"{type(reference).__name__}."
        )
    if grn.score_threshold is not None or grn.top_k_edges is not None:
        raise ValueError(
            "ChIP-seq evaluation requires an unfiltered PooledGRN; infer with "
            "`score_threshold=None` and `top_k_edges=None`."
        )

    def map_labels(labels: tuple[str, ...], axis_name: str) -> torch.LongTensor:
        mapped: list[int] = []
        missing: list[str] = []
        for label in labels:
            token_index = _map_gene(
                label,
                symbol_to_index=reference._symbol_to_index,
                ensembl_to_index=reference._ensembl_to_index,
            )
            if token_index is None:
                missing.append(label)
            else:
                mapped.append(token_index)
        if missing:
            raise ValueError(
                f"{axis_name} contains genes absent from the model vocabulary; "
                f"examples: {missing[:5]}."
            )
        if len(set(mapped)) != len(mapped):
            raise ValueError(f"{axis_name} contains duplicate mapped genes.")
        return torch.tensor(mapped, dtype=torch.long)

    source_tokens = map_labels(grn.source_genes, "PooledGRN.source_genes")
    target_tokens = map_labels(grn.target_genes, "PooledGRN.target_genes")
    supported_sources = torch.isin(
        source_tokens,
        reference._supported_tf_token_ids,
    )
    candidate_mask = supported_sources.unsqueeze(1) & (
        source_tokens.unsqueeze(1) != target_tokens.unsqueeze(0)
    )
    n_candidates = int(candidate_mask.sum().item())
    if n_candidates <= 0:
        raise ValueError(
            "Pooled GRN contains no non-self candidates for ChIP-supported TFs."
        )

    pair_keys = (
        source_tokens.unsqueeze(1) * reference._pair_key_base
        + target_tokens.unsqueeze(0)
    )
    labels = torch.isin(pair_keys, reference._pair_keys)[candidate_mask]
    scores = grn.scores[candidate_mask]
    n_positive_edges = int(labels.sum().item())
    if n_positive_edges <= 0:
        raise ValueError(
            "No mapped positive ChIP-seq edges remain in the pooled GRN "
            "candidate universe."
        )

    return ChIPSeqEvaluation(
        auprc=_binary_auprc(scores, labels),
        early_precision=_early_precision(scores, labels),
        n_candidates=n_candidates,
        n_positive_edges=n_positive_edges,
        positive_rate=float(n_positive_edges) / float(n_candidates),
    )
