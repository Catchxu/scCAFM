"""GEARS-reference condition splits and ontology graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData


def canonicalize_perturbation(condition: str, control_label: str = "ctrl") -> str:
    """Return a gene-only name for an intervention."""
    genes = sorted(g for g in str(condition).split("+") if g != control_label)
    return "+".join(genes) if genes else control_label


def _simulation_split(
    conditions: list[str], gene_fraction: float, combo_fraction: float,
    random_state: int, control_label: str,
):
    """Reproduce DataSplitter.get_simulation_split without global RNG changes."""
    components = {p: p.split("+") for p in conditions}
    genes = np.unique([g for p in conditions for g in components[p] if g != control_label])
    seen = np.random.RandomState(random_state).choice(
        genes, int(len(genes) * gene_fraction), replace=False
    )
    unseen = np.setdiff1d(genes, seen)
    singles = [p for p in conditions if control_label in components[p]]
    combos = [p for p in conditions if control_label not in components[p]]
    train_singles = [p for p in singles if any(g in seen for g in components[p])]
    seen_combos = [p for p in combos if any(g in seen for g in components[p])]
    seen1 = [p for p in seen_combos if sum(g in seen for g in components[p]) == 1]
    remaining = np.setdiff1d(seen_combos, seen1)
    train_combos = np.random.RandomState(random_state).choice(
        remaining, int(len(remaining) * combo_fraction), replace=False
    )
    seen2 = np.setdiff1d(remaining, train_combos).tolist()
    unseen_singles = [p for p in singles if any(g in unseen for g in components[p])]
    seen0 = [p for p in combos if sum(g in seen for g in components[p]) == 0]
    return (
        train_singles + train_combos.tolist(),
        seen1 + seen2 + unseen_singles + seen0,
        {"combo_seen0": seen0, "combo_seen1": seen1,
         "combo_seen2": seen2, "unseen_single": unseen_singles},
    )


def prepare_gene_perturbation_data(
    adata: AnnData, vocabulary: pd.DataFrame, *,
    perturbation_genes: list[str] | tuple[str, ...],
    n_genes: int = 2047,
    condition_key: str = "condition", gene_symbol_key: str = "gene_name",
    control_label: str = "ctrl", train_gene_fraction: float = 0.75,
    train_combo_fraction: float = 0.75,
    validation_gene_fraction: float = 0.9,
    validation_combo_fraction: float = 0.9,
    random_state: int = 1, cell_type: str, species: str = "human",
) -> AnnData:
    """Prepare the reference GEARS train/validation/test condition protocol.

    X is normalized log expression prepared by the caller. Perturbation labels
    are retained; GEARS observation fields are derived here. Attach source DE
    rankings to the returned object or calculate fresh rankings before fitting.
    Interventions outside the external ontology node universe are excluded, as
    in PertData.load(). Controls belong to train
    and supply the common input pool for training, validation and prediction.

    The tutorial's EFM-compatible panel retains every intervention target plus
    training-control HVGs. This explicit panel is a tutorial input choice; the
    reference split and decoder do not enforce a 2,047-gene limit.
    """
    supported = set(perturbation_genes)
    condition = adata.obs[condition_key].astype(str)
    keep = condition.map(
        lambda p: all(g in supported for g in p.split("+") if g != control_label)
    ).to_numpy()
    obs = adata.obs.loc[keep].copy()
    obs["source_condition"] = obs[condition_key].astype(str)
    obs["source_cell_type"] = obs["cell_type"].astype(str)
    obs["cell_type"] = cell_type
    obs["species"] = species
    obs["control"] = obs[condition_key].astype(str).eq(control_label).astype(int)
    obs["dose_val"] = np.where(obs[condition_key].astype(str).str.contains("+", regex=False), "1+1", "1")
    obs["condition_name"] = (
        obs["cell_type"].astype(str) + "_" + obs[condition_key].astype(str) + "_" + obs["dose_val"]
    )
    conditions = [p for p in obs[condition_key].astype(str).unique() if p != control_label]
    outer_train, test, test_groups = _simulation_split(
        conditions, train_gene_fraction, train_combo_fraction,
        random_state, control_label,
    )
    train, validation, validation_groups = _simulation_split(
        outer_train, validation_gene_fraction, validation_combo_fraction,
        random_state, control_label,
    )
    train.append(control_label)
    split = {p: "train" for p in train}
    split.update({p: "val" for p in validation})
    split.update({p: "test" for p in test})
    obs["split"] = obs[condition_key].astype(str).map(split)

    controls = np.flatnonzero(obs[condition_key].astype(str).eq(control_label).to_numpy())
    generator = np.random.RandomState(random_state)
    control_ids = np.empty(len(obs), dtype=object)
    # These pairs are sampled once, matching the cached GEARS cell graphs.
    for pert in obs[condition_key].astype(str).unique():
        rows = np.flatnonzero(obs[condition_key].astype(str).eq(pert).to_numpy())
        paired = rows if pert == control_label else controls[
            generator.randint(0, len(controls), size=len(rows))
        ]
        control_ids[rows] = obs.index[paired].to_numpy()
    obs["control_cell_id"] = control_ids

    targets = sorted({g for p in conditions for g in p.split("+") if g != control_label})
    available = vocabulary.set_index("gene_id")
    eligible = adata.var_names.isin(available.index)
    target_ids = pd.Series(
        adata.var_names, index=adata.var[gene_symbol_key].astype(str)
    ).loc[targets].tolist()
    missing = set(target_ids) - set(available.index)
    if missing:
        raise ValueError(f"Perturbation targets are absent from the EFM vocabulary: {missing}")
    if n_genes < len(target_ids):
        raise ValueError("The gene panel must retain every perturbation target.")
    source_controls = adata.obs_names.get_indexer(obs.index[controls])
    hvg = AnnData(
        X=sp.csr_matrix(adata[source_controls, eligible].X),
        var=adata.var.loc[eligible].copy(),
    )
    sc.pp.highly_variable_genes(hvg, flavor="seurat", n_top_genes=n_genes)
    ranked = hvg.var["dispersions_norm"].sort_values(
        ascending=False, kind="stable", na_position="last"
    ).index
    fillers = [g for g in ranked if g not in set(target_ids)]
    selected = target_ids + fillers[:n_genes - len(target_ids)]
    if len(selected) != n_genes:
        raise ValueError("The vocabulary-matched panel is too small.")
    rows = adata.obs_names.get_indexer(obs.index)
    ordered = np.concatenate([
        np.flatnonzero(obs[condition_key].astype(str).eq(p).to_numpy())
        for p in train + validation + test
    ])
    obs = obs.iloc[ordered].copy()
    rows = rows[ordered]
    prepared = AnnData(
        X=sp.csr_matrix(adata[rows, selected].X),
        obs=obs, var=adata.var.loc[selected].copy(),
    )
    prepared.var["source_gene_symbol"] = prepared.var[gene_symbol_key].astype(str)
    prepared.var["gene_symbol"] = available.loc[selected, "gene_symbol"].to_numpy()
    prepared.var["perturbation_target"] = prepared.var_names.isin(target_ids)
    if "counts" in adata.layers:
        columns = adata.var_names.get_indexer(selected)
        prepared.layers["counts"] = sp.csr_matrix(adata.layers["counts"][rows][:, columns])
    prepared.uns["perturbation_split"] = {
        "protocol": "GEARS simulation with nested validation",
        "random_state": random_state, "control_label": control_label,
        "condition_key": condition_key,
        "train_gene_fraction": train_gene_fraction,
        "train_combo_fraction": train_combo_fraction,
        "validation_gene_fraction": validation_gene_fraction,
        "validation_combo_fraction": validation_combo_fraction,
        "perturbation_genes": targets,
        "conditions": {"train": train, "val": validation, "test": test},
        "test_subgroup": test_groups, "val_subgroup": validation_groups,
        "excluded_conditions": sorted(set(condition) - set(obs[condition_key].astype(str))),
        "expression_scale": "Normalized log expression supplied by the caller.",
        "gene_selection": "All supported intervention targets plus training-control HVGs.",
        "control_pairing": "One fixed input control per target; controls map to themselves.",
    }
    return prepared


@dataclass(frozen=True)
class PerturbationGraphs:
    """Reference GO graph with separate expression and ontology node axes."""

    gene_names: tuple[str, ...]
    perturbation_genes: tuple[str, ...]
    ontology_edges: torch.Tensor
    ontology_weights: torch.Tensor

    def state_dict(self) -> dict:
        return {
            "gene_names": self.gene_names,
            "perturbation_genes": self.perturbation_genes,
            "ontology_edges": self.ontology_edges,
            "ontology_weights": self.ontology_weights,
        }


def build_perturbation_graphs(
    adata: AnnData, perturbation_genes: list[str] | tuple[str, ...],
    go_edges: pd.DataFrame, *, top_k: int = 20,
) -> PerturbationGraphs:
    """Use the reference top k+1 incoming GO edges, including self-edges.

    The reference co-expression branch is inactive. No expression graph is built.
    The ontology universe is the external essential-gene prior, not the list of
    interventions observed in this particular assay.
    """
    import networkx as nx

    indices = go_edges.groupby("target", sort=True)["importance"].nlargest(
        top_k + 1
    ).index.get_level_values(-1)
    selected = go_edges.loc[indices]
    network = nx.from_pandas_edgelist(
        selected, source="source", target="target", edge_attr="importance",
        create_using=nx.DiGraph(),
    )
    nodes = tuple(perturbation_genes)
    lookup = {g: i for i, g in enumerate(nodes)}
    edges = list(network.edges)
    return PerturbationGraphs(
        tuple(adata.var_names), nodes,
        torch.tensor([(lookup[a], lookup[b]) for a, b in edges], dtype=torch.long).T,
        torch.tensor([network[a][b]["importance"] for a, b in edges], dtype=torch.float32),
    )
