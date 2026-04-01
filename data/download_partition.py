import cellxgene_census
import pandas as pd
import numpy as np
from data_config import DEFAULT_HOMOLOGY_PATH, VERSION, normalize_organism, organism_output_name
from typing import List
import os
import argparse
import warnings


parser = argparse.ArgumentParser(
                    description='Download a given partition cell of the query in h5ad')

parser.add_argument("--query-name",
    type=str,
    required=True,
    help="query name to build the index",
)

parser.add_argument("--partition-idx",
    type=int,
    required=True,
    help="partition index to download",
)
parser.add_argument("--output-dir",
    type=str,
    required=True,
    help="Directory to store the output h4ad file",
)

parser.add_argument("--index-dir",
    type=str,
    required=True,
    help="Directory to find the index file",
)

parser.add_argument("--max-partition-size",
    type=int,
    required=True,
    help="The max partition size for each partition(chunk)",
)

parser.add_argument(
    "--organism",
    type=str,
    default="Homo sapiens",
    help="Organism to download: Homo sapiens or Mus musculus.",
)

parser.add_argument(
    "--token-dict-path",
    type=str,
    default=None,
    help="Optional path to token_dict.csv. If provided, keep only genes found in the token dictionary.",
)
parser.add_argument(
    "--homology-path",
    type=str,
    default=str(DEFAULT_HOMOLOGY_PATH),
    help="Path to homologous.csv used to map mouse genes onto the human vocabulary.",
)


args = parser.parse_args()

# suppress noisy anndata index coercion warning during Census -> AnnData conversion
warnings.filterwarnings(
    "ignore",
    message=".*Transforming to str index.*",
)




def define_partition(partition_idx, id_list, partition_size) -> List[str]:
    """
    This function is used to define the partition for each job

    partition_idx is the partition index, which is an integer, and 0 <= partition_idx <= len(id_list) // MAX_PARTITION_SIZE
    """
    i = partition_idx * partition_size
    return id_list[i:i + partition_size]


def load2list(query_name, soma_id_dir, organism) -> List[int]:
    """
    This function is used to load the idx list from file
    """
    file_path = os.path.join(soma_id_dir, organism_output_name(organism), f"{query_name}.idx")
    with open(file_path, 'r') as fp:
        idx_list = fp.readlines()
    idx_list = [int(x.strip()) for x in idx_list]
    return idx_list


def _normalize_symbol(x) -> str:
    return str(x).strip().upper()


def _normalize_ensembl(x) -> str:
    value = str(x).strip().upper()
    if value.startswith("ENSG") or value.startswith("ENSMUSG"):
        value = value.split(".", 1)[0]
    return value


def _load_token_dict(path: str) -> pd.DataFrame:
    token_df = pd.read_csv(path)
    required_columns = {"gene_symbol", "gene_id"}
    missing = required_columns - set(token_df.columns)
    if missing:
        raise ValueError(
            f"`token_dict` is missing required columns: {sorted(missing)}. "
            "Expected at least ['gene_symbol', 'gene_id']."
        )
    return token_df


def _extract_source_feature_ids(adata) -> list[str]:
    if "feature_id" in adata.var.columns:
        return adata.var["feature_id"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _extract_source_feature_names(adata) -> list[str]:
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _extract_var_gene_names(adata) -> list[str]:
    if "feature_id" in adata.var.columns:
        return adata.var["feature_id"].astype(str).tolist()
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _set_canonical_var_names(adata) -> None:
    if "feature_id" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["feature_id"].astype(str), dtype=str)
    elif "feature_name" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["feature_name"].astype(str), dtype=str)
    else:
        adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.var_names.name = None


def _drop_unneeded_metadata_columns(adata) -> None:
    obs_drop_cols = [
        column
        for column in adata.obs.columns
        if column == "soma_joinid" or str(column).endswith("_term_id")
    ]
    var_drop_cols = [column for column in adata.var.columns if column == "soma_joinid"]

    if obs_drop_cols:
        adata.obs = adata.obs.drop(columns=obs_drop_cols)
    if var_drop_cols:
        adata.var = adata.var.drop(columns=var_drop_cols)


def _load_mouse_homology_map(path: str, token_dict: pd.DataFrame | None = None) -> pd.DataFrame:
    homology_df = pd.read_csv(path)
    required_columns = {
        "mouse_EnsemblID",
        "mouse_Symbol",
        "human_EnsemblID",
        "human_Symbol",
    }
    missing = required_columns - set(homology_df.columns)
    if missing:
        raise ValueError(
            f"`homologous.csv` is missing required columns: {sorted(missing)}."
        )

    working = homology_df.loc[:, sorted(required_columns)].copy()
    for column in ("mouse_EnsemblID", "human_EnsemblID"):
        working[column] = working[column].map(_normalize_ensembl)
        working.loc[~working[column].str.startswith(("ENSG", "ENSMUSG")), column] = pd.NA

    for column in ("mouse_Symbol", "human_Symbol"):
        working[column] = working[column].astype(str).str.strip()
        working.loc[working[column].isin({"", "nan", "None", "<NA>"})] = pd.NA

    working = working.dropna(subset=["mouse_EnsemblID", "human_EnsemblID"]).copy()

    if token_dict is not None:
        valid_human_ids = {
            _normalize_ensembl(gene_id)
            for gene_id in token_dict["gene_id"].dropna().tolist()
            if str(gene_id).strip() and not str(gene_id).startswith("<")
        }
        working["human_in_token_dict"] = working["human_EnsemblID"].isin(valid_human_ids)
    else:
        working["human_in_token_dict"] = True

    working["has_human_symbol"] = working["human_Symbol"].notna()
    working = working.sort_values(
        by=["human_in_token_dict", "has_human_symbol", "human_EnsemblID"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    working = working.drop_duplicates(subset=["mouse_EnsemblID"], keep="first")
    return working.set_index("mouse_EnsemblID")


def _apply_mouse_homology_map(adata, homology_map: pd.DataFrame) -> None:
    source_feature_ids = _extract_source_feature_ids(adata)
    source_feature_names = _extract_source_feature_names(adata)
    mouse_ids = pd.Index([_normalize_ensembl(x) for x in source_feature_ids], dtype=str)
    mapped = homology_map.reindex(mouse_ids)
    keep_mask = mapped["human_EnsemblID"].notna().to_numpy(dtype=bool)

    if not keep_mask.any():
        raise ValueError("No mouse genes in the downloaded partition were found in `homologous.csv`.")

    adata._inplace_subset_var(keep_mask)
    mapped = mapped.iloc[keep_mask].reset_index(names="mouse_EnsemblID")

    kept_source_ids = np.asarray(source_feature_ids, dtype=object)[keep_mask]
    kept_source_names = np.asarray(source_feature_names, dtype=object)[keep_mask]
    human_feature_ids = mapped["human_EnsemblID"].astype(str).to_numpy(dtype=object)
    human_feature_names = mapped["human_Symbol"].copy()
    human_feature_names = human_feature_names.where(
        human_feature_names.notna(),
        pd.Series(kept_source_names),
    )

    adata.var["source_feature_id"] = kept_source_ids
    adata.var["source_feature_name"] = kept_source_names
    adata.var["feature_id"] = human_feature_ids
    adata.var["feature_name"] = human_feature_names.astype(str).to_numpy(dtype=object)
    adata.var["mapped_from_species"] = "mouse"


def _filter_genes_by_token_dict(adata, token_dict: pd.DataFrame) -> None:
    gene_names = _extract_var_gene_names(adata)
    is_ensembl = all(
        str(gene_name).strip().upper().startswith(("ENSG", "ENSMUSG"))
        for gene_name in gene_names[: min(len(gene_names), 100)]
    ) if gene_names else False

    if is_ensembl:
        valid_genes = {
            _normalize_ensembl(gene_id)
            for gene_id in token_dict["gene_id"].dropna().tolist()
            if str(gene_id).strip() and not str(gene_id).startswith("<")
        }
        keep_mask = np.array(
            [_normalize_ensembl(gene_name) in valid_genes for gene_name in gene_names],
            dtype=bool,
        )
    else:
        valid_genes = {
            _normalize_symbol(symbol)
            for symbol in token_dict["gene_symbol"].dropna().tolist()
            if str(symbol).strip()
        }
        keep_mask = np.array(
            [_normalize_symbol(gene_name) in valid_genes for gene_name in gene_names],
            dtype=bool,
        )

    if not keep_mask.any():
        raise ValueError("No genes in downloaded partition were found in `token_dict`.")

    adata._inplace_subset_var(keep_mask)


def download_partition(
    partition_idx,
    query_name,
    output_dir,
    index_dir,
    partition_size,
    organism,
    token_dict_path=None,
    homology_path=None,
):
    """
    This function is used to download the partition_idx partition of the query_name
    """
    # define id partition
    normalized_organism = normalize_organism(organism)
    id_list = load2list(query_name, index_dir, normalized_organism)
    id_partition =  define_partition(partition_idx, id_list, partition_size)
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        adata = cellxgene_census.get_anndata(census,    
                                            organism=normalized_organism,
                                            obs_coords=id_partition,
                                            )
    token_dict = _load_token_dict(token_dict_path) if token_dict_path else None
    if normalized_organism == "Mus musculus" and homology_path:
        _apply_mouse_homology_map(
            adata,
            _load_mouse_homology_map(homology_path, token_dict=token_dict),
        )
    if token_dict is not None:
        _filter_genes_by_token_dict(adata, token_dict)
    # Use stable feature identifiers as var_names when available.
    _set_canonical_var_names(adata)
    _drop_unneeded_metadata_columns(adata)
    # Ensure h5ad-compatible string indices explicitly.
    adata.obs_names = adata.obs_names.astype(str)
    # Add a normalized per-cell species label for downstream tokenization.
    adata.obs["species"] = organism_output_name(normalized_organism)
    # prepare the query dir if not exist
    query_dir = os.path.join(output_dir, organism_output_name(normalized_organism), query_name)
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    adata.write_h5ad(query_adata_path, compression="gzip")
    return query_adata_path

def del_partition(partition_idx, query_name, output_dir, index_dir, partition_size, organism):
    query_dir = os.path.join(output_dir, organism_output_name(organism), query_name)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    os.remove(query_adata_path)


if __name__ == "__main__":

    download_partition(partition_idx=args.partition_idx,
                    query_name=args.query_name,
                    output_dir=args.output_dir,
                    index_dir=args.index_dir,
                    partition_size=args.max_partition_size,
                    organism=args.organism,
                    token_dict_path=args.token_dict_path,
                    homology_path=args.homology_path,
                    )
