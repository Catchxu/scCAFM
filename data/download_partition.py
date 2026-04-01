import cellxgene_census
import pandas as pd
import numpy as np
from data_config import VERSION, normalize_organism, organism_output_name
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


def _filter_genes_by_token_dict(adata, token_dict: pd.DataFrame) -> None:
    gene_names = adata.var_names.astype(str).tolist()
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


def download_partition(partition_idx, query_name, output_dir, index_dir, partition_size, organism, token_dict_path=None):
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
    if token_dict_path:
        token_dict = _load_token_dict(token_dict_path)
        _filter_genes_by_token_dict(adata, token_dict)
    # Ensure h5ad-compatible string indices explicitly.
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
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
                    )
