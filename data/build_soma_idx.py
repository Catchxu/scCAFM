### This script is used to retrieve cell soma ids from cellxgene census

import cellxgene_census
from data_config import ORGANISM_TO_CENSUS_KEY, VALUE_FILTER, VERSION, normalize_organism, organism_output_name
from typing import List
import os
import argparse

parser = argparse.ArgumentParser(
                    description='Build soma index list based on query')


parser.add_argument("--query-name",
    type=str,
    required=True,
    help="query name to build the index",
)

parser.add_argument("--output-dir",
    type=str,
    required=True,
    help="Directory to store the output idx file",
)

parser.add_argument(
    "--organism",
    type=str,
    default="Homo sapiens",
    help="Organism to query: Homo sapiens or Mus musculus.",
)

args = parser.parse_args()
# print(args)


def retrieve_soma_idx(query_name, organism) -> List[str]:
    """
    This function is used to retrieve cell soma ids from cellxgene census based on the query name
    """
    normalized_organism = normalize_organism(organism)
    organism_key = ORGANISM_TO_CENSUS_KEY[normalized_organism]
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        cell_metadata = census["census_data"][organism_key].obs.read(
            value_filter=VALUE_FILTER[query_name],
            column_names=["soma_joinid"],
        )
        # Consume iterator while census handle is still open.
        cell_metadata = cell_metadata.concat()
        cell_metadata = cell_metadata.to_pandas()
        return cell_metadata["soma_joinid"].to_list()

def convert2file(idx_list: List[str], query_name: str, output_dir: str, organism: str) -> None:
    """
    This function is used to convert the retrieved idx_list to file by query_name
    """

    # set up the dir
    output_root = os.path.join(output_dir, organism_output_name(organism))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    file_path = os.path.join(output_root, f"{query_name}.idx")

    # write to the file
    with open(file_path, 'w') as fp:
        for item in idx_list:
            fp.write("%s\n" % item)

def build_soma_idx(query_name, output_dir, organism) -> None:
    """
    This function is used to build the soma idx for cells under query_name
    """
    idx_list = retrieve_soma_idx(query_name, organism)
    convert2file(idx_list, query_name, output_dir, organism)


# if __name__ ==  "__main__":
#     build_soma_idx("heart")

build_soma_idx(args.query_name, args.output_dir, args.organism)
