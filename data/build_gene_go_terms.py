from pathlib import Path

import pandas as pd
from pybiomart import Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
TOKEN_DICT_PATH = ROOT_DIR / "assets" / "models" / "vocab.json"
OUTPUT_PATH = ROOT_DIR / "checkpoints" / "gene_go_terms.csv"
ENSEMBL_HOST = "http://www.ensembl.org"

from src.assets import load_vocab_json


def load_gene_tokens(path):
    token_path = Path(path).expanduser().resolve()
    if token_path.suffix.lower() == ".json":
        token_df = load_vocab_json(token_path)
    else:
        token_df = pd.read_csv(token_path)
    token_df = token_df.dropna(subset=["gene_id"]).copy()
    token_df["gene_id"] = token_df["gene_id"].astype(str).str.strip()
    token_df = token_df[token_df["gene_id"].str.startswith("ENSG")].copy()
    return token_df


def query_gene_terms():
    dataset = Dataset(name="hsapiens_gene_ensembl", host=ENSEMBL_HOST)
    return dataset.query(
        attributes=[
            "ensembl_gene_id",
            "external_gene_name",
            "go_id",
            "name_1006",
            "namespace_1003",
        ],
        only_unique=True,
        use_attr_names=True,
    ).drop_duplicates()


def build_gene_term_table(token_df, annotation_df):
    annotation_df = annotation_df.rename(
        columns={
            "ensembl_gene_id": "gene_id",
            "external_gene_name": "queried_gene_symbol",
            "go_id": "go_id",
            "name_1006": "go_term_name",
            "namespace_1003": "go_domain",
        }
    )

    merged_df = token_df.merge(annotation_df, on="gene_id", how="left")
    merged_df = merged_df.dropna(subset=["go_id", "go_term_name", "go_domain"])
    return merged_df.sort_values(["token_index", "go_domain", "go_id"], na_position="last")


def main():
    token_df = load_gene_tokens(TOKEN_DICT_PATH)
    annotation_df = query_gene_terms()
    output_df = build_gene_term_table(token_df, annotation_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(output_df)} rows to {OUTPUT_PATH}")




if __name__ == "__main__":
    main()
