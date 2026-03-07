#!/usr/bin/env python3
"""Replace placeholder descriptions for ENSG-only gene symbols."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_DESCRIPTION = "Not well characterized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update gene_descriptions.csv so genes with symbols starting with "
            "'ENSG' use a fallback description."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("resources/gene_descriptions.csv"),
        help="Input CSV path (default: resources/gene_descriptions.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, overwrite --input.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ENSG",
        help="Gene symbol prefix to match (default: ENSG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input

    df = pd.read_csv(args.input)
    required_columns = {"gene_symbol", "description"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    matched = df["gene_symbol"].astype(str).str.startswith(args.prefix)
    matched_count = int(matched.sum())
    df.loc[matched, "description"] = TARGET_DESCRIPTION
    df.to_csv(output_path, index=False)

    print(
        f"Updated {matched_count} rows where gene_symbol starts with "
        f"'{args.prefix}'. Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
