#!/usr/bin/env python3
"""
One-shot downloader for Cellxgene Census data.

This script orchestrates:
1) building soma index files per query
2) downloading all partitions per query as .h5ad
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path


def _load_queries(query_list_path: Path):
    queries = []
    with query_list_path.open("r", encoding="utf-8") as f:
        for raw in f:
            q = raw.strip()
            if not q or q.startswith("#"):
                continue
            queries.append(q)
    return queries


def _count_index_rows(index_file: Path) -> int:
    with index_file.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _run(cmd, cwd: Path):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser(
        description="Build soma index and download all query partitions in one run."
    )
    parser.add_argument(
        "--query-list",
        type=str,
        default="query_list.txt",
        help="Path to query list file (default: data/query_list.txt when run from repo root).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory to write/read <query>.idx files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write downloaded .h5ad partitions.",
    )
    parser.add_argument(
        "--max-partition-size",
        type=int,
        default=200000,
        help="Max cells per partition (default: 200000).",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip building soma index and use existing index files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a partition if its output .h5ad file already exists.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    query_list = Path(args.query_list).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not query_list.is_file():
        raise FileNotFoundError(f"query list not found: {query_list}")
    index_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(query_list)
    if not queries:
        raise ValueError(f"no queries found in: {query_list}")

    py = sys.executable

    if not args.skip_index:
        for q in queries:
            _run(
                [
                    py,
                    "build_soma_idx.py",
                    "--query-name",
                    q,
                    "--output-dir",
                    str(index_dir),
                ],
                cwd=script_dir,
            )

    for q in queries:
        idx_file = index_dir / f"{q}.idx"
        if not idx_file.is_file():
            raise FileNotFoundError(
                f"missing index for query '{q}': {idx_file}. "
                "Run without --skip-index or provide valid --index-dir."
            )

        total_cells = _count_index_rows(idx_file)
        if total_cells == 0:
            print(f"[skip] query={q} has 0 cells in index.")
            continue

        n_parts = math.ceil(total_cells / args.max_partition_size)
        print(f"[info] query={q}, cells={total_cells}, partitions={n_parts}")

        for part_idx in range(n_parts):
            out_file = output_dir / q / f"partition_{part_idx}.h5ad"
            if args.resume and out_file.exists():
                print(f"[skip] exists: {out_file}")
                continue

            _run(
                [
                    py,
                    "download_partition.py",
                    "--query-name",
                    q,
                    "--partition-idx",
                    str(part_idx),
                    "--output-dir",
                    str(output_dir),
                    "--index-dir",
                    str(index_dir),
                    "--max-partition-size",
                    str(args.max_partition_size),
                ],
                cwd=script_dir,
            )

    print("[done] all queries finished.")


if __name__ == "__main__":
    main()
