#!/usr/bin/env python3
"""
Check expected .h5ad partitions for integrity.
If a file is missing or broken, delete (if needed) and redownload.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import scanpy as sc


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


def _is_h5ad_valid(path: Path) -> bool:
    try:
        adata = sc.read_h5ad(path, backed="r")
        _ = adata.shape
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        return True
    except Exception:
        return False


def _run(cmd, cwd: Path):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser(
        description="Check partition .h5ad integrity and redownload missing/broken files."
    )
    parser.add_argument(
        "--query-list",
        type=str,
        default="query_list.txt",
        help="Path to query list file.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory containing <query>.idx files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing downloaded partition files.",
    )
    parser.add_argument(
        "--max-partition-size",
        type=int,
        default=200000,
        help="Max cells per partition (default: 200000).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    query_list = Path(args.query_list).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not query_list.is_file():
        raise FileNotFoundError(f"query list not found: {query_list}")
    if not index_dir.is_dir():
        raise FileNotFoundError(f"index dir not found: {index_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(query_list)
    if not queries:
        raise ValueError(f"no queries found in: {query_list}")

    checked = 0
    repaired = 0

    for q in queries:
        idx_file = index_dir / f"{q}.idx"
        if not idx_file.is_file():
            raise FileNotFoundError(f"missing index file: {idx_file}")

        total_cells = _count_index_rows(idx_file)
        if total_cells == 0:
            print(f"[skip] query={q} has 0 cells in index.")
            continue

        n_parts = math.ceil(total_cells / args.max_partition_size)
        print(f"[info] query={q}, expected_partitions={n_parts}")

        for part_idx in range(n_parts):
            checked += 1
            out_file = output_dir / q / f"partition_{part_idx}.h5ad"

            needs_redownload = False
            if out_file.exists():
                if not _is_h5ad_valid(out_file):
                    print(f"[broken] {out_file}")
                    out_file.unlink(missing_ok=True)
                    needs_redownload = True
            else:
                print(f"[missing] {out_file}")
                needs_redownload = True

            if needs_redownload:
                repaired += 1
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

    print(f"[done] checked={checked}, repaired={repaired}")


if __name__ == "__main__":
    main()
