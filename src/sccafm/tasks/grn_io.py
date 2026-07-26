from __future__ import annotations

import os
import tempfile

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .grn import CellSpecificGRNs, PooledGRN


def _write_tables_atomically(
    tables: Iterable[pd.DataFrame],
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    wrote_table = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            for table in tables:
                if not isinstance(table, pd.DataFrame):
                    raise TypeError("GRN table iterators must yield pandas DataFrames.")
                table.to_csv(
                    handle,
                    index=False,
                    header=not wrote_table,
                )
                wrote_table = True
        if not wrote_table:
            raise ValueError("No GRN result tables were produced.")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_cell_specific_grns_csv(
    result: CellSpecificGRNs,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Stream cell-specific edge tables to one CSV file."""

    if not isinstance(result, CellSpecificGRNs):
        raise TypeError("`result` must be a CellSpecificGRNs object.")
    return _write_tables_atomically(
        result.iter_edge_tables(),
        path,
        overwrite=overwrite,
    )


def write_pooled_grn_csv(
    result: PooledGRN,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one pooled GRN edge table to CSV."""

    if not isinstance(result, PooledGRN):
        raise TypeError("`result` must be a PooledGRN object.")
    return _write_tables_atomically(
        [result.to_edge_table()],
        path,
        overwrite=overwrite,
    )


__all__ = [
    "write_cell_specific_grns_csv",
    "write_pooled_grn_csv",
]
