"""Scope membership normalization and lineage checks."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

SCOPE_MEMBERS_SCHEMA = pl.Schema({"cell_id": pl.Int64})


def normalize_cell_ids(
    cells: pl.DataFrame | pl.Series | Iterable[int],
) -> pl.DataFrame:
    """Return sorted unique canonical Int64 membership."""
    if isinstance(cells, pl.DataFrame):
        if "cell_id" not in cells.columns:
            raise ValueError("Scope DataFrame requires a cell_id column")
        result = cells.select("cell_id")
    elif isinstance(cells, pl.Series):
        result = cells.rename("cell_id").to_frame()
    else:
        result = pl.DataFrame({"cell_id": list(cells)}, schema=SCOPE_MEMBERS_SCHEMA)
    if result.schema["cell_id"] != pl.Int64:
        raise TypeError("Scope cell_id must have Polars Int64 dtype")
    if result["cell_id"].null_count():
        raise ValueError("Scope cell_id values must be non-null")
    return result.unique(maintain_order=False).sort("cell_id")


def require_membership_subset(
    members: pl.DataFrame,
    container: pl.DataFrame,
    *,
    container_name: str,
) -> None:
    """Reject membership outside an authoritative or parent set."""
    outside = members.join(container.select("cell_id"), on="cell_id", how="anti")
    if not outside.is_empty():
        sample = outside.head(10)["cell_id"].to_list()
        raise ValueError(
            f"Scope contains {outside.height} cells outside {container_name}; sample={sample}"
        )
