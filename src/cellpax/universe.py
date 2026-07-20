"""Universe-specific validation and semantic-role normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from cellpax.identity import canonical_json_bytes
from cellpax.table_utils import validate_cell_table


def semantic_roles_json(frame: pl.DataFrame, roles: Mapping[str, Sequence[str]]) -> str:
    """Validate semantic column roles and return canonical JSON text."""
    normalized: dict[str, list[str]] = {}
    for role, columns in roles.items():
        if not isinstance(role, str) or not role:
            raise ValueError("Semantic role names must be non-empty strings")
        values = list(columns)
        if not values or not all(
            isinstance(column, str) and column for column in values
        ):
            raise ValueError(f"Semantic role {role!r} requires column names")
        if len(values) != len(set(values)):
            raise ValueError(f"Semantic role {role!r} contains duplicate columns")
        missing = set(values) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Semantic role {role!r} references missing columns: {sorted(missing)}"
            )
        normalized[role] = values
    return canonical_json_bytes(normalized).decode("utf-8")


def validate_universe_cells(
    cells: pl.DataFrame, *, nullable_columns: Sequence[str] = ()
) -> None:
    """Validate authoritative universe cells before materialization."""
    validate_cell_table(cells, nullable_columns=nullable_columns)
