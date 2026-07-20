"""Declarative pieces of a materialized CellPax table contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars import DataFrame, Series


@dataclass(frozen=True, slots=True)
class EnumConstraint:
    """Allowed non-null values for one column."""

    column: str
    values: frozenset[object]


@dataclass(frozen=True, slots=True)
class RangeConstraint:
    """Inclusive numerical bounds for one nullable or non-null column."""

    column: str
    minimum: int | float | None = None
    maximum: int | float | None = None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    """A direct foreign key between two materialized registry tables."""

    columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RowConstraint:
    """A named row predicate that returns True for invalid rows."""

    name: str
    message: str
    columns: tuple[str, ...]
    invalid: Callable[["DataFrame"], "Series"] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class TableContract:
    """Complete physical and row-level contract for one registry table."""

    name: str
    schema: pl.Schema
    nullable: frozenset[str] = frozenset()
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    enums: tuple[EnumConstraint, ...] = ()
    ranges: tuple[RangeConstraint, ...] = ()
    row_constraints: tuple[RowConstraint, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed contracts when modules are imported."""
        columns = set(self.schema.names())
        referenced = set(self.nullable)
        referenced.update(self.primary_key)
        for key in self.unique_keys:
            referenced.update(key)
        referenced.update(constraint.column for constraint in self.enums)
        referenced.update(constraint.column for constraint in self.ranges)
        for constraint in self.row_constraints:
            referenced.update(constraint.columns)
        for foreign_key in self.foreign_keys:
            referenced.update(foreign_key.columns)
        unknown = referenced - columns
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"Contract {self.name!r} references unknown columns: {names}"
            )
        if self.primary_key and set(self.primary_key) & self.nullable:
            raise ValueError(f"Contract {self.name!r} has a nullable primary key")


ContractRegistry = Mapping[str, TableContract]
