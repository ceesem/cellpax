"""Immutable taxonomy preparation and rich integer-enum bindings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import polars as pl

from cellpax.contracts import CONTRACTS, validate_table


@dataclass(frozen=True, slots=True)
class TaxonDefinition:
    """Concise input for one taxon; mechanical persisted fields are optional."""

    taxon_id: int
    key: str
    label: str | None = None
    parent_id: int | None = None
    long_name: str | None = None
    color: str | None = None
    description: str | None = None
    cluster_label: str | None = None
    short_name: str | None = None
    sort_order: int | None = None
    status: str = "active"
    introduced_in: str | None = None
    replaced_by: int | None = None


def taxonomy_table(
    name: str,
    version: str,
    taxa: Iterable[TaxonDefinition],
) -> pl.DataFrame:
    """Build the strict taxonomy table from concise typed definitions.

    Input order supplies ``sort_order`` by default. ``label`` supplies both the
    cluster and short display names, while the humanized key supplies defaults
    for any omitted display names. New taxa default to active and introduced in
    the taxonomy version being built.
    """
    definitions = list(taxa)
    if not definitions:
        raise ValueError("A taxonomy requires at least one taxon definition")
    if not all(isinstance(taxon, TaxonDefinition) for taxon in definitions):
        raise TypeError("taxa must contain TaxonDefinition values")

    rows = []
    for position, taxon in enumerate(definitions):
        display_name = re.sub(r"[_-]+", " ", taxon.key).strip().title()
        label = taxon.label or display_name
        rows.append(
            {
                "taxonomy_name": name,
                "taxonomy_version": version,
                "taxon_id": taxon.taxon_id,
                "key": taxon.key,
                "parent_id": taxon.parent_id,
                "cluster_label": taxon.cluster_label or label,
                "short_name": taxon.short_name or label,
                "long_name": taxon.long_name or display_name,
                "description": taxon.description,
                "color": taxon.color,
                "sort_order": (
                    position if taxon.sort_order is None else taxon.sort_order
                ),
                "status": taxon.status,
                "introduced_in": taxon.introduced_in or version,
                "replaced_by": taxon.replaced_by,
            }
        )
    frame = pl.DataFrame(rows, schema=CONTRACTS["taxonomy"].schema)
    validate_taxonomy(frame)
    return frame


def validate_taxonomy(frame: pl.DataFrame) -> tuple[str, str]:
    """Validate one complete taxonomy version and its parent tree."""
    validate_table(frame, CONTRACTS["taxonomy"])
    names = frame["taxonomy_name"].unique().to_list()
    versions = frame["taxonomy_version"].unique().to_list()
    if len(names) != 1 or not names[0] or len(versions) != 1 or not versions[0]:
        raise ValueError("A taxonomy frame must contain one non-empty name and version")
    ids = set(frame["taxon_id"].to_list())
    if any(taxon_id < 0 for taxon_id in ids):
        raise ValueError("taxon_id values must be non-negative")
    if frame.filter(pl.col("key").str.strip_chars().str.len_chars() == 0).height:
        raise ValueError("Taxonomy keys must be non-empty")
    parents = set(frame["parent_id"].drop_nulls().to_list())
    replacements = set(frame["replaced_by"].drop_nulls().to_list())
    if not parents <= ids:
        raise ValueError("Taxonomy parent_id values must exist in the same version")
    if not replacements <= ids:
        raise ValueError("Taxonomy replaced_by values must exist in the same version")
    parent_by_id = {
        row["taxon_id"]: row["parent_id"] for row in frame.iter_rows(named=True)
    }
    for taxon_id in ids:
        seen: set[int] = set()
        cursor: int | None = taxon_id
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Taxonomy parent hierarchy contains a cycle")
            seen.add(cursor)
            cursor = parent_by_id[cursor]
    return names[0], versions[0]


class RichTaxon(IntEnum):
    """Base for generated taxonomy enums carrying display metadata."""

    @property
    def metadata(self) -> dict[str, object]:
        return dict(type(self).__taxonomy_rows__[int(self)])

    @property
    def key(self) -> str:
        return str(self.metadata["key"])

    @property
    def short_name(self) -> str:
        return str(self.metadata["short_name"])

    @property
    def long_name(self) -> str:
        return str(self.metadata["long_name"])

    @property
    def color(self) -> str | None:
        value = self.metadata["color"]
        return None if value is None else str(value)

    @property
    def parent_id(self) -> int | None:
        value = self.metadata["parent_id"]
        return None if value is None else int(value)


def _enum_identifier(value: str, *, prefix: str) -> str:
    identifier = re.sub(r"\W+", "_", value).strip("_")
    if not identifier or identifier[0].isdigit():
        identifier = f"{prefix}_{identifier}"
    return identifier


def taxonomy_enum(
    frame: pl.DataFrame, *, class_name: str | None = None
) -> type[RichTaxon]:
    """Generate a rich IntEnum binding for one validated taxonomy version."""
    name, version = validate_taxonomy(frame)
    members: dict[str, int] = {}
    rows: dict[int, dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        member = _enum_identifier(row["key"], prefix="TAXON").upper()
        if member in members:
            raise ValueError(f"Taxonomy keys collide as enum member {member!r}")
        members[member] = row["taxon_id"]
        rows[row["taxon_id"]] = row
    generated = RichTaxon(
        class_name or _enum_identifier(f"{name}_{version}_Taxon", prefix="Taxonomy"),
        members,
    )
    generated.__taxonomy_rows__ = rows
    return generated
