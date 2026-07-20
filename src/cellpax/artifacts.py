"""Thin helpers around DataFolio 2 logical item references."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from cellpax.identity import new_entity_id

if TYPE_CHECKING:
    from datafolio import DataFolio

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_item_ref(ref: str) -> str:
    """Validate a namespaced DataFolio logical item name and return it."""
    if not isinstance(ref, str) or not ref:
        raise TypeError("DataFolio item refs must be non-empty strings")
    segments = ref.split("/")
    if any(not _SEGMENT.fullmatch(segment) or segment == ".." for segment in segments):
        raise ValueError(f"Invalid DataFolio item ref: {ref!r}")
    return ref


def new_item_ref(kind: str, *, collection: str = "objects") -> str:
    """Create an immutable internal DataFolio item ref for a CellPax payload."""
    validate_item_ref(kind)
    validate_item_ref(collection)
    return f"cellpax/{collection}/{kind}/{new_entity_id()}"


def commit_ref(commit_id: str) -> str:
    """Return the deterministic DataFolio item ref for a CellPax commit."""
    return validate_item_ref(f"cellpax/commits/{commit_id}")


def component_manifest_ref(kind: str, component_id: str) -> str:
    """Return the deterministic ref for a content component's manifest."""
    validate_item_ref(kind)
    return validate_item_ref(f"cellpax/manifests/{kind}/{component_id}")


def resolve_checksum(folio: "DataFolio", ref: str) -> str:
    """Resolve the exact-byte checksum recorded by DataFolio 2."""
    validate_item_ref(ref)
    info = folio.item_info(ref)
    try:
        checksum = info["checksum"]
    except KeyError as error:
        raise ValueError(
            f"DataFolio item {ref!r} has no owned-payload checksum"
        ) from error
    if not isinstance(checksum, str) or not checksum:
        raise ValueError(f"DataFolio item {ref!r} has an invalid checksum")
    return checksum


def validate_item_refs(folio: "DataFolio", refs: Iterable[str]) -> None:
    """Verify selected refs using DataFolio's public integrity report."""
    requested = tuple(dict.fromkeys(validate_item_ref(ref) for ref in refs))
    report = folio.validate()
    missing = [ref for ref in requested if ref not in report]
    invalid = [ref for ref in requested if ref in report and not report[ref]]
    if missing or invalid:
        raise ValueError(
            f"DataFolio item validation failed; missing={missing}, invalid={invalid}"
        )
