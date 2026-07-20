"""Importable architecture invariants for contract and mutation tests."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from cellpax.contracts.core import TableContract
from cellpax.contracts.schemas import CONTRACTS
from cellpax.contracts.validation import validate_table

DOWNSTREAM_CONTRACTS = frozenset(
    {
        "kept_revision",
        "decision",
        "taxonomy",
        "assignment_set",
        "assignment",
        "annotation_release",
    }
)
METHOD_SPECIFIC_COLUMN_TOKENS = (
    "coclustering",
    "similarity_matrix",
    "cut_threshold",
    "cluster_threshold",
)
CONTENT_COMPONENT_CONTRACTS = frozenset(
    {
        "universe",
        "feature_block",
        "feature_selection",
        "scope",
        "feature_space",
        "representation",
        "clustering_run",
        "candidate_set",
        "propagation_run",
    }
)


def assert_no_method_specific_leaks(
    contracts: Mapping[str, TableContract] = CONTRACTS,
) -> None:
    """Ensure downstream contracts do not expose generator-specific columns."""
    failures: list[str] = []
    for name in sorted(DOWNSTREAM_CONTRACTS):
        table_contract = contracts[name]
        for column in table_contract.schema.names():
            lowered = column.lower()
            if any(token in lowered for token in METHOD_SPECIFIC_COLUMN_TOKENS):
                failures.append(f"{name}.{column}")
    if failures:
        raise AssertionError(
            "Method-specific columns leaked downstream: " + ", ".join(failures)
        )


def assert_content_separate_from_naming(
    contracts: Mapping[str, TableContract] = CONTRACTS,
) -> None:
    """Ensure content component registries have no human naming field."""
    failures = [
        name
        for name in sorted(CONTENT_COMPONENT_CONTRACTS)
        if "name" in contracts[name].schema
    ]
    if failures:
        raise AssertionError(
            "Content components must be named only by kept revisions: "
            + ", ".join(failures)
        )


def assert_registry_extension(
    previous: pl.DataFrame,
    current: pl.DataFrame,
    contract: TableContract,
) -> None:
    """Ensure a registry snapshot only appends immutable keyed rows."""
    if not contract.primary_key:
        raise ValueError(f"Contract {contract.name!r} has no primary key")
    validate_table(previous, contract)
    validate_table(current, contract)

    keys = contract.primary_key

    def indexed(frame: pl.DataFrame) -> dict[tuple[object, ...], dict[str, object]]:
        result: dict[tuple[object, ...], dict[str, object]] = {}
        for row in frame.iter_rows(named=True):
            key = tuple(row[column] for column in keys)
            result[key] = row
        return result

    before = indexed(previous)
    after = indexed(current)
    missing = before.keys() - after.keys()
    changed = [key for key in before.keys() & after.keys() if before[key] != after[key]]
    if missing or changed:
        raise AssertionError(
            f"Registry {contract.name!r} is not an immutable extension; "
            f"removed={sorted(missing)!r}, changed={sorted(changed)!r}"
        )


def assert_append_only_decisions(previous: pl.DataFrame, current: pl.DataFrame) -> None:
    """Enforce append-only behavior for the curation ledger."""
    assert_registry_extension(previous, current, CONTRACTS["decision"])


def assert_architecture_contracts() -> None:
    """Run architecture invariants that can be checked from schemas alone."""
    assert_no_method_specific_leaks()
    assert_content_separate_from_naming()
