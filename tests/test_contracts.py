from datetime import datetime, timezone

import polars as pl
import pytest

from cellpax.contracts import (
    CONTRACTS,
    ContractValidationError,
    contract,
    validate_registries,
    validate_table,
)
from cellpax.contracts.core import TableContract

NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def frame(name: str, rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=CONTRACTS[name].schema)


def test_every_contract_accepts_an_empty_typed_frame() -> None:
    assert len(CONTRACTS) == 21
    for table_contract in CONTRACTS.values():
        empty = pl.DataFrame(schema=table_contract.schema)
        assert validate_table(empty, table_contract) == ()


def test_malformed_contract_definitions_fail_at_import_boundary() -> None:
    with pytest.raises(ValueError, match="unknown columns"):
        TableContract(
            name="bad",
            schema=pl.Schema({"id": pl.String}),
            nullable=frozenset({"missing"}),
        )
    with pytest.raises(ValueError, match="nullable primary key"):
        TableContract(
            name="bad",
            schema=pl.Schema({"id": pl.String}),
            nullable=frozenset({"id"}),
            primary_key=("id",),
        )


def test_unknown_contract_name_and_registry_are_reported() -> None:
    with pytest.raises(KeyError, match="Unknown CellPax contract"):
        contract("missing")

    issues = validate_registries({"missing": pl.DataFrame()}, raise_on_error=False)
    assert issues[0].code == "unknown_contract"


def test_schema_columns_and_dtypes_are_exact() -> None:
    malformed = pl.DataFrame({"universe_id": [1]})

    issues = validate_table(malformed, CONTRACTS["universe"], raise_on_error=False)

    assert issues[0].code == "columns"
    with pytest.raises(ContractValidationError, match="column mismatch"):
        validate_table(malformed, CONTRACTS["universe"])


def test_nullability_primary_key_enum_and_range_are_reported_together() -> None:
    malformed = frame(
        "candidate_membership",
        [
            {
                "candidate_set_id": "set",
                "cell_id": 1,
                "candidate_id": -1,
                "membership_strength": 1.5,
            },
            {
                "candidate_set_id": "set",
                "cell_id": 1,
                "candidate_id": None,
                "membership_strength": None,
            },
        ],
    )

    issues = validate_table(
        malformed, CONTRACTS["candidate_membership"], raise_on_error=False
    )

    assert {issue.code for issue in issues} == {"primary_key", "range"}


def test_feature_space_requires_exactly_one_source() -> None:
    malformed = frame(
        "feature_space",
        [
            {
                "feature_space_id": "space",
                "scope_id": "scope",
                "fit_scope_id": "scope",
                "parent_feature_space_id": None,
                "input_feature_block_ids": None,
                "feature_selection_id": "selection",
                "transform": "raw_join",
                "params_json": "{}",
                "fitted_state_ref": None,
                "values_ref": None,
                "missing_policy": "drop",
                "seed": None,
                "created_at": NOW,
                "created_by": "tester",
            }
        ],
    )

    with pytest.raises(ContractValidationError, match="exactly one"):
        validate_table(malformed, CONTRACTS["feature_space"])


def test_decision_requires_one_target_and_known_action() -> None:
    malformed = frame(
        "decision",
        [
            {
                "decision_id": "decision",
                "review_branch": "main",
                "revision_id": "revision",
                "parent_decision_id": None,
                "action": "invent",
                "target_kind": "cells",
                "target_ids": None,
                "target_ref": None,
                "params_json": "{}",
                "taxon_id": None,
                "rationale": "test",
                "evidence_refs": [],
                "author": "tester",
                "created_at": NOW,
            }
        ],
    )

    issues = validate_table(malformed, CONTRACTS["decision"], raise_on_error=False)

    assert {issue.code for issue in issues} == {"enum", "one_decision_target"}


def test_assignment_provenance_and_status_rules() -> None:
    malformed = frame(
        "assignment",
        [
            {
                "assignment_set_id": "set",
                "cell_id": 1,
                "taxon_id": None,
                "assignment_status": "leaf",
                "assignment_source": "propagated",
                "source_revision_id": "revision",
                "decision_id": None,
                "propagation_run_id": None,
                "coverage_feature_space_id": None,
                "coverage": None,
                "confidence": None,
                "alternatives_json": None,
                "created_at": NOW,
            }
        ],
    )

    issues = validate_table(malformed, CONTRACTS["assignment"], raise_on_error=False)

    assert {issue.code for issue in issues} == {
        "propagation_provenance",
        "taxon_assignment_status",
    }


def test_registry_foreign_keys_are_checked_when_targets_are_present() -> None:
    universes = frame(
        "universe",
        [
            {
                "universe_id": "u1",
                "cells_ref": "cellpax/objects/cells/one",
                "schema_hash": "hash",
                "semantic_roles_json": "{}",
                "source_refs": [],
                "created_at": NOW,
                "created_by": "tester",
            }
        ],
    )
    blocks = frame(
        "feature_block",
        [
            {
                "feature_block_id": "b1",
                "universe_id": "missing",
                "values_ref": "cellpax/objects/features/one",
                "schema_hash": "hash",
                "source_refs": [],
                "created_at": NOW,
                "created_by": "tester",
            }
        ],
    )

    issues = validate_registries(
        {"universe": universes, "feature_block": blocks}, raise_on_error=False
    )

    assert len(issues) == 1
    assert issues[0].code == "foreign_key"


def test_missing_foreign_target_can_be_optional_or_required() -> None:
    blocks = pl.DataFrame(schema=CONTRACTS["feature_block"].schema)

    assert validate_registries({"feature_block": blocks}) == ()
    issues = validate_registries(
        {"feature_block": blocks},
        require_foreign_targets=True,
        raise_on_error=False,
    )

    assert issues[0].code == "foreign_key_target_missing"
