from datetime import datetime, timezone

import polars as pl
import pytest

from cellpax.contracts import CONTRACTS
from cellpax.contracts.core import TableContract
from cellpax.contracts.invariants import (
    assert_append_only_decisions,
    assert_architecture_contracts,
    assert_content_separate_from_naming,
    assert_no_method_specific_leaks,
    assert_registry_extension,
)

NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def decision(decision_id: str, rationale: str = "because") -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "review_branch": "main",
        "revision_id": "revision",
        "parent_decision_id": None,
        "action": "exclude",
        "target_kind": "candidates",
        "target_ids": [1],
        "target_ref": None,
        "params_json": "{}",
        "taxon_id": None,
        "rationale": rationale,
        "evidence_refs": [],
        "author": "tester",
        "created_at": NOW,
    }


def decision_frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=CONTRACTS["decision"].schema)


def test_builtin_architecture_contracts_pass() -> None:
    assert_architecture_contracts()


def test_method_specific_downstream_column_is_rejected() -> None:
    original = CONTRACTS["decision"]
    bad_schema = pl.Schema(
        {**dict(original.schema.items()), "coclustering_threshold": pl.Float64}
    )
    bad = TableContract(name="decision", schema=bad_schema)
    contracts = {**CONTRACTS, "decision": bad}

    with pytest.raises(AssertionError, match="Method-specific"):
        assert_no_method_specific_leaks(contracts)


def test_human_name_on_content_component_is_rejected() -> None:
    original = CONTRACTS["scope"]
    bad = TableContract(
        name="scope",
        schema=pl.Schema({**dict(original.schema.items()), "name": pl.String}),
    )
    contracts = {**CONTRACTS, "scope": bad}

    with pytest.raises(AssertionError, match="named only by kept revisions"):
        assert_content_separate_from_naming(contracts)


def test_registry_extension_allows_only_new_rows() -> None:
    previous = decision_frame(decision("d1"))
    extended = decision_frame(decision("d1"), decision("d2"))

    assert_registry_extension(previous, extended, CONTRACTS["decision"])
    assert_append_only_decisions(previous, extended)


def test_registry_extension_rejects_update_and_delete() -> None:
    previous = decision_frame(decision("d1"))
    changed = decision_frame(decision("d1", rationale="rewritten"))
    deleted = decision_frame()

    with pytest.raises(AssertionError, match="changed"):
        assert_append_only_decisions(previous, changed)
    with pytest.raises(AssertionError, match="removed"):
        assert_append_only_decisions(previous, deleted)
