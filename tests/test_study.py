from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from datafolio import DataFolio

from cellpax import Study
from cellpax.artifacts import commit_ref, component_manifest_ref
from cellpax.contracts import CONTRACTS, SCHEMA_VERSION, ContractValidationError
from cellpax.identity import canonical_hash


def universe_cells() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell_id": pl.Series([30, 10, 20], dtype=pl.Int64),
            "root_id": pl.Series([300, 100, 200], dtype=pl.Int64),
            "soma_x": pl.Series([3.0, 1.0, 2.0], dtype=pl.Float64),
            "label": pl.Series(["c", None, "b"], dtype=pl.String),
        }
    )


def feature_values() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell_id": pl.Series([10, 20, 30], dtype=pl.Int64),
            "area": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float64),
            "branches": pl.Series([2, 4, 6], dtype=pl.Int32),
        }
    )


def feature_catalog(*, family: str = "morphology") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_id": ["area", "branch-count"],
            "column_name": ["area", "branches"],
            "modality": ["morphology", "morphology"],
            "family": [family, family],
            "units": pl.Series(["um2", None], dtype=pl.String),
            "description": pl.Series(["surface area", None], dtype=pl.String),
            "raw_or_derived": ["raw", "raw"],
        }
    )


def create_substrate(path: Path) -> tuple[Study, object, object]:
    study = Study.create(path, created_by="test-user", metadata={"project": "tiny"})
    universe = study.register_universe(
        universe_cells(),
        semantic_roles={"identifier_alias": ["root_id"], "position": ["soma_x"]},
        nullable_columns=["label"],
    )
    block = study.register_feature_block(feature_values(), feature_catalog())
    return study, universe, block


def ordered_selection(block_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_block_id": pl.Series([block_id, block_id], dtype=pl.String),
            "feature_id": pl.Series(["branch-count", "area"], dtype=pl.String),
        }
    )


def test_slice_one_preview_keep_reopen_validate_and_lineage(tmp_path: Path) -> None:
    path = tmp_path / "study"
    study, universe, block = create_substrate(path)

    selection = study.preview_feature_selection(
        ordered_selection(block.feature_block_id), derivation_text="notebook columns"
    )
    root_scope = study.preview_scope(
        pl.Series("cell_id", [30, 10, 30], dtype=pl.Int64),
        derivation_text="inhibitory cells",
    )
    assert study.registry("scope").is_empty()
    assert study.registry("feature_selection").is_empty()

    revision = study.keep(
        "inh-core",
        scope=root_scope,
        feature_selection=selection,
        notes="first durable revision",
    )
    child = study.preview_scope([10], derivation_text="CGE branch", parent=root_scope)
    child_revision = study.keep(
        "cge", scope=child, parent_revision=revision, notes="local branch"
    )

    assert root_scope.n_cells == 2
    assert study.folio.get(root_scope.members_ref, frame="polars")[
        "cell_id"
    ].to_list() == [10, 30]
    members = study.folio.get(selection.members_ref, frame="polars")
    assert members["position"].to_list() == [0, 1]
    assert members["feature_id"].to_list() == ["branch-count", "area"]
    assert child_revision.parent_revision_id == revision.revision_id
    assert study.registry("kept_revision")["name"].to_list() == [
        "inh-core",
        "cge",
    ]
    study.validate()

    head = study.head_commit()
    assert head is not None
    assert set(study.folio.get_inputs(commit_ref(head.commit_id))) == set(
        head.registries.values()
    )
    scope_inputs = study.folio.get_inputs(head.registries["scope"])
    previous_scope_registry = next(
        ref for ref in scope_inputs if ref.startswith("cellpax/registries/scope/")
    )
    assert root_scope.members_ref in study.folio.get_inputs(previous_scope_registry)
    assert child.members_ref in scope_inputs
    assert selection.members_ref in study.folio.get_inputs(
        head.registries["feature_selection"]
    )
    assert study.folio.get_inputs(
        component_manifest_ref("universe", universe.universe_id)
    ) == [universe.cells_ref]
    assert study.folio.get_inputs(
        component_manifest_ref("feature-block", block.feature_block_id)
    ) == [block.values_ref]

    reopened = Study.open(path)
    reopened.validate()
    assert reopened.study_id == study.study_id
    assert reopened.get_revision(child_revision.revision_id).name == "cge"
    assert reopened.get_scope(child.scope_id).parent_scope_id == root_scope.scope_id
    assert (
        reopened.get_feature_selection(selection.feature_selection_id).n_features == 2
    )


def test_reopen_and_validate_in_fresh_process(tmp_path: Path) -> None:
    path = tmp_path / "study"
    study, _, block = create_substrate(path)
    selection = study.preview_feature_selection(
        ordered_selection(block.feature_block_id), derivation_text="ordered"
    )
    scope = study.preview_scope([10, 20], derivation_text="root")
    revision = study.keep("kept", scope=scope, feature_selection=selection)

    code = """
import json
import sys
from cellpax import Study

study = Study.open(sys.argv[1], read_only=True)
study.validate()
revision = study.registry("kept_revision").row(0, named=True)
selection = study.registry("feature_selection").row(0, named=True)
print(json.dumps({
    "head": study.head_commit_id,
    "revision": revision["revision_id"],
    "selection_size": selection["n_features"],
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout)
    assert loaded == {
        "head": study.head_commit_id,
        "revision": revision.revision_id,
        "selection_size": 2,
    }


def test_failed_keep_rolls_back_commit_and_registry_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _, _ = create_substrate(tmp_path / "study")
    scope = study.preview_scope([10, 20], derivation_text="preview")
    before_head = study.head_commit_id
    before_contents = study.folio.list_contents()
    original_add = DataFolio.add

    def failing_add(self: DataFolio, name: str, obj: object, **kwargs: object):
        if name.startswith("cellpax/commits/"):
            raise RuntimeError("injected commit failure")
        return original_add(self, name, obj, **kwargs)

    monkeypatch.setattr(DataFolio, "add", failing_add)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        study.keep("should-not-exist", scope=scope)

    assert study.head_commit_id == before_head
    assert study.folio.list_contents() == before_contents
    assert study.registry("scope").is_empty()
    assert study.registry("kept_revision").is_empty()


def test_registration_is_immutable_and_idempotent(tmp_path: Path) -> None:
    study, universe, block = create_substrate(tmp_path / "study")
    contents = study.folio.list_contents()

    same_universe = study.register_universe(
        universe_cells(),
        semantic_roles={"position": ["soma_x"], "identifier_alias": ["root_id"]},
        nullable_columns=["label"],
    )
    same_block = study.register_feature_block(feature_values(), feature_catalog())
    assert same_universe == universe
    assert same_block == block
    assert study.folio.list_contents() == contents
    assert study.registry("feature_block").height == 1

    changed = universe_cells().with_columns(pl.col("soma_x") + 1)
    with pytest.raises(ValueError, match="cannot be replaced"):
        study.register_universe(
            changed,
            semantic_roles={
                "identifier_alias": ["root_id"],
                "position": ["soma_x"],
            },
            nullable_columns=["label"],
        )
    with pytest.raises(ValueError, match="different semantic catalog"):
        study.register_feature_block(
            feature_values(), feature_catalog(family="different")
        )


def test_scope_identity_tracks_derivation_and_parent_rules(tmp_path: Path) -> None:
    study, _, _ = create_substrate(tmp_path / "study")
    left = study.preview_scope([20, 10], derivation_text="left recipe")
    right = study.preview_scope([10, 20], derivation_text="right recipe")
    assert left.membership_hash == right.membership_hash
    assert left.scope_id != right.scope_id

    study.keep("left", scope=left)
    with pytest.raises(ValueError, match="outside the parent scope"):
        study.preview_scope([30], derivation_text="invalid child", parent=left)
    with pytest.raises(ValueError, match="already exists"):
        study.keep("left", scope=right)
    with pytest.raises(KeyError, match="Unknown scope"):
        study.get_scope("missing")
    with pytest.raises(KeyError, match="Unknown kept revision"):
        study.get_revision("missing")


def test_revision_entity_identity_is_separate_from_state_and_config(
    tmp_path: Path,
) -> None:
    study, _, _ = create_substrate(tmp_path / "study")
    scope = study.preview_scope([10, 20], derivation_text="same state")
    first = study.keep("first name", scope=scope)
    second = study.keep("second name", scope=scope)
    assert first.revision_id != second.revision_id
    assert first.name != second.name
    assert first.state_hash == second.state_hash
    assert first.config_hash == second.config_hash


def test_keep_rejects_unrooted_parent_and_foreign_preview_handles(
    tmp_path: Path,
) -> None:
    study, _, block = create_substrate(tmp_path / "local")
    parent = study.preview_scope([10, 20], derivation_text="preview parent")
    child = study.preview_scope([10], derivation_text="preview child", parent=parent)
    with pytest.raises(ValueError, match="parent must be kept"):
        study.keep("child too early", scope=child)

    foreign, _, foreign_block = create_substrate(tmp_path / "foreign")
    foreign_scope = foreign.preview_scope([10], derivation_text="foreign")
    with pytest.raises(ValueError, match="does not belong to this study"):
        study.keep("foreign scope", scope=foreign_scope)

    local_scope = study.preview_scope([10], derivation_text="local")
    foreign_selection = foreign.preview_feature_selection(
        ordered_selection(foreign_block.feature_block_id), derivation_text="foreign"
    )
    with pytest.raises(ValueError, match="does not belong to this study"):
        study.keep(
            "foreign selection",
            scope=local_scope,
            feature_selection=foreign_selection,
        )

    foreign_revision = foreign.keep("foreign revision", scope=foreign_scope)
    with pytest.raises(ValueError, match="Parent revision does not belong"):
        study.keep(
            "foreign parent", scope=local_scope, parent_revision=foreign_revision
        )
    assert block.feature_block_id == foreign_block.feature_block_id


def test_registration_and_selection_reject_invalid_inputs(tmp_path: Path) -> None:
    study = Study.create(tmp_path / "study", created_by="test-user")
    with pytest.raises(ValueError, match="one universe"):
        study.universe()
    with pytest.raises(ValueError, match="duplicate"):
        study.register_universe(
            universe_cells(),
            semantic_roles={"position": ["soma_x"]},
            nullable_columns=["label"],
            source_refs=["source/data", "source/data"],
        )

    study.register_universe(
        universe_cells(),
        semantic_roles={"position": ["soma_x"]},
        nullable_columns=["label"],
    )
    with pytest.raises(ValueError, match="at least one feature"):
        study.register_feature_block(
            feature_values().select("cell_id"), feature_catalog().head(0)
        )
    outside = feature_values().with_columns(
        pl.when(pl.col("cell_id") == 30)
        .then(pl.lit(999, dtype=pl.Int64))
        .otherwise(pl.col("cell_id"))
        .alias("cell_id")
    )
    with pytest.raises(ValueError, match="outside the universe"):
        study.register_feature_block(outside, feature_catalog())
    with pytest.raises(ValueError, match="describe every physical"):
        study.register_feature_block(feature_values(), feature_catalog().head(1))

    block = study.register_feature_block(feature_values(), feature_catalog())
    with pytest.raises(ValueError, match="unknown features"):
        study.preview_feature_selection(
            pl.DataFrame(
                {
                    "feature_block_id": [block.feature_block_id],
                    "feature_id": ["missing"],
                }
            ),
            derivation_text="bad",
        )
    with pytest.raises(TypeError, match="String dtype"):
        study.preview_feature_selection(
            pl.DataFrame(
                {
                    "feature_block_id": [1],
                    "feature_id": [2],
                }
            ),
            derivation_text="bad dtypes",
        )
    with pytest.raises(ValueError, match="duplicate feature"):
        duplicated = ordered_selection(block.feature_block_id).head(1)
        study.preview_feature_selection(
            pl.concat([duplicated, duplicated]), derivation_text="duplicates"
        )
    with pytest.raises(KeyError, match="Unknown feature selection"):
        study.get_feature_selection("missing")


def test_preview_preconditions_and_existing_content_reuse(tmp_path: Path) -> None:
    study = Study.create(tmp_path / "study", created_by="user")
    study.validate()
    study.register_universe(
        universe_cells(),
        semantic_roles={"position": ["soma_x"]},
        nullable_columns=["label"],
    )
    with pytest.raises(ValueError, match="before registering"):
        study.preview_feature_selection(
            pl.DataFrame(
                {
                    "feature_block_id": pl.Series([], dtype=pl.String),
                    "feature_id": pl.Series([], dtype=pl.String),
                }
            ),
            derivation_text="empty",
        )
    block = study.register_feature_block(feature_values(), feature_catalog())
    with pytest.raises(ValueError, match="derivation_text"):
        study.preview_feature_selection(
            ordered_selection(block.feature_block_id), derivation_text=""
        )
    with pytest.raises(ValueError, match="derivation_text"):
        study.preview_scope([10], derivation_text="")

    selection = study.preview_feature_selection(
        ordered_selection(block.feature_block_id), derivation_text="first"
    )
    scope = study.preview_scope([10, 20], derivation_text="root")
    revision = study.keep("root", scope=scope, feature_selection=selection)
    assert (
        study.preview_feature_selection(
            ordered_selection(block.feature_block_id), derivation_text="second"
        )
        == selection
    )
    assert study.preview_scope([20, 10], derivation_text="root") == scope
    child = study.preview_scope([10], derivation_text="child", parent=scope.scope_id)
    second = study.keep(
        "child",
        scope=child,
        feature_selection=selection.feature_selection_id,
        parent_revision=revision.revision_id,
    )
    assert second.parent_revision_id == revision.revision_id
    third = study.keep(
        "child-again", scope=child.scope_id, parent_revision=second.revision_id
    )
    assert third.scope_id == child.scope_id
    with pytest.raises(ValueError, match="requires a name"):
        study.keep("", scope=scope)


def test_preheld_duplicate_scope_preview_has_clear_stale_error(tmp_path: Path) -> None:
    study, _, _ = create_substrate(tmp_path / "study")
    first = study.preview_scope([10, 20], derivation_text="same scope")
    stale = study.preview_scope([20, 10], derivation_text="same scope")
    assert first.scope_id == stale.scope_id
    assert first.members_ref != stale.members_ref

    study.keep("first", scope=first)
    with pytest.raises(ValueError, match="Stale scope preview handle"):
        study.keep("second", scope=stale)


def test_catalog_shape_types_and_duplicate_columns_are_rejected(tmp_path: Path) -> None:
    study = Study.create(tmp_path / "study", created_by="user")
    study.register_universe(
        universe_cells(),
        semantic_roles={"position": ["soma_x"]},
        nullable_columns=["label"],
    )
    with pytest.raises(ValueError, match="columns mismatch"):
        study.register_feature_block(
            feature_values(), feature_catalog().drop("description")
        )
    with pytest.raises(TypeError, match="must have String"):
        study.register_feature_block(
            feature_values(), feature_catalog().with_columns(pl.lit(1).alias("family"))
        )
    duplicated = feature_catalog().with_columns(pl.lit("area").alias("column_name"))
    with pytest.raises(ValueError, match="must be unique"):
        study.register_feature_block(feature_values(), duplicated)
    null_optionals = feature_catalog().with_columns(
        pl.lit(None).alias("units"), pl.lit(None).alias("description")
    )
    block = study.register_feature_block(feature_values(), null_optionals)
    assert block.feature_block_id


def test_source_refs_are_canonical_and_validated(tmp_path: Path) -> None:
    study = Study.create(tmp_path / "study", created_by="user")
    study.folio.add("source/z", {"source": "z"})
    study.folio.add("source/a", {"source": "a"})
    universe = study.register_universe(
        universe_cells(),
        semantic_roles={"position": ["soma_x"]},
        nullable_columns=["label"],
        source_refs=["source/z", "source/a"],
    )
    assert universe.source_refs == ("source/a", "source/z")
    block = study.register_feature_block(
        feature_values(), feature_catalog(), source_refs=["source/z"]
    )
    assert block.source_refs == ("source/z",)
    study.validate()


def _durable_substrate(path: Path) -> tuple[Study, object, object, object]:
    study, _, block = create_substrate(path)
    selection = study.preview_feature_selection(
        ordered_selection(block.feature_block_id), derivation_text="selection"
    )
    scope = study.preview_scope([10, 20], derivation_text="scope")
    revision = study.keep("durable", scope=scope, feature_selection=selection)
    return study, scope, selection, revision


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("scope_order", "membership is not canonical"),
        ("scope_count", "n_cells mismatch"),
        ("scope_hash", "membership hash mismatch"),
        ("selection_position", "members are not canonical"),
        ("selection_catalog", "catalog hash mismatch"),
        ("selection_count", "count mismatch"),
        ("revision_state", "state hash mismatch"),
    ],
)
def test_validation_detects_retained_artifact_corruption(
    tmp_path: Path, corruption: str, message: str
) -> None:
    study, scope, selection, _ = _durable_substrate(tmp_path / corruption)
    head = study.head_commit()
    assert head is not None
    if corruption == "scope_order":
        members = study.folio.get(scope.members_ref, frame="polars").reverse()
        study.folio.add(scope.members_ref, members, overwrite=True)
    elif corruption in {"scope_count", "scope_hash"}:
        ref = head.registries["scope"]
        registry = study.folio.get(ref, frame="polars")
        column = "n_cells" if corruption == "scope_count" else "membership_hash"
        value = (
            pl.lit(999, dtype=pl.Int64)
            if corruption == "scope_count"
            else pl.lit("bad")
        )
        study.folio.add(ref, registry.with_columns(value.alias(column)), overwrite=True)
    elif corruption == "selection_position":
        members = study.folio.get(selection.members_ref, frame="polars")
        study.folio.add(
            selection.members_ref,
            members.with_columns((pl.col("position") + 1).alias("position")),
            overwrite=True,
        )
    elif corruption in {"selection_catalog", "selection_count"}:
        ref = head.registries["feature_selection"]
        registry = study.folio.get(ref, frame="polars")
        column = "catalog_hash" if corruption == "selection_catalog" else "n_features"
        value = "bad" if corruption == "selection_catalog" else 999
        study.folio.add(
            ref, registry.with_columns(pl.lit(value).alias(column)), overwrite=True
        )
    else:
        ref = head.registries["kept_revision"]
        registry = study.folio.get(ref, frame="polars")
        study.folio.add(
            ref,
            registry.with_columns(pl.lit("bad").alias("state_hash")),
            overwrite=True,
        )
    with pytest.raises(ValueError, match=message):
        study.validate()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"unexpected": True}, "Malformed universe manifest"),
        (
            {
                "component_kind": "feature_block",
                "schema_version": "1",
                "cells_checksum": "x",
                "schema_hash": "x",
                "semantic_roles": {},
                "nullable_columns": [],
                "source_refs": [],
            },
            "Manifest kind mismatch",
        ),
    ],
)
def test_validation_detects_component_manifest_corruption(
    tmp_path: Path, replacement: dict[str, object], message: str
) -> None:
    study, universe, _ = create_substrate(tmp_path / message.replace(" ", "-"))
    ref = component_manifest_ref("universe", universe.universe_id)
    study.folio.add(ref, replacement, overwrite=True)
    with pytest.raises(ValueError, match=message):
        study.validate()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"bad": "shape"}, "Malformed CellPax commit"),
        (
            {"schema_version": "other", "parent_commit_id": None, "registry_refs": {}},
            "unsupported schema version",
        ),
        (
            {
                "schema_version": SCHEMA_VERSION,
                "parent_commit_id": None,
                "registry_refs": [],
            },
            "registry_refs must be a mapping",
        ),
        (
            {
                "schema_version": SCHEMA_VERSION,
                "parent_commit_id": None,
                "registry_refs": {"unknown": "cellpax/registries/unknown/value"},
            },
            "unknown registry",
        ),
    ],
)
def test_head_commit_rejects_malformed_manifests(
    tmp_path: Path, manifest: dict[str, object], message: str
) -> None:
    study = Study.create(tmp_path / message.replace(" ", "-"), created_by="user")
    commit_id = canonical_hash(manifest)
    study.folio.add(commit_ref(commit_id), manifest)
    study.folio.metadata["cellpax_head_commit_id"] = commit_id
    with pytest.raises(ValueError, match=message):
        study.head_commit()


def test_validation_requires_foreign_key_target_registries(tmp_path: Path) -> None:
    study, _, _, _ = _durable_substrate(tmp_path / "missing-foreign-target")
    head = study.head_commit()
    assert head is not None
    registry_refs = head.registries
    registry_refs.pop("scope")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "parent_commit_id": head.commit_id,
        "registry_refs": dict(sorted(registry_refs.items())),
    }
    commit_id = canonical_hash(manifest)
    study.folio.add(
        commit_ref(commit_id),
        manifest,
        inputs=list(manifest["registry_refs"].values()),
    )
    study.folio.metadata["cellpax_head_commit_id"] = commit_id

    with pytest.raises(ContractValidationError, match="kept_revision.foreign_key"):
        study.validate()


def test_registry_snapshots_are_cached_per_commit_head(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "registry-cache"
    study, _, _, revision = _durable_substrate(path)
    reopened = Study.open(path)
    head = reopened.head_commit()
    assert head is not None
    revision_ref = head.registries["kept_revision"]
    original_get = DataFolio.get
    registry_loads = 0

    def counted_get(folio, name, *args, **kwargs):
        nonlocal registry_loads
        if folio is reopened.folio and name == revision_ref:
            registry_loads += 1
        return original_get(folio, name, *args, **kwargs)

    monkeypatch.setattr(DataFolio, "get", counted_get)
    for _ in range(20):
        assert reopened.get_revision(revision.revision_id) == revision
    assert registry_loads == 1


def test_study_creation_metadata_read_only_and_contract_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="created_by"):
        Study.create(tmp_path / "missing-author", created_by="")
    with pytest.raises(ValueError, match="protected keys"):
        Study.create(
            tmp_path / "protected",
            created_by="user",
            metadata={"cellpax_schema_version": "other"},
        )

    path = tmp_path / "study"
    study = Study.create(path, created_by="user")
    assert study.registry("scope").schema == CONTRACTS["scope"].schema
    with pytest.raises(KeyError, match="Unknown CellPax registry"):
        study.registry("not-a-registry")

    read_only = Study.open(path, read_only=True)
    with pytest.raises(RuntimeError, match="read-only"):
        read_only.register_universe(
            universe_cells(),
            semantic_roles={"position": ["soma_x"]},
            nullable_columns=["label"],
        )

    plain = DataFolio(tmp_path / "plain")
    plain.add("thing", {"value": 1})
    with pytest.raises(ValueError, match="Not a CellPax study"):
        Study.open(tmp_path / "plain")
