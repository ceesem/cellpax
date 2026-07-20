from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from test_views import build_view_study

import cellpax.adapters.trajan as trajan_adapter
from cellpax import Study
from cellpax.adapters.trajan import (
    add_to_connectivity_table,
    add_to_synapse_table,
    annotation_frame,
    decorate_cells,
)
from cellpax.release import (
    RELEASE_ARTIFACT_NAMES,
    RELEASE_QUALITY,
    ReleaseBundle,
    validate_release_manifest,
)


def test_self_contained_release_and_trajan_consumption(tmp_path: Path) -> None:
    study, _, _, revision, _, assignment_set, _, _ = build_view_study(
        tmp_path / "release"
    )
    release = study.create_annotation_release(
        "synthetic-v1",
        source_revision=revision,
        assignment_set=assignment_set,
    )
    assert study.get_annotation_release(release.annotation_release_id) == release
    assert study.get_annotation_release("synthetic-v1") == release
    assert study.registry("annotation_release").height == 1

    bundle = study.load_annotation_release(release)
    assert isinstance(bundle, ReleaseBundle)
    assert set(bundle.manifest["artifacts"]) == set(RELEASE_ARTIFACT_NAMES)
    assert bundle.taxonomy.height == 3
    assert bundle.assignments.height == 6
    assert bundle.decisions.height == 3
    assert bundle.quality_summary.schema == RELEASE_QUALITY.schema
    assert bundle.quality_summary["n_taxon_assigned"][0] == 4
    assert bundle.quality_summary["n_outside_taxonomy"][0] == 2
    assert bundle.quality_summary["n_decisions"][0] == 3
    assert bundle.recipe["assignment_set_id"] == assignment_set.assignment_set_id
    assert bundle.recipe["source_revision_id"] == revision.revision_id
    assert bundle.recipe["decisions"][-1]["action"] == "exclude"
    assert "generator_ref" not in str(bundle.recipe)
    assert release.annotation_release_id in bundle.replay_script
    compile(bundle.replay_script, "release_replay.py", "exec")

    Taxon = bundle.taxonomy_enum()
    assert Taxon.TYPE_A.long_name == "Type A"
    namespace: dict[str, object] = {}
    exec(bundle.enum_binding, namespace)
    generated = namespace["cells_1_0_0_Taxon"]
    assert generated.TYPE_B.short_name == "B"

    consumer = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 8), dtype=pl.Int64),
            "consumer_value": range(7),
        }
    )
    decorated = decorate_cells(consumer, bundle)
    assert decorated.height == consumer.height
    assert decorated.columns[:2] == consumer.columns
    assert decorated.filter(pl.col("taxon_id") == 10).height == 2
    assert (
        decorated.filter(pl.col("assignment_status") == "outside_taxonomy").height == 2
    )
    assert decorated.filter(pl.col("cell_id") == 7)["taxon_id"][0] is None

    aliased = decorate_cells(
        consumer.rename({"cell_id": "root_id"}),
        bundle,
        cell_id_column="root_id",
    )
    assert aliased["root_id"].to_list() == consumer["cell_id"].to_list()

    annotation = annotation_frame(bundle)
    assert annotation.height == bundle.assignments.height
    assert annotation["cell_id"].n_unique() == annotation.height

    class ConnectivityTable:
        call: tuple[tuple[object, ...], dict[str, object]] | None = None

        def add_annotation(self, *args: object, **kwargs: object) -> object:
            self.call = (args, kwargs)
            return self

    connectivity = ConnectivityTable()
    assert (
        add_to_connectivity_table(
            connectivity,
            bundle,
            name="cell-types",
            is_universe=True,
            side="source",
        )
        is connectivity
    )
    assert connectivity.call is not None
    connectivity_args, connectivity_kwargs = connectivity.call
    assert connectivity_args[0] == "cell-types"
    assert isinstance(connectivity_args[1], pl.DataFrame)
    assert connectivity_kwargs == {
        "cell_id_col": "cell_id",
        "is_universe": True,
        "side": "source",
    }

    class SynapseTable:
        call: tuple[tuple[object, ...], dict[str, object]] | None = None

        def add_cell_annotation(self, *args: object, **kwargs: object) -> object:
            self.call = (args, kwargs)
            return self

    synapses = SynapseTable()
    assert add_to_synapse_table(synapses, bundle, name="cell-types") is synapses
    assert synapses.call is not None
    synapse_args, synapse_kwargs = synapses.call
    assert synapse_args[0] == "cell-types"
    assert isinstance(synapse_args[1], pl.DataFrame)
    assert synapse_kwargs == {"cell_id_col": "cell_id", "is_universe": False}

    study.validate_annotation_release(release)
    study.validate()
    reopened = Study.open(tmp_path / "release", read_only=True)
    reopened.validate()
    reopened_bundle = reopened.load_annotation_release(release.annotation_release_id)
    assert reopened_bundle.assignments.equals(bundle.assignments)
    assert reopened_bundle.recipe == bundle.recipe

    code = """
import json, sys
from datafolio import DataFolio
from cellpax import Study

def reject_models(*args, **kwargs):
    raise AssertionError("release validation must not load model payloads")

DataFolio.get_model = reject_models
study = Study.open(sys.argv[1], read_only=True)
release = study.get_annotation_release(sys.argv[2])
study.validate_annotation_release(release)
bundle = study.load_annotation_release(release)
print(json.dumps({
    "release": bundle.release.name,
    "assignments": bundle.assignments.height,
    "recipe_version": bundle.recipe["schema_version"],
}))
"""
    output = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "release"),
            release.annotation_release_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.stdout) == {
        "release": "synthetic-v1",
        "assignments": 6,
        "recipe_version": "1.0",
    }


def test_release_and_consumer_preconditions(tmp_path: Path) -> None:
    study, initial, represented, revision, _, assignment_set, _, _ = build_view_study(
        tmp_path / "release-errors"
    )
    with pytest.raises(KeyError, match="Unknown annotation release"):
        study.get_annotation_release("missing")
    with pytest.raises(ValueError, match="requires a name"):
        study.create_annotation_release(
            "", source_revision=revision, assignment_set=assignment_set
        )
    with pytest.raises(ValueError, match="must be kept"):
        study.create_annotation_release(
            "foreign-revision",
            source_revision=replace(revision, revision_id="foreign"),
            assignment_set=assignment_set,
        )
    with pytest.raises(ValueError, match="must belong"):
        study.create_annotation_release(
            "foreign-assignments",
            source_revision=revision,
            assignment_set=replace(assignment_set, assignment_set_id="foreign"),
        )
    with pytest.raises(ValueError, match="stale or foreign"):
        study.create_annotation_release(
            "stale-revision",
            source_revision=replace(revision, name="changed"),
            assignment_set=assignment_set,
        )
    with pytest.raises(ValueError, match="stale or foreign"):
        study.create_annotation_release(
            "stale-assignments",
            source_revision=revision,
            assignment_set=replace(assignment_set, state_hash="changed"),
        )

    small_scope = study.preview_scope(
        [1, 2],
        derivation_text="release scope too small",
        parent=study.get_scope(revision.scope_id),
    )
    small_revision = study.keep(
        "small release source", scope=small_scope, parent_revision=revision
    )
    with pytest.raises(ValueError, match="release source revision scope"):
        study.create_annotation_release(
            "scope-mismatch",
            source_revision=small_revision,
            assignment_set=assignment_set,
        )

    unrelated_revision = study.keep(
        "unrelated release source",
        scope=study.get_scope(revision.scope_id),
        parent_revision=initial,
    )
    with pytest.raises(ValueError, match="revision ancestry"):
        study.create_annotation_release(
            "unrelated-provenance",
            source_revision=unrelated_revision,
            assignment_set=assignment_set,
        )

    release = study.create_annotation_release(
        "valid-release",
        source_revision=revision,
        assignment_set=assignment_set,
    )
    with pytest.raises(ValueError, match="already exists"):
        study.create_annotation_release(
            "valid-release",
            source_revision=revision,
            assignment_set=assignment_set,
        )
    with pytest.raises(ValueError, match="does not belong"):
        study.validate_annotation_release(
            replace(release, annotation_release_id="foreign")
        )
    with pytest.raises(ValueError, match="stale or foreign"):
        study.load_annotation_release(replace(release, name="changed"))

    bundle = study.load_annotation_release(release)
    valid_cells = pl.DataFrame({"cell_id": pl.Series(range(1, 7), dtype=pl.Int64)})
    with pytest.raises(TypeError, match="Polars"):
        decorate_cells([], bundle)
    with pytest.raises(ValueError, match="missing id column"):
        decorate_cells(pl.DataFrame({"other": [1]}), bundle)
    with pytest.raises(TypeError, match="Int64"):
        decorate_cells(
            pl.DataFrame({"cell_id": pl.Series([1], dtype=pl.Int32)}), bundle
        )
    with pytest.raises(ValueError, match="non-null and unique"):
        decorate_cells(
            pl.DataFrame({"cell_id": pl.Series([1, 1], dtype=pl.Int64)}), bundle
        )
    with pytest.raises(ValueError, match="already contains"):
        decorate_cells(valid_cells.with_columns(pl.lit(1).alias("taxon_id")), bundle)
    with pytest.raises(ValueError, match="missing 1 released cells"):
        decorate_cells(valid_cells.filter(pl.col("cell_id") != 6), bundle)
    partial = decorate_cells(
        valid_cells.filter(pl.col("cell_id") != 6),
        bundle,
        require_all_release_cells=False,
    )
    assert partial.height == 5
    with pytest.raises(TypeError, match="add_annotation"):
        add_to_connectivity_table(object(), bundle)
    with pytest.raises(TypeError, match="add_cell_annotation"):
        add_to_synapse_table(object(), bundle)

    manifest = study.folio.get(release.manifest_ref)
    enum_ref = manifest["artifacts"]["enum_binding"]["ref"]
    enum_source = study.folio.get(enum_ref)
    study.folio.add(enum_ref, f"{enum_source}\n# corrupted", overwrite=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        study.validate_annotation_release(release)
    study.folio.add(enum_ref, enum_source, overwrite=True)
    study.validate_annotation_release(release)

    study.folio.add(
        release.manifest_ref,
        {**manifest, "component_kind": "other"},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="kind mismatch"):
        study.validate_annotation_release(release)

    assert represented.candidate_set_id is None


def test_release_manifest_shape_validation() -> None:
    artifacts = {
        name: {"ref": f"release/{name}", "checksum": f"checksum-{name}"}
        for name in RELEASE_ARTIFACT_NAMES
    }
    manifest = {
        "component_kind": "annotation_release",
        "schema_version": "1.0",
        "annotation_release_id": "release",
        "name": "name",
        "taxonomy_name": "cells",
        "taxonomy_version": "1.0.0",
        "assignment_set_id": "assignments",
        "assignment_set_state_hash": "assignment-state",
        "source_revision_id": "revision",
        "source_revision_state_hash": "revision-state",
        "decision_head_id": "decision",
        "artifacts": artifacts,
    }
    assert validate_release_manifest(manifest) == manifest
    with pytest.raises(ValueError, match="Malformed"):
        validate_release_manifest([])
    with pytest.raises(ValueError, match="malformed artifacts"):
        validate_release_manifest({**manifest, "artifacts": {}})
    bad_artifacts = {**artifacts, "recipe": {"ref": "", "checksum": "ok"}}
    with pytest.raises(ValueError, match="descriptor 'recipe'"):
        validate_release_manifest({**manifest, "artifacts": bad_artifacts})


def test_trajan_adapter_has_a_release_only_cellpax_boundary() -> None:
    source = Path(trajan_adapter.__file__).read_text()
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert "cellpax.release" in imports
    assert (
        not {
            "cellpax.study",
            "cellpax.clustering",
            "cellpax.review",
            "cellpax.propagation",
        }
        & imports
    )
