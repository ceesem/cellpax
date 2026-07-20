from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
import pytest

from cellpax import (
    CandidateCutConfig,
    CandidatePartition,
    ClusteringConfig,
    FeatureSpaceConfig,
    GeneratorArtifacts,
    RepresentationConfig,
    Study,
)
from cellpax.contracts import CONTRACTS
from cellpax.plotting import embedding_scatter
from cellpax.views import (
    VIEW_CONTRACTS,
    cells,
    comparison,
    embedding,
    feature_profiles,
    history,
    release_summary,
    serialize_view,
    stability,
    taxonomy,
    validate_view,
)


class ThreeCandidateStub:
    method = "view_stub"

    def compute(self, coordinates, cell_ids, config):
        return GeneratorArtifacts(payload={"n_cells": len(cell_ids)})

    def cut(self, payload, config):
        evidence = pl.DataFrame(
            {
                "candidate_id_a": pl.Series([0], dtype=pl.Int32),
                "candidate_id_b": pl.Series([1], dtype=pl.Int32),
                "metric": ["synthetic_boundary"],
                "value": [0.04],
                "n_permutations": pl.Series([100], dtype=pl.Int32),
                "evidence_ref": pl.Series([None], dtype=pl.String),
            }
        )
        return CandidatePartition(
            candidate_ids=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            boundary_evidence=evidence,
        )


def taxonomy_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "taxonomy_name": "cells",
                "taxonomy_version": "1.0.0",
                "taxon_id": 1,
                "key": "root",
                "parent_id": None,
                "cluster_label": "Root",
                "short_name": "Root",
                "long_name": "Root",
                "description": None,
                "color": None,
                "sort_order": 0,
                "status": "active",
                "introduced_in": "1.0.0",
                "replaced_by": None,
            },
            {
                "taxonomy_name": "cells",
                "taxonomy_version": "1.0.0",
                "taxon_id": 10,
                "key": "type_a",
                "parent_id": 1,
                "cluster_label": "A",
                "short_name": "A",
                "long_name": "Type A",
                "description": None,
                "color": "#ff0000",
                "sort_order": 1,
                "status": "active",
                "introduced_in": "1.0.0",
                "replaced_by": None,
            },
            {
                "taxonomy_name": "cells",
                "taxonomy_version": "1.0.0",
                "taxon_id": 11,
                "key": "type_b",
                "parent_id": 1,
                "cluster_label": "B",
                "short_name": "B",
                "long_name": "Type B",
                "description": None,
                "color": "#0000ff",
                "sort_order": 2,
                "status": "active",
                "introduced_in": "1.0.0",
                "replaced_by": None,
            },
        ],
        schema=CONTRACTS["taxonomy"].schema,
    )


def build_view_study(path: Path):
    study = Study.create(path, created_by="viewer")
    cell_ids = list(range(1, 7))
    study.register_universe(
        pl.DataFrame({"cell_id": pl.Series(cell_ids, dtype=pl.Int64)}),
        semantic_roles={},
    )
    values = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "x": [0.0, 0.1, 10.0, 10.1, 0.2, 10.2],
            "y": [0.0, 0.1, 0.0, 0.1, 0.2, 0.2],
        }
    )
    catalog = pl.DataFrame(
        {
            "feature_id": ["x", "y"],
            "column_name": ["x", "y"],
            "modality": ["synthetic", "synthetic"],
            "family": ["position", "position"],
            "units": ["um", "um"],
            "description": pl.Series([None, None], dtype=pl.String),
            "raw_or_derived": ["raw", "raw"],
        }
    )
    block = study.register_feature_block(values, catalog)
    selection = study.preview_feature_selection(
        pl.DataFrame(
            {
                "feature_block_id": [block.feature_block_id] * 2,
                "feature_id": ["x", "y"],
            }
        ),
        derivation_text="view features",
    )
    scope = study.preview_scope(cell_ids, derivation_text="view scope")
    initial = study.keep("view inputs", scope=scope, feature_selection=selection)
    space = study.preview_feature_space(
        scope=scope,
        fit_scope=scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(transform="raw_join"),
    )
    representation = study.preview_representation(
        scope=scope,
        fit_scope=scope,
        feature_space=space,
        config=RepresentationConfig.resolve(method="scaled_passthrough"),
    )
    represented = study.keep(
        "view representation",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        visualization_representation=representation,
        parent_revision=initial,
    )
    generator = ThreeCandidateStub()
    run = study.preview_clustering_run(
        scope=scope,
        representation=representation,
        config=ClusteringConfig.external(method=generator.method, compute_params={}),
        generator=generator,
    )
    candidate_set = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.external(cut_method="native", cut_params={}),
        generator=generator,
    )
    revision = study.keep(
        "view candidates",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        visualization_representation=representation,
        candidate_set=candidate_set,
        parent_revision=represented,
    )
    study.register_taxonomy(taxonomy_frame())
    assign_a = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="candidates",
        target_ids=[0],
        taxon_id=10,
        rationale="Type A candidate.",
    )
    assign_b = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="candidates",
        target_ids=[1],
        taxon_id=11,
        rationale="Type B candidate.",
    )
    excluded = study.append_decision(
        revision=revision,
        review_branch="main",
        action="exclude",
        target_kind="candidates",
        target_ids=[2],
        rationale="Outside the taxonomy.",
    )
    assignment_set = study.create_assignment_set_from_decisions(
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="main",
        decision_head=excluded,
    )
    comparison_revision = study.keep(
        "view comparison",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        visualization_representation=representation,
        candidate_set=candidate_set,
        parent_revision=revision,
    )
    return (
        study,
        initial,
        represented,
        revision,
        comparison_revision,
        assignment_set,
        assign_a,
        assign_b,
    )


class FakeAxes:
    def __init__(self) -> None:
        self.scatter_call = None
        self.xlabel = None
        self.ylabel = None
        self.aspect = None

    def scatter(self, x, y, **kwargs):
        self.scatter_call = (x, y, kwargs)

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def set_aspect(self, value, **kwargs):
        self.aspect = (value, kwargs)


def test_slice_five_views_plot_and_serialized_component(tmp_path: Path) -> None:
    study, _, _, revision, other, assignment_set, _, _ = build_view_study(
        tmp_path / "views"
    )

    cell_view = cells(study, revision, assignment_set=assignment_set)
    assert cell_view.schema == VIEW_CONTRACTS["cells"].schema
    assert cell_view.height == 6
    assert (
        cell_view.filter(pl.col("assignment_status") == "outside_taxonomy").height == 2
    )

    embedding_view = embedding(study, revision, assignment_set=assignment_set)
    assert embedding_view.schema == VIEW_CONTRACTS["embedding"].schema
    assert embedding_view["short_name"].to_list() == ["A", "A", "B", "B", None, None]
    axes = FakeAxes()
    assert embedding_scatter(embedding_view, ax=axes) is axes
    assert axes.scatter_call is not None
    assert axes.scatter_call[2]["c"] == [
        "#ff0000",
        "#ff0000",
        "#0000ff",
        "#0000ff",
        "#808080",
        "#808080",
    ]

    serialized = serialize_view("embedding", embedding_view)
    assert serialized == serialize_view("embedding", embedding_view)
    assert serialized == serialize_view("embedding", embedding_view.reverse())
    component = json.loads(serialized)
    assert component["component_kind"] == "cellpax_view"
    assert component["view_name"] == "embedding"
    assert component["row_count"] == 6
    assert len(component["rows"]) == 6

    candidate_profiles = feature_profiles(study, revision)
    assert candidate_profiles.schema == VIEW_CONTRACTS["feature_profiles"].schema
    assert candidate_profiles.height == 6
    assert candidate_profiles["n_cells"].to_list() == [2] * 6
    assert feature_profiles(study, revision, group_by="all").height == 2
    assert (
        feature_profiles(
            study, revision, group_by="taxon", assignment_set=assignment_set
        ).height
        == 4
    )

    stability_view = stability(study, revision)
    assert stability_view.height == 3
    assert stability_view["membership_mean"].null_count() == 3
    assert stability_view["boundary_evidence_count"].to_list() == [1, 1, 0]
    assert stability_view["boundary_min_value"].to_list() == [0.04, 0.04, None]

    comparison_view = comparison(study, revision, other)
    assert comparison_view.height == 3
    assert comparison_view["n_cells"].to_list() == [2, 2, 2]

    taxonomy_view = taxonomy(study, assignment_set=assignment_set)
    assert taxonomy_view["assignment_count"].to_list() == [0, 2, 2]
    assert taxonomy_view["feature_based_count"].to_list() == [0, 2, 2]
    assert taxonomy(study, name="cells", version="1.0.0")["assignment_count"].sum() == 0

    history_view = history(study, other)
    assert history_view["name"].to_list() == [
        "view inputs",
        "view representation",
        "view candidates",
        "view comparison",
    ]
    assert history_view["depth"].to_list() == [0, 1, 2, 3]

    summary = release_summary(study, revision, assignment_set=assignment_set)
    assert summary.schema == VIEW_CONTRACTS["release_summary"].schema
    assert summary["n_scope_cells"][0] == 6
    assert summary["n_candidates"][0] == 3
    assert summary["n_taxon_assigned"][0] == 4
    assert summary["n_outside_taxonomy"][0] == 2

    study.validate()
    reopened = Study.open(tmp_path / "views", read_only=True)
    assert embedding(
        reopened, revision.revision_id, assignment_set=assignment_set.assignment_set_id
    ).equals(embedding_view)


def test_view_and_plotting_preconditions(tmp_path: Path, monkeypatch) -> None:
    study, initial, represented, revision, _, assignment_set, _, _ = build_view_study(
        tmp_path / "view-errors"
    )
    with pytest.raises(KeyError, match="Unknown view contract"):
        validate_view("invented", pl.DataFrame())
    with pytest.raises(ValueError, match="distinct non-negative"):
        embedding(study, revision, dimensions=(0, 0))
    with pytest.raises(ValueError, match="Exactly two"):
        embedding(study, revision, dimensions=(0,))
    with pytest.raises(ValueError, match="Exactly two"):
        embedding(study, revision, dimensions=(0, 1, 2))
    with pytest.raises(ValueError, match="Exactly two"):
        embedding(study, revision, dimensions=(0, True))
    with pytest.raises(ValueError, match="unavailable"):
        embedding(study, revision, dimensions=(0, 9))
    with pytest.raises(ValueError, match="no representation"):
        embedding(study, initial)
    representation_id = revision.visualization_representation_id
    assert representation_id is not None
    representation = study.get_representation(representation_id)
    assert embedding(study, revision, representation=representation_id).height == 6
    assert embedding(study, revision, representation=representation).height == 6
    with pytest.raises(ValueError, match="revision scope"):
        embedding(
            study,
            revision,
            representation=replace(representation, scope_id="another-scope"),
        )
    with pytest.raises(ValueError, match="group_by"):
        feature_profiles(study, revision, group_by="family")
    with pytest.raises(ValueError, match="no feature space"):
        feature_profiles(study, initial)
    with pytest.raises(ValueError, match="candidate set"):
        feature_profiles(study, represented)
    with pytest.raises(ValueError, match="assignment set"):
        feature_profiles(study, revision, group_by="taxon")
    with pytest.raises(ValueError, match="candidate set"):
        stability(study, represented)
    with pytest.raises(ValueError, match="requires name/version"):
        taxonomy(study)
    with pytest.raises(ValueError, match="conflicts"):
        taxonomy(study, assignment_set=assignment_set, name="other")
    with pytest.raises(ValueError, match="conflicts"):
        taxonomy(study, assignment_set=assignment_set, version="other")
    with pytest.raises(ValueError, match="candidate set"):
        comparison(study, initial, revision)

    frame = embedding(study, revision, assignment_set=assignment_set)
    axes = FakeAxes()
    with pytest.raises(ValueError, match="color_by"):
        embedding_scatter(frame, ax=axes, color_by="family")
    with pytest.raises(ValueError, match="point_size"):
        embedding_scatter(frame, ax=axes, point_size=0)
    with pytest.raises(ValueError, match="alpha"):
        embedding_scatter(frame, ax=axes, alpha=2)
    assert embedding_scatter(frame, ax=axes, color_by="candidate") is axes
    assert embedding_scatter(frame, ax=axes, color_by="none") is axes

    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda: (object(), axes)
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    assert embedding_scatter(frame) is axes

    unassigned_embedding = embedding(study, revision)
    assert unassigned_embedding["taxon_id"].null_count() == 6

    bare_cells = cells(study, initial)
    assert bare_cells["candidate_id"].null_count() == bare_cells.height
    bare_summary = release_summary(study, initial)
    assert bare_summary["n_candidates"][0] == 0
    assert bare_summary["n_assignment_rows"][0] == 0

    feature_space_id = revision.feature_space_id
    assert feature_space_id is not None
    space = study.get_feature_space(feature_space_id)
    assert space.values_ref is not None
    values = study.folio.get(space.values_ref, frame="polars")
    study.folio.add(
        space.values_ref,
        values.rename({"feature_00000": "drifted_feature"}),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="metadata columns do not match"):
        feature_profiles(study, revision)
