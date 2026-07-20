from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cellpax import (
    CandidateCutConfig,
    CandidatePartition,
    ClusteringConfig,
    FeatureSpaceConfig,
    GeneratorArtifacts,
    PropagationConfig,
    RepresentationConfig,
    Study,
)
from cellpax.contracts import CONTRACTS
from cellpax.review import DecisionActionConfig, validate_decision_semantics
from cellpax.taxonomy import (
    TaxonDefinition,
    taxonomy_enum,
    taxonomy_table,
    validate_taxonomy,
)


class ThreeCandidateStub:
    method = "three_candidate_stub"

    def compute(self, coordinates, cell_ids, config):
        return GeneratorArtifacts(payload={"n_cells": len(cell_ids)})

    def cut(self, payload, config):
        return CandidatePartition(
            candidate_ids=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        )


def build_review_study(path: Path):
    study = Study.create(path, created_by="reviewer")
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
            "units": pl.Series([None, None], dtype=pl.String),
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
        derivation_text="review features",
    )
    scope = study.preview_scope(cell_ids, derivation_text="review scope")
    inputs = study.keep("review inputs", scope=scope, feature_selection=selection)
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
        "review representation",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        parent_revision=inputs,
    )
    stub = ThreeCandidateStub()
    run = study.preview_clustering_run(
        scope=scope,
        representation=representation,
        config=ClusteringConfig.external(method=stub.method, compute_params={}),
        generator=stub,
    )
    candidate_set = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.external(cut_method="native", cut_params={}),
        generator=stub,
    )
    revision = study.keep(
        "review candidates",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        candidate_set=candidate_set,
        parent_revision=represented,
    )
    return study, scope, space, revision


def taxonomy_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "taxonomy_name": "cells",
                "taxonomy_version": "1.0.0",
                "taxon_id": 1,
                "key": "inhibitory",
                "parent_id": None,
                "cluster_label": "Inh",
                "short_name": "Inh",
                "long_name": "Inhibitory",
                "description": None,
                "color": "#777777",
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


@pytest.mark.parametrize(
    ("action", "params", "error"),
    [
        ("invent", None, "Unsupported"),
        ("assign", {"extra": True}, "accepts no parameters"),
        ("rename", {}, "non-empty string name"),
        ("rename", {"name": 7}, "non-empty string name"),
        ("attach_local_revision", {}, "revision_id"),
        ("split", {}, "non-empty parts list"),
        ("split", {"parts": [{}]}, "target_ref and taxon_id"),
        (
            "split",
            {"parts": [{"target_ref": "", "taxon_id": 10}]},
            "target_ref",
        ),
        (
            "split",
            {"parts": [{"target_ref": "members", "taxon_id": True}]},
            "taxon_id",
        ),
    ],
)
def test_decision_action_config_rejects_malformed_payloads(
    action: str, params: dict[str, object] | None, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        DecisionActionConfig.resolve(action=action, params=params)


def test_decision_action_config_exposes_canonical_params() -> None:
    config = DecisionActionConfig.resolve(action="rename", params={"name": "Type A"})
    assert config.params == {"name": "Type A"}


@pytest.mark.parametrize(
    ("action", "target_kind", "target_count", "taxon_id", "error"),
    [
        ("exclude", "unknown", 1, None, "target kind"),
        ("assign", "cells", 1, None, "requires taxon_id"),
        ("merge", "candidates", 2, None, "requires taxon_id"),
        ("merge", "candidates", 1, 10, "at least two"),
        ("merge", "cells", 2, 10, "at least two"),
        ("split", "candidates", 2, None, "exactly one"),
        ("rename", "cells", 1, None, "taxon target"),
        ("attach_local_revision", "taxon", 1, None, "cell targets"),
        ("exclude", "cells", 1, -1, "taxon_id"),
        ("exclude", "cells", 1, True, "taxon_id"),
    ],
)
def test_decision_semantics_reject_invalid_cross_field_combinations(
    action: str,
    target_kind: str,
    target_count: int | None,
    taxon_id: int | None,
    error: str,
) -> None:
    params = (
        {"name": "renamed"}
        if action == "rename"
        else {"revision_id": "local"}
        if action == "attach_local_revision"
        else {"parts": [{"target_ref": "part", "taxon_id": 10}]}
        if action == "split"
        else None
    )
    config = DecisionActionConfig.resolve(action=action, params=params)
    with pytest.raises((TypeError, ValueError), match=error):
        validate_decision_semantics(
            config=config,
            target_kind=target_kind,
            target_count=target_count,
            taxon_id=taxon_id,
        )


def test_taxonomy_validation_and_enum_edge_cases() -> None:
    valid = taxonomy_frame()
    generated = taxonomy_enum(
        valid.with_columns(
            pl.when(pl.col("taxon_id") == 10)
            .then(pl.lit("10 / special"))
            .otherwise(pl.col("key"))
            .alias("key")
        )
    )
    assert generated.TAXON_10_SPECIAL.color == "#ff0000"
    assert generated.INHIBITORY.color == "#777777"
    assert generated.TAXON_10_SPECIAL.short_name == "A"
    assert generated.TAXON_10_SPECIAL.long_name == "Type A"

    mutations = [
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 10)
                .then(pl.lit("other"))
                .otherwise(pl.col("taxonomy_name"))
                .alias("taxonomy_name")
            ),
            "one non-empty name and version",
        ),
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 10)
                .then(pl.lit(-10))
                .otherwise(pl.col("taxon_id"))
                .alias("taxon_id")
            ),
            "non-negative",
        ),
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 10)
                .then(pl.lit("  "))
                .otherwise(pl.col("key"))
                .alias("key")
            ),
            "non-empty",
        ),
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 10)
                .then(pl.lit(99, dtype=pl.Int64))
                .otherwise(pl.col("parent_id"))
                .alias("parent_id")
            ),
            "parent_id",
        ),
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 10)
                .then(pl.lit(99, dtype=pl.Int64))
                .otherwise(pl.col("replaced_by"))
                .alias("replaced_by")
            ),
            "replaced_by",
        ),
        (
            valid.with_columns(
                pl.when(pl.col("taxon_id") == 1)
                .then(pl.lit(10, dtype=pl.Int64))
                .otherwise(pl.col("parent_id"))
                .alias("parent_id")
            ),
            "cycle",
        ),
    ]
    for frame, error in mutations:
        with pytest.raises(ValueError, match=error):
            validate_taxonomy(frame)

    colliding = valid.with_columns(
        pl.when(pl.col("taxon_id") == 11)
        .then(pl.lit("type-a"))
        .otherwise(pl.col("key"))
        .alias("key")
    )
    with pytest.raises(ValueError, match="collide"):
        taxonomy_enum(colliding)


def test_concise_taxonomy_table_fills_persisted_defaults() -> None:
    frame = taxonomy_table(
        "cells",
        "1.0.0",
        [
            TaxonDefinition(1, "inhibitory", label="Inh", color="#777777"),
            TaxonDefinition(
                10,
                "type_a",
                label="A",
                parent_id=1,
                description="A child type.",
            ),
        ],
    )

    assert frame.schema == CONTRACTS["taxonomy"].schema
    assert frame["sort_order"].to_list() == [0, 1]
    assert frame["cluster_label"].to_list() == ["Inh", "A"]
    assert frame["short_name"].to_list() == ["Inh", "A"]
    assert frame["long_name"].to_list() == ["Inhibitory", "Type A"]
    assert frame["introduced_in"].to_list() == ["1.0.0", "1.0.0"]
    assert frame["status"].to_list() == ["active", "active"]
    assert frame["parent_id"].to_list() == [None, 1]

    with pytest.raises(ValueError, match="at least one"):
        taxonomy_table("cells", "1.0.0", [])
    with pytest.raises(TypeError, match="TaxonDefinition"):
        taxonomy_table("cells", "1.0.0", [{"taxon_id": 1}])


@pytest.mark.parametrize(
    ("params", "seed", "error"),
    [
        ({"n_neighbors": 0}, None, "positive"),
        ({"weights": "ranked"}, None, "weights"),
        ({}, True, "seed"),
    ],
)
def test_propagation_config_rejects_invalid_values(
    params: dict[str, object], seed: int | None, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        PropagationConfig.resolve(params=params, seed=seed)


def test_study_review_api_rejects_invalid_ledger_and_assignment_states(
    tmp_path: Path,
) -> None:
    study, _, _, revision = build_review_study(tmp_path / "review-errors")
    study.register_taxonomy(taxonomy_frame())

    with pytest.raises(KeyError, match="Unknown decision"):
        study.get_decision("missing")
    with pytest.raises(KeyError, match="Unknown taxonomy"):
        study.get_taxonomy("cells", "missing")
    with pytest.raises(KeyError, match="Unknown assignment set"):
        study.get_assignment_set("missing")
    with pytest.raises(KeyError, match="Unknown propagation run"):
        study.get_propagation_run("missing")

    common = {
        "revision": revision,
        "review_branch": "invalid",
        "action": "assign",
        "target_kind": "candidates",
        "rationale": "Validation exercise.",
        "target_ids": [0],
        "taxon_id": 10,
    }
    with pytest.raises(ValueError, match="review_branch"):
        study.append_decision(**{**common, "review_branch": ""})
    with pytest.raises(ValueError, match="rationale"):
        study.append_decision(**{**common, "rationale": ""})
    with pytest.raises(ValueError, match="Exactly one"):
        study.append_decision(**{**common, "target_ids": None})
    with pytest.raises(ValueError, match="Exactly one"):
        study.append_decision(**common, target_cells=[1])
    with pytest.raises(TypeError, match="target_ids"):
        study.append_decision(**{**common, "target_ids": []})
    with pytest.raises(ValueError, match="unknown candidates"):
        study.append_decision(**{**common, "target_ids": [999]})

    no_candidates = study.get_revision(
        study.registry("kept_revision").filter(pl.col("candidate_set_id").is_null())[
            "revision_id"
        ][0]
    )
    with pytest.raises(ValueError, match="candidate set"):
        study.append_decision(**{**common, "revision": no_candidates})

    first = study.append_decision(**{**common, "review_branch": "main"})
    second = study.append_decision(
        **{**common, "review_branch": "main", "target_ids": [1]}
    )
    with pytest.raises(ValueError, match="current branch head"):
        study.append_decision(
            **{**common, "review_branch": "main"}, parent_decision=first
        )
    with pytest.raises(ValueError, match="same review branch"):
        study.append_decision(
            **{**common, "review_branch": "other"}, parent_decision=second
        )

    changed = taxonomy_frame().with_columns(pl.lit("changed").alias("description"))
    with pytest.raises(ValueError, match="immutable"):
        study.register_taxonomy(changed)

    rows = study.reduce_decisions(
        second, taxonomy_name="cells", taxonomy_version="1.0.0"
    )
    base = study.create_assignment_set(
        rows,
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="main",
        decision_head=second,
    )
    third = study.append_decision(
        **{**common, "review_branch": "main", "target_ids": [0]}
    )
    current_rows = study.reduce_decisions(
        third, taxonomy_name="cells", taxonomy_version="1.0.0"
    )
    create_kwargs = {
        "taxonomy_name": "cells",
        "taxonomy_version": "1.0.0",
        "review_branch": "main",
        "decision_head": third,
    }
    with pytest.raises(ValueError, match="branch and decision head"):
        study.create_assignment_set(
            current_rows, **{**create_kwargs, "review_branch": "other"}
        )
    with pytest.raises(ValueError, match="current branch head"):
        study.create_assignment_set(rows, **{**create_kwargs, "decision_head": second})
    with pytest.raises(ValueError, match="columns mismatch"):
        study.create_assignment_set(current_rows.drop("confidence"), **create_kwargs)
    with pytest.raises(ValueError, match="outside the taxonomy"):
        study.create_assignment_set(
            current_rows.with_columns(pl.lit(999, dtype=pl.Int64).alias("taxon_id")),
            **create_kwargs,
        )
    with pytest.raises(ValueError, match="parent taxon"):
        study.create_assignment_set(
            current_rows.with_columns(pl.lit("parent_only").alias("assignment_status")),
            **create_kwargs,
        )
    with pytest.raises(ValueError, match="must be paired"):
        study.create_assignment_set(
            current_rows.with_columns(pl.lit(None).cast(pl.Float32).alias("coverage")),
            **create_kwargs,
        )
    with pytest.raises(ValueError, match="Unknown propagation run"):
        study.create_assignment_set(
            current_rows.with_columns(
                pl.lit("missing-run").alias("propagation_run_id"),
                pl.lit("propagated").alias("assignment_source"),
            ),
            **create_kwargs,
        )

    taxonomy_v2 = taxonomy_frame().with_columns(
        pl.lit("2.0.0").alias("taxonomy_version")
    )
    study.register_taxonomy(taxonomy_v2)
    with pytest.raises(ValueError, match="another taxonomy version"):
        study.reduce_decisions(
            third,
            taxonomy_name="cells",
            taxonomy_version="2.0.0",
            base_assignment_set=base,
        )

    ambiguous = study.append_decision(
        revision=revision,
        review_branch="ambiguous",
        action="mark_ambiguous",
        target_kind="cells",
        target_ids=[6],
        rationale="No prior claim exists.",
    )
    with pytest.raises(ValueError, match="existing or explicit taxon"):
        study.reduce_decisions(
            ambiguous, taxonomy_name="cells", taxonomy_version="1.0.0"
        )

    outside_ref = "review/split/outside"
    study.folio.add(
        outside_ref, pl.DataFrame({"cell_id": pl.Series([3], dtype=pl.Int64)})
    )
    split = study.append_decision(
        revision=revision,
        review_branch="bad-split",
        action="split",
        target_kind="candidates",
        target_ids=[0],
        params={"parts": [{"target_ref": outside_ref, "taxon_id": 10}]},
        rationale="The part is deliberately outside candidate zero.",
    )
    with pytest.raises(ValueError, match="outside its candidate"):
        study.reduce_decisions(split, taxonomy_name="cells", taxonomy_version="1.0.0")


def test_assignment_set_integrity_lookups_are_deduplicated(
    tmp_path: Path, monkeypatch
) -> None:
    study, _, _, revision = build_review_study(tmp_path / "deduplicated-lookups")
    study.register_taxonomy(taxonomy_frame())
    decision = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="candidates",
        target_ids=[0, 1, 2],
        taxon_id=10,
        rationale="Exercise repeated assignment provenance.",
    )
    rows = study.reduce_decisions(
        decision, taxonomy_name="cells", taxonomy_version="1.0.0"
    )
    assert rows.height == 6

    calls = {"revision": 0, "decision": 0, "feature_space": 0}
    original_revision = study.get_revision
    original_decision = study.get_decision
    original_feature_space = study.get_feature_space

    def counted_revision(identifier):
        calls["revision"] += 1
        return original_revision(identifier)

    def counted_decision(identifier):
        calls["decision"] += 1
        return original_decision(identifier)

    def counted_feature_space(identifier):
        calls["feature_space"] += 1
        return original_feature_space(identifier)

    monkeypatch.setattr(study, "get_revision", counted_revision)
    monkeypatch.setattr(study, "get_decision", counted_decision)
    monkeypatch.setattr(study, "get_feature_space", counted_feature_space)
    study.create_assignment_set(
        rows,
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="main",
        decision_head=decision,
    )

    assert calls == {"revision": 1, "decision": 1, "feature_space": 1}


def test_decision_ledger_taxonomy_and_assignment_only_corrections(
    tmp_path: Path,
) -> None:
    study, _, _, revision = build_review_study(tmp_path / "review")
    taxonomy = study.register_taxonomy(taxonomy_frame())
    assert study.register_taxonomy(taxonomy_frame()).equals(taxonomy)
    Taxon = study.taxonomy_enum("cells", "1.0.0")
    assert Taxon.TYPE_A == 10
    assert Taxon.TYPE_A.long_name == "Type A"
    assert Taxon.TYPE_A.parent_id == 1

    assign_a = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="candidates",
        target_ids=[0],
        taxon_id=10,
        rationale="Candidate zero is coherent type A.",
    )
    assign_b = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="candidates",
        target_ids=[1],
        taxon_id=11,
        rationale="Candidate one is coherent type B.",
    )
    exclude = study.append_decision(
        revision=revision,
        review_branch="main",
        action="exclude",
        target_kind="candidates",
        target_ids=[2],
        rationale="Diffuse artifact candidate; deliberately outside taxonomy.",
    )
    assert exclude.parent_decision_id == assign_b.decision_id
    assert study.decision_head("main") == exclude
    assert [decision.decision_id for decision in study.decision_lineage(exclude)] == [
        assign_a.decision_id,
        assign_b.decision_id,
        exclude.decision_id,
    ]

    base = study.create_assignment_set_from_decisions(
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="main",
        decision_head=exclude,
    )
    base_rows = study.assignments(base)
    assert base_rows.filter(pl.col("taxon_id") == 10).height == 2
    assert base_rows.filter(pl.col("taxon_id") == 11).height == 2
    assert (
        base_rows.filter(pl.col("assignment_status") == "outside_taxonomy").height == 2
    )

    ambiguous = study.append_decision(
        revision=revision,
        review_branch="main",
        action="mark_ambiguous",
        target_kind="cells",
        target_ids=[3],
        rationale="Borderline cell; retain parent evidence but mark ambiguity.",
    )
    corrected = study.append_decision(
        revision=revision,
        review_branch="main",
        action="assign",
        target_kind="cells",
        target_cells=[4],
        taxon_id=10,
        rationale="Manual review corrects this cell to type A.",
    )
    assert corrected.target_ref is not None
    revised = study.create_assignment_set_from_decisions(
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="main",
        decision_head=corrected,
        base_assignment_set=base,
    )
    revised_rows = study.assignments(revised)
    assert revised.assignment_set_id != base.assignment_set_id
    assert revised.taxonomy_version == base.taxonomy_version == "1.0.0"
    assert revised_rows.filter(pl.col("cell_id") == 3)["assignment_status"][0] == (
        "ambiguous"
    )
    assert revised_rows.filter(pl.col("cell_id") == 4)["taxon_id"][0] == 10
    assert study.registry("taxonomy").height == 3
    study.validate()

    reopened = Study.open(tmp_path / "review", read_only=True)
    reopened.validate()
    assert reopened.assignments(revised.assignment_set_id).equals(revised_rows)


def test_merge_split_and_propagated_noncore_assignments(tmp_path: Path) -> None:
    study, scope, space, revision = build_review_study(tmp_path / "propagation")
    study.register_taxonomy(taxonomy_frame())
    with pytest.raises(ValueError, match="requires taxon_id"):
        study.append_decision(
            revision=revision,
            review_branch="invalid-merge",
            action="merge",
            target_kind="candidates",
            target_ids=[0, 1],
            rationale="A merge without a destination taxon must not be recorded.",
        )
    assert study.decision_head("invalid-merge") is None
    merge = study.append_decision(
        revision=revision,
        review_branch="merge",
        action="merge",
        target_kind="candidates",
        target_ids=[0, 1],
        taxon_id=10,
        rationale="Synthetic merge exercise.",
    )
    assert merge.action == "merge"
    left_ref = "review/split/left"
    right_ref = "review/split/right"
    study.folio.add(left_ref, pl.DataFrame({"cell_id": pl.Series([1], dtype=pl.Int64)}))
    study.folio.add(
        right_ref, pl.DataFrame({"cell_id": pl.Series([2], dtype=pl.Int64)})
    )
    split = study.append_decision(
        revision=revision,
        review_branch="split",
        action="split",
        target_kind="candidates",
        target_ids=[0],
        params={
            "parts": [
                {"target_ref": left_ref, "taxon_id": 10},
                {"target_ref": right_ref, "taxon_id": 11},
            ]
        },
        rationale="Synthetic split exercise.",
    )
    split_set = study.create_assignment_set_from_decisions(
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="split",
        decision_head=split,
    )
    assert study.assignments(split_set)["taxon_id"].to_list() == [10, 11]

    assign_a = study.append_decision(
        revision=revision,
        review_branch="propagate",
        action="assign",
        target_kind="candidates",
        target_ids=[0],
        taxon_id=10,
        rationale="Core A.",
    )
    assign_b = study.append_decision(
        revision=revision,
        review_branch="propagate",
        action="assign",
        target_kind="candidates",
        target_ids=[1],
        taxon_id=11,
        rationale="Core B.",
    )
    source = study.create_assignment_set_from_decisions(
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="propagate",
        decision_head=assign_b,
    )
    fit_scope = study.preview_scope(
        [1, 2, 3, 4], derivation_text="feature-based core", parent=scope
    )
    application_scope = study.preview_scope(
        [5, 6], derivation_text="non-core propagation", parent=scope
    )
    fit_revision = study.keep(
        "propagation fit scope", scope=fit_scope, parent_revision=revision
    )
    study.keep(
        "propagation application scope",
        scope=application_scope,
        parent_revision=fit_revision,
    )
    propagation_config = PropagationConfig.resolve(params={"n_neighbors": 2})
    with pytest.raises(ValueError, match="non-null"):
        study.preview_propagation_run(
            fit_scope=fit_scope,
            application_scope=application_scope,
            feature_space=space,
            source_assignment_set=source,
            config=propagation_config,
            coverage=pl.DataFrame(
                {
                    "cell_id": pl.Series([5, 6], dtype=pl.Int64),
                    "coverage": pl.Series([0.4, None], dtype=pl.Float32),
                }
            ),
        )
    run = study.preview_propagation_run(
        fit_scope=fit_scope,
        application_scope=application_scope,
        feature_space=space,
        source_assignment_set=source,
        config=propagation_config,
        coverage=pl.DataFrame(
            {
                "cell_id": pl.Series([5, 6], dtype=pl.Int64),
                "coverage": pl.Series([0.4, 0.4], dtype=pl.Float32),
            }
        ),
    )
    assert not hasattr(run, "predictions_ref")
    propagated = study.propagated_assignment_rows(run, source_revision=revision)
    assert propagated["assignment_source"].to_list() == ["propagated", "propagated"]
    assert propagated["coverage"].to_list() == pytest.approx([0.4, 0.4])
    combined = pl.concat(
        [
            study.assignments(source).drop("assignment_set_id", "created_at"),
            propagated,
        ]
    )
    propagated_set = study.create_assignment_set(
        combined,
        taxonomy_name="cells",
        taxonomy_version="1.0.0",
        review_branch="propagate",
        decision_head=assign_b,
        propagation_runs=[run],
    )
    rows = study.assignments(propagated_set)
    assert rows.filter(pl.col("assignment_source") == "feature_based").height == 4
    assert rows.filter(pl.col("assignment_source") == "propagated").height == 2
    assert study.propagation_quality(run).height == 2
    assert study.registry("propagation_run").height == 1
    study.validate(trusted_models=True)
