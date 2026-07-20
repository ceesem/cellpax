import random
from datetime import datetime, timezone

import polars as pl

from cellpax.contracts import CONTRACTS, validate_registries
from cellpax.contracts.invariants import assert_architecture_contracts

NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def frame(name: str, rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=CONTRACTS[name].schema)


def test_random_candidate_stub_crosses_the_algorithm_agnostic_seam() -> None:
    candidate_a, candidate_b = random.Random(42).sample(range(100, 1000), 2)

    frames = {
        "clustering_run": frame(
            "clustering_run",
            [
                {
                    "clustering_run_id": "run",
                    "scope_id": "scope",
                    "representation_id": "representation",
                    "method": "synthetic_stub",
                    "compute_params_json": "{}",
                    "spatial_input_json": None,
                    "generator_ref": "cellpax/objects/generator/one",
                    "hierarchy_nodes_ref": "cellpax/objects/hierarchy-nodes/one",
                    "hierarchy_members_ref": "cellpax/objects/hierarchy-members/one",
                    "seed": 42,
                    "recompute_deterministic": True,
                    "software_versions": "{}",
                    "created_at": NOW,
                    "created_by": "stub",
                }
            ],
        ),
        "candidate_hierarchy_node": frame(
            "candidate_hierarchy_node",
            [
                {
                    "clustering_run_id": "run",
                    "node_id": "root",
                    "parent_node_id": None,
                    "level": 0,
                    "merge_score": None,
                    "n_cells": 3,
                    "metadata_json": "{}",
                },
                {
                    "clustering_run_id": "run",
                    "node_id": "left",
                    "parent_node_id": "root",
                    "level": 1,
                    "merge_score": 0.2,
                    "n_cells": 1,
                    "metadata_json": "{}",
                },
                {
                    "clustering_run_id": "run",
                    "node_id": "right",
                    "parent_node_id": "root",
                    "level": 1,
                    "merge_score": 0.2,
                    "n_cells": 1,
                    "metadata_json": "{}",
                },
            ],
        ),
        "candidate_hierarchy_membership": frame(
            "candidate_hierarchy_membership",
            [
                {"clustering_run_id": "run", "cell_id": 1, "leaf_node_id": "left"},
                {"clustering_run_id": "run", "cell_id": 2, "leaf_node_id": "right"},
            ],
        ),
        "candidate_set": frame(
            "candidate_set",
            [
                {
                    "candidate_set_id": "set",
                    "clustering_run_id": "run",
                    "scope_id": "scope",
                    "cut_method": "native",
                    "cut_params_json": "{}",
                    "definitions_ref": "cellpax/objects/definitions/one",
                    "membership_ref": "cellpax/objects/membership/one",
                    "created_at": NOW,
                    "created_by": "stub",
                }
            ],
        ),
        "candidate_definition": frame(
            "candidate_definition",
            [
                {
                    "candidate_set_id": "set",
                    "candidate_id": candidate_a,
                    "hierarchy_node_id": "left",
                    "n_cells": 1,
                },
                {
                    "candidate_set_id": "set",
                    "candidate_id": candidate_b,
                    "hierarchy_node_id": "right",
                    "n_cells": 1,
                },
            ],
        ),
        "candidate_membership": frame(
            "candidate_membership",
            [
                {
                    "candidate_set_id": "set",
                    "cell_id": 1,
                    "candidate_id": candidate_a,
                    "membership_strength": 0.9,
                },
                {
                    "candidate_set_id": "set",
                    "cell_id": 2,
                    "candidate_id": candidate_b,
                    "membership_strength": 0.8,
                },
                {
                    "candidate_set_id": "set",
                    "cell_id": 3,
                    "candidate_id": None,
                    "membership_strength": None,
                },
            ],
        ),
        "candidate_boundary_evidence": frame(
            "candidate_boundary_evidence",
            [
                {
                    "candidate_set_id": "set",
                    "candidate_id_a": candidate_a,
                    "candidate_id_b": candidate_b,
                    "metric": "fake_pvalue",
                    "value": 0.04,
                    "n_permutations": 100,
                    "evidence_ref": None,
                }
            ],
        ),
        "kept_revision": frame(
            "kept_revision",
            [
                {
                    "revision_id": "revision",
                    "name": "stub revision",
                    "parent_revision_id": None,
                    "scope_id": "scope",
                    "feature_space_id": None,
                    "clustering_representation_id": None,
                    "visualization_representation_id": None,
                    "candidate_set_id": "set",
                    "state_hash": "state",
                    "config_hash": "config",
                    "software_versions": "{}",
                    "created_by": "stub",
                    "created_at": NOW,
                    "notes": None,
                }
            ],
        ),
        "decision": frame(
            "decision",
            [
                {
                    "decision_id": "decision",
                    "review_branch": "main",
                    "revision_id": "revision",
                    "parent_decision_id": None,
                    "action": "exclude",
                    "target_kind": "candidates",
                    "target_ids": [candidate_b],
                    "target_ref": None,
                    "params_json": "{}",
                    "taxon_id": None,
                    "rationale": "Synthetic exclusion",
                    "evidence_refs": [],
                    "author": "stub",
                    "created_at": NOW,
                }
            ],
        ),
        "taxonomy": frame(
            "taxonomy",
            [
                {
                    "taxonomy_name": "stub",
                    "taxonomy_version": "1.0.0",
                    "taxon_id": 7,
                    "key": "stub_cell",
                    "parent_id": None,
                    "cluster_label": "Stub",
                    "short_name": "Stub",
                    "long_name": "Synthetic stub cell",
                    "description": None,
                    "color": None,
                    "sort_order": 0,
                    "status": "active",
                    "introduced_in": "1.0.0",
                    "replaced_by": None,
                }
            ],
        ),
        "assignment_set": frame(
            "assignment_set",
            [
                {
                    "assignment_set_id": "assignments",
                    "taxonomy_name": "stub",
                    "taxonomy_version": "1.0.0",
                    "review_branch": "main",
                    "decision_head_id": "decision",
                    "assignments_ref": "cellpax/objects/assignments/one",
                    "state_hash": "assignment-state",
                    "created_at": NOW,
                    "created_by": "stub",
                }
            ],
        ),
        "assignment": frame(
            "assignment",
            [
                {
                    "assignment_set_id": "assignments",
                    "cell_id": 1,
                    "taxon_id": 7,
                    "assignment_status": "leaf",
                    "assignment_source": "manual",
                    "source_revision_id": "revision",
                    "decision_id": "decision",
                    "propagation_run_id": None,
                    "coverage_feature_space_id": None,
                    "coverage": None,
                    "confidence": 0.95,
                    "alternatives_json": None,
                    "created_at": NOW,
                },
                {
                    "assignment_set_id": "assignments",
                    "cell_id": 2,
                    "taxon_id": None,
                    "assignment_status": "outside_taxonomy",
                    "assignment_source": "manual",
                    "source_revision_id": "revision",
                    "decision_id": "decision",
                    "propagation_run_id": None,
                    "coverage_feature_space_id": None,
                    "coverage": None,
                    "confidence": None,
                    "alternatives_json": None,
                    "created_at": NOW,
                },
            ],
        ),
        "annotation_release": frame(
            "annotation_release",
            [
                {
                    "annotation_release_id": "release",
                    "name": "stub release",
                    "taxonomy_name": "stub",
                    "taxonomy_version": "1.0.0",
                    "assignment_set_id": "assignments",
                    "source_revision_id": "revision",
                    "manifest_ref": "cellpax/objects/release/one",
                    "created_at": NOW,
                    "created_by": "stub",
                }
            ],
        ),
    }

    assert validate_registries(frames) == ()
    assert_architecture_contracts()
