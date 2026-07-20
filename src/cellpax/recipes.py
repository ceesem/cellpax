"""Resolved, language-neutral recipes and replay instructions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from cellpax.records import AssignmentSet, KeptRevision

if TYPE_CHECKING:
    from cellpax.study import Study


def resolved_release_recipe(
    study: Study,
    *,
    source_revision: KeptRevision,
    assignment_set: AssignmentSet,
) -> dict[str, Any]:
    """Resolve release-relevant computation and review state without model payloads."""
    revisions = list(study.revision_lineage(source_revision))

    feature_space_ids = sorted(
        {record.feature_space_id for record in revisions if record.feature_space_id}
    )
    representation_ids = sorted(
        {
            representation_id
            for record in revisions
            for representation_id in (
                record.clustering_representation_id,
                record.visualization_representation_id,
            )
            if representation_id
        }
    )
    candidate_set_ids = sorted(
        {record.candidate_set_id for record in revisions if record.candidate_set_id}
    )
    candidate_sets = [study.get_candidate_set(value) for value in candidate_set_ids]
    clustering_run_ids = sorted(
        {candidate.clustering_run_id for candidate in candidate_sets}
    )
    assignments = study.assignments(assignment_set)
    propagation_run_ids = sorted(
        assignments["propagation_run_id"].drop_nulls().unique().to_list()
    )
    decisions = study.decision_lineage(assignment_set.decision_head_id)

    return {
        "schema_version": "1.0",
        "source_revision_id": source_revision.revision_id,
        "assignment_set_id": assignment_set.assignment_set_id,
        "taxonomy": {
            "name": assignment_set.taxonomy_name,
            "version": assignment_set.taxonomy_version,
        },
        "revision_history": [
            {
                "revision_id": record.revision_id,
                "name": record.name,
                "parent_revision_id": record.parent_revision_id,
                "scope_id": record.scope_id,
                "feature_space_id": record.feature_space_id,
                "clustering_representation_id": record.clustering_representation_id,
                "visualization_representation_id": record.visualization_representation_id,
                "candidate_set_id": record.candidate_set_id,
                "state_hash": record.state_hash,
                "config_hash": record.config_hash,
            }
            for record in revisions
        ],
        "feature_spaces": [
            {
                "feature_space_id": record.feature_space_id,
                "scope_id": record.scope_id,
                "fit_scope_id": record.fit_scope_id,
                "parent_feature_space_id": record.parent_feature_space_id,
                "input_feature_block_ids": list(record.input_feature_block_ids or ()),
                "feature_selection_id": record.feature_selection_id,
                "transform": record.transform,
                "params": json.loads(record.params_json),
                "missing_policy": record.missing_policy,
                "seed": record.seed,
            }
            for record in (
                study.get_feature_space(value) for value in feature_space_ids
            )
        ],
        "representations": [
            {
                "representation_id": record.representation_id,
                "scope_id": record.scope_id,
                "fit_scope_id": record.fit_scope_id,
                "method": record.method,
                "input_feature_space_id": record.input_feature_space_id,
                "spatial_input": (
                    None
                    if record.spatial_input_json is None
                    else json.loads(record.spatial_input_json)
                ),
                "n_components": record.n_components,
                "params": json.loads(record.params_json),
                "seed": record.seed,
                "recompute_deterministic": record.recompute_deterministic,
                "software_versions": json.loads(record.software_versions),
            }
            for record in (
                study.get_representation(value) for value in representation_ids
            )
        ],
        "clustering_runs": [
            {
                "clustering_run_id": record.clustering_run_id,
                "scope_id": record.scope_id,
                "representation_id": record.representation_id,
                "method": record.method,
                "compute_params": json.loads(record.compute_params_json),
                "spatial_input": (
                    None
                    if record.spatial_input_json is None
                    else json.loads(record.spatial_input_json)
                ),
                "seed": record.seed,
                "recompute_deterministic": record.recompute_deterministic,
                "software_versions": json.loads(record.software_versions),
            }
            for record in (
                study.get_clustering_run(value) for value in clustering_run_ids
            )
        ],
        "candidate_sets": [
            {
                "candidate_set_id": record.candidate_set_id,
                "clustering_run_id": record.clustering_run_id,
                "scope_id": record.scope_id,
                "cut_method": record.cut_method,
                "cut_params": json.loads(record.cut_params_json),
            }
            for record in candidate_sets
        ],
        "decisions": [
            {
                "decision_id": record.decision_id,
                "parent_decision_id": record.parent_decision_id,
                "review_branch": record.review_branch,
                "revision_id": record.revision_id,
                "action": record.action,
                "target_kind": record.target_kind,
                "target_ids": (
                    None if record.target_ids is None else list(record.target_ids)
                ),
                "target_ref": record.target_ref,
                "params": json.loads(record.params_json),
                "taxon_id": record.taxon_id,
                "rationale": record.rationale,
                "evidence_refs": list(record.evidence_refs),
                "author": record.author,
            }
            for record in decisions
        ],
        "propagation_runs": [
            {
                "propagation_run_id": record.propagation_run_id,
                "fit_scope_id": record.fit_scope_id,
                "application_scope_id": record.application_scope_id,
                "input_feature_space_id": record.input_feature_space_id,
                "source_assignment_set_id": record.source_assignment_set_id,
                "method": record.method,
                "params": json.loads(record.params_json),
                "seed": record.seed,
                "recompute_deterministic": record.recompute_deterministic,
            }
            for record in (
                study.get_propagation_run(value) for value in propagation_run_ids
            )
        ],
    }


def render_release_replay_script(annotation_release_id: str) -> str:
    """Render a portable validation/inspection script for one retained release."""
    return f'''"""Validate and inspect a CellPax annotation release."""

from pathlib import Path
import sys

from cellpax import Study

study = Study.open(Path(sys.argv[1]), read_only=True)
release = study.get_annotation_release({annotation_release_id!r})
study.validate_annotation_release(release)
bundle = study.load_annotation_release(release)
print(bundle.quality_summary)
'''
