"""Immutable public handles for persisted and previewed CellPax entities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Universe:
    """Registered authoritative cell universe."""

    universe_id: str
    cells_ref: str
    schema_hash: str
    semantic_roles_json: str
    source_refs: tuple[str, ...]
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        """Return the registry-row representation."""
        return {
            "universe_id": self.universe_id,
            "cells_ref": self.cells_ref,
            "schema_hash": self.schema_hash,
            "semantic_roles_json": self.semantic_roles_json,
            "source_refs": list(self.source_refs),
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """Registered immutable cell-keyed feature input."""

    feature_block_id: str
    universe_id: str
    values_ref: str
    schema_hash: str
    source_refs: tuple[str, ...]
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        """Return the registry-row representation."""
        return {
            "feature_block_id": self.feature_block_id,
            "universe_id": self.universe_id,
            "values_ref": self.values_ref,
            "schema_hash": self.schema_hash,
            "source_refs": list(self.source_refs),
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """An exact ordered feature selection, previewed or registered."""

    feature_selection_id: str
    catalog_hash: str
    members_ref: str
    n_features: int
    derivation_text: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        """Return the registry-row representation."""
        return {
            "feature_selection_id": self.feature_selection_id,
            "catalog_hash": self.catalog_hash,
            "members_ref": self.members_ref,
            "n_features": self.n_features,
            "derivation_text": self.derivation_text,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class Scope:
    """An exact retained cell membership, previewed or registered."""

    scope_id: str
    universe_id: str
    parent_scope_id: str | None
    members_ref: str
    membership_hash: str
    n_cells: int
    derivation_text: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        """Return the registry-row representation."""
        return {
            "scope_id": self.scope_id,
            "universe_id": self.universe_id,
            "parent_scope_id": self.parent_scope_id,
            "members_ref": self.members_ref,
            "membership_hash": self.membership_hash,
            "n_cells": self.n_cells,
            "derivation_text": self.derivation_text,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class FeatureSpace:
    """One immutable fitted or raw feature layer."""

    feature_space_id: str
    scope_id: str
    fit_scope_id: str
    parent_feature_space_id: str | None
    input_feature_block_ids: tuple[str, ...] | None
    feature_selection_id: str
    transform: str
    params_json: str
    fitted_state_ref: str | None
    values_ref: str | None
    missing_policy: str
    seed: int | None
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "feature_space_id": self.feature_space_id,
            "scope_id": self.scope_id,
            "fit_scope_id": self.fit_scope_id,
            "parent_feature_space_id": self.parent_feature_space_id,
            "input_feature_block_ids": (
                None
                if self.input_feature_block_ids is None
                else list(self.input_feature_block_ids)
            ),
            "feature_selection_id": self.feature_selection_id,
            "transform": self.transform,
            "params_json": self.params_json,
            "fitted_state_ref": self.fitted_state_ref,
            "values_ref": self.values_ref,
            "missing_policy": self.missing_policy,
            "seed": self.seed,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class Representation:
    """Versioned coordinates for clustering or visualization."""

    representation_id: str
    scope_id: str
    fit_scope_id: str
    method: str
    input_feature_space_id: str
    spatial_input_json: str | None
    n_components: int | None
    params_json: str
    fitted_state_ref: str | None
    coords_ref: str
    seed: int | None
    recompute_deterministic: bool
    software_versions: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "representation_id": self.representation_id,
            "scope_id": self.scope_id,
            "fit_scope_id": self.fit_scope_id,
            "method": self.method,
            "input_feature_space_id": self.input_feature_space_id,
            "spatial_input_json": self.spatial_input_json,
            "n_components": self.n_components,
            "params_json": self.params_json,
            "fitted_state_ref": self.fitted_state_ref,
            "coords_ref": self.coords_ref,
            "seed": self.seed,
            "recompute_deterministic": self.recompute_deterministic,
            "software_versions": self.software_versions,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class ClusteringRun:
    """One immutable expensive clustering computation."""

    clustering_run_id: str
    scope_id: str
    representation_id: str
    method: str
    compute_params_json: str
    spatial_input_json: str | None
    generator_ref: str
    hierarchy_nodes_ref: str | None
    hierarchy_members_ref: str | None
    seed: int | None
    recompute_deterministic: bool
    software_versions: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "clustering_run_id": self.clustering_run_id,
            "scope_id": self.scope_id,
            "representation_id": self.representation_id,
            "method": self.method,
            "compute_params_json": self.compute_params_json,
            "spatial_input_json": self.spatial_input_json,
            "generator_ref": self.generator_ref,
            "hierarchy_nodes_ref": self.hierarchy_nodes_ref,
            "hierarchy_members_ref": self.hierarchy_members_ref,
            "seed": self.seed,
            "recompute_deterministic": self.recompute_deterministic,
            "software_versions": self.software_versions,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """One cheap flat partition derived from a clustering run."""

    candidate_set_id: str
    clustering_run_id: str
    scope_id: str
    cut_method: str
    cut_params_json: str
    definitions_ref: str
    membership_ref: str
    created_at: datetime
    created_by: str
    boundary_evidence_ref: str | None = None
    clustering_run_record: ClusteringRun | None = None

    def row(self) -> dict[str, object]:
        return {
            "candidate_set_id": self.candidate_set_id,
            "clustering_run_id": self.clustering_run_id,
            "scope_id": self.scope_id,
            "cut_method": self.cut_method,
            "cut_params_json": self.cut_params_json,
            "definitions_ref": self.definitions_ref,
            "membership_ref": self.membership_ref,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """One append-only review action with rationale and evidence."""

    decision_id: str
    review_branch: str
    revision_id: str
    parent_decision_id: str | None
    action: str
    target_kind: str
    target_ids: tuple[int, ...] | None
    target_ref: str | None
    params_json: str
    taxon_id: int | None
    rationale: str
    evidence_refs: tuple[str, ...]
    author: str
    created_at: datetime

    def row(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "review_branch": self.review_branch,
            "revision_id": self.revision_id,
            "parent_decision_id": self.parent_decision_id,
            "action": self.action,
            "target_kind": self.target_kind,
            "target_ids": None if self.target_ids is None else list(self.target_ids),
            "target_ref": self.target_ref,
            "params_json": self.params_json,
            "taxon_id": self.taxon_id,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
            "author": self.author,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class AssignmentSet:
    """One immutable collection of claims against a taxonomy version."""

    assignment_set_id: str
    taxonomy_name: str
    taxonomy_version: str
    review_branch: str
    decision_head_id: str
    assignments_ref: str
    state_hash: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "assignment_set_id": self.assignment_set_id,
            "taxonomy_name": self.taxonomy_name,
            "taxonomy_version": self.taxonomy_version,
            "review_branch": self.review_branch,
            "decision_head_id": self.decision_head_id,
            "assignments_ref": self.assignments_ref,
            "state_hash": self.state_hash,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class AnnotationRelease:
    """One published pairing of taxonomy vocabulary and cell assignments."""

    annotation_release_id: str
    name: str
    taxonomy_name: str
    taxonomy_version: str
    assignment_set_id: str
    source_revision_id: str
    manifest_ref: str
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "annotation_release_id": self.annotation_release_id,
            "name": self.name,
            "taxonomy_name": self.taxonomy_name,
            "taxonomy_version": self.taxonomy_version,
            "assignment_set_id": self.assignment_set_id,
            "source_revision_id": self.source_revision_id,
            "manifest_ref": self.manifest_ref,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class PropagationRun:
    """One content-addressed fitted label-propagation computation."""

    propagation_run_id: str
    fit_scope_id: str
    application_scope_id: str
    input_feature_space_id: str
    source_assignment_set_id: str
    method: str
    params_json: str
    model_ref: str
    quality_summary_ref: str
    seed: int | None
    recompute_deterministic: bool
    created_at: datetime
    created_by: str

    def row(self) -> dict[str, object]:
        return {
            "propagation_run_id": self.propagation_run_id,
            "fit_scope_id": self.fit_scope_id,
            "application_scope_id": self.application_scope_id,
            "input_feature_space_id": self.input_feature_space_id,
            "source_assignment_set_id": self.source_assignment_set_id,
            "method": self.method,
            "params_json": self.params_json,
            "model_ref": self.model_ref,
            "quality_summary_ref": self.quality_summary_ref,
            "seed": self.seed,
            "recompute_deterministic": self.recompute_deterministic,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class KeptRevision:
    """Named immutable pointer set persisted in the study history."""

    revision_id: str
    name: str
    parent_revision_id: str | None
    scope_id: str
    feature_space_id: str | None
    clustering_representation_id: str | None
    visualization_representation_id: str | None
    candidate_set_id: str | None
    state_hash: str
    config_hash: str
    software_versions: str
    created_by: str
    created_at: datetime
    notes: str | None

    def row(self) -> dict[str, object]:
        """Return the registry-row representation."""
        return {
            "revision_id": self.revision_id,
            "name": self.name,
            "parent_revision_id": self.parent_revision_id,
            "scope_id": self.scope_id,
            "feature_space_id": self.feature_space_id,
            "clustering_representation_id": self.clustering_representation_id,
            "visualization_representation_id": self.visualization_representation_id,
            "candidate_set_id": self.candidate_set_id,
            "state_hash": self.state_hash,
            "config_hash": self.config_hash,
            "software_versions": self.software_versions,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class StudyCommit:
    """Immutable CellPax registry-snapshot commit."""

    commit_id: str
    parent_commit_id: str | None
    registry_refs: tuple[tuple[str, str], ...]

    @property
    def registries(self) -> dict[str, str]:
        """Return registry refs as a new mapping."""
        return dict(self.registry_refs)
