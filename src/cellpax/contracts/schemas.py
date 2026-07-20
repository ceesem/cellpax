"""Version 1 Polars schemas and constraints for CellPax registries."""

from __future__ import annotations

import polars as pl

from cellpax.contracts.core import (
    EnumConstraint,
    ForeignKey,
    RangeConstraint,
    RowConstraint,
    TableContract,
)

SCHEMA_VERSION = "1.0"
AUDIT_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")


def _present(column: pl.Expr) -> pl.Expr:
    """Return whether a scalar or list-valued column is populated."""
    return column.is_not_null()


def _feature_space_source_invalid(frame: pl.DataFrame) -> pl.Series:
    parent = _present(pl.col("parent_feature_space_id"))
    blocks = _present(pl.col("input_feature_block_ids"))
    return frame.select((parent == blocks).alias("invalid")).to_series()


def _decision_target_invalid(frame: pl.DataFrame) -> pl.Series:
    ids = _present(pl.col("target_ids"))
    ref = _present(pl.col("target_ref"))
    return frame.select((ids == ref).alias("invalid")).to_series()


def _propagation_source_invalid(frame: pl.DataFrame) -> pl.Series:
    propagated = pl.col("assignment_source") == "propagated"
    has_run = pl.col("propagation_run_id").is_not_null()
    return frame.select((propagated != has_run).alias("invalid")).to_series()


def _taxon_status_invalid(frame: pl.DataFrame) -> pl.Series:
    unassigned = pl.col("assignment_status").is_in(["unassigned", "outside_taxonomy"])
    has_taxon = pl.col("taxon_id").is_not_null()
    return frame.select((unassigned == has_taxon).alias("invalid")).to_series()


UNIVERSE = TableContract(
    name="universe",
    schema=pl.Schema(
        {
            "universe_id": pl.String,
            "cells_ref": pl.String,
            "schema_hash": pl.String,
            "semantic_roles_json": pl.String,
            "source_refs": pl.List(pl.String),
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("universe_id",),
)

FEATURE_BLOCK = TableContract(
    name="feature_block",
    schema=pl.Schema(
        {
            "feature_block_id": pl.String,
            "universe_id": pl.String,
            "values_ref": pl.String,
            "schema_hash": pl.String,
            "source_refs": pl.List(pl.String),
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("feature_block_id",),
    foreign_keys=(ForeignKey(("universe_id",), "universe", ("universe_id",)),),
)

FEATURE_CATALOG = TableContract(
    name="feature_catalog",
    schema=pl.Schema(
        {
            "feature_id": pl.String,
            "feature_block_id": pl.String,
            "column_name": pl.String,
            "modality": pl.String,
            "family": pl.String,
            "units": pl.String,
            "description": pl.String,
            "raw_or_derived": pl.String,
        }
    ),
    nullable=frozenset({"units", "description"}),
    primary_key=("feature_block_id", "feature_id"),
    unique_keys=(("feature_block_id", "column_name"),),
    enums=(EnumConstraint("raw_or_derived", frozenset({"raw", "derived"})),),
    foreign_keys=(
        ForeignKey(("feature_block_id",), "feature_block", ("feature_block_id",)),
    ),
)

FEATURE_SELECTION = TableContract(
    name="feature_selection",
    schema=pl.Schema(
        {
            "feature_selection_id": pl.String,
            "catalog_hash": pl.String,
            "members_ref": pl.String,
            "n_features": pl.Int32,
            "derivation_text": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("feature_selection_id",),
    ranges=(RangeConstraint("n_features", minimum=0),),
)

SCOPE = TableContract(
    name="scope",
    schema=pl.Schema(
        {
            "scope_id": pl.String,
            "universe_id": pl.String,
            "parent_scope_id": pl.String,
            "members_ref": pl.String,
            "membership_hash": pl.String,
            "n_cells": pl.Int64,
            "derivation_text": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    nullable=frozenset({"parent_scope_id"}),
    primary_key=("scope_id",),
    ranges=(RangeConstraint("n_cells", minimum=0),),
    foreign_keys=(
        ForeignKey(("universe_id",), "universe", ("universe_id",)),
        ForeignKey(("parent_scope_id",), "scope", ("scope_id",)),
    ),
)

FEATURE_SPACE = TableContract(
    name="feature_space",
    schema=pl.Schema(
        {
            "feature_space_id": pl.String,
            "scope_id": pl.String,
            "fit_scope_id": pl.String,
            "parent_feature_space_id": pl.String,
            "input_feature_block_ids": pl.List(pl.String),
            "feature_selection_id": pl.String,
            "transform": pl.String,
            "params_json": pl.String,
            "fitted_state_ref": pl.String,
            "values_ref": pl.String,
            "missing_policy": pl.String,
            "seed": pl.Int64,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    nullable=frozenset(
        {
            "parent_feature_space_id",
            "input_feature_block_ids",
            "fitted_state_ref",
            "values_ref",
            "seed",
        }
    ),
    primary_key=("feature_space_id",),
    row_constraints=(
        RowConstraint(
            "one_feature_space_source",
            "exactly one of parent_feature_space_id and input_feature_block_ids must be populated",
            ("parent_feature_space_id", "input_feature_block_ids"),
            _feature_space_source_invalid,
        ),
    ),
    foreign_keys=(
        ForeignKey(("scope_id",), "scope", ("scope_id",)),
        ForeignKey(("fit_scope_id",), "scope", ("scope_id",)),
        ForeignKey(
            ("parent_feature_space_id",), "feature_space", ("feature_space_id",)
        ),
        ForeignKey(
            ("feature_selection_id",),
            "feature_selection",
            ("feature_selection_id",),
        ),
    ),
)

REPRESENTATION = TableContract(
    name="representation",
    schema=pl.Schema(
        {
            "representation_id": pl.String,
            "scope_id": pl.String,
            "fit_scope_id": pl.String,
            "method": pl.String,
            "input_feature_space_id": pl.String,
            "spatial_input_json": pl.String,
            "n_components": pl.Int32,
            "params_json": pl.String,
            "fitted_state_ref": pl.String,
            "coords_ref": pl.String,
            "seed": pl.Int64,
            "recompute_deterministic": pl.Boolean,
            "software_versions": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    nullable=frozenset(
        {"spatial_input_json", "n_components", "fitted_state_ref", "seed"}
    ),
    primary_key=("representation_id",),
    ranges=(RangeConstraint("n_components", minimum=1),),
    foreign_keys=(
        ForeignKey(("scope_id",), "scope", ("scope_id",)),
        ForeignKey(("fit_scope_id",), "scope", ("scope_id",)),
        ForeignKey(("input_feature_space_id",), "feature_space", ("feature_space_id",)),
    ),
)

CLUSTERING_RUN = TableContract(
    name="clustering_run",
    schema=pl.Schema(
        {
            "clustering_run_id": pl.String,
            "scope_id": pl.String,
            "representation_id": pl.String,
            "method": pl.String,
            "compute_params_json": pl.String,
            "spatial_input_json": pl.String,
            "generator_ref": pl.String,
            "hierarchy_nodes_ref": pl.String,
            "hierarchy_members_ref": pl.String,
            "seed": pl.Int64,
            "recompute_deterministic": pl.Boolean,
            "software_versions": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    nullable=frozenset(
        {"spatial_input_json", "hierarchy_nodes_ref", "hierarchy_members_ref", "seed"}
    ),
    primary_key=("clustering_run_id",),
    foreign_keys=(
        ForeignKey(("scope_id",), "scope", ("scope_id",)),
        ForeignKey(("representation_id",), "representation", ("representation_id",)),
    ),
)

CANDIDATE_HIERARCHY_NODE = TableContract(
    name="candidate_hierarchy_node",
    schema=pl.Schema(
        {
            "clustering_run_id": pl.String,
            "node_id": pl.String,
            "parent_node_id": pl.String,
            "level": pl.Int32,
            "merge_score": pl.Float64,
            "n_cells": pl.Int64,
            "metadata_json": pl.String,
        }
    ),
    nullable=frozenset({"parent_node_id", "merge_score"}),
    primary_key=("clustering_run_id", "node_id"),
    ranges=(RangeConstraint("level", minimum=0), RangeConstraint("n_cells", minimum=0)),
    foreign_keys=(
        ForeignKey(("clustering_run_id",), "clustering_run", ("clustering_run_id",)),
        ForeignKey(
            ("clustering_run_id", "parent_node_id"),
            "candidate_hierarchy_node",
            ("clustering_run_id", "node_id"),
        ),
    ),
)

CANDIDATE_HIERARCHY_MEMBERSHIP = TableContract(
    name="candidate_hierarchy_membership",
    schema=pl.Schema(
        {
            "clustering_run_id": pl.String,
            "cell_id": pl.Int64,
            "leaf_node_id": pl.String,
        }
    ),
    primary_key=("clustering_run_id", "cell_id"),
    foreign_keys=(
        ForeignKey(("clustering_run_id",), "clustering_run", ("clustering_run_id",)),
        ForeignKey(
            ("clustering_run_id", "leaf_node_id"),
            "candidate_hierarchy_node",
            ("clustering_run_id", "node_id"),
        ),
    ),
)

CANDIDATE_SET = TableContract(
    name="candidate_set",
    schema=pl.Schema(
        {
            "candidate_set_id": pl.String,
            "clustering_run_id": pl.String,
            "scope_id": pl.String,
            "cut_method": pl.String,
            "cut_params_json": pl.String,
            "definitions_ref": pl.String,
            "membership_ref": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("candidate_set_id",),
    enums=(
        EnumConstraint(
            "cut_method", frozenset({"distance", "resolution", "tree_prune", "native"})
        ),
    ),
    foreign_keys=(
        ForeignKey(("clustering_run_id",), "clustering_run", ("clustering_run_id",)),
        ForeignKey(("scope_id",), "scope", ("scope_id",)),
    ),
)

CANDIDATE_DEFINITION = TableContract(
    name="candidate_definition",
    schema=pl.Schema(
        {
            "candidate_set_id": pl.String,
            "candidate_id": pl.Int32,
            "hierarchy_node_id": pl.String,
            "n_cells": pl.Int64,
        }
    ),
    nullable=frozenset({"hierarchy_node_id"}),
    primary_key=("candidate_set_id", "candidate_id"),
    ranges=(
        RangeConstraint("candidate_id", minimum=0),
        RangeConstraint("n_cells", minimum=0),
    ),
    foreign_keys=(
        ForeignKey(("candidate_set_id",), "candidate_set", ("candidate_set_id",)),
    ),
)

CANDIDATE_MEMBERSHIP = TableContract(
    name="candidate_membership",
    schema=pl.Schema(
        {
            "candidate_set_id": pl.String,
            "cell_id": pl.Int64,
            "candidate_id": pl.Int32,
            "membership_strength": pl.Float32,
        }
    ),
    nullable=frozenset({"candidate_id", "membership_strength"}),
    primary_key=("candidate_set_id", "cell_id"),
    ranges=(
        RangeConstraint("candidate_id", minimum=0),
        RangeConstraint("membership_strength", minimum=0.0, maximum=1.0),
    ),
    foreign_keys=(
        ForeignKey(("candidate_set_id",), "candidate_set", ("candidate_set_id",)),
        ForeignKey(
            ("candidate_set_id", "candidate_id"),
            "candidate_definition",
            ("candidate_set_id", "candidate_id"),
        ),
    ),
)

CANDIDATE_BOUNDARY_EVIDENCE = TableContract(
    name="candidate_boundary_evidence",
    schema=pl.Schema(
        {
            "candidate_set_id": pl.String,
            "candidate_id_a": pl.Int32,
            "candidate_id_b": pl.Int32,
            "metric": pl.String,
            "value": pl.Float64,
            "n_permutations": pl.Int32,
            "evidence_ref": pl.String,
        }
    ),
    nullable=frozenset({"n_permutations", "evidence_ref"}),
    primary_key=("candidate_set_id", "candidate_id_a", "candidate_id_b", "metric"),
    ranges=(
        RangeConstraint("candidate_id_a", minimum=0),
        RangeConstraint("candidate_id_b", minimum=0),
        RangeConstraint("n_permutations", minimum=0),
    ),
    foreign_keys=(
        ForeignKey(("candidate_set_id",), "candidate_set", ("candidate_set_id",)),
    ),
)

KEPT_REVISION = TableContract(
    name="kept_revision",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "name": pl.String,
            "parent_revision_id": pl.String,
            "scope_id": pl.String,
            "feature_space_id": pl.String,
            "clustering_representation_id": pl.String,
            "visualization_representation_id": pl.String,
            "candidate_set_id": pl.String,
            "state_hash": pl.String,
            "config_hash": pl.String,
            "software_versions": pl.String,
            "created_by": pl.String,
            "created_at": AUDIT_DATETIME,
            "notes": pl.String,
        }
    ),
    nullable=frozenset(
        {
            "parent_revision_id",
            "feature_space_id",
            "clustering_representation_id",
            "visualization_representation_id",
            "candidate_set_id",
            "notes",
        }
    ),
    primary_key=("revision_id",),
    unique_keys=(("name",),),
    foreign_keys=(
        ForeignKey(("parent_revision_id",), "kept_revision", ("revision_id",)),
        ForeignKey(("scope_id",), "scope", ("scope_id",)),
        ForeignKey(("feature_space_id",), "feature_space", ("feature_space_id",)),
        ForeignKey(
            ("clustering_representation_id",),
            "representation",
            ("representation_id",),
        ),
        ForeignKey(
            ("visualization_representation_id",),
            "representation",
            ("representation_id",),
        ),
        ForeignKey(("candidate_set_id",), "candidate_set", ("candidate_set_id",)),
    ),
)

DECISION = TableContract(
    name="decision",
    schema=pl.Schema(
        {
            "decision_id": pl.String,
            "review_branch": pl.String,
            "revision_id": pl.String,
            "parent_decision_id": pl.String,
            "action": pl.String,
            "target_kind": pl.String,
            "target_ids": pl.List(pl.Int64),
            "target_ref": pl.String,
            "params_json": pl.String,
            "taxon_id": pl.Int64,
            "rationale": pl.String,
            "evidence_refs": pl.List(pl.String),
            "author": pl.String,
            "created_at": AUDIT_DATETIME,
        }
    ),
    nullable=frozenset({"parent_decision_id", "target_ids", "target_ref", "taxon_id"}),
    primary_key=("decision_id",),
    enums=(
        EnumConstraint(
            "action",
            frozenset(
                {
                    "merge",
                    "split",
                    "exclude",
                    "mark_ambiguous",
                    "assign",
                    "assign_parent_only",
                    "rename",
                    "attach_local_revision",
                }
            ),
        ),
        EnumConstraint("target_kind", frozenset({"candidates", "cells", "taxon"})),
    ),
    row_constraints=(
        RowConstraint(
            "one_decision_target",
            "exactly one of target_ids and target_ref must be populated",
            ("target_ids", "target_ref"),
            _decision_target_invalid,
        ),
    ),
    foreign_keys=(
        ForeignKey(("revision_id",), "kept_revision", ("revision_id",)),
        ForeignKey(("parent_decision_id",), "decision", ("decision_id",)),
    ),
)

TAXONOMY = TableContract(
    name="taxonomy",
    schema=pl.Schema(
        {
            "taxonomy_name": pl.String,
            "taxonomy_version": pl.String,
            "taxon_id": pl.Int64,
            "key": pl.String,
            "parent_id": pl.Int64,
            "cluster_label": pl.String,
            "short_name": pl.String,
            "long_name": pl.String,
            "description": pl.String,
            "color": pl.String,
            "sort_order": pl.Int32,
            "status": pl.String,
            "introduced_in": pl.String,
            "replaced_by": pl.Int64,
        }
    ),
    nullable=frozenset({"parent_id", "description", "color", "replaced_by"}),
    primary_key=("taxonomy_name", "taxonomy_version", "taxon_id"),
    unique_keys=(("taxonomy_name", "taxonomy_version", "key"),),
    enums=(
        EnumConstraint("status", frozenset({"provisional", "active", "deprecated"})),
    ),
)

PROPAGATION_RUN = TableContract(
    name="propagation_run",
    schema=pl.Schema(
        {
            "propagation_run_id": pl.String,
            "fit_scope_id": pl.String,
            "application_scope_id": pl.String,
            "input_feature_space_id": pl.String,
            "source_assignment_set_id": pl.String,
            "method": pl.String,
            "params_json": pl.String,
            "model_ref": pl.String,
            "quality_summary_ref": pl.String,
            "seed": pl.Int64,
            "recompute_deterministic": pl.Boolean,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    nullable=frozenset({"seed"}),
    primary_key=("propagation_run_id",),
    foreign_keys=(
        ForeignKey(("fit_scope_id",), "scope", ("scope_id",)),
        ForeignKey(("application_scope_id",), "scope", ("scope_id",)),
        ForeignKey(("input_feature_space_id",), "feature_space", ("feature_space_id",)),
        ForeignKey(
            ("source_assignment_set_id",),
            "assignment_set",
            ("assignment_set_id",),
        ),
    ),
)

ASSIGNMENT_SET = TableContract(
    name="assignment_set",
    schema=pl.Schema(
        {
            "assignment_set_id": pl.String,
            "taxonomy_name": pl.String,
            "taxonomy_version": pl.String,
            "review_branch": pl.String,
            "decision_head_id": pl.String,
            "assignments_ref": pl.String,
            "state_hash": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("assignment_set_id",),
    foreign_keys=(ForeignKey(("decision_head_id",), "decision", ("decision_id",)),),
)

ASSIGNMENT = TableContract(
    name="assignment",
    schema=pl.Schema(
        {
            "assignment_set_id": pl.String,
            "cell_id": pl.Int64,
            "taxon_id": pl.Int64,
            "assignment_status": pl.String,
            "assignment_source": pl.String,
            "source_revision_id": pl.String,
            "decision_id": pl.String,
            "propagation_run_id": pl.String,
            "coverage_feature_space_id": pl.String,
            "coverage": pl.Float32,
            "confidence": pl.Float32,
            "alternatives_json": pl.String,
            "created_at": AUDIT_DATETIME,
        }
    ),
    nullable=frozenset(
        {
            "taxon_id",
            "decision_id",
            "propagation_run_id",
            "coverage_feature_space_id",
            "coverage",
            "confidence",
            "alternatives_json",
        }
    ),
    primary_key=("assignment_set_id", "cell_id"),
    enums=(
        EnumConstraint(
            "assignment_status",
            frozenset(
                {"leaf", "parent_only", "ambiguous", "unassigned", "outside_taxonomy"}
            ),
        ),
        EnumConstraint(
            "assignment_source", frozenset({"feature_based", "propagated", "manual"})
        ),
    ),
    ranges=(
        RangeConstraint("coverage", minimum=0.0, maximum=1.0),
        RangeConstraint("confidence", minimum=0.0, maximum=1.0),
    ),
    row_constraints=(
        RowConstraint(
            "propagation_provenance",
            "propagated assignments require propagation_run_id and other sources forbid it",
            ("assignment_source", "propagation_run_id"),
            _propagation_source_invalid,
        ),
        RowConstraint(
            "taxon_assignment_status",
            "unassigned/outside_taxonomy rows must not have taxon_id; other statuses require it",
            ("assignment_status", "taxon_id"),
            _taxon_status_invalid,
        ),
    ),
    foreign_keys=(
        ForeignKey(("assignment_set_id",), "assignment_set", ("assignment_set_id",)),
        ForeignKey(("source_revision_id",), "kept_revision", ("revision_id",)),
        ForeignKey(("decision_id",), "decision", ("decision_id",)),
        ForeignKey(("propagation_run_id",), "propagation_run", ("propagation_run_id",)),
        ForeignKey(
            ("coverage_feature_space_id",),
            "feature_space",
            ("feature_space_id",),
        ),
    ),
)

ANNOTATION_RELEASE = TableContract(
    name="annotation_release",
    schema=pl.Schema(
        {
            "annotation_release_id": pl.String,
            "name": pl.String,
            "taxonomy_name": pl.String,
            "taxonomy_version": pl.String,
            "assignment_set_id": pl.String,
            "source_revision_id": pl.String,
            "manifest_ref": pl.String,
            "created_at": AUDIT_DATETIME,
            "created_by": pl.String,
        }
    ),
    primary_key=("annotation_release_id",),
    unique_keys=(("name",),),
    foreign_keys=(
        ForeignKey(("assignment_set_id",), "assignment_set", ("assignment_set_id",)),
        ForeignKey(("source_revision_id",), "kept_revision", ("revision_id",)),
    ),
)

CONTRACTS: dict[str, TableContract] = {
    contract.name: contract
    for contract in (
        UNIVERSE,
        FEATURE_BLOCK,
        FEATURE_CATALOG,
        FEATURE_SELECTION,
        SCOPE,
        FEATURE_SPACE,
        REPRESENTATION,
        CLUSTERING_RUN,
        CANDIDATE_HIERARCHY_NODE,
        CANDIDATE_HIERARCHY_MEMBERSHIP,
        CANDIDATE_SET,
        CANDIDATE_DEFINITION,
        CANDIDATE_MEMBERSHIP,
        CANDIDATE_BOUNDARY_EVIDENCE,
        KEPT_REVISION,
        DECISION,
        TAXONOMY,
        PROPAGATION_RUN,
        ASSIGNMENT_SET,
        ASSIGNMENT,
        ANNOTATION_RELEASE,
    )
}


def contract(name: str) -> TableContract:
    """Return a named version 1 contract."""
    try:
        return CONTRACTS[name]
    except KeyError as error:
        available = ", ".join(CONTRACTS)
        raise KeyError(
            f"Unknown CellPax contract {name!r}; choose from {available}"
        ) from error
