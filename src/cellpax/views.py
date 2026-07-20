"""Revision-aware, read-only Polars view contracts.

Views are ordinary functions returning frames with fixed schemas.  They are not
persisted state and deliberately do not form a class hierarchy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl

from cellpax.contracts import CONTRACTS, validate_table
from cellpax.contracts.core import EnumConstraint, RangeConstraint, TableContract
from cellpax.identity import canonical_hash, canonical_json_bytes
from cellpax.records import AssignmentSet, KeptRevision, Representation

if TYPE_CHECKING:
    from cellpax.study import Study

_AUDIT_DATETIME = pl.Datetime("us", "UTC")
_ASSIGNMENT_STATUSES = frozenset(
    {"leaf", "parent_only", "ambiguous", "unassigned", "outside_taxonomy"}
)
_ASSIGNMENT_SOURCES = frozenset({"feature_based", "propagated", "manual"})


def _assignment_enums() -> tuple[EnumConstraint, ...]:
    return (
        EnumConstraint("assignment_status", _ASSIGNMENT_STATUSES),
        EnumConstraint("assignment_source", _ASSIGNMENT_SOURCES),
    )


CELLS_VIEW = TableContract(
    name="cells_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "cell_id": pl.Int64,
            "candidate_id": pl.Int32,
            "membership_strength": pl.Float32,
            "taxon_id": pl.Int64,
            "assignment_status": pl.String,
            "assignment_source": pl.String,
            "confidence": pl.Float32,
        }
    ),
    nullable=frozenset(
        {
            "candidate_id",
            "membership_strength",
            "taxon_id",
            "assignment_status",
            "assignment_source",
            "confidence",
        }
    ),
    primary_key=("revision_id", "cell_id"),
    enums=_assignment_enums(),
    ranges=(
        RangeConstraint("membership_strength", minimum=0.0, maximum=1.0),
        RangeConstraint("confidence", minimum=0.0, maximum=1.0),
    ),
)

EMBEDDING_VIEW = TableContract(
    name="embedding_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "representation_id": pl.String,
            "cell_id": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "candidate_id": pl.Int32,
            "taxon_id": pl.Int64,
            "assignment_status": pl.String,
            "assignment_source": pl.String,
            "confidence": pl.Float32,
            "short_name": pl.String,
            "color": pl.String,
        }
    ),
    nullable=frozenset(
        {
            "candidate_id",
            "taxon_id",
            "assignment_status",
            "assignment_source",
            "confidence",
            "short_name",
            "color",
        }
    ),
    primary_key=("revision_id", "cell_id"),
    enums=_assignment_enums(),
    ranges=(RangeConstraint("confidence", minimum=0.0, maximum=1.0),),
)

FEATURE_PROFILES_VIEW = TableContract(
    name="feature_profiles_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "feature_space_id": pl.String,
            "group_kind": pl.String,
            "group_id": pl.Int64,
            "feature_position": pl.Int32,
            "feature_block_id": pl.String,
            "feature_id": pl.String,
            "modality": pl.String,
            "family": pl.String,
            "units": pl.String,
            "n_cells": pl.Int64,
            "mean": pl.Float64,
            "std": pl.Float64,
            "median": pl.Float64,
        }
    ),
    nullable=frozenset({"group_id", "units", "std"}),
    unique_keys=(("revision_id", "group_kind", "group_id", "feature_position"),),
    enums=(EnumConstraint("group_kind", frozenset({"all", "candidate", "taxon"})),),
    ranges=(RangeConstraint("n_cells", minimum=1),),
)

STABILITY_VIEW = TableContract(
    name="stability_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "candidate_set_id": pl.String,
            "candidate_id": pl.Int32,
            "n_cells": pl.Int64,
            "membership_mean": pl.Float64,
            "membership_min": pl.Float64,
            "membership_max": pl.Float64,
            "boundary_evidence_count": pl.Int64,
            "boundary_min_value": pl.Float64,
        }
    ),
    nullable=frozenset(
        {
            "membership_mean",
            "membership_min",
            "membership_max",
            "boundary_min_value",
        }
    ),
    primary_key=("revision_id", "candidate_id"),
    ranges=(
        RangeConstraint("candidate_id", minimum=0),
        RangeConstraint("n_cells", minimum=0),
        RangeConstraint("membership_mean", minimum=0.0, maximum=1.0),
        RangeConstraint("membership_min", minimum=0.0, maximum=1.0),
        RangeConstraint("membership_max", minimum=0.0, maximum=1.0),
        RangeConstraint("boundary_evidence_count", minimum=0),
    ),
)

COMPARISON_VIEW = TableContract(
    name="comparison_view",
    schema=pl.Schema(
        {
            "left_revision_id": pl.String,
            "right_revision_id": pl.String,
            "left_candidate_id": pl.Int32,
            "right_candidate_id": pl.Int32,
            "n_cells": pl.Int64,
        }
    ),
    nullable=frozenset({"left_candidate_id", "right_candidate_id"}),
    unique_keys=(
        (
            "left_revision_id",
            "right_revision_id",
            "left_candidate_id",
            "right_candidate_id",
        ),
    ),
    ranges=(RangeConstraint("n_cells", minimum=1),),
)

TAXONOMY_VIEW = TableContract(
    name="taxonomy_view",
    schema=pl.Schema(
        {
            **dict(CONTRACTS["taxonomy"].schema.items()),
            "assignment_count": pl.Int64,
            "feature_based_count": pl.Int64,
            "propagated_count": pl.Int64,
            "manual_count": pl.Int64,
            "ambiguous_count": pl.Int64,
            "parent_only_count": pl.Int64,
        }
    ),
    nullable=CONTRACTS["taxonomy"].nullable,
    primary_key=("taxonomy_name", "taxonomy_version", "taxon_id"),
    ranges=(
        RangeConstraint("assignment_count", minimum=0),
        RangeConstraint("feature_based_count", minimum=0),
        RangeConstraint("propagated_count", minimum=0),
        RangeConstraint("manual_count", minimum=0),
        RangeConstraint("ambiguous_count", minimum=0),
        RangeConstraint("parent_only_count", minimum=0),
    ),
)

HISTORY_VIEW = TableContract(
    name="history_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "name": pl.String,
            "depth": pl.Int32,
            "parent_revision_id": pl.String,
            "scope_id": pl.String,
            "feature_space_id": pl.String,
            "clustering_representation_id": pl.String,
            "visualization_representation_id": pl.String,
            "candidate_set_id": pl.String,
            "created_by": pl.String,
            "created_at": _AUDIT_DATETIME,
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
    ranges=(RangeConstraint("depth", minimum=0),),
)

RELEASE_SUMMARY_VIEW = TableContract(
    name="release_summary_view",
    schema=pl.Schema(
        {
            "revision_id": pl.String,
            "revision_name": pl.String,
            "scope_id": pl.String,
            "feature_space_id": pl.String,
            "visualization_representation_id": pl.String,
            "candidate_set_id": pl.String,
            "assignment_set_id": pl.String,
            "taxonomy_name": pl.String,
            "taxonomy_version": pl.String,
            "n_scope_cells": pl.Int64,
            "n_candidates": pl.Int64,
            "n_assignment_rows": pl.Int64,
            "n_taxon_assigned": pl.Int64,
            "n_outside_taxonomy": pl.Int64,
            "n_ambiguous": pl.Int64,
            "n_feature_based": pl.Int64,
            "n_propagated": pl.Int64,
            "n_manual": pl.Int64,
            "mean_confidence": pl.Float64,
            "mean_coverage": pl.Float64,
        }
    ),
    nullable=frozenset(
        {
            "feature_space_id",
            "visualization_representation_id",
            "candidate_set_id",
            "assignment_set_id",
            "taxonomy_name",
            "taxonomy_version",
            "mean_confidence",
            "mean_coverage",
        }
    ),
    primary_key=("revision_id",),
    ranges=tuple(
        RangeConstraint(column, minimum=0)
        for column in (
            "n_scope_cells",
            "n_candidates",
            "n_assignment_rows",
            "n_taxon_assigned",
            "n_outside_taxonomy",
            "n_ambiguous",
            "n_feature_based",
            "n_propagated",
            "n_manual",
        )
    )
    + (
        RangeConstraint("mean_confidence", minimum=0.0, maximum=1.0),
        RangeConstraint("mean_coverage", minimum=0.0, maximum=1.0),
    ),
)

VIEW_CONTRACTS: Mapping[str, TableContract] = {
    "cells": CELLS_VIEW,
    "embedding": EMBEDDING_VIEW,
    "feature_profiles": FEATURE_PROFILES_VIEW,
    "stability": STABILITY_VIEW,
    "comparison": COMPARISON_VIEW,
    "taxonomy": TAXONOMY_VIEW,
    "history": HISTORY_VIEW,
    "release_summary": RELEASE_SUMMARY_VIEW,
}


def validate_view(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and return one typed view frame."""
    try:
        contract = VIEW_CONTRACTS[name]
    except KeyError as error:
        raise KeyError(f"Unknown view contract {name!r}") from error
    validate_table(frame, contract)
    return frame


def _revision(study: Study, revision: KeptRevision | str) -> KeptRevision:
    return study.get_revision(revision) if isinstance(revision, str) else revision


def _assignment_set(
    study: Study, assignment_set: AssignmentSet | str | None
) -> AssignmentSet | None:
    if isinstance(assignment_set, str):
        return study.get_assignment_set(assignment_set)
    return assignment_set


def _empty_columns(
    frame: pl.DataFrame, columns: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    return frame.with_columns(
        *(pl.lit(None, dtype=dtype).alias(name) for name, dtype in columns.items())
    )


def cells(
    study: Study,
    revision: KeptRevision | str,
    *,
    assignment_set: AssignmentSet | str | None = None,
) -> pl.DataFrame:
    """Project one revision's scope, candidates, and optional assignments."""
    record = _revision(study, revision)
    scope = study.get_scope(record.scope_id)
    frame = study.folio.get(scope.members_ref, frame="polars").with_columns(
        pl.lit(record.revision_id).alias("revision_id")
    )
    if record.candidate_set_id is None:
        frame = _empty_columns(
            frame,
            {"candidate_id": pl.Int32, "membership_strength": pl.Float32},
        )
    else:
        frame = frame.join(
            study.candidate_membership(record.candidate_set_id).select(
                "cell_id", "candidate_id", "membership_strength"
            ),
            on="cell_id",
            how="left",
        )
    assignment = _assignment_set(study, assignment_set)
    if assignment is None:
        frame = _empty_columns(
            frame,
            {
                "taxon_id": pl.Int64,
                "assignment_status": pl.String,
                "assignment_source": pl.String,
                "confidence": pl.Float32,
            },
        )
    else:
        frame = frame.join(
            study.assignments(assignment).select(
                "cell_id",
                "taxon_id",
                "assignment_status",
                "assignment_source",
                "confidence",
            ),
            on="cell_id",
            how="left",
        )
    result = frame.select(*CELLS_VIEW.schema.names()).cast(CELLS_VIEW.schema)
    return validate_view("cells", result)


def embedding(
    study: Study,
    revision: KeptRevision | str,
    *,
    assignment_set: AssignmentSet | str | None = None,
    representation: Representation | str | None = None,
    dimensions: tuple[int, int] = (0, 1),
) -> pl.DataFrame:
    """Project two named dimensions plus generic candidate/taxonomy display data."""
    record = _revision(study, revision)
    if (
        len(dimensions) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in dimensions
        )
        or dimensions[0] == dimensions[1]
        or min(dimensions) < 0
    ):
        raise ValueError(
            "Exactly two distinct non-negative integer embedding dimensions are required"
        )
    if representation is None:
        representation_id = (
            record.visualization_representation_id
            or record.clustering_representation_id
        )
        if representation_id is None:
            raise ValueError("Revision has no representation for an embedding view")
        representation_record = study.get_representation(representation_id)
    else:
        representation_record = (
            study.get_representation(representation)
            if isinstance(representation, str)
            else representation
        )
    if representation_record.scope_id != record.scope_id:
        raise ValueError("Embedding representation must cover the revision scope")
    x_column, y_column = (f"dim_{index}" for index in dimensions)
    coords = study.folio.get(representation_record.coords_ref, frame="polars")
    missing = {x_column, y_column} - set(coords.columns)
    if missing:
        raise ValueError(f"Embedding dimensions are unavailable: {sorted(missing)}")
    frame = coords.select(
        "cell_id",
        pl.col(x_column).cast(pl.Float64).alias("x"),
        pl.col(y_column).cast(pl.Float64).alias("y"),
    ).join(
        cells(study, record, assignment_set=assignment_set).select(
            "cell_id",
            "candidate_id",
            "taxon_id",
            "assignment_status",
            "assignment_source",
            "confidence",
        ),
        on="cell_id",
        how="left",
    )
    assignment = _assignment_set(study, assignment_set)
    if assignment is None:
        frame = _empty_columns(frame, {"short_name": pl.String, "color": pl.String})
    else:
        taxonomy = study.get_taxonomy(
            assignment.taxonomy_name, assignment.taxonomy_version
        )
        frame = frame.join(
            taxonomy.select("taxon_id", "short_name", "color"),
            on="taxon_id",
            how="left",
        )
    result = frame.with_columns(
        pl.lit(record.revision_id).alias("revision_id"),
        pl.lit(representation_record.representation_id).alias("representation_id"),
    ).select(*EMBEDDING_VIEW.schema.names())
    return validate_view("embedding", result)


def feature_profiles(
    study: Study,
    revision: KeptRevision | str,
    *,
    group_by: str = "candidate",
    assignment_set: AssignmentSet | str | None = None,
) -> pl.DataFrame:
    """Summarize materialized feature values by candidate, taxon, or all cells."""
    if group_by not in {"all", "candidate", "taxon"}:
        raise ValueError("group_by must be 'all', 'candidate', or 'taxon'")
    record = _revision(study, revision)
    if record.feature_space_id is None:
        raise ValueError("Revision has no feature space for feature profiles")
    space = study.get_feature_space(record.feature_space_id)
    if space.values_ref is None:
        raise ValueError("Feature-profile feature space has no materialized values")
    values = study.folio.get(space.values_ref, frame="polars")
    value_columns = [column for column in values.columns if column != "cell_id"]
    selection = study.get_feature_selection(space.feature_selection_id)
    members = study.folio.get(selection.members_ref, frame="polars")
    catalog = study.feature_catalog()
    metadata = (
        members.with_columns(
            pl.concat_str(
                pl.lit("feature_"),
                pl.col("position").cast(pl.String).str.pad_start(5, "0"),
            ).alias("value_column")
        )
        .join(
            catalog.select(
                "feature_block_id",
                "feature_id",
                "column_name",
                "modality",
                "family",
                "units",
            ),
            on=["feature_block_id", "feature_id"],
            how="left",
            validate="1:1",
        )
        .rename({"position": "feature_position"})
    )
    metadata_columns = metadata["value_column"].to_list()
    if (
        metadata.height != members.height
        or metadata["column_name"].null_count()
        or len(metadata_columns) != len(set(metadata_columns))
        or set(metadata_columns) != set(value_columns)
    ):
        raise ValueError(
            "Feature metadata columns do not match materialized feature values"
        )
    metadata = metadata.select(
        "value_column",
        "feature_position",
        "feature_block_id",
        "feature_id",
        "modality",
        "family",
        "units",
    )
    if group_by == "all":
        groups = values.select("cell_id").with_columns(
            pl.lit(None, dtype=pl.Int64).alias("group_id")
        )
    elif group_by == "candidate":
        if record.candidate_set_id is None:
            raise ValueError("Candidate profiles require a revision candidate set")
        groups = (
            study.candidate_membership(record.candidate_set_id)
            .filter(pl.col("candidate_id").is_not_null())
            .select("cell_id", pl.col("candidate_id").cast(pl.Int64).alias("group_id"))
        )
    else:
        assignment = _assignment_set(study, assignment_set)
        if assignment is None:
            raise ValueError("Taxon profiles require an assignment set")
        groups = (
            study.assignments(assignment)
            .filter(pl.col("taxon_id").is_not_null())
            .select("cell_id", pl.col("taxon_id").alias("group_id"))
        )
    selected = values.join(groups, on="cell_id", how="inner")
    if selected.is_empty():
        return validate_view(
            "feature_profiles", pl.DataFrame(schema=FEATURE_PROFILES_VIEW.schema)
        )
    result = (
        selected.unpivot(
            index=["cell_id", "group_id"],
            on=value_columns,
            variable_name="value_column",
            value_name="value",
        )
        .group_by("group_id", "value_column")
        .agg(
            pl.len().cast(pl.Int64).alias("n_cells"),
            pl.col("value").mean().cast(pl.Float64).alias("mean"),
            pl.col("value").std().cast(pl.Float64).alias("std"),
            pl.col("value").median().cast(pl.Float64).alias("median"),
        )
        .join(metadata, on="value_column", validate="m:1")
        .with_columns(
            pl.lit(record.revision_id).alias("revision_id"),
            pl.lit(space.feature_space_id).alias("feature_space_id"),
            pl.lit(group_by).alias("group_kind"),
        )
        .select(*FEATURE_PROFILES_VIEW.schema.names())
        .sort("group_id", "feature_position", nulls_last=False)
    )
    return validate_view("feature_profiles", result)


def stability(study: Study, revision: KeptRevision | str) -> pl.DataFrame:
    """Project generic candidate strength and boundary-evidence summaries."""
    record = _revision(study, revision)
    if record.candidate_set_id is None:
        raise ValueError("Stability view requires a revision candidate set")
    definitions = study.candidate_definitions(record.candidate_set_id)
    membership = study.candidate_membership(record.candidate_set_id)
    strengths = (
        membership.filter(pl.col("candidate_id").is_not_null())
        .group_by("candidate_id")
        .agg(
            pl.col("membership_strength")
            .mean()
            .cast(pl.Float64)
            .alias("membership_mean"),
            pl.col("membership_strength")
            .min()
            .cast(pl.Float64)
            .alias("membership_min"),
            pl.col("membership_strength")
            .max()
            .cast(pl.Float64)
            .alias("membership_max"),
        )
    )
    evidence = study.candidate_boundary_evidence(record.candidate_set_id)
    if evidence.is_empty():
        boundary = pl.DataFrame(
            schema={
                "candidate_id": pl.Int32,
                "boundary_evidence_count": pl.Int64,
                "boundary_min_value": pl.Float64,
            }
        )
    else:
        incident = pl.concat(
            [
                evidence.select(
                    pl.col("candidate_id_a").alias("candidate_id"), "value"
                ),
                evidence.select(
                    pl.col("candidate_id_b").alias("candidate_id"), "value"
                ),
            ]
        )
        boundary = incident.group_by("candidate_id").agg(
            pl.len().cast(pl.Int64).alias("boundary_evidence_count"),
            pl.col("value").min().cast(pl.Float64).alias("boundary_min_value"),
        )
    result = (
        definitions.select("candidate_id", "n_cells")
        .join(strengths, on="candidate_id", how="left")
        .join(boundary, on="candidate_id", how="left")
        .with_columns(
            pl.lit(record.revision_id).alias("revision_id"),
            pl.lit(record.candidate_set_id).alias("candidate_set_id"),
            pl.col("boundary_evidence_count").fill_null(0).cast(pl.Int64),
        )
        .select(*STABILITY_VIEW.schema.names())
        .sort("candidate_id")
    )
    return validate_view("stability", result)


def comparison(
    study: Study,
    left: KeptRevision | str,
    right: KeptRevision | str,
) -> pl.DataFrame:
    """Project a typed candidate contingency table for two revisions."""
    left_record = _revision(study, left)
    right_record = _revision(study, right)
    result = (
        study.compare_revisions(left_record, right_record)
        .with_columns(
            pl.lit(left_record.revision_id).alias("left_revision_id"),
            pl.lit(right_record.revision_id).alias("right_revision_id"),
            pl.col("n_cells").cast(pl.Int64),
        )
        .select(*COMPARISON_VIEW.schema.names())
    )
    return validate_view("comparison", result)


def taxonomy(
    study: Study,
    *,
    assignment_set: AssignmentSet | str | None = None,
    name: str | None = None,
    version: str | None = None,
) -> pl.DataFrame:
    """Project taxonomy metadata with optional assignment counts per taxon."""
    assignment = _assignment_set(study, assignment_set)
    if assignment is not None:
        if name is not None and name != assignment.taxonomy_name:
            raise ValueError("Taxonomy name conflicts with the assignment set")
        if version is not None and version != assignment.taxonomy_version:
            raise ValueError("Taxonomy version conflicts with the assignment set")
        name = assignment.taxonomy_name
        version = assignment.taxonomy_version
    if name is None or version is None:
        raise ValueError("Taxonomy view requires name/version or an assignment set")
    frame = study.get_taxonomy(name, version)
    count_columns = {
        "assignment_count": pl.Int64,
        "feature_based_count": pl.Int64,
        "propagated_count": pl.Int64,
        "manual_count": pl.Int64,
        "ambiguous_count": pl.Int64,
        "parent_only_count": pl.Int64,
    }
    if assignment is None:
        result = frame.with_columns(
            *(
                pl.lit(0, dtype=dtype).alias(column)
                for column, dtype in count_columns.items()
            )
        )
    else:
        counts = (
            study.assignments(assignment)
            .filter(pl.col("taxon_id").is_not_null())
            .group_by("taxon_id")
            .agg(
                pl.len().cast(pl.Int64).alias("assignment_count"),
                (pl.col("assignment_source") == "feature_based")
                .sum()
                .cast(pl.Int64)
                .alias("feature_based_count"),
                (pl.col("assignment_source") == "propagated")
                .sum()
                .cast(pl.Int64)
                .alias("propagated_count"),
                (pl.col("assignment_source") == "manual")
                .sum()
                .cast(pl.Int64)
                .alias("manual_count"),
                (pl.col("assignment_status") == "ambiguous")
                .sum()
                .cast(pl.Int64)
                .alias("ambiguous_count"),
                (pl.col("assignment_status") == "parent_only")
                .sum()
                .cast(pl.Int64)
                .alias("parent_only_count"),
            )
        )
        result = frame.join(counts, on="taxon_id", how="left").with_columns(
            *(
                pl.col(column).fill_null(0).cast(dtype)
                for column, dtype in count_columns.items()
            )
        )
    result = result.select(*TAXONOMY_VIEW.schema.names()).sort("sort_order", "taxon_id")
    return validate_view("taxonomy", result)


def history(study: Study, revision: KeptRevision | str) -> pl.DataFrame:
    """Project one revision's root-to-head ancestry as a typed frame."""
    cursor = _revision(study, revision)
    lineage: list[KeptRevision] = []
    seen: set[str] = set()
    while True:
        if cursor.revision_id in seen:
            raise ValueError("Revision history contains a cycle")
        seen.add(cursor.revision_id)
        lineage.append(cursor)
        if cursor.parent_revision_id is None:
            break
        cursor = study.get_revision(cursor.parent_revision_id)
    rows = []
    for depth, record in enumerate(reversed(lineage)):
        rows.append(
            {
                "revision_id": record.revision_id,
                "name": record.name,
                "depth": depth,
                "parent_revision_id": record.parent_revision_id,
                "scope_id": record.scope_id,
                "feature_space_id": record.feature_space_id,
                "clustering_representation_id": record.clustering_representation_id,
                "visualization_representation_id": record.visualization_representation_id,
                "candidate_set_id": record.candidate_set_id,
                "created_by": record.created_by,
                "created_at": record.created_at,
                "notes": record.notes,
            }
        )
    return validate_view("history", pl.DataFrame(rows, schema=HISTORY_VIEW.schema))


def release_summary(
    study: Study,
    revision: KeptRevision | str,
    *,
    assignment_set: AssignmentSet | str | None = None,
) -> pl.DataFrame:
    """Project one release-readiness summary without creating a release."""
    record = _revision(study, revision)
    assignment = _assignment_set(study, assignment_set)
    cell_frame = cells(study, record, assignment_set=assignment)
    if record.candidate_set_id is None:
        n_candidates = 0
    else:
        n_candidates = study.candidate_definitions(record.candidate_set_id).height
    if assignment is None:
        assignment_rows = pl.DataFrame(schema=CONTRACTS["assignment"].schema)
    else:
        scope_cells = cell_frame.select("cell_id")
        assignment_rows = study.assignments(assignment).join(
            scope_cells, on="cell_id", how="inner"
        )

    def count(column: str, value: str) -> int:
        if assignment_rows.is_empty():
            return 0
        return assignment_rows.filter(pl.col(column) == value).height

    row = {
        "revision_id": record.revision_id,
        "revision_name": record.name,
        "scope_id": record.scope_id,
        "feature_space_id": record.feature_space_id,
        "visualization_representation_id": record.visualization_representation_id,
        "candidate_set_id": record.candidate_set_id,
        "assignment_set_id": None
        if assignment is None
        else assignment.assignment_set_id,
        "taxonomy_name": None if assignment is None else assignment.taxonomy_name,
        "taxonomy_version": None if assignment is None else assignment.taxonomy_version,
        "n_scope_cells": cell_frame.height,
        "n_candidates": n_candidates,
        "n_assignment_rows": assignment_rows.height,
        "n_taxon_assigned": (
            0
            if assignment_rows.is_empty()
            else assignment_rows.filter(pl.col("taxon_id").is_not_null()).height
        ),
        "n_outside_taxonomy": count("assignment_status", "outside_taxonomy"),
        "n_ambiguous": count("assignment_status", "ambiguous"),
        "n_feature_based": count("assignment_source", "feature_based"),
        "n_propagated": count("assignment_source", "propagated"),
        "n_manual": count("assignment_source", "manual"),
        "mean_confidence": (
            None if assignment_rows.is_empty() else assignment_rows["confidence"].mean()
        ),
        "mean_coverage": (
            None if assignment_rows.is_empty() else assignment_rows["coverage"].mean()
        ),
    }
    frame = pl.DataFrame([row], schema=RELEASE_SUMMARY_VIEW.schema)
    return validate_view("release_summary", frame)


def serialize_view(name: str, frame: pl.DataFrame) -> bytes:
    """Serialize a validated view as a self-describing deterministic JSON component."""
    validate_view(name, frame)
    contract = VIEW_CONTRACTS[name]
    ordering = contract.primary_key or contract.unique_keys[0]
    ordered = frame.sort(list(ordering), nulls_last=True)
    rows = json.loads(ordered.write_json())
    schema = [
        {
            "name": column,
            "dtype": str(dtype),
            "nullable": column in contract.nullable,
        }
        for column, dtype in contract.schema.items()
    ]
    payload = {
        "component_kind": "cellpax_view",
        "schema_version": "1.0",
        "view_name": name,
        "view_contract_hash": canonical_hash(schema),
        "row_count": ordered.height,
        "rows_hash": canonical_hash(rows),
        "schema": schema,
        "rows": rows,
    }
    return canonical_json_bytes(payload)
