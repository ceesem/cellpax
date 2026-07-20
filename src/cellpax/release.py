"""Language-neutral annotation-release products and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl

from cellpax.contracts.core import RangeConstraint, TableContract
from cellpax.records import AnnotationRelease
from cellpax.taxonomy import taxonomy_enum, validate_taxonomy

RELEASE_ARTIFACT_NAMES = (
    "taxonomy",
    "assignments",
    "decisions",
    "quality_summary",
    "recipe",
    "replay_script",
    "enum_binding",
)

RELEASE_MANIFEST_KEYS = frozenset(
    {
        "component_kind",
        "schema_version",
        "annotation_release_id",
        "name",
        "taxonomy_name",
        "taxonomy_version",
        "assignment_set_id",
        "assignment_set_state_hash",
        "source_revision_id",
        "source_revision_state_hash",
        "decision_head_id",
        "artifacts",
    }
)

RELEASE_QUALITY = TableContract(
    name="release_quality",
    schema=pl.Schema(
        {
            "annotation_release_id": pl.String,
            "n_scope_cells": pl.Int64,
            "n_assignment_rows": pl.Int64,
            "n_taxon_assigned": pl.Int64,
            "n_outside_taxonomy": pl.Int64,
            "n_ambiguous": pl.Int64,
            "n_parent_only": pl.Int64,
            "n_feature_based": pl.Int64,
            "n_propagated": pl.Int64,
            "n_manual": pl.Int64,
            "n_decisions": pl.Int64,
            "n_propagation_runs": pl.Int64,
            "mean_confidence": pl.Float64,
            "min_confidence": pl.Float64,
            "mean_coverage": pl.Float64,
            "min_coverage": pl.Float64,
        }
    ),
    nullable=frozenset(
        {"mean_confidence", "min_confidence", "mean_coverage", "min_coverage"}
    ),
    primary_key=("annotation_release_id",),
    ranges=tuple(
        RangeConstraint(column, minimum=0)
        for column in (
            "n_scope_cells",
            "n_assignment_rows",
            "n_taxon_assigned",
            "n_outside_taxonomy",
            "n_ambiguous",
            "n_parent_only",
            "n_feature_based",
            "n_propagated",
            "n_manual",
            "n_decisions",
            "n_propagation_runs",
        )
    )
    + (
        RangeConstraint("mean_confidence", minimum=0.0, maximum=1.0),
        RangeConstraint("min_confidence", minimum=0.0, maximum=1.0),
        RangeConstraint("mean_coverage", minimum=0.0, maximum=1.0),
        RangeConstraint("min_coverage", minimum=0.0, maximum=1.0),
    ),
)


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    """A loaded release whose products require no clustering implementation."""

    release: AnnotationRelease
    manifest: Mapping[str, Any]
    taxonomy: pl.DataFrame
    assignments: pl.DataFrame
    decisions: pl.DataFrame
    quality_summary: pl.DataFrame
    recipe: Mapping[str, Any]
    replay_script: str
    enum_binding: str

    def taxonomy_enum(self):
        """Build the runtime rich enum from the language-neutral taxonomy table."""
        return taxonomy_enum(self.taxonomy)


def validate_release_manifest(manifest: object) -> Mapping[str, Any]:
    """Validate the fixed top-level and artifact-ref/checksum manifest shape."""
    if not isinstance(manifest, Mapping) or set(manifest) != RELEASE_MANIFEST_KEYS:
        raise ValueError("Malformed annotation-release manifest")
    if manifest["component_kind"] != "annotation_release":
        raise ValueError("Annotation-release manifest kind mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        RELEASE_ARTIFACT_NAMES
    ):
        raise ValueError("Annotation-release manifest has malformed artifacts")
    for name in RELEASE_ARTIFACT_NAMES:
        descriptor = artifacts[name]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"ref", "checksum"}
            or not isinstance(descriptor["ref"], str)
            or not descriptor["ref"]
            or not isinstance(descriptor["checksum"], str)
            or not descriptor["checksum"]
        ):
            raise ValueError(
                f"Annotation-release artifact descriptor {name!r} is malformed"
            )
    return manifest


def release_quality_summary(
    *,
    annotation_release_id: str,
    assignments: pl.DataFrame,
    n_scope_cells: int,
    n_decisions: int,
) -> pl.DataFrame:
    """Build the fixed one-row quality product for an annotation release."""

    def count(column: str, value: str) -> int:
        return assignments.filter(pl.col(column) == value).height

    propagation_ids = assignments["propagation_run_id"].drop_nulls().unique()
    row = {
        "annotation_release_id": annotation_release_id,
        "n_scope_cells": n_scope_cells,
        "n_assignment_rows": assignments.height,
        "n_taxon_assigned": assignments.filter(pl.col("taxon_id").is_not_null()).height,
        "n_outside_taxonomy": count("assignment_status", "outside_taxonomy"),
        "n_ambiguous": count("assignment_status", "ambiguous"),
        "n_parent_only": count("assignment_status", "parent_only"),
        "n_feature_based": count("assignment_source", "feature_based"),
        "n_propagated": count("assignment_source", "propagated"),
        "n_manual": count("assignment_source", "manual"),
        "n_decisions": n_decisions,
        "n_propagation_runs": len(propagation_ids),
        "mean_confidence": assignments["confidence"].mean(),
        "min_confidence": assignments["confidence"].min(),
        "mean_coverage": assignments["coverage"].mean(),
        "min_coverage": assignments["coverage"].min(),
    }
    return pl.DataFrame([row], schema=RELEASE_QUALITY.schema)


def generate_enum_binding(taxonomy: pl.DataFrame) -> str:
    """Generate deterministic importable Python source for a rich taxon enum."""
    validate_taxonomy(taxonomy)
    generated = taxonomy_enum(taxonomy)
    rows = {
        row["taxon_id"]: row for row in taxonomy.sort("taxon_id").iter_rows(named=True)
    }
    lines = [
        '"""Generated CellPax taxonomy binding."""',
        "",
        "from cellpax.taxonomy import RichTaxon",
        "",
        "",
        f"class {generated.__name__}(RichTaxon):",
    ]
    lines.extend(f"    {member.name} = {int(member)}" for member in generated)
    lines.extend(
        [
            "",
            f"{generated.__name__}.__taxonomy_rows__ = {rows!r}",
            "",
        ]
    )
    return "\n".join(lines)
