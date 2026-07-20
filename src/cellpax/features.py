"""Feature catalog and ordered-selection substrate helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from cellpax.contracts import CONTRACTS, validate_table
from cellpax.identity import canonical_hash

CATALOG_INPUT_COLUMNS = (
    "feature_id",
    "column_name",
    "modality",
    "family",
    "units",
    "description",
    "raw_or_derived",
)

_CATALOG_INPUT_SCHEMA = pl.Schema(
    {column: pl.String for column in CATALOG_INPUT_COLUMNS}
)


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Concise input for one catalog feature; mechanical fields are optional.

    ``column_name`` defaults to ``feature_id`` (the common case where the physical
    value column matches the semantic id).
    """

    feature_id: str
    modality: str
    family: str
    column_name: str | None = None
    units: str | None = None
    description: str | None = None
    raw_or_derived: str = "raw"


def feature_catalog(features: Iterable[FeatureDefinition]) -> pl.DataFrame:
    """Build the strict catalog input frame from concise typed definitions."""
    definitions = list(features)
    if not definitions:
        raise ValueError("A feature catalog requires at least one FeatureDefinition")
    if not all(isinstance(feature, FeatureDefinition) for feature in definitions):
        raise TypeError("features must contain FeatureDefinition values")
    rows = [
        {
            "feature_id": feature.feature_id,
            "column_name": feature.column_name or feature.feature_id,
            "modality": feature.modality,
            "family": feature.family,
            "units": feature.units,
            "description": feature.description,
            "raw_or_derived": feature.raw_or_derived,
        }
        for feature in definitions
    ]
    return pl.DataFrame(rows, schema=_CATALOG_INPUT_SCHEMA)


FEATURE_SELECTION_MEMBERS_SCHEMA = pl.Schema(
    {
        "position": pl.Int32,
        "feature_block_id": pl.String,
        "feature_id": pl.String,
    }
)


def prepare_feature_catalog(
    catalog: pl.DataFrame,
    *,
    feature_block_id: str,
    value_columns: list[str],
) -> pl.DataFrame:
    """Add block identity and validate a catalog against physical value columns."""
    missing = set(CATALOG_INPUT_COLUMNS) - set(catalog.columns)
    extra = set(catalog.columns) - set(CATALOG_INPUT_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"Feature catalog columns mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    ordered = catalog.select(CATALOG_INPUT_COLUMNS)
    for column in CATALOG_INPUT_COLUMNS:
        dtype = ordered.schema[column]
        if dtype == pl.String:
            continue
        if column in {"units", "description"} and dtype == pl.Null:
            continue
        raise TypeError(f"Feature catalog column {column!r} must have String dtype")
    result = ordered.select(
        "feature_id",
        pl.lit(feature_block_id).alias("feature_block_id"),
        *CATALOG_INPUT_COLUMNS[1:],
    ).cast(CONTRACTS["feature_catalog"].schema)
    validate_table(result, CONTRACTS["feature_catalog"])

    described = result["column_name"].to_list()
    if len(described) != len(set(described)):
        raise ValueError("Feature catalog column_name values must be unique")
    if set(described) != set(value_columns):
        raise ValueError(
            "Feature catalog must describe every physical feature column exactly once; "
            f"catalog_only={sorted(set(described) - set(value_columns))}, "
            f"values_only={sorted(set(value_columns) - set(described))}"
        )
    return result


def feature_catalog_hash(catalog: pl.DataFrame) -> str:
    """Hash sorted semantic feature-catalog records."""
    validate_table(catalog, CONTRACTS["feature_catalog"])
    ordered = catalog.sort("feature_block_id", "feature_id")
    return canonical_hash(ordered.to_dicts())


def prepare_feature_selection_members(
    selected: pl.DataFrame, catalog: pl.DataFrame
) -> pl.DataFrame:
    """Validate selected feature pairs and retain their exact input order."""
    required = {"feature_block_id", "feature_id"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Feature selection is missing columns: {sorted(missing)}")
    wrong_dtypes = [
        column for column in sorted(required) if selected.schema[column] != pl.String
    ]
    if wrong_dtypes:
        raise TypeError(
            f"Feature selection identifiers must have String dtype: {wrong_dtypes}"
        )
    pairs = selected.select("feature_block_id", "feature_id")
    if pairs.is_duplicated().any():
        raise ValueError("Feature selection cannot contain duplicate feature ids")
    known = catalog.select("feature_block_id", "feature_id")
    unknown = pairs.join(
        known,
        on=["feature_block_id", "feature_id"],
        how="anti",
    )
    if not unknown.is_empty():
        raise ValueError(
            f"Feature selection contains unknown features: {unknown.to_dicts()}"
        )
    return (
        pairs.with_row_index("position")
        .with_columns(pl.col("position").cast(pl.Int32))
        .select(FEATURE_SELECTION_MEMBERS_SCHEMA.names())
    )
