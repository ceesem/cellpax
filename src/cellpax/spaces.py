"""Pure table assembly and fitting helpers for Slice 2 spaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from cellpax.config import FeatureSpaceConfig, RepresentationConfig
from cellpax.transforms import ClippedScaler

MISSINGNESS_SCHEMA = pl.Schema(
    {
        "position": pl.Int32,
        "feature_block_id": pl.String,
        "feature_id": pl.String,
        "missing_scope": pl.Int64,
        "missing_fit_scope": pl.Int64,
    }
)


class MissingValuesError(ValueError):
    """Raised when an error missing policy encounters null feature values."""

    def __init__(self, report: pl.DataFrame) -> None:
        self.report = report
        total = int(report["missing_scope"].sum()) + int(
            report["missing_fit_scope"].sum()
        )
        super().__init__(f"Feature values contain {total} missing entries")


class ScopeReductionRequiredError(ValueError):
    """Raised when ``drop`` would silently shrink the application scope."""

    def __init__(self, report: pl.DataFrame, retained_members: pl.DataFrame) -> None:
        self.report = report
        self.retained_members = retained_members
        super().__init__(
            "drop would reduce the declared application scope; create a child "
            "scope from retained_members and retry"
        )


def feature_column(position: int) -> str:
    """Return the stable physical column name for an ordered feature position."""
    return f"feature_{position:05d}"


def assemble_selected_values(
    members: pl.DataFrame,
    selection_members: pl.DataFrame,
    catalog: pl.DataFrame,
    blocks: Mapping[str, pl.DataFrame],
) -> pl.DataFrame:
    """Left-align selected block columns to a canonical scope membership."""
    result = members.select("cell_id")
    for selected in selection_members.sort("position").iter_rows(named=True):
        catalog_row = catalog.filter(
            (pl.col("feature_block_id") == selected["feature_block_id"])
            & (pl.col("feature_id") == selected["feature_id"])
        )
        if catalog_row.height != 1:
            raise ValueError(f"Selection feature is not uniquely cataloged: {selected}")
        column_name = catalog_row["column_name"][0]
        block = blocks[selected["feature_block_id"]]
        if not block.schema[column_name].is_numeric():
            raise TypeError(
                f"Selected feature {selected['feature_id']!r} must be numeric"
            )
        result = result.join(
            block.select(
                "cell_id",
                pl.col(column_name)
                .cast(pl.Float64)
                .alias(feature_column(selected["position"])),
            ),
            on="cell_id",
            how="left",
        )
    return result.sort("cell_id")


def missingness_report(
    scope_values: pl.DataFrame,
    fit_values: pl.DataFrame,
    selection_members: pl.DataFrame,
) -> pl.DataFrame:
    """Report missing counts for every ordered feature in both scopes."""
    rows = []
    for selected in selection_members.sort("position").iter_rows(named=True):
        column = feature_column(selected["position"])
        rows.append(
            {
                "position": selected["position"],
                "feature_block_id": selected["feature_block_id"],
                "feature_id": selected["feature_id"],
                "missing_scope": scope_values[column].null_count(),
                "missing_fit_scope": fit_values[column].null_count(),
            }
        )
    return pl.DataFrame(rows, schema=MISSINGNESS_SCHEMA)


def _transformer(config: FeatureSpaceConfig) -> Any | None:
    params = config.params
    if config.transform == "raw_join":
        return None
    if config.transform == "standard_scaler":
        return StandardScaler(**params)
    if config.transform == "robust_scaler":
        params["quantile_range"] = tuple(params["quantile_range"])
        return RobustScaler(**params)
    if config.transform == "clipped_scaler":
        return ClippedScaler(**params)
    raise ValueError(f"Unsupported transform {config.transform!r}")


def fit_feature_space(
    scope_values: pl.DataFrame,
    fit_values: pl.DataFrame,
    selection_members: pl.DataFrame,
    config: FeatureSpaceConfig,
) -> tuple[pl.DataFrame, Any | None, pl.DataFrame]:
    """Apply missing policy, fit one stage, and return materialized values."""
    report = missingness_report(scope_values, fit_values, selection_members)
    feature_columns = [
        feature_column(position) for position in selection_members["position"].to_list()
    ]
    if config.missing_policy == "error" and (
        scope_values.select(feature_columns).null_count().sum_horizontal()[0]
        or fit_values.select(feature_columns).null_count().sum_horizontal()[0]
    ):
        raise MissingValuesError(report)

    scope_ready = scope_values
    fit_ready = fit_values
    if config.missing_policy == "drop":
        retained_scope = scope_ready.drop_nulls(feature_columns)
        if retained_scope.height != scope_ready.height:
            raise ScopeReductionRequiredError(report, retained_scope.select("cell_id"))
        fit_ready = fit_ready.drop_nulls(feature_columns)
    if scope_ready.is_empty() or fit_ready.is_empty():
        raise ValueError("Missing-value handling left no cells to transform")

    steps: list[tuple[str, Any]] = []
    if config.missing_policy == "median":
        steps.append(
            (
                "missing",
                SimpleImputer(strategy="median", keep_empty_features=True),
            )
        )
    transform = _transformer(config)
    if transform is not None:
        steps.append(("transform", transform))
    estimator = Pipeline(steps) if steps else None

    fit_matrix = fit_ready.select(feature_columns).to_numpy()
    scope_matrix = scope_ready.select(feature_columns).to_numpy()
    if estimator is None:
        output = np.asarray(scope_matrix, dtype=np.float64)
    else:
        estimator.fit(fit_matrix)
        output = np.asarray(estimator.transform(scope_matrix), dtype=np.float64)
    values = pl.DataFrame(output, schema=feature_columns).with_columns(
        pl.Series("cell_id", scope_ready["cell_id"], dtype=pl.Int64)
    )
    return values.select("cell_id", *feature_columns), estimator, report


def fit_representation(
    scope_values: pl.DataFrame,
    fit_values: pl.DataFrame,
    config: RepresentationConfig,
) -> tuple[pl.DataFrame, Any | None]:
    """Fit and materialize one coordinate representation."""
    feature_columns = [column for column in scope_values.columns if column != "cell_id"]
    if feature_columns != [
        column for column in fit_values.columns if column != "cell_id"
    ]:
        raise ValueError("Representation fit and application schemas differ")
    if scope_values.select(feature_columns).null_count().sum_horizontal()[0]:
        raise ValueError("Representation input contains null values")
    if fit_values.select(feature_columns).null_count().sum_horizontal()[0]:
        raise ValueError("Representation fit input contains null values")

    scope_matrix = scope_values.select(feature_columns).to_numpy()
    fit_matrix = fit_values.select(feature_columns).to_numpy()
    if config.method == "scaled_passthrough":
        estimator = None
        output = np.asarray(scope_matrix, dtype=np.float64)
    else:
        if config.n_components is None or config.n_components > min(fit_matrix.shape):
            raise ValueError("PCA n_components exceeds fit data dimensions")
        estimator = PCA(
            n_components=config.n_components,
            random_state=config.seed,
            **config.params,
        )
        estimator.fit(fit_matrix)
        output = np.asarray(estimator.transform(scope_matrix), dtype=np.float64)
    coord_columns = [f"dim_{index}" for index in range(output.shape[1])]
    coords = pl.DataFrame(output, schema=coord_columns).with_columns(
        pl.Series("cell_id", scope_values["cell_id"], dtype=pl.Int64)
    )
    return coords.select("cell_id", *coord_columns), estimator
