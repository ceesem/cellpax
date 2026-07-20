"""Trajan-facing annotation decoration without analysis-layer imports."""

from __future__ import annotations

import polars as pl

from cellpax.release import ReleaseBundle

_OUTPUT_COLUMNS = {
    "taxon_id",
    "taxon_key",
    "taxon_short_name",
    "taxon_long_name",
    "taxon_parent_id",
    "taxon_color",
    "assignment_status",
    "assignment_source",
    "assignment_confidence",
    "assignment_coverage",
}


def annotation_frame(release: ReleaseBundle) -> pl.DataFrame:
    """Project a release into the cell annotation contract consumed by Trajan."""
    return release.assignments.select(
        "cell_id",
        "taxon_id",
        "assignment_status",
        "assignment_source",
        pl.col("confidence").alias("assignment_confidence"),
        pl.col("coverage").alias("assignment_coverage"),
    ).join(
        release.taxonomy.select(
            "taxon_id",
            pl.col("key").alias("taxon_key"),
            pl.col("short_name").alias("taxon_short_name"),
            pl.col("long_name").alias("taxon_long_name"),
            pl.col("parent_id").alias("taxon_parent_id"),
            pl.col("color").alias("taxon_color"),
        ),
        on="taxon_id",
        how="left",
    )


def add_to_connectivity_table(
    table: object,
    release: ReleaseBundle,
    *,
    name: str = "cellpax",
    is_universe: bool = False,
    side: str = "both",
) -> object:
    """Register a release with a Trajan ConnectivityTable or EdgeList."""
    add_annotation = getattr(table, "add_annotation", None)
    if not callable(add_annotation):
        raise TypeError("Trajan table must provide a callable add_annotation method")
    return add_annotation(
        name,
        annotation_frame(release),
        cell_id_col="cell_id",
        is_universe=is_universe,
        side=side,
    )


def add_to_synapse_table(
    table: object,
    release: ReleaseBundle,
    *,
    name: str = "cellpax",
    is_universe: bool = False,
) -> object:
    """Register a release with a Trajan SynapseTable."""
    add_cell_annotation = getattr(table, "add_cell_annotation", None)
    if not callable(add_cell_annotation):
        raise TypeError(
            "Trajan table must provide a callable add_cell_annotation method"
        )
    return add_cell_annotation(
        name,
        annotation_frame(release),
        cell_id_col="cell_id",
        is_universe=is_universe,
    )


def decorate_cells(
    cells: pl.DataFrame,
    release: ReleaseBundle,
    *,
    cell_id_column: str = "cell_id",
    require_all_release_cells: bool = True,
) -> pl.DataFrame:
    """Left-join released assignments and taxonomy metadata onto consumer cells."""
    if not isinstance(cells, pl.DataFrame):
        raise TypeError("Trajan cell input must be a Polars DataFrame")
    if cell_id_column not in cells.columns:
        raise ValueError(f"Cell input is missing id column {cell_id_column!r}")
    if cells.schema[cell_id_column] != pl.Int64:
        raise TypeError("Trajan cell ids must have Int64 dtype")
    if (
        cells[cell_id_column].null_count()
        or cells[cell_id_column].n_unique() != cells.height
    ):
        raise ValueError("Trajan cell ids must be non-null and unique")
    collisions = sorted(_OUTPUT_COLUMNS & set(cells.columns))
    if collisions:
        raise ValueError(
            f"Cell input already contains annotation columns: {collisions}"
        )

    assignment_cells = release.assignments.select("cell_id")
    available = cells.select(pl.col(cell_id_column).alias("cell_id"))
    if require_all_release_cells:
        missing = assignment_cells.join(available, on="cell_id", how="anti")
        if not missing.is_empty():
            raise ValueError(
                f"Consumer universe is missing {missing.height} released cells"
            )

    annotations = annotation_frame(release)
    if cell_id_column != "cell_id":
        annotations = annotations.rename({"cell_id": cell_id_column})
    return cells.join(annotations, on=cell_id_column, how="left")
