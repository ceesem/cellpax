"""Canonical hashing and validation for dynamic payload tables."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from cellpax.identity import canonical_hash

_PRIMITIVE_DTYPES = {
    pl.Boolean: "bool",
    pl.Int8: "int8",
    pl.Int16: "int16",
    pl.Int32: "int32",
    pl.Int64: "int64",
    pl.UInt8: "uint8",
    pl.UInt16: "uint16",
    pl.UInt32: "uint32",
    pl.UInt64: "uint64",
    pl.Float32: "float32",
    pl.Float64: "float64",
    pl.String: "string",
    pl.Binary: "binary",
    pl.Date: "date",
    pl.Time: "time",
    pl.Null: "null",
}


def canonical_dtype(dtype: pl.DataType) -> object:
    """Return a versioned JSON value for a supported Polars dtype."""
    primitive = _PRIMITIVE_DTYPES.get(dtype)
    if primitive is not None:
        return primitive
    if isinstance(dtype, pl.Datetime):
        return {
            "datetime": {
                "time_unit": dtype.time_unit,
                "time_zone": dtype.time_zone,
            }
        }
    if isinstance(dtype, pl.Duration):
        return {"duration": {"time_unit": dtype.time_unit}}
    if isinstance(dtype, pl.List):
        return {"list": canonical_dtype(dtype.inner)}
    if isinstance(dtype, pl.Array):
        return {"array": {"inner": canonical_dtype(dtype.inner), "shape": dtype.shape}}
    if isinstance(dtype, pl.Struct):
        return {
            "struct": [
                {"name": field.name, "dtype": canonical_dtype(field.dtype)}
                for field in dtype.fields
            ]
        }
    raise TypeError(f"Unsupported Polars dtype for canonical hashing: {dtype!s}")


def table_schema_hash(
    frame: pl.DataFrame, *, nullable_columns: Iterable[str] = ()
) -> str:
    """Hash ordered physical columns, canonical dtypes, and declared nullability."""
    nullable = frozenset(nullable_columns)
    unknown = nullable - set(frame.columns)
    if unknown:
        raise ValueError(f"Unknown nullable columns: {sorted(unknown)}")
    fields = [
        {
            "name": name,
            "dtype": canonical_dtype(dtype),
            "nullable": name in nullable,
        }
        for name, dtype in frame.schema.items()
    ]
    return canonical_hash(fields)


def validate_cell_table(
    frame: pl.DataFrame,
    *,
    nullable_columns: Iterable[str] = (),
    require_unique: bool = True,
) -> None:
    """Validate a dynamic table keyed by canonical Int64 ``cell_id``."""
    if "cell_id" not in frame.columns:
        raise ValueError("Cell-keyed tables require a cell_id column")
    if frame.schema["cell_id"] != pl.Int64:
        raise TypeError("cell_id must have Polars Int64 dtype")
    if frame["cell_id"].null_count():
        raise ValueError("cell_id must be non-null")
    if require_unique and frame.select("cell_id").is_duplicated().any():
        raise ValueError("cell_id must be unique")

    nullable = frozenset(nullable_columns)
    if "cell_id" in nullable:
        raise ValueError("cell_id cannot be declared nullable")
    unknown = nullable - set(frame.columns)
    if unknown:
        raise ValueError(f"Unknown nullable columns: {sorted(unknown)}")
    for column in frame.columns:
        if column not in nullable and frame[column].null_count():
            raise ValueError(f"Column {column!r} contains nulls but is non-nullable")
