from __future__ import annotations

import polars as pl
import pytest

from cellpax.scopes import normalize_cell_ids
from cellpax.table_utils import canonical_dtype, table_schema_hash, validate_cell_table
from cellpax.universe import semantic_roles_json


def test_schema_hash_covers_order_dtype_and_nullability() -> None:
    frame = pl.DataFrame(
        {
            "cell_id": pl.Series([1], dtype=pl.Int64),
            "value": pl.Series([1.0], dtype=pl.Float32),
        }
    )
    digest = table_schema_hash(frame)
    assert digest != table_schema_hash(frame.select("value", "cell_id"))
    assert digest != table_schema_hash(frame, nullable_columns=["value"])
    assert digest != table_schema_hash(
        frame.with_columns(pl.col("value").cast(pl.Float64))
    )


def test_canonical_dtype_supports_nested_and_temporal_types() -> None:
    assert canonical_dtype(pl.Datetime("us", "UTC")) == {
        "datetime": {"time_unit": "us", "time_zone": "UTC"}
    }
    assert canonical_dtype(pl.Duration("ns")) == {"duration": {"time_unit": "ns"}}
    assert canonical_dtype(pl.List(pl.Int16)) == {"list": "int16"}
    assert canonical_dtype(pl.Array(pl.Float32, 3)) == {
        "array": {"inner": "float32", "shape": (3,)}
    }
    assert canonical_dtype(pl.Struct({"name": pl.String})) == {
        "struct": [{"name": "name", "dtype": "string"}]
    }
    with pytest.raises(TypeError, match="Unsupported Polars dtype"):
        canonical_dtype(pl.Categorical)


@pytest.mark.parametrize(
    ("frame", "nullable", "message"),
    [
        (pl.DataFrame({"value": [1]}), (), "require a cell_id"),
        (
            pl.DataFrame({"cell_id": pl.Series([1], dtype=pl.Int32)}),
            (),
            "Int64",
        ),
        (
            pl.DataFrame({"cell_id": pl.Series([1, 1], dtype=pl.Int64)}),
            (),
            "unique",
        ),
        (
            pl.DataFrame(
                {
                    "cell_id": pl.Series([1], dtype=pl.Int64),
                    "value": pl.Series([None], dtype=pl.String),
                }
            ),
            (),
            "non-nullable",
        ),
    ],
)
def test_validate_cell_table_rejects_invalid_frames(
    frame: pl.DataFrame, nullable: tuple[str, ...], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_cell_table(frame, nullable_columns=nullable)


def test_nullable_declarations_are_strict() -> None:
    frame = pl.DataFrame(
        {
            "cell_id": pl.Series([1], dtype=pl.Int64),
            "value": pl.Series([None], dtype=pl.String),
        }
    )
    validate_cell_table(frame, nullable_columns=["value"])
    with pytest.raises(ValueError, match="cell_id cannot"):
        validate_cell_table(frame, nullable_columns=["cell_id", "value"])
    with pytest.raises(ValueError, match="Unknown nullable"):
        validate_cell_table(frame, nullable_columns=["missing"])
    with pytest.raises(ValueError, match="Unknown nullable"):
        table_schema_hash(frame, nullable_columns=["missing"])


def test_cell_id_nulls_are_rejected() -> None:
    frame = pl.DataFrame({"cell_id": pl.Series([None], dtype=pl.Int64)})
    with pytest.raises(ValueError, match="non-null"):
        validate_cell_table(frame)


def test_scope_normalization_dataframe_validation() -> None:
    assert normalize_cell_ids(
        pl.DataFrame({"cell_id": pl.Series([2, 1, 2], dtype=pl.Int64)})
    )["cell_id"].to_list() == [1, 2]
    with pytest.raises(ValueError, match="requires a cell_id"):
        normalize_cell_ids(pl.DataFrame({"other": [1]}))
    with pytest.raises(TypeError, match="Int64"):
        normalize_cell_ids(pl.Series("cell_id", [1], dtype=pl.Int32))
    with pytest.raises(ValueError, match="non-null"):
        normalize_cell_ids(pl.Series("cell_id", [None], dtype=pl.Int64))


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ({"": ["value"]}, "non-empty"),
        ({"position": []}, "requires column names"),
        ({"position": ["value", "value"]}, "duplicate columns"),
        ({"position": ["missing"]}, "missing columns"),
    ],
)
def test_semantic_roles_are_strict(roles: dict[str, list[str]], message: str) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": pl.Series([1], dtype=pl.Int64),
            "value": pl.Series([1.0], dtype=pl.Float64),
        }
    )
    with pytest.raises(ValueError, match=message):
        semantic_roles_json(frame, roles)
