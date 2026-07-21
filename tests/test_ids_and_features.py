"""Bringing in a cell_id via a mapping, and joining additional static features."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable


def _ossify_output(n: int = 12) -> pl.DataFrame:
    """An ossify-style extraction keyed on root_id, with no static cell_id."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "root_id": pl.Series(range(1000, 1000 + n), dtype=pl.Int64),
            "axon_len": rng.normal(5, 1, n),
            "dend_vol": rng.normal(50, 5, n),
        }
    )


def test_id_map_at_construction() -> None:
    df = _ossify_output(6)
    id_map = pl.DataFrame(
        {
            "root_id": pl.Series(range(1000, 1006), dtype=pl.Int64),
            "cell_id": pl.Series(range(1, 7), dtype=pl.Int64),
        }
    )
    ft = FeatureTable(df, features=["axon_len", "dend_vol"], id_map=id_map)
    assert ft.id_column == "cell_id"
    assert ft.n_cells == 6
    assert sorted(ft.dataframe()["cell_id"].to_list()) == list(range(1, 7))
    assert "root_id" in ft.columns  # the original key is kept as metadata


def test_id_map_errors() -> None:
    df = _ossify_output(4)
    with pytest.raises(ValueError, match="no 'cell_id' in id_map"):
        FeatureTable(
            df,
            features=["axon_len"],
            id_map=pl.DataFrame(
                {"root_id": pl.Series([1000, 1001], dtype=pl.Int64), "cell_id": [1, 2]}
            ),
        )
    with pytest.raises(ValueError, match="duplicate 'root_id'"):
        FeatureTable(
            df,
            features=["axon_len"],
            id_map=pl.DataFrame(
                {
                    "root_id": pl.Series([1000, 1000, 1001, 1002], dtype=pl.Int64),
                    "cell_id": [1, 2, 3, 4],
                }
            ),
        )


def test_set_id_column_after_construction() -> None:
    # build keyed on root_id, then re-key to cell_id via a mapping
    df = _ossify_output(6)
    ft = FeatureTable(df, features=["axon_len", "dend_vol"], id_column="root_id")
    assert ft.id_column == "root_id"
    id_map = pl.DataFrame(
        {
            "root_id": pl.Series(range(1000, 1006), dtype=pl.Int64),
            "cell_id": pl.Series(range(1, 7), dtype=pl.Int64),
        }
    )
    ft.set_id_column("cell_id", id_map=id_map)
    assert ft.id_column == "cell_id"
    ft.add_mask("first", pl.col("cell_id") <= 3)
    assert ft.dataframe("first").height == 3


def test_add_features_from_another_source() -> None:
    df = _ossify_output(6).with_columns(
        pl.Series("cell_id", range(1, 7), dtype=pl.Int64)
    )
    ft = FeatureTable(df, features=["axon_len", "dend_vol"])
    rng = np.random.default_rng(1)
    extra = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 7), dtype=pl.Int64),
            "syn_density": rng.normal(size=6),
            "input_count": rng.integers(0, 50, 6),
        }
    )
    meta = pl.DataFrame(
        {"feature_id": ["syn_density", "input_count"], "family": ["conn", "conn"]}
    )
    ft.add_features(extra, ["syn_density", "input_count"], feature_metadata=meta)
    assert set(ft.feature_columns) == {
        "axon_len",
        "dend_vol",
        "syn_density",
        "input_count",
    }
    ft.define_features("conn", family="conn")
    assert set(ft.collections["conn"].columns) == {"syn_density", "input_count"}
    # the new features participate in scaling / clustering
    assert ft.features(columns="conn", scaled=True).shape == (6, 2)


def test_add_features_defines_a_collection() -> None:
    df = _ossify_output(6).with_columns(
        pl.Series("cell_id", range(1, 7), dtype=pl.Int64)
    )
    ft = FeatureTable(df, features=["axon_len", "dend_vol"])
    rng = np.random.default_rng(2)
    extra = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 7), dtype=pl.Int64),
            "syn_density": rng.normal(size=6),
            "input_count": rng.normal(size=6),
        }
    )
    ft.add_features(extra, ["syn_density", "input_count"], collection="connectivity")
    assert ft.collections["connectivity"].columns == ("syn_density", "input_count")
    assert ft.features(columns="connectivity", scaled=True).shape == (6, 2)


def test_add_features_coverage_and_validation() -> None:
    df = _ossify_output(6).with_columns(
        pl.Series("cell_id", range(1, 7), dtype=pl.Int64)
    )
    ft = FeatureTable(df, features=["axon_len"])
    partial = pl.DataFrame(
        {"cell_id": pl.Series([1, 2, 3], dtype=pl.Int64), "extra": [0.1, 0.2, 0.3]}
    )
    with pytest.raises(ValueError, match="does not cover every cell"):
        ft.add_features(partial, ["extra"])
    # allow_missing keeps the uncovered cells as nulls
    ft.add_features(partial, ["extra"], allow_missing=True)
    assert ft.dataframe()["extra"].null_count() == 3
    # re-adding an existing feature is rejected
    clash = pl.DataFrame(
        {"cell_id": pl.Series(range(1, 7), dtype=pl.Int64), "axon_len": range(6)}
    )
    with pytest.raises(ValueError, match="already present"):
        ft.add_features(clash, ["axon_len"])
