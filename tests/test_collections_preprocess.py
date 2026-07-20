"""Step 2: composable feature collections + the unified preprocess layer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureCollection, FeatureTable


def _table_with_metadata(n: int = 200) -> FeatureTable:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "axon_len": rng.lognormal(3, 1.2, n),  # wide, positive -> ihs
            "axon_tort": rng.normal(1.2, 0.1, n),  # tight -> leave
            "dend_vol": rng.lognormal(1, 1.8, n),  # wide, positive -> ihs
            "dend_bias": rng.normal(0, 2, n),  # symmetric, has negatives
        }
    )
    meta = pl.DataFrame(
        {
            "feature_id": ["axon_len", "axon_tort", "dend_vol", "dend_bias"],
            "family": ["axon", "axon", "dend", "dend"],
            "modality": ["morphology"] * 4,
        }
    )
    return FeatureTable(
        df, features=meta["feature_id"].to_list(), feature_metadata=meta
    )


def test_feature_collection_set_algebra() -> None:
    a = FeatureCollection("a", ("x", "y"))
    b = FeatureCollection("b", ("y", "z"))
    assert (a | b).columns == ("x", "y", "z")
    assert (a & b).columns == ("y",)
    assert (a - b).columns == ("x",)
    assert list(a) == ["x", "y"] and len(a) == 2


def test_define_features_by_family_predicate_and_columns() -> None:
    ft = _table_with_metadata()
    ft.define_features("axon", family="axon")
    ft.define_features("dend", family="dend")
    assert ft.collections["axon"].columns == ("axon_len", "axon_tort")
    combo = ft.collections["axon"] | ft.collections["dend"]
    assert set(combo.columns) == {"axon_len", "axon_tort", "dend_vol", "dend_bias"}

    ft.define_features("explicit", columns=["axon_len", "dend_vol"])
    assert ft.collections["explicit"].columns == ("axon_len", "dend_vol")
    ft.define_features("wide", predicate=pl.col("family") == "dend")
    assert set(ft.collections["wide"].columns) == {"dend_vol", "dend_bias"}

    assert "axon" in ft.collections
    with pytest.raises(ValueError, match="exactly one"):
        ft.define_features("bad", family="axon", columns=["axon_len"])
    with pytest.raises(ValueError, match="non-feature"):
        ft.define_features("bad", columns=["cell_id"])
    with pytest.raises(KeyError, match="Unknown collection"):
        ft.collections["missing"]


def test_scaling_with_a_collection_subset() -> None:
    ft = _table_with_metadata()
    ft.define_features("axon", family="axon")
    # only the axon features are scaled; dend columns stay raw
    df = ft.dataframe(columns="axon", scaled=True)
    assert abs(df["axon_len"].mean()) < 1e-9
    assert df["dend_vol"].to_list() == ft.dataframe()["dend_vol"].to_list()
    assert ft.features(columns="axon", scaled=True).shape == (200, 2)


def test_preprocess_skew_screen_ihs() -> None:
    ft = _table_with_metadata()
    ft.preprocess()  # ihs default, threshold 1.5

    t = ft.transforms
    assert t["axon_len"] == "ihs"
    assert t["dend_vol"] == "ihs"
    assert t["axon_tort"] is None  # tight
    assert t["dend_bias"] is None  # symmetric

    # scaled features now reflect the ihs transform before standardization
    raw = ft.features(columns=["axon_len"])[:, 0]
    scaled = ft.features(columns=["axon_len"], scaled=True)[:, 0]
    # monotone with arcsinh(raw), not with raw linearly -> check rank correlation is 1
    order_ihs = np.argsort(np.arcsinh(raw))
    order_scaled = np.argsort(scaled)
    assert np.array_equal(order_ihs, order_scaled)
    # standardized: about zero mean
    assert abs(scaled.mean()) < 1e-9


def test_preprocess_skips_log_on_negative_but_ihs_ok() -> None:
    ft = _table_with_metadata()
    # force a skewed negative-containing feature to test the log-skip path
    ft.preprocess(method="log")
    # dend_bias has negatives; even if skewed, log is skipped
    assert ft.transforms["dend_bias"] is None
    with pytest.raises(ValueError, match="method must be"):
        ft.preprocess(method="zscore")
