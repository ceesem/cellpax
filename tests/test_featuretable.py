"""Step 1: FeatureTable core — construction, columns, masks, scaled views."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable


def _table(n: int = 40) -> FeatureTable:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": rng.normal(0, 1, n),
            "m1": rng.normal(5, 2, n),
            "region": ["L"] * (n // 2) + ["R"] * (n - n // 2),
        }
    )
    return FeatureTable(df, features=["m0", "m1"])


def test_construction_and_accessors() -> None:
    ft = _table()
    assert ft.n_cells == 40
    assert ft.n_features == 2
    assert ft.feature_columns == ["m0", "m1"]
    assert ft.id_column == "cell_id"
    assert ft.masks == ["all"]
    assert "region" in ft.columns
    assert not any(c.startswith("_mask_") for c in ft.columns)


def test_construction_rejects_bad_input() -> None:
    good = pl.DataFrame({"cell_id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="non-null and unique"):
        FeatureTable(pl.DataFrame({"cell_id": [1, 1], "x": [1.0, 2.0]}), features=["x"])
    with pytest.raises(ValueError, match="not found"):
        FeatureTable(good, features=["missing"])
    with pytest.raises(ValueError, match="at least one feature"):
        FeatureTable(good, features=[])
    with pytest.raises(TypeError, match="must be numeric"):
        FeatureTable(pl.DataFrame({"cell_id": [1, 2], "s": ["a", "b"]}), features=["s"])
    with pytest.raises(ValueError, match="reserved"):
        FeatureTable(
            pl.DataFrame({"cell_id": [1], "x": [1.0], "_mask_all": [True]}),
            features=["x"],
        )


def test_add_column_writes_masked_subset() -> None:
    ft = _table(6)
    ft.add_mask("left", pl.col("region") == "L")
    ft.add_column([10, 11, 12], "score", mask="left", fill_value=-1)
    df = ft.dataframe()
    assert df.sort("cell_id")["score"].to_list() == [10, 11, 12, -1, -1, -1]
    # None fill yields nulls with an inferred float dtype
    ft.add_column([1.5, 2.5, 3.5], "val", mask="left")
    assert ft.dataframe()["val"].null_count() == 3
    with pytest.raises(ValueError, match="mask selects"):
        ft.add_column([1, 2], "bad", mask="left")


def test_masks_hierarchical_based_on() -> None:
    ft = _table(6)
    ft.add_mask("left", pl.col("region") == "L")  # cells 1-3
    ft.add_column([0, 0, 1], "grp", mask="left", fill_value=-1)
    ft.add_mask("left_g0", pl.col("grp") == 0, based_on="left")
    assert set(ft.masks) == {"all", "left", "left_g0"}
    # child is nested within parent
    assert ft.mask_series("left_g0").sum() == 2
    assert ft.dataframe("left_g0").height == 2
    with pytest.raises(ValueError, match="Invalid mask name"):
        ft.add_mask("all", pl.col("region") == "L")
    with pytest.raises(TypeError, match="boolean"):
        ft.add_mask("bad", pl.col("m0"))


def test_drop_mask_removes_column_and_invalidates_scaler() -> None:
    ft = _table(10)
    ft.add_mask("sub", pl.col("cell_id") <= 5)
    ft.features("sub", scaled=True)  # populate the scaler cache for "sub"
    assert ("sub", ("m0", "m1")) in ft._scaler_cache

    ft.drop_mask("sub")
    assert "sub" not in ft.masks
    assert not ft._scaler_cache  # cached scaler for the dropped mask is gone
    with pytest.raises(KeyError, match="Unknown mask"):
        ft.mask_series("sub")
    with pytest.raises(KeyError, match="Unknown mask"):
        ft.drop_mask("sub")
    with pytest.raises(ValueError, match="implicit"):
        ft.drop_mask("all")


def test_add_mask_redefinition_does_not_reuse_stale_scaler() -> None:
    # regression: redefining a mask must refit its scaler on the NEW subset,
    # not silently reuse one cached under the old definition
    from sklearn.preprocessing import StandardScaler

    ft = _table(10)
    ft.add_mask("sub", pl.col("cell_id") <= 5)
    ft.features("sub", scaled=True)  # populate the cache under the old definition

    ft.add_mask("sub", pl.col("cell_id") > 5)
    raw = ft.dataframe("sub").select("m0", "m1").to_numpy()
    expected = StandardScaler().fit_transform(raw)
    assert np.allclose(ft.features("sub", scaled=True), expected)


def test_dataframe_raw_vs_scaled_and_masking() -> None:
    ft = _table(40)
    ft.add_mask("right", pl.col("region") == "R")

    full = ft.dataframe()
    assert full.height == 40
    assert "region" in full.columns and "m0" in full.columns

    right = ft.dataframe("right")
    assert right.height == 20
    assert set(right["region"].to_list()) == {"R"}

    raw = ft.dataframe("right")
    scaled = ft.dataframe("right", scaled=True)
    # labels/metadata identical; feature columns change
    assert raw["region"].to_list() == scaled["region"].to_list()
    assert not np.allclose(raw["m0"].to_numpy(), scaled["m0"].to_numpy())
    # standard-scaled features on this mask are ~zero-mean, ~unit-std
    assert abs(scaled["m0"].mean()) < 1e-9
    assert scaled["m1"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_features_matrix_and_lazy_per_mask_scaler() -> None:
    ft = _table(40)
    ft.add_mask("left", pl.col("region") == "L")
    assert ft.features().shape == (40, 2)
    assert ft.features("left").shape == (20, 2)
    # per-mask scaling: the left mask's scaled matrix is centered on the left mask
    scaled_left = ft.features("left", scaled=True)
    assert np.allclose(scaled_left.mean(axis=0), 0.0, atol=1e-9)


def test_pandas_input_is_accepted() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"cell_id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    ft = FeatureTable(df, features=["x"])
    assert ft.n_cells == 3 and ft.feature_columns == ["x"]


def test_error_paths_and_repr() -> None:
    ft = _table(6)
    with pytest.raises(KeyError, match="Unknown mask"):
        ft.mask_series("nope")
    with pytest.raises(ValueError, match="cannot start with"):
        ft.add_column([1] * 6, "_mask_x")
    with pytest.raises(ValueError, match="id_column"):
        FeatureTable(pl.DataFrame({"x": [1.0]}), features=["x"], id_column="cell_id")
    assert "FeatureTable(" in repr(ft)


def test_add_mask_accepts_array_predicate() -> None:
    ft = _table(6)
    ft.add_mask("first_three", np.array([True, True, True, False, False, False]))
    assert ft.dataframe("first_three").height == 3
    with pytest.raises(ValueError, match="length must match"):
        ft.add_mask("bad", np.array([True, False]))


def test_fitted_scaler_transforms() -> None:
    from cellpax.featuretable import FittedScaler

    class _Identity:
        def transform(self, x):
            return x

    x = np.array([[0.0, 1.0, 4.0], [3.0, 8.0, 9.0]])
    fs = FittedScaler(scaler=_Identity(), transforms=["ihs", "log", "sqrt"])
    out = fs.transform(x)
    assert np.allclose(out[:, 0], np.arcsinh(x[:, 0]))  # ihs
    assert np.allclose(out[:, 2], np.sqrt(x[:, 2]))  # sqrt
    assert out[0, 1] < out[1, 1]  # log is monotone increasing
    with pytest.raises(ValueError, match="Unknown per-feature transform"):
        FittedScaler(scaler=_Identity(), transforms=["cube"]).transform(x[:, :1])
