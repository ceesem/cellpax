"""Gradient: a coordinate along a continuum instead of pretended modes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.gradient import Gradient, fit_principal_curve, twonn_dimension


def _arc(n: int = 300, noise: float = 0.05, seed: int = 0):
    """Noisy sine arch — curved enough that a straight axis is the wrong model."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, np.pi, n))
    points = np.column_stack([t, np.sin(t), np.zeros(n)]) + rng.normal(0, noise, (n, 3))
    return points, t


def _arc_table(n: int = 300, seed: int = 0) -> tuple[FeatureTable, np.ndarray]:
    points, t = _arc(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": points[:, 0],
            "m1": points[:, 1],
            # a third informative feature: per-feature standardization gives
            # every column unit variance, so a pure-noise column would be
            # inflated into a full-strength noise axis
            "m2": 0.5 * t + rng.normal(0, 0.05, n),
            "depth": t + rng.normal(0, 0.05, n),
            "completeness": rng.uniform(0, 1, n),  # independent of the axis
        }
    )
    ft = FeatureTable(df, features=["m0", "m1", "m2"])
    return ft, t


# -- the curve fit -----------------------------------------------------------------


def test_the_curve_recovers_the_generating_parameter() -> None:
    from scipy.stats import spearmanr

    points, t = _arc()
    coordinate, curve = fit_principal_curve(points)
    rho = abs(spearmanr(coordinate, t).statistic)
    assert rho > 0.98
    assert coordinate.min() == 0.0 and coordinate.max() == 1.0
    assert curve.shape[1] == 3


def test_the_curve_follows_the_bend_a_straight_axis_cannot() -> None:
    """On a three-quarter circle the first PC folds; arc length does not."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1.5 * np.pi, 400))
    points = np.column_stack([np.cos(t), np.sin(t), np.zeros(400)]) + rng.normal(
        0, 0.03, (400, 3)
    )
    coordinate, _ = fit_principal_curve(points)
    centered = points - points.mean(axis=0)
    pc1 = centered @ np.linalg.svd(centered, full_matrices=False)[2][0]

    curve_rho = abs(spearmanr(coordinate, t).statistic)
    pc_rho = abs(spearmanr(pc1, t).statistic)
    assert curve_rho > 0.99
    assert pc_rho < 0.95  # the fold is real
    assert curve_rho > pc_rho


def test_twonn_tells_a_curve_from_a_ball() -> None:
    rng = np.random.default_rng(0)
    points, _ = _arc(noise=0.02)
    ball = rng.normal(0, 1, (300, 3))
    assert twonn_dimension(points) < 1.8
    assert twonn_dimension(ball) > 2.4


# -- the Gradient object -----------------------------------------------------------


def test_bin_by_quantiles_gives_balanced_named_intervals() -> None:
    points, _ = _arc()
    coordinate, _ = fit_principal_curve(points)
    gradient = Gradient(np.arange(1, 301), coordinate, name="axis")

    labels = gradient.bin(3, names=["low", "mid", "high"])
    counts = labels.counts()
    assert labels.n_clusters == 3
    assert max(counts.values()) - min(counts.values()) <= 2  # quantile-balanced

    # default names declare the cuts instead of hiding them
    unnamed = gradient.bin(2)
    assert all("[" in n and "]" in n for n in unnamed.names)


def test_bin_with_explicit_edges_and_validation() -> None:
    gradient = Gradient(np.arange(5), np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    labels = gradient.bin([0.4, 0.8], names=["a", "b", "c"])
    assert labels.codes.tolist() == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError, match="ascending"):
        gradient.bin([0.8, 0.4])
    with pytest.raises(ValueError, match="expected 3 names"):
        gradient.bin([0.4, 0.8], names=["a"])


def test_coordinate_for_aligns_by_id_with_nan_off_manifold() -> None:
    gradient = Gradient(np.array([10, 20, 30]), np.array([0.0, 0.5, 1.0]))
    aligned = gradient.coordinate_for(np.array([30, 99, 10]))
    assert aligned[0] == 1.0 and aligned[2] == 0.0
    assert np.isnan(aligned[1])


# -- ft.parametrize ----------------------------------------------------------------


def test_parametrize_orients_records_and_attaches() -> None:
    from scipy.stats import spearmanr

    ft, t = _arc_table()
    gradient = ft.parametrize(orient_by="depth", nuisance=["completeness"], name="axis")

    rho = spearmanr(gradient.coordinate, t).statistic
    assert rho > 0.98  # oriented: positive, not just strong
    assert gradient.params["orient_by"] == "depth"
    assert gradient.intrinsic_dimension < 2.5

    loadings = gradient.loadings()
    assert loadings.columns == ["feature", "spearman_rho"]
    assert loadings["feature"][0] == "m0"  # the axis IS m0, up to the bend

    ft.attach(gradient)
    assert "axis" in ft.columns
    assert ft.dataframe()["axis"].null_count() == 0
    with pytest.raises(ValueError, match="overwrite=True"):
        ft.attach(gradient)
    ft.attach(gradient, overwrite=True)


def test_the_nuisance_tripwire_fires_on_a_completeness_gradient() -> None:
    """The trap this exists for: graded truncation masquerading as an axis."""
    ft, t = _arc_table()
    # completeness that *tracks* the manifold parameter — the artifact case
    ft.add_column(list(t + np.random.default_rng(2).normal(0, 0.1, len(t))), "frac")
    with pytest.warns(UserWarning, match="tracks nuisance column 'frac'"):
        ft.parametrize(nuisance=["frac"])


def test_the_dimension_gate_warns_on_a_ball() -> None:
    rng = np.random.default_rng(3)
    n = 200
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": rng.normal(0, 1, n) for i in range(5)},
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(5)])
    with pytest.warns(UserWarning, match="does not look like a curve"):
        ft.parametrize(pca=False)


def test_parametrize_restricted_to_named_clusters() -> None:
    from cellpax.labels import LabelSet

    ft, t = _arc_table()
    half = len(t) // 2
    kinds = LabelSet(
        ft._cell_ids(),
        np.array([0] * half + [1] * (len(t) - half)),
        names=["low", "high"],
        name="kind",
    )
    gradient = ft.parametrize(labels=kinds, clusters=["low"])
    assert len(gradient) == half
    with pytest.raises(ValueError, match="not in the label set"):
        ft.parametrize(labels=kinds, clusters=["nope"])
    with pytest.raises(ValueError, match="needs labels="):
        ft.parametrize(clusters=["low"])
