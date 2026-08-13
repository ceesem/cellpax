"""``FittedSpace``: one PCA fit, truncation and whitening as views over it."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA

from cellpax.featuretable import FeatureTable
from cellpax.space import FittedSpace


def _redundant_block(n: int = 400, dim: int = 12, seed: int = 0):
    """Independent features plus a block of three that are one measurement three ways.

    The shape the ``alpha`` parameter exists for: soma volume / area / radius, or a
    family of depth percentiles — features that count once but contribute k times to a
    Euclidean distance.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, dim - 3))
    block = base[:, :1] * np.array([1.0, 2.0, 3.0]) + rng.normal(
        scale=0.01, size=(n, 3)
    )
    return np.column_stack([base, block])


def _columns(data: np.ndarray) -> list[str]:
    return [f"c{i}" for i in range(data.shape[1])]


def _fit(data: np.ndarray, **kwargs) -> FittedSpace:
    return FittedSpace.fit(data, columns=_columns(data), **kwargs)


# -- fitting and truncation ------------------------------------------------------


def test_keeps_every_component_and_truncates_as_a_view() -> None:
    data = _redundant_block()
    space = _fit(data)
    assert space.n_total_components == data.shape[1]
    assert space.n_components < space.n_total_components
    assert space.eigenvalues_.shape == (space.n_total_components,)
    assert space.transform_scaled(data).shape == (data.shape[0], space.n_components)


def test_component_count_matches_sklearn_float_semantics() -> None:
    """Switching a PCA(n_components=0.95) call over must not change the width."""
    data = _redundant_block()
    reference = PCA(n_components=0.95, svd_solver="full").fit(data)
    assert _fit(data).n_components == reference.n_components_


def test_explained_variance_may_be_an_integer_count() -> None:
    data = _redundant_block()
    assert _fit(data, explained_variance=5).n_components == 5
    with pytest.raises(ValueError, match="outside 1"):
        _fit(data, explained_variance=999)


def test_rejects_a_variance_target_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="explained_variance"):
        _fit(_redundant_block(), explained_variance=1.5)


# -- views share one fit ---------------------------------------------------------


def test_with_alpha_shares_the_fitted_arrays() -> None:
    """The guarantee a whitening sweep rests on: one PCA behind every alpha."""
    space = _fit(_redundant_block())
    for alpha in (0.25, 0.5, 0.75, 1.0):
        view = space.with_alpha(alpha)
        assert view.components_ is space.components_
        assert view.eigenvalues_ is space.eigenvalues_
        assert view.alpha == alpha


def test_with_components_shares_the_fitted_arrays() -> None:
    space = _fit(_redundant_block())
    view = space.with_components(4)
    assert view.components_ is space.components_
    assert view.n_components == 4
    with pytest.raises(ValueError, match="outside 1"):
        space.with_components(999)


def test_alpha_zero_is_bitwise_the_plain_projection() -> None:
    """alpha=0 must skip the weighting, not apply lambda**0 and pick up float noise."""
    data = _redundant_block()
    space = _fit(data)
    plain = (data - space.mean_) @ space.components_[: space.n_components].T
    np.testing.assert_array_equal(space.transform_scaled(data), plain)
    np.testing.assert_array_equal(space.with_alpha(0.0).transform_scaled(data), plain)


# -- what whitening actually does ------------------------------------------------


def test_full_whitening_equalises_component_variance() -> None:
    data = _redundant_block()
    space = _fit(data)
    unweighted = space.transform_scaled(data).var(axis=0, ddof=1)
    whitened = space.with_alpha(1.0).transform_scaled(data).var(axis=0, ddof=1)
    assert unweighted.max() / unweighted.min() > 5.0
    np.testing.assert_allclose(whitened, 1.0)


def test_partial_whitening_lands_between_the_two() -> None:
    data = _redundant_block()
    space = _fit(data)
    spread = [
        (lambda v: v.max() / v.min())(
            space.with_alpha(a).transform_scaled(data).var(axis=0, ddof=1)
        )
        for a in (0.0, 0.5, 1.0)
    ]
    assert spread[0] > spread[1] > spread[2]


def test_eigenvalue_floor_bounds_the_smallest_component_amplification() -> None:
    """Whitening amplifies the *smallest* retained component most; the floor caps that."""
    data = _redundant_block()
    space = _fit(data)
    unfloored = space.with_alpha(1.0).transform_scaled(data)
    floored = space.with_alpha(
        1.0, eigenvalue_floor=space.noise_floor
    ).transform_scaled(data)
    last = space.n_components - 1
    assert np.abs(floored[:, last]).max() < np.abs(unfloored[:, last]).max()
    # and it leaves the leading component, whose eigenvalue dwarfs the floor, alone
    np.testing.assert_allclose(floored[:, 0], unfloored[:, 0], rtol=1e-3)


def test_noise_floor_is_the_median_discarded_eigenvalue() -> None:
    space = _fit(_redundant_block())
    discarded = space.eigenvalues_[space.n_components :]
    assert space.noise_floor == pytest.approx(float(np.median(discarded)))


def test_weighting_a_degenerate_spectrum_raises_with_the_remedy() -> None:
    """A rank-deficient direction has a near-zero eigenvalue, not a zero one.

    Whitening it divides by the square root of round-off, amplifying noise by orders of
    magnitude — so the guard has to be a relative rank tolerance, not a check for
    non-positive values.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 2))
    data = np.column_stack([base, base, base])  # rank 2 in 6 columns
    space = _fit(data, explained_variance=6, alpha=1.0)
    assert space.eigenvalues_[-1] > 0  # not exactly zero, which is the point
    with pytest.raises(ValueError, match="eigenvalue_floor"):
        space.transform_scaled(data)
    # truncating to the true rank, or flooring, both make it well posed
    space.with_components(2).transform_scaled(data)
    _fit(data, explained_variance=6, alpha=1.0, eigenvalue_floor=1e-6).transform_scaled(
        data
    )


# -- diagnostics -----------------------------------------------------------------


def test_condition_numbers_are_reported_in_both_senses() -> None:
    space = _fit(_redundant_block())
    assert space.covariance_condition_number > space.condition_number
    assert space.condition_number == pytest.approx(
        np.sqrt(space.covariance_condition_number)
    )


def test_a_redundant_block_shows_up_as_a_large_condition_number() -> None:
    plain = _fit(np.random.default_rng(1).normal(size=(400, 12)))
    redundant = _fit(_redundant_block())
    assert redundant.condition_number > 10 * plain.condition_number


def test_spectrum_flags_a_leptokurtic_discarded_component() -> None:
    """A near-Gaussian discarded PC is noise; a sharply peaked one is a small group.

    Plants 12 cells offset along a deliberately low-variance direction, so the 0.95 cut
    discards it *because* few cells are involved — which is backwards if they are a real
    rare type, and is exactly what the kurtosis column exists to surface.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 8))
    data[:, 7] *= 0.01  # a low-variance direction the truncation will drop
    data[:12, 7] += 1.5  # ...carrying a small tight group
    space = _fit(data, explained_variance=0.95)

    spectrum = space.spectrum(data)
    assert spectrum.height == space.n_total_components
    assert set(spectrum.columns) >= {"excess_kurtosis", "n_tail_cells", "retained"}

    discarded = spectrum.filter(~pl.col("retained"))
    assert not discarded.is_empty()
    worst = discarded.sort("excess_kurtosis", descending=True).row(0, named=True)
    assert worst["excess_kurtosis"] > 5.0
    assert worst["n_tail_cells"] == 12


def test_spectrum_without_data_omits_the_kurtosis_columns() -> None:
    spectrum = _fit(_redundant_block()).spectrum()
    assert "excess_kurtosis" not in spectrum.columns
    assert spectrum["cumulative_ev"].to_numpy()[-1] == pytest.approx(1.0)


def test_label_records_the_representation() -> None:
    space = _fit(_redundant_block())
    assert space.label == "pca(0.95)"
    assert space.with_alpha(0.5).label == "pca(0.95, alpha=0.5)"
    assert "floor=" in space.with_alpha(0.5, eigenvalue_floor=0.25).label
    assert _fit(_redundant_block(), explained_variance=4).label == "pca(n=4)"


# -- transform on new rows -------------------------------------------------------


def test_transform_runs_the_frozen_scaler_and_checks_the_width() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(10.0, 3.0, size=(200, 6))
    ft = FeatureTable(
        pl.DataFrame(
            {
                "cell_id": pl.Series(range(1, 201), dtype=pl.Int64),
                **{f"m{i}": raw[:, i] for i in range(6)},
            }
        ),
        features=[f"m{i}" for i in range(6)],
    )
    space = ft.space()
    np.testing.assert_allclose(
        space.transform(raw), space.transform_scaled(ft.features(scaled=True))
    )
    with pytest.raises(ValueError, match="expected 6 features"):
        space.transform_scaled(np.zeros((3, 5)))


# -- serialization ---------------------------------------------------------------


def test_records_round_trip_without_pickling_an_estimator() -> None:
    data = _redundant_block()
    space = _fit(data, alpha=0.5)
    meta, arrays = space.to_records()
    restored = FittedSpace.from_records(meta, arrays)

    np.testing.assert_array_equal(restored.components_, space.components_)
    np.testing.assert_array_equal(restored.eigenvalues_, space.eigenvalues_)
    assert restored.n_components == space.n_components
    assert restored.alpha == space.alpha
    assert restored.label == space.label
    np.testing.assert_allclose(
        restored.transform_scaled(data), space.transform_scaled(data)
    )


def test_records_carry_the_scaler_parameters_not_the_object() -> None:
    from cellpax.clustering import make_clipped_scaler
    from cellpax.featuretable import FittedScaler

    rng = np.random.default_rng(0)
    raw = rng.normal(5.0, 2.0, size=(100, 4))
    scaler = FittedScaler(scaler=make_clipped_scaler(), transforms=[None] * 4)
    scaler.scaler.fit(raw)
    space = FittedSpace.fit(scaler.transform(raw), columns=_columns(raw), scaler=scaler)

    meta, arrays = space.to_records()
    assert meta["scaler"]["kind"] == "robust_percentile"
    assert "clipper__lower_bounds_" in arrays

    restored = FittedSpace.from_records(meta, arrays)
    np.testing.assert_allclose(restored.transform(raw), space.transform(raw))


def test_a_sigma_clip_contributes_no_fitted_bounds() -> None:
    """The reason to prefer it when freezing: nothing for a new dataset to shift."""
    from cellpax.clustering import make_clipped_scaler
    from cellpax.featuretable import FittedScaler

    rng = np.random.default_rng(0)
    raw = rng.normal(5.0, 2.0, size=(100, 4))
    scaler = FittedScaler(
        scaler=make_clipped_scaler(mode="sigma", n_sigma=4.0), transforms=[None] * 4
    )
    scaler.scaler.fit(raw)
    space = FittedSpace.fit(scaler.transform(raw), columns=_columns(raw), scaler=scaler)

    meta, arrays = space.to_records()
    assert meta["scaler"]["kind"] == "robust_sigma"
    assert meta["scaler"]["n_sigma"] == 4.0
    assert not any(key.startswith("clipper__") for key in arrays)
    np.testing.assert_allclose(
        FittedSpace.from_records(meta, arrays).transform(raw), space.transform(raw)
    )


def test_an_unfreezable_scaler_raises_rather_than_downgrading() -> None:
    from cellpax.featuretable import FittedScaler

    class _Odd:
        def transform(self, x):
            return x

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 3))
    space = FittedSpace.fit(
        raw,
        columns=_columns(raw),
        scaler=FittedScaler(scaler=_Odd(), transforms=[None] * 3),
    )
    with pytest.raises(TypeError, match="cannot freeze a scaler"):
        space.to_records()


# -- FeatureTable wiring ---------------------------------------------------------


def test_ft_space_caches_the_fit_and_not_the_weighting() -> None:
    """alpha is out of the cache key on purpose, so a sweep cannot refit the space."""
    data = _redundant_block(n=120, dim=10)
    ft = FeatureTable(
        pl.DataFrame(
            {
                "cell_id": pl.Series(range(1, 121), dtype=pl.Int64),
                **{f"m{i}": data[:, i] for i in range(10)},
            }
        ),
        features=[f"m{i}" for i in range(10)],
    )
    first = ft.space()
    assert ft.space() is first
    for alpha in (0.25, 0.5, 1.0):
        assert ft.space(alpha=alpha).components_ is first.components_


def test_ft_space_is_dropped_when_preprocessing_changes() -> None:
    data = _redundant_block(n=120, dim=10)
    ft = FeatureTable(
        pl.DataFrame(
            {
                "cell_id": pl.Series(range(1, 121), dtype=pl.Int64),
                **{f"m{i}": np.abs(data[:, i]) for i in range(10)},
            }
        ),
        features=[f"m{i}" for i in range(10)],
    )
    first = ft.space()
    ft.preprocess()
    assert ft.space() is not first


def test_features_pca_projects_through_the_space() -> None:
    data = _redundant_block(n=120, dim=10)
    ft = FeatureTable(
        pl.DataFrame(
            {
                "cell_id": pl.Series(range(1, 121), dtype=pl.Int64),
                **{f"m{i}": data[:, i] for i in range(10)},
            }
        ),
        features=[f"m{i}" for i in range(10)],
    )
    space = ft.space()
    np.testing.assert_array_equal(
        ft.features_pca(), space.transform_scaled(ft.features(scaled=True))
    )
    assert ft.features_pca(alpha=1.0).shape == ft.features_pca().shape
    with pytest.raises(ValueError, match="needs pca to be on"):
        ft.cluster(pca=False, alpha=0.5, n_neighbors=5, seed=0, n_jobs=1)


# -- feature weights -------------------------------------------------------------


def test_weights_are_applied_before_the_fit_and_by_transform() -> None:
    """mean_ and the rotation are fit on the weighted data, so transform must match."""
    data = _redundant_block()
    w = np.linspace(0.5, 2.0, data.shape[1])
    space = _fit(data, feature_weights=w)

    np.testing.assert_allclose(space.feature_weights, w)
    expected = (data * w - space.mean_) @ space.components_[: space.n_components].T
    np.testing.assert_allclose(space.transform_scaled(data), expected)


def test_weighting_changes_the_fit_not_just_the_projection() -> None:
    data = _redundant_block()
    plain = _fit(data)
    weighted = _fit(data, feature_weights=np.linspace(0.5, 2.0, data.shape[1]))
    assert not np.allclose(np.abs(plain.components_), np.abs(weighted.components_))


def test_weights_demote_a_redundant_block() -> None:
    """The point: eleven columns measuring one thing should not count eleven times."""
    from cellpax.diagnostics import block_weights

    data = _redundant_block()
    names = _columns(data)
    plain = _fit(data)
    weighted = _fit(data, feature_weights=block_weights(data, names))
    assert weighted.explained_variance_ratio_[0] < plain.explained_variance_ratio_[0]


def test_scores_are_weighted_too() -> None:
    data = _redundant_block()
    w = np.linspace(0.5, 2.0, data.shape[1])
    space = _fit(data, feature_weights=w)
    expected = (data * w - space.mean_) @ space.components_.T
    np.testing.assert_allclose(space.scores(data), expected)


def test_weights_must_match_the_feature_count() -> None:
    data = _redundant_block()
    with pytest.raises(ValueError, match="feature_weights has 3 entries"):
        _fit(data, feature_weights=np.ones(3))


def test_weights_survive_alpha_and_component_views() -> None:
    data = _redundant_block()
    w = np.linspace(0.5, 2.0, data.shape[1])
    space = _fit(data, feature_weights=w)
    for view in (space.with_alpha(0.5), space.with_components(3)):
        np.testing.assert_allclose(view.feature_weights, w)


def test_the_label_records_weighting() -> None:
    data = _redundant_block()
    w = np.ones(data.shape[1])
    assert _fit(data, feature_weights=w).label == "pca(0.95, weighted)"
    assert (
        _fit(data, feature_weights=w, alpha=0.5).label
        == "pca(0.95, weighted, alpha=0.5)"
    )
    assert _fit(data).label == "pca(0.95)"


def test_weights_round_trip_through_records() -> None:
    data = _redundant_block()
    w = np.linspace(0.5, 2.0, data.shape[1])
    space = _fit(data, feature_weights=w)
    meta, arrays = space.to_records()
    assert "feature_weights" in arrays

    restored = FittedSpace.from_records(meta, arrays)
    np.testing.assert_allclose(restored.feature_weights, w)
    np.testing.assert_allclose(
        restored.transform_scaled(data), space.transform_scaled(data)
    )


def test_an_unweighted_space_stores_no_weights() -> None:
    _meta, arrays = _fit(_redundant_block()).to_records()
    assert "feature_weights" not in arrays
