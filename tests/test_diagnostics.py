"""Pre-clustering diagnostics: redundant blocks, ties, and what clipping takes."""

from __future__ import annotations

import numpy as np
import pytest

from cellpax.clustering import SortedMatrix
from cellpax.diagnostics import (
    clip_comparison,
    covariate_sensitivity,
    duplicate_rows,
    feature_correlation,
    stratum_shift,
    tie_report,
)


def _with_redundant_block(n: int = 400, seed: int = 0):
    """Three independent features, a block of four that are one measurement, two counts.

    The shape of the real feature set: a depth-percentile family measuring one profile
    four ways, alongside low-cardinality integer counts.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    family = base[:, :1] + rng.normal(scale=0.05, size=(n, 4))
    counts = rng.integers(0, 5, size=(n, 2)).astype(float)
    data = np.column_stack([base, family, counts])
    names = [
        "indep_a",
        "indep_b",
        "indep_c",
        "depth_p10",
        "depth_p30",
        "depth_p50",
        "depth_p90",
        "n_branch",
        "n_bifurc",
    ]
    return data, names


# -- correlation structure -------------------------------------------------------


def test_a_redundant_family_lands_in_one_contiguous_block() -> None:
    """Which is the whole reason to order the matrix: unordered, it is invisible."""
    data, names = _with_redundant_block()
    sorted_matrix = feature_correlation(data, names)

    assert isinstance(sorted_matrix, SortedMatrix)
    assert sorted_matrix.matrix.shape == (len(names), len(names))
    assert sorted_matrix.sizes.sum() == len(names)
    assert int(sorted_matrix.sizes.max()) >= 4

    ordered = list(sorted_matrix.cell_ids)
    positions = [ordered.index(f"depth_p{p}") for p in (10, 30, 50, 90)]
    assert max(positions) - min(positions) <= 4, "the family should be contiguous"


def test_boundaries_and_codes_describe_the_blocks() -> None:
    data, names = _with_redundant_block()
    sorted_matrix = feature_correlation(data, names)
    np.testing.assert_array_equal(
        np.diff(sorted_matrix.boundaries), sorted_matrix.sizes
    )
    assert sorted_matrix.boundaries[0] == 0
    assert sorted_matrix.boundaries[-1] == len(names)
    assert len(sorted_matrix.names) == len(sorted_matrix.sizes)
    # codes change exactly at the boundaries
    changes = np.flatnonzero(np.diff(sorted_matrix.codes)) + 1
    np.testing.assert_array_equal(changes, sorted_matrix.boundaries[1:-1])


def test_block_threshold_controls_how_loosely_features_are_grouped() -> None:
    data, names = _with_redundant_block()
    loose = feature_correlation(data, names, block_threshold=0.9)
    tight = feature_correlation(data, names, block_threshold=0.1)
    assert len(loose.sizes) <= len(tight.sizes)


def test_spearman_is_available_and_a_constant_feature_is_tolerated() -> None:
    data, names = _with_redundant_block(n=100)
    data[:, 0] = 3.0  # constant: no correlation with anything, and a nan if unhandled
    sorted_matrix = feature_correlation(data, names, method="spearman")
    assert np.isfinite(sorted_matrix.matrix).all()


def test_correlation_input_is_validated() -> None:
    data, names = _with_redundant_block(n=50)
    with pytest.raises(ValueError, match="names were given"):
        feature_correlation(data, names[:-1])
    with pytest.raises(ValueError, match="at least two features"):
        feature_correlation(data[:, :1], names[:1])
    with pytest.raises(ValueError, match="method must be"):
        feature_correlation(data, names, method="kendall")


# -- ties ------------------------------------------------------------------------


def test_tie_report_singles_out_the_low_cardinality_features() -> None:
    data, names = _with_redundant_block()
    report = tie_report(data, names)
    assert report.height == len(names)
    # sorted by frac_distinct, so the integer counts come first
    assert set(report["feature"].head(2)) == {"n_branch", "n_bifurc"}
    assert report["n_distinct"].head(2).max() <= 5
    assert report.filter(report["feature"] == "indep_a")["frac_distinct"][0] == 1.0


def test_duplicate_rows_finds_none_across_continuous_features() -> None:
    """The expected answer on real morphometrics — worth pinning so it is not assumed."""
    data, _ = _with_redundant_block()
    report = duplicate_rows(data)
    assert report["n_duplicate_rows"][0] == 0
    assert report["max_tie_block"][0] == 1


def test_duplicate_rows_finds_collisions_among_counts_alone() -> None:
    """And it becomes real on a small subset of discrete features, which is the caveat."""
    data, names = _with_redundant_block()
    counts = data[:, [names.index("n_branch"), names.index("n_bifurc")]]
    report = duplicate_rows(counts)
    assert report["n_duplicate_rows"][0] > 0
    assert report["max_tie_block"][0] > 1
    assert report["n_unique_rows"][0] <= 25  # 5 x 5 possible coordinates


def test_tie_report_counts_extremes_from_the_finite_values_despite_nans() -> None:
    data = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, np.nan], [np.nan, 1.0]])
    report = tie_report(data, ["f0", "f1"])
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["f0"]["n_at_min"] == 1
    assert by_name["f0"]["n_at_max"] == 1
    assert by_name["f1"]["n_at_min"] == 3  # a constant column: min == max
    assert by_name["f1"]["n_at_max"] == 3


def test_tie_report_validates_its_names() -> None:
    data, names = _with_redundant_block(n=50)
    with pytest.raises(ValueError, match="names were given"):
        tie_report(data, names[:2])


# -- clipping --------------------------------------------------------------------


def _one_outlier(n: int, magnitude: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    column = rng.normal(size=(n, 1))
    column[0, 0] = magnitude
    return column


def test_percentile_clipping_scales_with_n_and_sigma_clipping_does_not() -> None:
    """The comparison the function exists for, as two numbers rather than an argument."""
    small = clip_comparison(_one_outlier(500), ["f"], n_sigma=5.0)
    large = clip_comparison(_one_outlier(20000), ["f"], n_sigma=5.0)

    assert large["n_clipped_percentile"][0] > 10 * small["n_clipped_percentile"][0]
    assert small["n_clipped_sigma"][0] == large["n_clipped_sigma"][0] == 1


def test_expected_percentile_cells_states_what_the_rule_takes_by_construction() -> None:
    report = clip_comparison(_one_outlier(20000), ["f"], lower=0.1)
    assert report["expected_percentile_cells"][0] == 20
    assert report["n_rows"][0] == 20000


def test_max_abs_sigma_says_whether_clipping_is_warranted_at_all() -> None:
    clean = clip_comparison(np.random.default_rng(0).normal(size=(500, 1)), ["f"])
    assert clean["max_abs_sigma"][0] < 5.0
    assert clean["n_clipped_sigma"][0] == 0
    # ...while the percentile rule clips regardless of there being anything extreme
    assert clean["n_clipped_percentile"][0] > 0


def test_the_gaussian_equivalent_of_n_sigma_is_reported() -> None:
    """So ``n_sigma=5`` in IQR units is not misread as five standard deviations."""
    report = clip_comparison(_one_outlier(500), ["f"], n_sigma=5.0)
    assert report["n_sigma_gaussian_equivalent"][0] == pytest.approx(6.745, rel=1e-3)


def test_clip_comparison_ignores_nans_and_names_the_features_carrying_them() -> None:
    data = np.column_stack([_one_outlier(500), _one_outlier(500)])
    data[5, 1] = np.nan  # one bad entry, not the outlier
    with pytest.warns(RuntimeWarning, match="f_nan"):
        report = clip_comparison(data, ["f_clean", "f_nan"], n_sigma=5.0)
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert np.isfinite(by_name["f_nan"]["max_abs_sigma"])
    assert by_name["f_nan"]["n_clipped_sigma"] == 1  # the outlier is still seen
    assert by_name["f_clean"]["n_clipped_sigma"] == 1


def test_clip_comparison_covers_every_feature() -> None:
    data, names = _with_redundant_block(n=600)
    report = clip_comparison(data, names)
    assert report.height == len(names)
    assert set(report["feature"]) == set(names)
    with pytest.raises(ValueError, match="names were given"):
        clip_comparison(data, names[:3])


# -- block weights ---------------------------------------------------------------


def test_block_weights_demote_a_block_and_leave_singletons_alone() -> None:
    from cellpax.diagnostics import block_weights

    data, names = _with_redundant_block()
    w = block_weights(data, names)

    assert w.shape == (len(names),)
    family = [names.index(f"depth_p{p}") for p in (10, 30, 50, 90)]
    lonely = [names.index("n_branch"), names.index("n_bifurc")]
    assert w[family].max() < 0.7, "a redundant family should be demoted"
    np.testing.assert_allclose(w[lonely], 1.0)


def test_mfa_is_adaptive_where_sqrt_k_is_not() -> None:
    """A loosely correlated block should be punished less than a tight one of equal size."""
    from cellpax.diagnostics import block_weights

    rng = np.random.default_rng(0)
    n = 600
    tight_base, loose_base = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    tight = tight_base + rng.normal(scale=0.02, size=(n, 4))  # r ~ 1
    loose = loose_base + rng.normal(scale=1.20, size=(n, 4))  # r ~ 0.4
    data = np.column_stack([tight, loose])
    names = [f"t{i}" for i in range(4)] + [f"l{i}" for i in range(4)]

    mfa = block_weights(data, names, method="mfa")
    sqrt_k = block_weights(data, names, method="sqrt_k")

    assert mfa[:4].mean() < mfa[4:].mean(), "tight block should be demoted harder"
    # sqrt_k cannot tell them apart when they land in equal-sized blocks
    assert len(set(np.round(sqrt_k[:4], 6))) == 1


def test_sqrt_k_matches_its_definition() -> None:
    from cellpax.diagnostics import block_weights

    data, names = _with_redundant_block()
    w = block_weights(data, names, method="sqrt_k")
    for value in np.unique(np.round(w, 9)):
        k = round(1.0 / value**2)
        assert np.isclose(value, 1.0 / np.sqrt(k))


def test_block_weights_validates_its_arguments() -> None:
    from cellpax.diagnostics import block_weights

    data, names = _with_redundant_block(n=100)
    with pytest.raises(ValueError, match="method must be"):
        block_weights(data, names, method="pca")
    with pytest.raises(TypeError, match="not the string"):
        block_weights(data, "analysis")


# -- the bare-string guard -------------------------------------------------------


def test_a_collection_name_is_rejected_rather_than_split_into_letters() -> None:
    """``list("analysis")`` is eight one-character names, and the width check only
    catches that when the widths happen to differ."""
    from cellpax.diagnostics import clip_comparison, feature_correlation, tie_report

    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 8))  # 8 columns, same as len("analysis")
    for fn in (feature_correlation, tie_report, clip_comparison):
        with pytest.raises(TypeError, match="not the string 'analysis'"):
            fn(data, "analysis")


# -- covariate sensitivity ---------------------------------------------------------


def _truncated_cohort(n: int = 400, seed: int = 0):
    """A completeness covariate, one feature that tracks it, one that is blind to it.

    The truncation situation: axon length is measurable on every cell but is really
    measuring how much axon the volume kept; soma size does not care.
    """
    rng = np.random.default_rng(seed)
    completeness = rng.uniform(0.2, 1.0, size=n)
    axon_length = 120.0 * completeness + rng.normal(scale=6.0, size=n)
    soma_size = rng.normal(loc=10.0, scale=2.0, size=n)
    data = np.column_stack([axon_length, soma_size])
    return data, completeness, ["axon_length", "soma_size"]


def test_a_truncation_tracking_feature_ranks_above_a_biological_one() -> None:
    data, completeness, names = _truncated_cohort()
    report = covariate_sensitivity(data, completeness, names)
    assert report["feature"][0] == "axon_length"
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["axon_length"]["spearman_rho"] > 0.9
    assert abs(by_name["soma_size"]["spearman_rho"]) < 0.2
    assert by_name["axon_length"]["p_value"] < 1e-6


def test_features_below_min_finite_get_null_rows_not_silently_dropped() -> None:
    data, completeness, names = _truncated_cohort()
    data[5:, 1] = np.nan  # soma_size keeps only five finite values
    report = covariate_sensitivity(data, completeness, names, min_finite=10)
    assert report.height == len(names), "sparse features stay in the frame"
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["soma_size"]["spearman_rho"] is None
    assert by_name["soma_size"]["p_value"] is None
    assert by_name["soma_size"]["n_finite"] == 5
    assert by_name["axon_length"]["spearman_rho"] is not None


def test_a_non_finite_covariate_warns_and_its_cells_are_left_out() -> None:
    data, completeness, names = _truncated_cohort()
    completeness[:3] = np.nan
    with pytest.warns(RuntimeWarning, match="3 non-finite"):
        report = covariate_sensitivity(data, completeness, names)
    n_finite = report.filter(report["feature"] == "axon_length")["n_finite"][0]
    assert n_finite == len(completeness) - 3


def test_feature_nans_reduce_n_finite_without_blinding_the_correlation() -> None:
    """No NaN blindness: the finite rows still carry the signal."""
    data, completeness, names = _truncated_cohort()
    data[::7, 0] = np.nan
    report = covariate_sensitivity(data, completeness, names)
    row = report.filter(report["feature"] == "axon_length").to_dicts()[0]
    assert row["n_finite"] == int(np.isfinite(data[:, 0]).sum())
    assert row["n_finite"] < len(completeness)
    assert row["spearman_rho"] > 0.9


def test_covariate_sensitivity_validates_shapes_and_names() -> None:
    data, completeness, names = _truncated_cohort(n=50)
    with pytest.raises(ValueError, match="one value per cell"):
        covariate_sensitivity(data, completeness[:-1], names)
    with pytest.raises(TypeError, match="not the string"):
        covariate_sensitivity(data, completeness, "analysis")


# -- stratum shift -----------------------------------------------------------------


def _strata_cohort(shift: float = 3.0, n_per: int = 200, seed: int = 0):
    """Three strata drawn identically, except 'c' is location-shifted in one feature.

    The portability situation: 'a' and 'b' could be one dataset; 'c' is the volume
    whose extraction ran differently.
    """
    rng = np.random.default_rng(seed)
    shifty = rng.normal(size=3 * n_per)
    stable = rng.normal(size=3 * n_per)
    strata = np.array(["a"] * n_per + ["b"] * n_per + ["c"] * n_per)
    shifty[strata == "c"] += shift
    data = np.column_stack([shifty, stable])
    return data, strata, ["shifty", "stable"]


def test_a_deliberately_shifted_stratum_tops_the_report() -> None:
    data, strata, names = _strata_cohort()
    report = stratum_shift(data, strata, names)
    assert (report["feature"][0], report["stratum"][0]) == ("shifty", "c")
    assert report["ks_statistic"][0] > 0.8
    assert report.height == len(names) * 3  # long form: every (feature, stratum) pair


def test_location_shift_reads_in_both_columns_and_an_identical_stratum_in_neither() -> (
    None
):
    """The pair of columns is the point: KS alone cannot tell 'recenterable' from
    'shape-level', and an identical stratum should read near zero in both."""
    data, strata, names = _strata_cohort(shift=2.0)
    report = stratum_shift(data, strata, names, reference="a")
    rows = {(r["feature"], r["stratum"]): r for r in report.to_dicts()}

    shifted = rows[("shifty", "c")]
    assert shifted["ks_statistic"] > 0.5
    assert abs(shifted["median_shift_iqr"]) > 1.0  # 2 sigma is ~1.5 IQR units

    identical = rows[("shifty", "b")]
    assert identical["ks_statistic"] < 0.2
    assert abs(identical["median_shift_iqr"]) < 0.3


def test_a_named_reference_stratum_gets_a_null_row_with_its_count() -> None:
    data, strata, names = _strata_cohort()
    report = stratum_shift(data, strata, names, reference="a")
    rows = {(r["feature"], r["stratum"]): r for r in report.to_dicts()}
    ref = rows[("shifty", "a")]
    assert ref["ks_statistic"] is None
    assert ref["p_value"] is None
    assert ref["median_shift_iqr"] is None
    assert ref["n_cells"] == 200


def test_small_strata_get_null_statistics_with_n_cells_still_reported() -> None:
    data, strata, names = _strata_cohort()
    strata = strata.copy()
    strata[:10] = "d"  # a ten-cell stratum, below min_cells=20
    report = stratum_shift(data, strata, names)
    rows = {(r["feature"], r["stratum"]): r for r in report.to_dicts()}
    small = rows[("shifty", "d")]
    assert small["ks_statistic"] is None
    assert small["n_cells"] == 10
    # ...and the surviving strata are unaffected
    assert rows[("shifty", "c")]["ks_statistic"] is not None


def test_a_zero_iqr_reference_nulls_the_shift_but_not_the_ks() -> None:
    data, strata, names = _strata_cohort()
    data[strata == "a", 0] = 5.0  # constant reference: IQR is 0
    report = stratum_shift(data, strata, names, reference="a")
    rows = {(r["feature"], r["stratum"]): r for r in report.to_dicts()}
    row = rows[("shifty", "c")]
    assert row["median_shift_iqr"] is None
    assert row["ks_statistic"] is not None


def test_stratum_nans_reduce_n_cells_without_nulling_the_statistics() -> None:
    """No NaN blindness here either: finite values within the stratum still compare."""
    data, strata, names = _strata_cohort()
    hit = np.flatnonzero(strata == "c")[:30]
    data[hit, 0] = np.nan
    report = stratum_shift(data, strata, names)
    rows = {(r["feature"], r["stratum"]): r for r in report.to_dicts()}
    row = rows[("shifty", "c")]
    assert row["n_cells"] == 170
    assert row["ks_statistic"] > 0.8


def test_stratum_shift_validates_reference_and_names() -> None:
    data, strata, names = _strata_cohort(n_per=50)
    with pytest.raises(ValueError, match="not one of the strata"):
        stratum_shift(data, strata, names, reference="z")
    with pytest.raises(TypeError, match="not the string"):
        stratum_shift(data, strata, "analysis")
