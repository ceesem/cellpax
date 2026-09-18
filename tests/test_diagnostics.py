"""Pre-clustering diagnostics: redundant blocks, ties, and what clipping takes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import SortedMatrix
from cellpax.diagnostics import (
    clip_comparison,
    covariate_sensitivity,
    discriminative_features,
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


# -- information imbalance ---------------------------------------------------------


def test_identical_spaces_have_near_zero_imbalance_both_ways() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 3))
    a_to_b, b_to_a = information_imbalance(a, a.copy())
    assert a_to_b < 0.1
    assert b_to_a < 0.1


def test_independent_spaces_have_near_one_imbalance_both_ways() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 3))
    b = rng.normal(size=(300, 3))
    a_to_b, b_to_a = information_imbalance(a, b)
    assert a_to_b > 0.6
    assert b_to_a > 0.6


def test_a_space_predicts_its_own_shadow_better_than_the_reverse() -> None:
    """The asymmetry the statistic exists for. b is a's first coordinate: being near
    in the full 3-D space forces being near in that coordinate, so Delta(a -> b) is
    small; being near in one coordinate says little about the other two, so
    Delta(b -> a) is large. A small-then-large pair reads as 'a contains b'."""
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 3))
    b = a[:, :1]
    a_to_b, b_to_a = information_imbalance(a, b)
    assert a_to_b + 0.2 < b_to_a, "the richer space should predict its projection"
    assert a_to_b < 0.4


def test_a_one_dimensional_array_is_accepted_as_a_single_coordinate() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 3))
    flat = information_imbalance(a, a[:, 0])
    column = information_imbalance(a, a[:, :1])
    assert flat == column


def test_larger_k_still_reads_correctly_at_both_extremes() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 3))
    same_to, same_from = information_imbalance(a, a.copy(), k=5)
    assert same_to < 0.1 and same_from < 0.1
    other_to, other_from = information_imbalance(a, rng.normal(size=(300, 3)), k=5)
    assert other_to > 0.6 and other_from > 0.6


def test_imbalance_is_deterministic() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(120, 4))
    b = rng.normal(size=(120, 2))
    assert information_imbalance(a, b, k=3) == information_imbalance(a, b, k=3)


def test_imbalance_validates_row_alignment_cohort_size_and_k() -> None:
    from cellpax.diagnostics import information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 3))
    with pytest.raises(ValueError, match="row-aligned"):
        information_imbalance(a, rng.normal(size=(99, 3)))
    with pytest.raises(ValueError, match="fewer than 10"):
        information_imbalance(a[:5], a[:5])
    with pytest.raises(ValueError, match="k must be between"):
        information_imbalance(a, a, k=0)
    with pytest.raises(ValueError, match="k must be between"):
        information_imbalance(a, a, k=100)


def test_the_memory_guard_refuses_large_cohorts_and_suggests_subsampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real threshold is 20,000 cells; 20k^2 doubles is not something a test
    should allocate, so the constant is lowered instead of the data enlarged."""
    import cellpax.diagnostics as diagnostics
    from cellpax.diagnostics import feature_relevance, information_imbalance

    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    monkeypatch.setattr(diagnostics, "_MAX_CELLS", 30)
    with pytest.raises(ValueError, match="subsampl"):
        information_imbalance(a, a)
    with pytest.raises(ValueError, match="subsampl"):
        feature_relevance(a, ["f0", "f1", "f2"])


# -- feature relevance -------------------------------------------------------------


def _one_structural_feature(n: int = 300, seed: int = 0):
    """Two blobs separated along f0; f1 and f2 are low-amplitude noise.

    All the neighbourhood structure worth having lives in one feature. The native
    scales *are* the design — the blob separation dwarfs the noise — so the ranking
    tests pass ``standardize=False``; z-scoring would deliberately erase exactly this
    kind of dominance, which is what
    ``test_after_z_scoring_a_blob_bit_is_not_worth_more_than_a_continuous_coordinate``
    pins.
    """
    rng = np.random.default_rng(seed)
    blob = np.repeat([-4.0, 4.0], n // 2)
    f0 = blob + rng.normal(scale=0.5, size=n)
    noise = rng.normal(scale=0.4, size=(n, 2))
    return np.column_stack([f0, noise]), ["structure", "noise_a", "noise_b"]


def test_single_mode_ranks_the_only_structural_feature_first() -> None:
    from cellpax.diagnostics import feature_relevance

    data, names = _one_structural_feature()
    report = feature_relevance(data, names, mode="single", standardize=False)
    assert report.columns == ["feature", "delta_to_full", "delta_from_full"]
    assert report["feature"][0] == "structure"
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["structure"]["delta_to_full"] < 0.5
    assert by_name["noise_a"]["delta_to_full"] > 0.7
    assert by_name["noise_b"]["delta_to_full"] > 0.7


def test_drop_one_mode_says_removing_the_structural_feature_degrades_most() -> None:
    from cellpax.diagnostics import feature_relevance

    data, names = _one_structural_feature()
    report = feature_relevance(data, names, mode="drop_one", standardize=False)
    assert report.columns == ["feature", "delta_without"]
    assert report["feature"][0] == "structure"
    # sorted descending: most indispensable first
    deltas = report["delta_without"].to_list()
    assert deltas == sorted(deltas, reverse=True)
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["structure"]["delta_without"] > 0.6
    assert by_name["noise_a"]["delta_without"] < 0.4


def test_after_z_scoring_a_blob_bit_is_not_worth_more_than_a_continuous_coordinate() -> (
    None
):
    """The docstring's caveat, pinned: the imbalance sees *neighbourhood* information
    only. Z-scored, a balanced two-blob feature saturates at ~2 sigma of separation
    and resolves nothing within a blob, so its one bit of blob membership carries no
    more local information than a plain continuous noise coordinate — the single-mode
    deltas land in one band rather than the blob feature winning. Anyone expecting
    'the clustering feature must rank first' should read this test."""
    from cellpax.diagnostics import feature_relevance

    rng = np.random.default_rng(0)
    n = 300
    f0 = np.repeat([-4.0, 4.0], n // 2) + rng.normal(scale=0.5, size=n)
    data = np.column_stack([f0, rng.normal(size=(n, 2))])
    report = feature_relevance(
        data, ["structure", "noise_a", "noise_b"], mode="single", standardize=True
    )
    deltas = report["delta_to_full"].to_list()
    assert max(deltas) - min(deltas) < 0.2, "all three carry comparable local info"
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["structure"]["delta_to_full"] > 0.5, "blobs alone are not enough"


def test_a_duplicated_structural_feature_is_covered_for_on_drop_one() -> None:
    """The disagreement between the modes, pinned: a twin makes a feature look
    dispensable to drop-one even though it scores well alone."""
    from cellpax.diagnostics import feature_relevance

    rng = np.random.default_rng(0)
    n = 300
    unique = np.repeat([-4.0, 4.0], n // 2) + rng.normal(scale=0.5, size=n)
    twin = np.tile([-4.0, 4.0], n // 2) + rng.normal(scale=0.5, size=n)
    twin_copy = twin + rng.normal(scale=0.05, size=n)
    data = np.column_stack([unique, twin, twin_copy])
    names = ["unique", "twin_a", "twin_b"]

    report = feature_relevance(data, names, mode="drop_one")
    by_name = {row["feature"]: row for row in report.to_dicts()}
    assert by_name["unique"]["delta_without"] > by_name["twin_a"]["delta_without"]
    assert by_name["unique"]["delta_without"] > by_name["twin_b"]["delta_without"]


def test_standardize_strips_a_feature_of_a_purely_unit_borne_advantage() -> None:
    """The reason the default is on: a noise feature in big units dominates raw
    Euclidean distance and looks like it alone reproduces the space. Z-scoring takes
    exactly that advantage away and nothing else."""
    from cellpax.diagnostics import feature_relevance

    data, names = _one_structural_feature()
    data = data.copy()
    data[:, 1] *= 1000.0  # noise_a now out-scales everything

    raw = feature_relevance(data, names, mode="single", standardize=False)
    scaled = feature_relevance(data, names, mode="single", standardize=True)
    raw_by = {row["feature"]: row for row in raw.to_dicts()}
    scaled_by = {row["feature"]: row for row in scaled.to_dicts()}
    assert raw["feature"][0] == "noise_a", "raw distances follow scale, not structure"
    assert raw_by["noise_a"]["delta_to_full"] < 0.2
    assert scaled_by["noise_a"]["delta_to_full"] > 0.5, "the unit advantage is gone"


def test_feature_relevance_works_with_larger_k_and_is_deterministic() -> None:
    from cellpax.diagnostics import feature_relevance

    data, names = _one_structural_feature()
    first = feature_relevance(data, names, mode="single", k=4, standardize=False)
    second = feature_relevance(data, names, mode="single", k=4, standardize=False)
    assert first.equals(second)
    assert first["feature"][0] == "structure"


def test_feature_relevance_validates_mode_names_and_width() -> None:
    from cellpax.diagnostics import feature_relevance

    data, names = _one_structural_feature(n=50)
    with pytest.raises(ValueError, match="mode must be"):
        feature_relevance(data, names, mode="loo")
    with pytest.raises(TypeError, match="not the string"):
        feature_relevance(data, "analysis")
    with pytest.raises(ValueError, match="names were given"):
        feature_relevance(data, names[:-1])
    with pytest.raises(ValueError, match="at least two features"):
        feature_relevance(data[:, :1], names[:1])


# -- discriminative_features ----------------------------------------------------


def _blocked_fixture(seed: int = 0):
    """30 near-duplicate 'arbor' columns and 3 independent 'soma' ones.

    Both separate the clusters; the arbor block carries one measurement across
    thirty columns. A flat ranking hands every slot to it.
    """
    rng = np.random.default_rng(seed)
    n = 300
    codes = np.array([0] * n + [1] * n)
    signal = np.where(codes == 1, 2.0, 0.0) + rng.normal(0, 1, 2 * n)
    arbor = signal[:, None] + rng.normal(0, 0.15, (2 * n, 30))
    soma = np.where(codes == 1, 1.4, 0.0)[:, None] + rng.normal(0, 1, (2 * n, 3))
    noise = rng.normal(0, 1, (2 * n, 20))
    features = np.hstack([arbor, soma, noise])
    names = (
        [f"arbor_{i}" for i in range(30)]
        + [f"soma_{i}" for i in range(3)]
        + [f"n_{i}" for i in range(20)]
    )
    return features, codes, names


def test_flat_ranking_is_swamped_by_the_correlated_block():
    """The behaviour the block-aware default exists to avoid."""
    features, codes, names = _blocked_fixture()
    flat = discriminative_features(features, codes, names, per_block=None, top=12)
    assert all(f.startswith("arbor_") for f in flat["feature"])
    assert flat["block"].n_unique() == 1


def test_one_representative_per_block_surfaces_the_other_measurements():
    features, codes, names = _blocked_fixture()
    out = discriminative_features(features, codes, names, top=8)

    assert out["feature"].n_unique() == out.height
    assert out["block"].n_unique() == out.height  # one row per block
    assert sum(f.startswith("arbor_") for f in out["feature"]) == 1
    assert sum(f.startswith("soma_") for f in out["feature"]) == 3
    # the big block is reported as big, so the collapse is visible not hidden
    arbor = out.filter(pl.col("feature").str.starts_with("arbor_"))
    assert arbor["block_size"][0] == 30
    # real signal separates from noise by an order of magnitude
    assert out["f_stat"][3] > 10 * out["f_stat"][4]


def test_direction_columns_say_which_way():
    features, codes, names = _blocked_fixture()
    out = discriminative_features(
        features, codes, names, top=4, cluster_names={0: "A", 1: "B"}
    )
    real = out.filter(~pl.col("feature").str.starts_with("n_"))
    assert set(real["high"]) == {"B"}  # cluster 1 carries the signal
    assert (real["high_z"] > real["low_z"]).all()


def test_a_supplied_blocking_is_reused():
    features, codes, names = _blocked_fixture()
    blocks = feature_correlation(features, names, block_threshold=0.5)
    assert discriminative_features(features, codes, names, blocks=blocks, top=5).equals(
        discriminative_features(features, codes, names, top=5)
    )


def test_misaligned_inputs_are_refused():
    features, codes, names = _blocked_fixture()
    with pytest.raises(ValueError, match="row-aligned"):
        discriminative_features(features, codes[:10], names)
    with pytest.raises(ValueError, match="names"):
        discriminative_features(features, codes, names[:5])
    with pytest.raises(ValueError, match="at least two clusters"):
        discriminative_features(features, np.zeros(features.shape[0]), names)


def test_unassigned_cells_are_ignored():
    features, codes, names = _blocked_fixture()
    dropped = codes.copy()
    dropped[:50] = -1
    out = discriminative_features(features, dropped, names, top=3)
    assert set(out["high"]) <= {"0", "1"}
