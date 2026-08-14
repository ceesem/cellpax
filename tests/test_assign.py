"""Conformal assignment: evidence once, alpha at read time, honesty per class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.labels import LabelSet


def _blob_table(
    n_per: tuple[int, ...] = (120, 120),
    *,
    separation: float = 8.0,
    core_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[FeatureTable, LabelSet, np.ndarray]:
    """Well-separated blobs with a curated core; returns (table, core, truth)."""
    rng = np.random.default_rng(seed)
    n = sum(n_per)
    truth = np.concatenate([np.full(k, i) for i, k in enumerate(n_per)])
    coords = rng.normal(0, 1.0, (n, 6))
    for i in range(len(n_per)):
        coords[truth == i, :3] += separation * i
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(6)},
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(6)])
    is_core = rng.uniform(size=n) < core_fraction
    codes = np.where(is_core, truth, -1)
    core = LabelSet(
        ft._cell_ids(),
        codes,
        names=[f"c{i}" for i in range(len(n_per))],
        name="kind",
    )
    return ft, core, truth


def test_sets_cover_the_truth_and_singletons_dominate_on_easy_data() -> None:
    ft, core, truth = _blob_table()
    assignment = ft.assign(core)

    sets = assignment.prediction_set(alpha=0.1)
    column_of = {int(k): j for j, k in enumerate(assignment.class_ids)}
    covered = np.array([sets[i, column_of[int(truth[i])]] for i in range(len(truth))])
    assert covered.mean() > 0.85  # 1 - alpha, minus finite-sample slack
    assert (assignment.set_sizes(alpha=0.1) == 1).mean() > 0.8  # easy data: decisive


def test_alpha_is_a_read_time_parameter_and_sets_are_nested() -> None:
    ft, core, _ = _blob_table()
    assignment = ft.assign(core)

    strict = assignment.prediction_set(alpha=0.02)  # higher confidence
    loose = assignment.prediction_set(alpha=0.3)
    assert (loose <= strict).all()  # higher confidence -> supersets
    assert strict.sum() >= loose.sum()


def test_to_labelset_abstains_on_ambiguity_instead_of_guessing() -> None:
    """Overlapping blobs: boundary cells get multi-label sets, hence -1."""
    ft, core, truth = _blob_table(separation=2.5, seed=1)
    assignment = ft.assign(core)

    labels = assignment.to_labelset(alpha=0.1)
    codes = labels.codes
    assigned = codes != -1
    assert 0 < assigned.sum() < len(codes)  # some abstentions on real overlap
    # among the cells it does assign, it is right far more often than chance
    assert (codes[assigned] == truth[assigned]).mean() > 0.9
    # names and identity come from the reference
    assert set(labels.names) <= {"c0", "c1"}


def test_mondrian_holds_coverage_for_the_rare_class_too() -> None:
    ft, core, truth = _blob_table((260, 40), separation=6.0, seed=2)
    assignment = ft.assign(core)

    report = assignment.coverage(alpha=0.2)
    assert report.height == 2
    assert report["coverage"].min() > 0.7  # both classes near 1 - alpha, not on average


def test_a_starved_class_is_named_when_alpha_outruns_its_calibration() -> None:
    ft, core, _ = _blob_table((200, 16), seed=3)
    assignment = ft.assign(core)
    # the small class has ~2 calibration cells; alpha=0.05 needs 19
    with pytest.warns(UserWarning, match="c1"):
        assignment.prediction_set(alpha=0.05)


def test_the_evidence_is_deterministic_and_params_are_recorded() -> None:
    ft1, core1, _ = _blob_table()
    ft2, core2, _ = _blob_table()
    a = ft1.assign(core1)
    b = ft2.assign(core2)
    assert np.allclose(a.p_values, b.p_values)  # derived seed, seeded smoothing
    assert a.params["mondrian"] == "class"
    assert isinstance(a.params["seed"], int)
    assert a.params["classifier"] == "RandomForestClassifier"


def test_frame_carries_evidence_and_optional_verdict() -> None:
    ft, core, _ = _blob_table()
    assignment = ft.assign(core)

    evidence = assignment.frame()
    assert "p_c0" in evidence.columns and "prob_c0" in evidence.columns
    assert "plausibility" in evidence.columns
    with_verdict = assignment.frame(alpha=0.1)
    assert "set_size" in with_verdict.columns
    assert with_verdict["assigned"].null_count() < with_verdict.height


def test_probabilities_ride_along_and_sum_to_one() -> None:
    ft, core, _ = _blob_table()
    assignment = ft.assign(core)
    probabilities = assignment.probabilities
    assert probabilities is not None
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_a_single_cell_class_is_excluded_with_a_warning() -> None:
    ft, core, _ = _blob_table()
    codes = core.codes
    keep_one = np.flatnonzero(codes == 1)[1:]
    codes[keep_one] = -1  # class 1 down to a single reference cell
    lone = LabelSet(core.cell_ids, codes, names=["c0", "c1"], name="kind")
    with pytest.warns(UserWarning, match="at least 2"):
        assignment = ft.assign(lone)
    assert list(assignment.class_ids) == [0]


def test_validity_domains_gate_the_guarantee_language() -> None:
    ft, core, _ = _blob_table()
    ft.add_mask("half", pl.col("cell_id") <= 120)
    ft.set_validity(columns=["m0"], where="half")
    with pytest.warns(UserWarning, match="exchangeability"):
        ft.assign(core)
    with pytest.raises(ValueError, match="exchangeability"):
        ft.assign(core, on_invalid="raise")


def test_reference_outside_the_target_mask_is_refused() -> None:
    ft, core, _ = _blob_table()
    ft.add_mask("late", pl.col("cell_id") > 60)
    with pytest.raises(ValueError, match="outside mask"):
        ft.assign(core, to="late")
