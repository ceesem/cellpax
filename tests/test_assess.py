"""Crossfit assessment: out-of-fold evidence for the labelling itself."""

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


def _own_label_p(assessment, core) -> np.ndarray:
    """Each reference cell's p-value for its own curated label."""
    truth = core.codes_for(assessment.cell_ids)
    column_of = {int(k): j for j, k in enumerate(assessment.class_ids)}
    is_reference = truth != -1
    rows = np.flatnonzero(is_reference)
    columns = np.array([column_of[int(c)] for c in truth[rows]])
    return assessment.p_values[rows, columns]


def test_reference_cells_are_judged_out_of_fold_not_flattered() -> None:
    """assign memorises its training cells; assess_labels must not.

    Overlapping blobs: a memorising classifier is confidently right about
    its training cells while honestly unsure about held-out ones, which is
    exactly the gap crossfitting removes.
    """
    ft, core, _ = _blob_table(separation=2.5, seed=1)
    assessment = ft.assess_labels(core)
    single_split = ft.assign(core)

    p_assess = _own_label_p(assessment, core)
    p_assign = _own_label_p(single_split, core)
    # a fairly judged genuine member's own-label p-value is a uniform rank:
    # mean near 0.5. The single split's training cells drag its mean up.
    assert 0.35 < p_assess.mean() < 0.65
    assert p_assign.mean() > p_assess.mean() + 0.1


def test_coverage_is_a_genuine_crossvalidated_check() -> None:
    ft, core, truth = _blob_table((260, 80), separation=6.0, seed=2)
    assessment = ft.assess_labels(core)

    report = assessment.coverage(alpha=0.2)
    assert report.height == 2
    # every reference cell's set is now out-of-fold, so this is honest
    assert report["coverage"].min() > 0.7


def test_all_cells_in_the_mask_are_covered_and_decisively() -> None:
    ft, core, truth = _blob_table()
    assessment = ft.assess_labels(core)

    assert assessment.p_values.shape[0] == len(truth)
    labels = assessment.to_labelset(alpha=0.1)
    is_core = core.codes_for(assessment.cell_ids) != -1
    outside = ~is_core
    codes = labels.codes[outside]
    assigned = codes != -1
    # easy data: the pooled fold evidence assigns most non-reference cells,
    # and correctly
    assert assigned.mean() > 0.8
    assert (codes[assigned] == truth[outside][assigned]).mean() > 0.9


def test_a_class_smaller_than_folds_is_excluded_with_a_warning() -> None:
    ft, core, _ = _blob_table((150, 4, 150), core_fraction=1.0, seed=3)
    with pytest.warns(UserWarning, match="class id 1 .* excluded"):
        assessment = ft.assess_labels(core, folds=5)
    assert list(assessment.class_ids) == [0, 2]


def test_fewer_than_two_folds_is_an_error() -> None:
    ft, core, _ = _blob_table()
    with pytest.raises(ValueError, match="folds"):
        ft.assess_labels(core, folds=1)


def test_the_evidence_is_deterministic_and_params_are_recorded() -> None:
    ft1, core1, _ = _blob_table()
    ft2, core2, _ = _blob_table()
    a = ft1.assess_labels(core1)
    b = ft2.assess_labels(core2)
    assert np.allclose(a.p_values, b.p_values)
    assert a.params["folds"] == 5
    assert a.params["mondrian"] == "class"
    assert isinstance(a.params["seed"], int)
    assert a.name == "kind_assess"


def test_probabilities_ride_along_and_frame_reads_as_usual() -> None:
    ft, core, _ = _blob_table()
    assessment = ft.assess_labels(core)
    frame = assessment.frame(alpha=0.1)
    assert "p_c0" in frame.columns and "prob_c0" in frame.columns
    assert "set_size" in frame.columns


def test_calibration_counts_are_full_class_counts() -> None:
    ft, core, _ = _blob_table()
    assessment = ft.assess_labels(core)
    truth = core.codes_for(assessment.cell_ids)
    for class_id, count in assessment.calibration_counts.items():
        assert count == int((truth == class_id).sum())


def test_mondrian_none_pools_the_calibration() -> None:
    ft, core, truth = _blob_table()
    assessment = ft.assess_labels(core, mondrian=None)
    sets = assessment.prediction_set(alpha=0.1)
    column_of = {int(k): j for j, k in enumerate(assessment.class_ids)}
    covered = np.array([sets[i, column_of[int(truth[i])]] for i in range(len(truth))])
    assert covered.mean() > 0.85
