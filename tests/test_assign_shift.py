"""The truncation-shift benchmark: whose coverage survives the periphery?

The setup every safeguard used to be silent about, now as a measured fixture:
features drift with reconstruction completeness, the curated core is biased
toward complete cells, and coverage is scored *per completeness tier*. Vanilla
conformal calibrates on the core and undercovers the truncated tier; the two
shift-aware candidates — in-library weighted conformal (Tibshirani) and
MAPIE's conditional conformal (Gibbs–Cherian–Candès) — are supposed to
restore it. These tests pin the qualitative ordering; the exact numbers feed
the overcomplete-then-prune decision.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.labels import LabelSet

ALPHA = 0.1


def _shifted_table(
    n: int = 900, *, drift: float = 3.0, seed: int = 0
) -> tuple[FeatureTable, LabelSet, np.ndarray, np.ndarray]:
    """Two classes whose features drift toward each other as completeness drops.

    ``completeness ~ U(0.2, 1)``; each cell's informative features move toward
    the between-class midpoint by ``drift * (1 - completeness)`` — graded
    truncation, the fake-continuum generator. The curated core is completeness-
    biased (mostly > 0.7) but keeps a thin tail down to ~0.3, so weighting has
    support to work with. Returns (table, core, truth, completeness).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    truth = np.array([0] * half + [1] * (n - half))
    completeness = rng.uniform(0.2, 1.0, n)
    centers = np.array([0.0, 6.0])
    midpoint = centers.mean()

    coords = rng.normal(0, 1.0, (n, 6))
    for i in (0, 1):
        rows = truth == i
        base = centers[i]
        pulled = base + (midpoint - base) * np.clip(
            drift * (1 - completeness[rows]) / 3.0, 0, 0.95
        )
        coords[rows, :3] += pulled[:, None]

    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(6)},
            "completeness": completeness,
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(6)])

    curated_probability = np.clip((completeness - 0.3) / 0.7, 0, 1) ** 2 * 0.8
    is_core = rng.uniform(size=n) < curated_probability
    core = LabelSet(
        ft._cell_ids(),
        np.where(is_core, truth, -1),
        names=["a", "b"],
        name="kind",
    )
    return ft, core, truth, completeness


def _tier_coverage(
    sets: np.ndarray,
    class_ids: np.ndarray,
    truth: np.ndarray,
    completeness: np.ndarray,
    evaluate: np.ndarray,
) -> dict[str, float]:
    """Coverage of the true label per completeness tier, on ``evaluate`` cells."""
    column_of = {int(k): j for j, k in enumerate(class_ids)}
    covered = np.array([sets[i, column_of[int(truth[i])]] for i in range(len(truth))])
    tiers = {
        "low": completeness < 0.45,
        "mid": (completeness >= 0.45) & (completeness < 0.7),
        "high": completeness >= 0.7,
    }
    return {
        tier: float(covered[members & evaluate].mean())
        for tier, members in tiers.items()
    }


def test_vanilla_conformal_undercovers_the_truncated_tier() -> None:
    ft, core, truth, completeness = _shifted_table()
    assignment = ft.assign(core, pca=False)
    evaluate = core.codes == -1  # held-out cells with known truth

    coverage = _tier_coverage(
        assignment.prediction_set(ALPHA),
        assignment.class_ids,
        truth,
        completeness,
        evaluate,
    )
    # near-nominal where calibration lives, catastrophic on the periphery:
    # truncated cells score worse than every calibration cell, so their sets
    # come back empty — undercoverage by silent abstention (measured: ~0.84
    # high tier, ~0.26 low tier at nominal 0.90)
    assert coverage["high"] > 0.80
    assert coverage["low"] < 0.50
    assert coverage["low"] < coverage["high"] - 0.3


def test_weighted_conformal_restores_coverage_on_the_periphery() -> None:
    ft, core, truth, completeness = _shifted_table()
    evaluate = core.codes == -1

    vanilla = ft.assign(core, pca=False)
    weighted = ft.assign(core, pca=False, shift_covariates=["completeness"])

    vanilla_coverage = _tier_coverage(
        vanilla.prediction_set(ALPHA), vanilla.class_ids, truth, completeness, evaluate
    )
    weighted_coverage = _tier_coverage(
        weighted.prediction_set(ALPHA),
        weighted.class_ids,
        truth,
        completeness,
        evaluate,
    )
    assert weighted_coverage["low"] > vanilla_coverage["low"] + 0.3
    assert weighted_coverage["low"] > 0.9
    # the price is paid in set size: where the classes have genuinely drifted
    # together, covering means conceding both labels (measured: mean set size
    # ~2.0 on the low tier vs vanilla's ~0.26 of mostly-empty sets)
    assert (
        weighted.set_sizes(ALPHA)[completeness < 0.45].mean()
        > vanilla.set_sizes(ALPHA)[completeness < 0.45].mean() + 1.0
    )


def test_conditional_conformal_restores_coverage_too() -> None:
    pytest.importorskip("mapie")
    pytest.importorskip("cvxpy")
    from sklearn.ensemble import RandomForestClassifier

    from cellpax.assign import conditional_prediction_set

    ft, core, truth, completeness = _shifted_table()
    evaluate = core.codes == -1
    data = ft.features(scaled=True)
    codes = core.codes
    is_reference = codes != -1

    rng = np.random.default_rng(0)
    calibration = np.zeros(len(codes), dtype=bool)
    for class_id in (0, 1):
        members = np.flatnonzero(codes == class_id)
        calibration[rng.choice(members, size=members.size // 4, replace=False)] = True
    train = is_reference & ~calibration

    classifier = RandomForestClassifier(random_state=0).fit(data[train], codes[train])
    sets = conditional_prediction_set(
        classifier,
        data[calibration],
        codes[calibration],
        data,
        covariates_calibration=completeness[calibration].reshape(-1, 1),
        covariates_test=completeness.reshape(-1, 1),
        alpha=ALPHA,
    )
    coverage = _tier_coverage(sets, classifier.classes_, truth, completeness, evaluate)
    assert coverage["low"] > 0.82
    assert coverage["high"] > 0.85


def test_weighted_matches_vanilla_when_there_is_no_shift() -> None:
    """Weights near 1 must not distort an exchangeable population."""
    rng = np.random.default_rng(5)
    n = 400
    truth = np.array([0] * (n // 2) + [1] * (n // 2))
    coords = rng.normal(0, 1.0, (n, 6))
    coords[truth == 1, :3] += 6.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(6)},
            "completeness": rng.uniform(0.2, 1.0, n),  # independent of curation
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(6)])
    core = LabelSet(
        ft._cell_ids(),
        np.where(rng.uniform(size=n) < 0.5, truth, -1),
        names=["a", "b"],
        name="kind",
    )
    vanilla = ft.assign(core, pca=False)
    weighted = ft.assign(core, pca=False, shift_covariates=["completeness"])
    evaluate = core.codes == -1

    for assignment in (vanilla, weighted):
        sets = assignment.prediction_set(ALPHA)
        column_of = {int(k): j for j, k in enumerate(assignment.class_ids)}
        covered = np.array([sets[i, column_of[int(truth[i])]] for i in range(n)])[
            evaluate
        ]
        assert covered.mean() > 0.85


def test_unknown_shift_covariates_raise() -> None:
    ft, core, _, _ = _shifted_table(n=200)
    with pytest.raises(KeyError, match="shift covariate"):
        ft.assign(core, shift_covariates=["nope"])
