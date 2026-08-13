"""Validity domains: declaring where features inform, and what uses that."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.persist import load_feature_table


def _truncated_table(n: int = 120) -> FeatureTable:
    """Two types separated by every feature; axon features garbage off the core.

    Cells 1..n//2 are type 0, the rest type 1. ``complete`` marks the
    well-reconstructed core (first 3/4 of each type). For truncated cells the
    axon features are replaced with values drawn from the *other* type's
    distribution — measurable, plausible, and wrong, which is the failure mode
    validity domains exist for.
    """
    rng = np.random.default_rng(0)
    half = n // 2
    kind = np.array([0] * half + [1] * half)
    centers = np.array([0.0, 8.0])

    soma = rng.normal(centers[kind], 0.4)
    axon = rng.normal(centers[kind], 0.4)
    complete = np.array(
        ([True] * (3 * half // 4) + [False] * (half - 3 * half // 4)) * 2
    )
    axon[~complete] = rng.normal(centers[1 - kind[~complete]], 0.4)

    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "soma_vol": soma,
            "soma_area": soma + rng.normal(0, 0.05, n),
            "axon_len": axon,
            "axon_branches": axon + rng.normal(0, 0.05, n),
            "kind": kind,
            "complete": complete,
        }
    )
    ft = FeatureTable(
        df, features=["soma_vol", "soma_area", "axon_len", "axon_branches"]
    )
    ft.add_mask("complete", pl.col("complete"))
    ft.define_features("soma", columns=["soma_vol", "soma_area"])
    ft.define_features("axon", columns=["axon_len", "axon_branches"])
    ft.define_features("full", columns=ft.feature_columns)
    return ft


# -- the registry ---------------------------------------------------------------


def test_set_validity_records_and_clears_domains() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    assert ft.validity_domains == {
        "axon_len": "complete",
        "axon_branches": "complete",
    }
    ft.set_validity(columns=["axon_len"], where=None)
    assert ft.validity_domains == {"axon_branches": "complete"}


def test_set_validity_requires_an_existing_mask() -> None:
    ft = _truncated_table()
    with pytest.raises(KeyError, match="Unknown mask"):
        ft.set_validity(columns="axon", where="nope")
    with pytest.raises(ValueError, match="default domain"):
        ft.set_validity(columns="axon", where="all")


def test_define_features_can_declare_the_domain_in_one_call() -> None:
    ft = _truncated_table()
    ft.define_features(
        "axon2", columns=["axon_len", "axon_branches"], valid_where="complete"
    )
    assert ft.validity_domains["axon_len"] == "complete"


def test_validity_matrix_and_fully_valid_follow_the_domain() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    matrix = ft.validity()
    complete = ft.mask_series("complete").to_numpy()

    assert matrix.shape == (ft.n_cells, ft.n_features)
    assert matrix[:, :2].all()  # soma columns valid everywhere
    assert np.array_equal(matrix[:, 2], complete)

    assert ft.fully_valid(columns="soma").all()
    assert np.array_equal(ft.fully_valid(columns="full"), complete)


def test_validity_patterns_collapse_to_a_handful() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    patterns = ft.validity_patterns()

    assert patterns.height == 2
    assert patterns["n_invalid"].to_list() == [0, 2]  # largest pattern first
    assert patterns["invalid_features"][1].to_list() == [
        "axon_len",
        "axon_branches",
    ]


def test_a_mask_backing_a_domain_cannot_be_dropped() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    with pytest.raises(ValueError, match="validity domain"):
        ft.drop_mask("complete")
    ft.set_validity(columns="axon", where=None)
    ft.drop_mask("complete")  # now fine


def test_validity_survives_a_reload(tmp_path) -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    ft.save(tmp_path / "folio", "run")
    back = load_feature_table(tmp_path / "folio", "run")
    assert back.validity_domains == ft.validity_domains
    assert np.array_equal(
        back.fully_valid(columns="full"), ft.fully_valid(columns="full")
    )


# -- propagation integration ------------------------------------------------------


def _core_labels(ft: FeatureTable):
    from cellpax.labels import LabelSet

    frame = ft.dataframe("complete")
    return LabelSet(
        frame["cell_id"].to_numpy(),
        frame["kind"].to_numpy(),
        names=["a", "b"],
        name="kind",
    )


def test_propagating_on_invalid_columns_warns_and_can_raise() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    core = _core_labels(ft)

    with pytest.warns(UserWarning, match="outside the validity domain"):
        ft.propagate_labels(core, columns="full", n_neighbors=5)
    with pytest.raises(ValueError, match="outside the validity domain"):
        ft.propagate_labels(core, columns="full", n_neighbors=5, on_invalid="raise")

    # a valid collection and an explicit opt-out are both silent
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ft.propagate_labels(core, columns="soma", n_neighbors=5)
        ft.propagate_labels(core, columns="full", n_neighbors=5, on_invalid="ignore")


def test_ladder_gives_truncated_cells_their_valid_features() -> None:
    """The point of the ladder: full features where valid, safe ones elsewhere.

    Truncated cells carry *wrong-type* axon values, so a full-feature
    propagation labels many of them with the other type; the ladder lets them
    fall back to the soma features, which are honest everywhere.
    """
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    core = _core_labels(ft)
    truth = ft.dataframe()["kind"].to_numpy()

    result = ft.propagate_labels(
        core, ladder=["full", "soma"], n_neighbors=5, pca=False
    )
    codes = result.labels.codes_for(ft._cell_ids())
    assert (codes == truth).all()

    rungs = result.rungs
    complete = ft.mask_series("complete").to_numpy()
    assert result.rung_names == ["full", "soma"]
    assert (rungs[complete] == 0).all()  # core cells got the rich rung
    assert (rungs[~complete] == 1).all()  # truncated cells fell back
    assert set(result.frame()["kind_nn_rung"].unique().to_list()) == {"full", "soma"}


def test_full_feature_propagation_mislabels_what_the_ladder_saves() -> None:
    """The negative control: without the ladder, invalid features do damage."""
    ft = _truncated_table()
    core = _core_labels(ft)
    truth = ft.dataframe()["kind"].to_numpy()

    result = ft.propagate_labels(core, columns="full", n_neighbors=5, pca=False)
    codes = result.labels.codes_for(ft._cell_ids())
    complete = ft.mask_series("complete").to_numpy()
    assert (codes[~complete] != truth[~complete]).mean() > 0.3


def test_ladder_records_per_rung_recovery_and_params() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    core = _core_labels(ft)
    result = ft.propagate_labels(core, ladder=["full", "soma"], n_neighbors=5)

    assert set(result.rung_recovery) == {"full", "soma"}
    assert result.rung_recovery["soma"].agreement > 0.9
    assert result.params["ladder_names"] == ["full", "soma"]
    assert result.space == "ladder(full, soma)"


def test_a_cell_no_rung_covers_stays_unassigned() -> None:
    ft = _truncated_table()
    ft.set_validity(columns="axon", where="complete")
    # soma features also restricted, so truncated cells have no valid rung
    ft.set_validity(columns="soma", where="complete")
    core = _core_labels(ft)

    result = ft.propagate_labels(core, ladder=["full", "soma"], n_neighbors=5)
    complete = ft.mask_series("complete").to_numpy()
    codes = result.labels.codes_for(ft._cell_ids())
    assert (codes[~complete] == -1).all()
    assert (result.rungs[~complete] == -1).all()


def test_a_rung_without_valid_reference_cells_is_skipped_with_a_warning() -> None:
    ft = _truncated_table()
    ft.add_mask("nowhere", pl.col("cell_id") < 0)
    ft.set_validity(columns="axon", where="nowhere")  # full rung unusable
    core = _core_labels(ft)

    with pytest.warns(UserWarning, match="fewer than two valid reference"):
        result = ft.propagate_labels(core, ladder=["full", "soma"], n_neighbors=5)
    assert result.rung_recovery["full"] is None
    assert (result.rungs[result.rungs >= 0] == 1).all()


# -- score_cells ------------------------------------------------------------------


def test_score_cells_stores_a_column_and_is_deterministic() -> None:
    ft = _truncated_table()
    frame = ft.score_cells(columns="soma", name="iso")

    assert frame.columns == ["cell_id", "iso"]
    assert "iso" in ft.columns
    # deterministic without a seed argument: same call, same scores
    again = ft.score_cells(columns="soma", name="iso")
    assert np.allclose(frame["iso"].to_numpy(), again["iso"].to_numpy())


def test_score_cells_flags_an_extreme_cell() -> None:
    ft = _truncated_table()
    df = ft.dataframe()
    values = df["soma_vol"].to_numpy().copy()
    values[0] = 400.0
    ft.add_column(values, "soma_vol", overwrite=True)

    scores = ft.score_cells(columns="soma", name="iso")
    assert scores["iso"].to_numpy().argmin() == 0


def test_score_cells_is_mask_scoped_and_supports_lof() -> None:
    from sklearn.neighbors import LocalOutlierFactor

    ft = _truncated_table()
    frame = ft.score_cells(
        "complete", scorer=LocalOutlierFactor(n_neighbors=10), name="lof"
    )
    assert frame.height == int(ft.mask_series("complete").sum())
    # off-mask cells hold null, on-mask cells hold the score
    table = ft.dataframe()
    assert table["lof"].null_count() == ft.n_cells - frame.height


def test_score_cells_rejects_a_scoreless_estimator() -> None:
    class _NoScore:
        def fit(self, x):
            return self

    ft = _truncated_table()
    with pytest.raises(TypeError, match="score_samples"):
        ft.score_cells(scorer=_NoScore(), name="bad")
