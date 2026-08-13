"""Step 7: comparing clustering approaches across LabelSets."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.compare import compare, compare_many
from cellpax.labels import LabelSet


def test_contingency_and_alluvial() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="a")
    b = LabelSet([1, 2, 3, 4], [0, 1, 1, 1], name="b")
    cmp = compare(a, b)
    cont = cmp.contingency()
    assert set(cont.columns) == {"a", "a_id", "b", "b_id", "n"}
    assert int(cont["n"].sum()) == 4
    alluvial = cmp.alluvial_frame()
    assert set(alluvial.columns) == {
        "source",
        "source_id",
        "target",
        "target_id",
        "value",
    }


def test_contingency_normalize_reports_fractions_of_the_shared_cells() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="a")
    b = LabelSet([1, 2, 3, 4], [0, 1, 1, 1], name="b")
    cont = compare(a, b).contingency(normalize=True)
    assert cont["fraction"].sum() == pytest.approx(1.0)
    fractions = {(row["a_id"], row["b_id"]): row["fraction"] for row in cont.to_dicts()}
    assert fractions[(0, 0)] == pytest.approx(0.25)
    assert fractions[(0, 1)] == pytest.approx(0.25)
    assert fractions[(1, 1)] == pytest.approx(0.5)


def test_clusters_sharing_a_name_stay_apart_in_the_contingency() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="a").rename({0: "x", 1: "x"})
    b = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="b")
    cont = compare(a, b).contingency()
    assert cont.height == 2  # one row per id pair, not one silently merged blob
    assert set(cont["a_id"]) == {0, 1}
    assert cont["a"].to_list() == ["x (id 0)", "x (id 1)"]  # told apart on display

    alluvial = compare(a, b).alluvial_frame()
    assert alluvial["source"].n_unique() == 2  # separate flows in a sankey too


def test_agreement_identical_is_perfect() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="a")
    same = LabelSet(
        [1, 2, 3, 4], [5, 5, 9, 9], name="same"
    )  # relabeled but same partition
    scores = compare(a, same).agreement()
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["jaccard"] == pytest.approx(1.0)
    assert scores["n"] == 4
    assert scores["coverage"] == pytest.approx(1.0)

    disjoint = LabelSet([1, 2, 3, 4], [0, 1, 0, 1], name="d")
    assert compare(a, disjoint).agreement()["ari"] < 1.0


def test_agreement_reports_coverage_alongside_the_metrics() -> None:
    a = LabelSet([1, 2, 3, 4, 5], [0, 0, 1, 1, -1], name="a")
    b = LabelSet([1, 2, 3, 4, 5], [0, 0, 1, 1, 0], name="b")
    scores = compare(a, b).agreement()
    assert scores["ari"] == pytest.approx(1.0)  # perfect, over the co-assigned cells
    assert scores["n"] == 4  # ...which is only these
    assert scores["n_a_assigned"] == 4
    assert scores["n_b_assigned"] == 5
    assert scores["coverage"] == pytest.approx(4 / 5)


def test_unassigned_cells_appear_in_the_contingency_as_null_rows() -> None:
    a = LabelSet([1, 2, 3, 4, 5], [0, 0, 1, 1, -1], name="a")
    b = LabelSet([1, 2, 3, 4, 5], [0, 0, 1, 1, 0], name="b")
    cont = compare(a, b).contingency()
    null_rows = cont.filter(pl.col("a_id") == -1)
    assert null_rows["a"].to_list() == [None]  # unassigned reads as a null name
    assert null_rows["n"].to_list() == [1]
    assert cont["a_id"].to_list()[-1] == -1  # and sorts last, not first


def test_compare_aligns_on_shared_cells() -> None:
    a = LabelSet([1, 2, 3], [0, 0, 1], name="a")
    b = LabelSet([2, 3, 4], [0, 1, 1], name="b")  # overlaps on cells 2,3
    cmp = compare(a, b)
    assert int(cmp.contingency()["n"].sum()) == 2
    with pytest.raises(ValueError, match="share no cells"):
        compare(a, LabelSet([9, 10], [0, 1], name="x"))


def test_compare_many_matrix() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="run_a")
    b = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="run_b")
    c = LabelSet([1, 2, 3, 4], [0, 1, 0, 1], name="run_c")
    matrix = compare_many([a, b, c])
    assert matrix["label"].to_list() == ["run_a", "run_b", "run_c"]
    # diagonal is 1, a vs b identical
    assert matrix["run_a"][0] == pytest.approx(1.0)
    assert matrix["run_b"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="distinct names"):
        compare_many([a, a])


def test_compare_many_rejects_an_unknown_metric_naming_the_valid_ones() -> None:
    a = LabelSet([1, 2], [0, 1], name="run_a")
    b = LabelSet([1, 2], [0, 1], name="run_b")
    with pytest.raises(ValueError, match="'jaccard'"):
        compare_many([a, b], metric="rand")


def test_compare_many_n_diagonal_is_each_sets_own_assigned_count() -> None:
    a = LabelSet([1, 2, 3, 4], [0, 0, 1, -1], name="run_a")  # 3 assigned
    b = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="run_b")  # 4 assigned
    matrix = compare_many([a, b], metric="n")
    assert matrix["run_a"][0] == 3
    assert matrix["run_b"][1] == 4
    assert matrix["run_a"][1] == matrix["run_b"][0] == 3  # co-assigned off-diagonal


def test_ft_compare_convenience() -> None:
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.normal(0, 0.3, (30, 3)), rng.normal(8, 0.3, (30, 3))])
    from cellpax.featuretable import FeatureTable

    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 61), dtype=pl.Int64),
            "m0": coords[:, 0],
            "m1": coords[:, 1],
            "m2": coords[:, 2],
        }
    )
    ft = FeatureTable(df, features=["m0", "m1", "m2"])
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    loose = ft.label("run", distance_threshold=0.3, name="loose")
    tight = ft.label("run", distance_threshold=0.7, name="tight")
    scores = ft.compare(loose, tight).agreement()
    assert 0.0 <= scores["ari"] <= 1.0
