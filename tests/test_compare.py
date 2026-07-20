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
    assert set(cont.columns) == {"a", "b", "n"}
    assert int(cont["n"].sum()) == 4
    alluvial = cmp.alluvial_frame()
    assert set(alluvial.columns) == {"source", "target", "value"}


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

    disjoint = LabelSet([1, 2, 3, 4], [0, 1, 0, 1], name="d")
    assert compare(a, disjoint).agreement()["ari"] < 1.0


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
