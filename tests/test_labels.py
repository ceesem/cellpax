"""Step 4: LabelSet — clear labels, relabeling verbs, and ft.attach."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.labels import LabelSet


def test_labelset_construction_and_frame() -> None:
    ls = LabelSet([10, 20, 30, 40], [0, 0, 1, -1], name="subclass")
    assert ls.ids == [0, 1]
    assert ls.names == ["0", "1"]
    assert ls.counts() == {"0": 2, "1": 1}
    frame = ls.to_frame()
    assert frame.columns == ["cell_id", "subclass", "subclass_id"]
    assert frame["subclass"].to_list() == ["0", "0", "1", None]  # -1 -> null


def test_relabeling_verbs() -> None:
    ls = LabelSet([1, 2, 3, 4, 5], [0, 1, 2, 0, 1])
    ls.rename({0: "L2a", 1: "L2b", 2: "L3"})
    assert ls.names == ["L2a", "L2b", "L3"]
    ls.set_colors({"L2a": "#f00"})
    assert ls.label("L2a").color == "#f00"

    ls.merge(["L2a", "L2b"], into="L2")
    assert set(ls.names) == {"L2", "L3"}
    assert ls.counts()["L2"] == 4

    ls.reorder(["L3", "L2"])
    assert ls.names == ["L3", "L2"]  # renumbered 0,1 in that order
    with pytest.raises(ValueError, match="every cluster exactly once"):
        ls.reorder(["L3"])
    with pytest.raises(KeyError, match="Unknown label name"):
        ls.rename({"nope": "x"})
    with pytest.raises(ValueError, match="at least two"):
        ls.merge(["L2"], into="x")


def test_combine_disjoint_labelsets() -> None:
    a = LabelSet([1, 2], [0, 0], name="cls").rename({0: "exc"})
    b = LabelSet([3, 4], [0, 1], name="cls").rename({0: "inh_a", 1: "inh_b"})
    combined = a.combine(b)
    assert combined.to_frame().sort("cell_id")["cls"].to_list() == [
        "exc",
        "exc",
        "inh_a",
        "inh_b",
    ]
    with pytest.raises(ValueError, match="disjoint"):
        a.combine(LabelSet([2, 5], [0, 0]))


def _two_blobs(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 3)), rng.normal(8, 0.3, (n // 2, 3))]
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": coords[:, 0],
            "m1": coords[:, 1],
            "m2": coords[:, 2],
        }
    )
    return FeatureTable(df, features=["m0", "m1", "m2"])


def test_ft_label_and_attach_roundtrip() -> None:
    ft = _two_blobs(60)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    assert len(labels.ids) == 2
    labels.rename(dict(zip(labels.ids, ["A", "B"])))

    ft.attach(labels)
    df = ft.dataframe()
    assert "subclass" in df.columns
    assert set(df["subclass"].to_list()) == {"A", "B"}

    # a partial label set leaves other cells null
    partial = LabelSet([1, 2, 3], [0, 0, 0], name="partial").rename({0: "X"})
    ft.attach(partial)
    assert ft.dataframe()["partial"].null_count() == 57
