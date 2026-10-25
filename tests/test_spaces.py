"""Named, declared spaces: define_space and the consumers that take a space name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import clipped_scaler_factory, identity_scaler_factory
from cellpax.diagnostics import block_weights
from cellpax.featuretable import FeatureTable


def _table(per_type: int = 30) -> FeatureTable:
    """Three types with a core and a periphery, plus a redundant feature block."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0], [6, 6, 0], [0, 0, 6]], dtype=float)
    blocks, is_core = [], []
    for center in centers:
        blocks.append(rng.normal(center, 0.4, (per_type, 3)))
        blocks.append(rng.normal(center, 1.1, (per_type, 3)))
        is_core += [True] * per_type + [False] * per_type
    coords = np.vstack(blocks)
    n = coords.shape[0]
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": coords[:, 0] * 50.0,  # off-scale, so scaling visibly matters
            "m1": coords[:, 1],
            "m2": coords[:, 2],
            "m2b": coords[:, 2] + rng.normal(0, 0.05, n),  # redundant with m2
            "is_core": is_core,
        }
    )
    ft = FeatureTable(df, features=["m0", "m1", "m2", "m2b"])
    ft.add_mask("core", pl.col("is_core"))
    ft.define_features("feat", columns=["m0", "m1", "m2", "m2b"])
    return ft


def _core_labels(ft: FeatureTable):
    ft.cluster(mask="core", n_neighbors=10, n_times=5, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", mask="core", distance_threshold=0.5, name="subclass")
    return labels.rename([f"C{i}" for i in labels.ids])


def test_named_space_matches_the_hand_built_one() -> None:
    ft = _table()
    ft.define_space("core_feat", "core", columns="feat", explained_variance=1.0)
    by_name = ft.fitted_space("core_feat")
    by_hand = ft.space("core", columns="feat", explained_variance=1.0)
    raw = ft.features("core", columns="feat")
    np.testing.assert_allclose(by_name.transform(raw), by_hand.transform(raw))
    assert by_name.mask == "core"


def test_space_name_defaults_the_population() -> None:
    ft = _table()
    ft.define_space("core_feat", "core", columns="feat")
    clus = ft.cluster(
        space="core_feat", n_neighbors=10, n_times=2, seed=0, n_jobs=1, name="c"
    )
    assert clus.mask == "core"
    labels = _core_labels(ft)
    ft.define_space("all_feat", columns="feat")
    named = ft.propagate_labels(labels, space="all_feat")
    explicit = ft.propagate_labels(labels, to="all", space=ft.fitted_space("all_feat"))
    assert named.labels.mask == "all"
    assert named.self_agreement() == explicit.self_agreement()


def test_explicit_population_still_checked_against_the_space() -> None:
    ft = _table()
    ft.define_space("core_feat", "core", columns="feat")
    labels = _core_labels(ft)
    with pytest.raises(ValueError, match="fit on mask 'core'"):
        ft.propagate_labels(labels, to="all", space="core_feat")


def test_space_scaler_is_independent_of_the_table() -> None:
    ft = _table()
    ft.set_scaler_factory(clipped_scaler_factory(mode="sigma", base="standard"))
    ft.define_space("raw", "core", columns="feat", scaler=identity_scaler_factory())
    ft.define_space("default", "core", columns="feat")
    raw = ft.features("core", columns="feat")
    identity = ft.fitted_space("raw")
    np.testing.assert_allclose(identity.scaler.transform(raw), raw)
    # the default space follows the table rule, which standardizes the off-scale m0
    sigma = ft.fitted_space("default").scaler.transform(raw)
    assert sigma[:, 0].std() < raw[:, 0].std() / 10
    # and the identity space doesn't move inside a temporary table rule
    with ft.using_scaler_factory(clipped_scaler_factory()):
        assert ft.fitted_space("raw") is identity


def test_string_weights_are_recomputed_for_the_space_and_boosted() -> None:
    ft = _table()
    ft.define_space(
        "w", "core", columns="feat", weights="redundancy", boost={"m1": 2.0}
    )
    expected = block_weights(
        ft.features("core", scaled=True, columns="feat"),
        ["m0", "m1", "m2", "m2b"],
        method="redundancy",
    ) * np.array([1.0, 2.0, 1.0, 1.0])
    np.testing.assert_allclose(ft.fitted_space("w").feature_weights, expected)
    with pytest.raises(ValueError, match="not in the space"):
        ft.define_space("bad", "core", columns="feat", boost={"nope": 2.0})


def test_redefining_the_mask_refits_the_space() -> None:
    ft = _table()
    ft.define_space("raw", "core", columns="feat", scaler=identity_scaler_factory())
    before = ft.fitted_space("raw")
    ft.add_mask("core", pl.col("cell_id") <= 40)
    after = ft.fitted_space("raw")
    assert after is not before
    assert not np.allclose(after.mean_, before.mean_)


def test_mask_backing_a_space_cannot_be_dropped() -> None:
    ft = _table()
    ft.define_space("core_feat", "core", columns="feat")
    with pytest.raises(ValueError, match="population of space"):
        ft.drop_mask("core")
    ft.drop_space("core_feat")
    ft.drop_mask("core")


def test_spaces_catalog_and_describe() -> None:
    ft = _table()
    ft.define_space(
        "raw", "core", columns="feat", scaler=identity_scaler_factory()
    ).define_space("w", columns="feat", weights="mfa")
    cat = ft.spaces
    assert cat["name"].to_list() == ["raw", "w"]
    assert cat.filter(pl.col("name") == "raw")["scaler"].item() == "identity"
    assert cat.filter(pl.col("name") == "w")["mask"].item() == "all"
    assert "spaces (2)" in ft.describe()


def test_space_declarations_round_trip(tmp_path: Path) -> None:
    ft = _table()
    ft.define_space(
        "raw",
        "core",
        columns="feat",
        scaler=clipped_scaler_factory(mode="sigma"),
        weights="redundancy",
        explained_variance=1.0,
    )
    ft.define_space(
        "frozen", "core", columns=["m0", "m1"], weights=lambda x, c: np.ones(len(c))
    )
    ft.save(tmp_path / "folio", "a")
    loaded = FeatureTable.load(tmp_path / "folio", "a")
    raw = ft.features("core", columns="feat")
    np.testing.assert_allclose(
        loaded.fitted_space("raw").transform(raw), ft.fitted_space("raw").transform(raw)
    )
    np.testing.assert_allclose(loaded.fitted_space("frozen").feature_weights, [1, 1])
    assert loaded.spaces["name"].to_list() == ["raw", "frozen"]


def test_set_validity_warns_when_replacing_a_domain() -> None:
    ft = _table()
    ft.add_mask("half", pl.col("cell_id") <= 90)
    ft.set_validity(["m0", "m1"], where="core")
    with pytest.warns(UserWarning, match="replaced the domain of 1 column"):
        ft.set_validity(["m1", "m2"], where="half")
    assert ft.validity_domains == {"m0": "core", "m1": "half", "m2": "half"}


def test_recovery_by_label_uses_reference_names() -> None:
    ft = _table()
    labels = _core_labels(ft)
    prop = ft.propagate_labels(labels, to="all", columns="feat")
    table = prop.recovery.by_label()
    assert set(table["label"]) == set(labels.names)
    n_labeled = labels.catalog()["n_cells"].sum()
    assert table["n"].sum() == n_labeled
    weighted = (table["recovered"] * table["n"]).sum() / table["n"].sum()
    assert np.isclose(weighted, prop.self_agreement())
    confusion = prop.recovery.confusion()
    assert confusion.drop("label").sum_horizontal().sum() == n_labeled


def test_flatten_labels_with_string_ids() -> None:
    from cellpax.labels import LabelSet

    df = pl.DataFrame(
        {
            "uid": [f"ds{i % 2}_{i}" for i in range(8)],
            "m0": np.arange(8, dtype=float),
            "ds": [f"ds{i % 2}" for i in range(8)],
        }
    )
    ft = FeatureTable(df, features=["m0"], id_column="uid")
    ft.add_mask("ds0", pl.col("ds") == "ds0")
    ft.add_mask("ds1", pl.col("ds") == "ds1")
    a = LabelSet.from_labels(
        ft.dataframe("ds0")["uid"].to_numpy(),
        ["A", "A", "B", "B"],
        name="x",
        mask="ds0",
    )
    b = LabelSet.from_labels(
        ft.dataframe("ds1")["uid"].to_numpy(),
        ["A", "B", "B", None],
        name="y",
        mask="ds1",
    )
    flat = ft.flatten_labels([a, b], name="sub")
    frame = flat.to_frame(id_column="uid")
    got = dict(zip(frame["uid"], frame["sub"]))
    assert got["ds0_0"] == "A" and got["ds1_1"] == "A" and got["ds1_7"] is None
    # the disjoint check keys on the ids too; it keeps same-named clusters apart
    assert len(a.combine(b, mode="disjoint").cell_ids) == 8
