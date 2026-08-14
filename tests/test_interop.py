"""The AnnData bridge: to_anndata / from_anndata (see cellpax.interop)."""

from __future__ import annotations

import builtins
import sys

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.interop import from_anndata, to_anndata
from cellpax.labels import LabelSet

anndata = pytest.importorskip("anndata")


def _built_table(n: int = 40) -> FeatureTable:
    """Metadata, two masks, a collection, ihs preprocess, a pca embedding,
    an attached label, and a validity domain — one of everything the bridge
    has to carry."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, (n, 4))
    values[:, 3] = rng.exponential(2.0, n) ** 3  # heavy right tail -> ihs
    region = ["L"] * (n // 2) + ["R"] * (n - n // 2)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": values[:, i] for i in range(4)},
            "region": region,
            "depth": rng.uniform(0.0, 100.0, n),
        }
    )
    meta = pl.DataFrame(
        {
            "feature_id": [f"m{i}" for i in range(4)],
            "family": ["axon", "axon", "dend", "dend"],
        }
    )
    ft = FeatureTable(
        df, features=[f"m{i}" for i in range(4)], feature_metadata=meta, seed=7
    )
    ft.add_mask("left", pl.col("region") == "L")
    ft.add_mask("core", pl.col("depth") < 60)
    ft.define_features("axon", family="axon")
    ft.preprocess()
    ft.embed(method="pca", n_components=2)
    labels = LabelSet.from_labels(
        df["cell_id"].to_list(),
        ["L23IT" if r == "L" else "L5ET" for r in region],
        name="subclass",
    )
    ft.attach(labels)
    ft.set_validity(["m0", "m1"], where="core")
    return ft


# -- to_anndata: shapes, indexing, alignment ------------------------------------


def test_x_matches_features_and_obs_var_are_indexed_by_id() -> None:
    ft = _built_table()
    adata = to_anndata(ft)

    assert adata.shape == (40, 4)
    assert np.allclose(np.asarray(adata.X), ft.features())
    assert list(adata.var_names) == ft.feature_columns
    # obs is indexed by string ids, with the original id column kept as data
    assert list(adata.obs_names) == [str(i) for i in range(1, 41)]
    assert adata.obs["cell_id"].tolist() == list(range(1, 41))
    # metadata and attached labels ride along; feature columns do not
    assert adata.obs["region"].tolist() == ft.dataframe()["region"].to_list()
    assert "subclass" in adata.obs.columns and "subclass_id" in adata.obs.columns
    assert not set(ft.feature_columns) & set(adata.obs.columns)
    # var carries the family metadata, in X's column order
    assert adata.var.loc["m0", "family"] == "axon"
    assert adata.var.loc["m3", "family"] == "dend"


def test_masked_export_with_a_collection_narrows_x_and_says_so() -> None:
    ft = _built_table()
    adata = to_anndata(ft, "left", columns="axon")

    assert adata.shape == (20, 2)
    assert np.allclose(np.asarray(adata.X), ft.features("left", columns="axon"))
    assert list(adata.var_names) == ["m0", "m1"]
    assert adata.uns["cellpax"]["mask"] == "left"
    assert adata.uns["cellpax"]["features"] == ["m0", "m1"]


def test_obsm_rows_are_id_aligned_even_when_the_stored_frame_is_shuffled() -> None:
    ft = _built_table()
    expected = (
        ft.embedding(name="pca").sort("cell_id").select(["pca0", "pca1"]).to_numpy()
    )
    # stored embedding frames are id-keyed, not order-keyed: shuffle the rows
    # to prove the export aligns by cell id rather than trusting row order
    key = ("all", "pca")
    ft._embeddings[key] = ft._embeddings[key].sample(fraction=1.0, shuffle=True, seed=1)

    adata = to_anndata(ft)
    # mask order is cell_id 1..40 here, so id-sorted coordinates are X's order
    assert np.allclose(adata.obsm["X_pca"], expected)


def test_masks_export_as_boolean_obs_columns() -> None:
    ft = _built_table()
    adata = to_anndata(ft)

    assert adata.obs["mask_left"].tolist() == ft.mask_series("left").to_list()
    assert adata.obs["mask_core"].tolist() == ft.mask_series("core").to_list()
    assert "mask_all" not in adata.obs.columns

    # a masked export restricts the membership columns to the exported cells
    left = to_anndata(ft, "left")
    assert left.obs["mask_left"].all()
    member = ft.mask_series("left").to_numpy()
    assert (
        left.obs["mask_core"].tolist()
        == ft.mask_series("core").to_numpy()[member].tolist()
    )


def test_partial_coverage_embedding_warns_and_is_skipped() -> None:
    ft = _built_table()
    xy = np.column_stack([np.arange(20.0), np.arange(20.0)])
    ft.add_embedding(xy, "left", name="soma", space="anatomical")
    # broadening the mask strands the external coordinates: they survive the
    # redefinition (nothing derived them from the scaled features) but no
    # longer cover every cell the mask now holds
    ft.add_mask("left", (pl.col("region") == "L") | (pl.col("cell_id") <= 25))

    with pytest.warns(UserWarning, match="soma"):
        adata = to_anndata(ft, "left")
    assert adata.shape[0] == 25
    assert "X_soma" not in adata.obsm


def test_scaled_export_carries_the_scaled_matrix_and_records_it() -> None:
    ft = _built_table()
    adata = to_anndata(ft, scaled=True)

    assert np.allclose(np.asarray(adata.X), ft.features(scaled=True))
    assert adata.uns["cellpax"]["scaled"] is True
    # transforms resolved by preprocess travel too — the heavy-tailed feature
    # was ihs-transformed and the record says so
    assert adata.uns["cellpax"]["transforms"]["m3"] == "ihs"


# -- from_anndata: round trip and foreign objects --------------------------------


def test_round_trip_preserves_the_table() -> None:
    ft = _built_table()
    back = from_anndata(to_anndata(ft))

    assert back.n_cells == ft.n_cells
    assert back.feature_columns == ft.feature_columns
    assert np.allclose(back.features(), ft.features())
    # the original id column comes back with its original dtype
    assert back.id_column == "cell_id"
    assert back.dataframe()["cell_id"].to_list() == list(range(1, 41))
    # mask membership, consumed from the mask_* columns rather than kept as data
    assert set(back.masks) == set(ft.masks)
    for name in ft.masks:
        assert back.mask_series(name).to_list() == ft.mask_series(name).to_list()
    assert "mask_left" not in back.columns
    # transforms, validity, seed, metadata
    assert back.transforms == ft.transforms
    assert back.validity_domains == ft.validity_domains == {"m0": "core", "m1": "core"}
    assert back.seed == ft.seed == 7
    assert (
        back.dataframe()["subclass"].to_list() == ft.dataframe()["subclass"].to_list()
    )
    # embedding coordinates, re-registered under the stripped name
    assert ("all", "pca") in back.embeddings
    assert back.embedding(name="pca").equals(ft.embedding(name="pca"))


def test_round_trip_skips_a_validity_domain_whose_mask_did_not_survive() -> None:
    ft = _built_table()
    adata = to_anndata(ft)
    adata.obs = adata.obs.drop(columns=["mask_core"])

    with pytest.warns(UserWarning, match="core"):
        back = from_anndata(adata)
    assert back.validity_domains == {}
    assert "core" not in back.masks


def test_from_anndata_accepts_a_foreign_anndata() -> None:
    import pandas as pd

    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 3))
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"batch": ["a"] * 7 + ["b"] * 8}, index=[f"c{i}" for i in range(15)]
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.obsm["X_umap"] = rng.normal(size=(15, 2))

    ft = from_anndata(adata)
    assert ft.n_cells == 15
    assert ft.feature_columns == ["g0", "g1", "g2"]
    assert np.allclose(ft.features(), X)
    # no uns["cellpax"]: obs_names become string ids under the default name
    assert ft.id_column == "cell_id"
    assert ft.dataframe()["cell_id"].to_list() == [f"c{i}" for i in range(15)]
    assert ft.dataframe()["batch"].to_list() == ["a"] * 7 + ["b"] * 8
    assert ("all", "umap") in ft.embeddings
    # a working table, not a husk: the scaling machinery runs
    assert ft.features(scaled=True).shape == (15, 3)

    subset = from_anndata(adata, features=["g2", "g0"])
    assert subset.feature_columns == ["g2", "g0"]
    assert np.allclose(subset.features(), X[:, [2, 0]])


# -- the optional dependency ------------------------------------------------------


def test_missing_anndata_raises_naming_the_extra(monkeypatch) -> None:
    ft = _built_table()
    real_import = builtins.__import__

    def refuse_anndata(name, *args, **kwargs):
        if name == "anndata" or name.startswith("anndata."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_anndata)
    monkeypatch.delitem(sys.modules, "anndata", raising=False)

    with pytest.raises(ImportError, match=r"cellpax\[anndata\]"):
        to_anndata(ft)
    with pytest.raises(ImportError, match=r"cellpax\[anndata\]"):
        from_anndata(object())
