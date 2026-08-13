"""Step 6: first-class DataFolio save/load (many-per-folio, namespaced)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import make_clipped_scaler
from cellpax.featuretable import FeatureTable
from cellpax.persist import list_analyses


def _built_table(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 4)), rng.normal(8, 0.3, (n // 2, 4))]
    )
    meta = pl.DataFrame(
        {
            "feature_id": [f"m{i}" for i in range(4)],
            "family": ["axon", "axon", "dend", "dend"],
        }
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
            "region": ["L"] * (n // 2) + ["R"] * (n - n // 2),
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(4)], feature_metadata=meta)
    ft.add_mask("left", pl.col("region") == "L")
    ft.define_features("axon", family="axon")
    ft.preprocess()
    ft.embed(method="pca", n_components=2)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    labels.rename([f"C{i}" for i in labels.ids])
    labels.set_colors({labels.names[0]: "#1f77b4"})
    labels.set_descriptions({labels.names[0]: "the shallow one"})
    ft.attach(labels)
    return ft


def _assert_round_trip(a: FeatureTable, b: FeatureTable) -> None:
    assert b.feature_columns == a.feature_columns
    assert b.id_column == a.id_column
    assert set(b.masks) == set(a.masks)
    assert b.transforms == a.transforms
    assert set(b.collections) == set(a.collections)
    assert b.collections["axon"].columns == a.collections["axon"].columns
    # full table (features, metadata, masks, attached labels) preserved
    assert b.dataframe().sort("cell_id").equals(a.dataframe().sort("cell_id"))
    # scaled features reproduce (transforms + scaler restored)
    assert np.allclose(b.features(scaled=True), a.features(scaled=True))
    # embedding + clustering restored
    assert b.embedding(name="pca").equals(a.embedding(name="pca"))
    la = a.label("run", distance_threshold=0.5, name="subclass")
    lb = b.label("run", distance_threshold=0.5, name="subclass")
    assert np.array_equal(
        np.sort(lb.to_frame()["subclass_id"].to_numpy()),
        np.sort(la.to_frame()["subclass_id"].to_numpy()),
    )
    # attached label identities a column can't hold: colors, descriptions, mask
    before, after = a.labelset("subclass"), b.labelset("subclass")
    assert after.catalog().equals(before.catalog())
    assert after.color_map() == before.color_map() != {}
    assert after.mask == before.mask


def test_save_load_round_trip(tmp_path: Path) -> None:
    ft = _built_table()
    folio = tmp_path / "study_folio"
    ft.save(folio, "l23it")
    reloaded = FeatureTable.load(folio, "l23it")
    _assert_round_trip(ft, reloaded)


def test_many_analyses_and_user_content_coexist(tmp_path: Path) -> None:
    from datafolio import DataFolio

    folio = DataFolio(tmp_path / "shared")
    # user-added content in its own namespace
    folio.add("notes/readme", {"author": "casey", "note": "L2/3 IT analysis"})

    _built_table().save(folio, "analysis_a")
    _built_table(40).save(folio, "analysis_b")

    assert list_analyses(folio) == ["analysis_a", "analysis_b"]
    # user content is untouched and not mistaken for an analysis
    assert folio.get("notes/readme")["author"] == "casey"

    a = FeatureTable.load(folio, "analysis_a")
    b = FeatureTable.load(folio, "analysis_b")
    assert a.n_cells == 60 and b.n_cells == 40


def test_save_writes_item_descriptions(tmp_path: Path) -> None:
    from datafolio import DataFolio

    ft = _built_table()
    folio_path = tmp_path / "described"
    ft.save(folio_path, "l23it")

    folio = DataFolio(folio_path, allow_existing=True)
    table_desc = folio.data["l23it/table"].description
    assert "60 cells" in table_desc and "4 features" in table_desc

    embedding_desc = folio.data["l23it/embedding/all__pca"].description
    assert "pca" in embedding_desc and "2D" in embedding_desc

    # a clustering with runs stores the runs, not the matrix they imply
    partitions_desc = folio.data["l23it/partitions/run"].description
    assert "60 cells" in partitions_desc and "Leiden runs" in partitions_desc

    settings_desc = folio.data["l23it/settings/run"].description
    assert "resolution" in settings_desc

    manifest_desc = folio.data["l23it/manifest"].description
    assert "manifest" in manifest_desc.lower()


def test_clipped_scaler_tag_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 31), dtype=pl.Int64),
            "x": rng.normal(size=30),
            "y": rng.normal(size=30),
        }
    )
    ft = FeatureTable(df, features=["x", "y"], scaler_factory=make_clipped_scaler)
    ft.save(tmp_path / "f", "clip")
    reloaded = FeatureTable.load(tmp_path / "f", "clip")
    assert np.allclose(reloaded.features(scaled=True), ft.features(scaled=True))


# -- storing the runs rather than the matrix they imply -------------------------


def _run_table(n: int = 60, dim: int = 8) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1.0, (n, dim))
    coords[n // 2 :, :3] += 8.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


def test_partitions_are_stored_and_the_consensus_derived_on_load(tmp_path) -> None:
    from cellpax import load_feature_table, save_feature_table

    ft = _run_table()
    original = ft.cluster(
        graph_type=["knn", "umap_fuzzy"],
        n_neighbors=15,
        resolution=[0.3, 1.0],
        n_times=2,
        seed=0,
        n_jobs=1,
        name="run",
    )
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    restored = load_feature_table(path, name="t").clustering("run")
    np.testing.assert_array_equal(
        restored.partitions.labels, original.partitions.labels
    )
    np.testing.assert_array_equal(
        restored.partitions.graph_type, original.partitions.graph_type
    )
    np.testing.assert_allclose(
        restored.similarity_matrix.toarray(),
        original.similarity_matrix.toarray(),
        atol=1e-6,
    )


def test_the_stored_runs_are_far_smaller_than_the_triplet_form(tmp_path) -> None:
    """The reason for the change: a triplet table can exceed the eager-load limit.

    Triplets grow with the number of nonzero cell *pairs*, so they scale as n^2 while the
    runs scale as n x n_runs. The gap is what made a saved analysis unloadable.
    """
    from scipy.sparse import coo_matrix

    ft = _run_table()
    clus = ft.cluster(
        n_neighbors=15, resolution=[0.3, 1.0], n_times=4, seed=0, n_jobs=1
    )
    triplet_bytes = coo_matrix(clus.similarity_matrix).nnz * (8 + 8 + 4)
    partition_bytes = clus.partitions.n_cells * clus.partitions.n_runs * 4
    assert partition_bytes < triplet_bytes / 3


def test_a_version_one_triplet_clustering_still_loads(tmp_path) -> None:
    """Older folios must keep working, and come back without partitions as before."""
    from datafolio import DataFolio
    from scipy.sparse import coo_matrix

    from cellpax import load_feature_table, save_feature_table

    ft = _run_table()
    clus = ft.cluster(n_neighbors=15, n_times=2, seed=0, n_jobs=1, name="run")
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    # rewrite the folio in the v1 shape: triplets, no storage marker, no spaces
    folio = DataFolio(path, allow_existing=True)
    cx = coo_matrix(clus.similarity_matrix)
    folio.add(
        "t/clustering/run",
        pl.DataFrame(
            {
                "row": cx.row.astype(np.int64),
                "col": cx.col.astype(np.int64),
                "value": cx.data.astype(np.float64),
            }
        ),
        description="v1 triplets",
        overwrite=True,
    )
    manifest = folio.get("t/manifest")
    manifest["version"] = "1"
    manifest.pop("spaces", None)
    manifest["clusterings"]["run"].pop("storage", None)
    folio.add("t/manifest", manifest, overwrite=True)

    restored = load_feature_table(path, name="t").clustering("run")
    assert restored.partitions is None
    np.testing.assert_allclose(
        restored.similarity_matrix.toarray(),
        clus.similarity_matrix.toarray(),
        atol=1e-6,
    )


def test_a_clustering_without_runs_falls_back_to_the_matrix(tmp_path) -> None:
    """A reloaded v1 clustering, re-saved, has no runs to store."""
    from cellpax import load_feature_table, save_feature_table
    from cellpax.clustering import Clustering

    ft = _run_table()
    matrix = ft.cluster(n_neighbors=15, n_times=2, seed=0, n_jobs=1).similarity_matrix
    ft._clusterings["bare"] = Clustering(
        matrix, cell_ids=ft._cell_ids("all"), mask="all", normalized=True
    )
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    restored = load_feature_table(path, name="t").clustering("bare")
    assert restored.partitions is None
    np.testing.assert_allclose(
        restored.similarity_matrix.toarray(), matrix.toarray(), atol=1e-6
    )


# -- frozen spaces --------------------------------------------------------------


def test_a_fitted_space_is_reloaded_rather_than_refit(tmp_path) -> None:
    """The point of freezing: reapplying to a future dataset must not refit anything."""
    from cellpax import load_feature_table, save_feature_table

    ft = _run_table()
    ft.cluster(n_neighbors=15, n_times=2, seed=0, n_jobs=1, alpha=0.5, name="run")
    original = ft.space()

    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")
    restored = load_feature_table(path, name="t").space()

    np.testing.assert_array_equal(restored.components_, original.components_)
    np.testing.assert_array_equal(restored.eigenvalues_, original.eigenvalues_)
    assert restored.n_components == original.n_components
    raw = ft.features()[:5]
    np.testing.assert_allclose(restored.transform(raw), original.transform(raw))


# -- the scaler tag cannot silently change preprocessing ------------------------


def test_a_configured_clipped_factory_round_trips_its_bounds(tmp_path) -> None:
    from cellpax import load_feature_table, save_feature_table
    from cellpax.clustering import clipped_scaler_factory

    for key, factory in (
        ("sigma4", clipped_scaler_factory(mode="sigma", n_sigma=4.0)),
        ("pct1", clipped_scaler_factory(1.0, 99.0)),
    ):
        rng = np.random.default_rng(0)
        df = pl.DataFrame(
            {
                "cell_id": pl.Series(range(1, 41), dtype=pl.Int64),
                **{f"m{i}": rng.normal(size=40) for i in range(4)},
            }
        )
        ft = FeatureTable(
            df, features=[f"m{i}" for i in range(4)], scaler_factory=factory
        )
        before = ft.features(scaled=True)
        path = tmp_path / f"{key}.zarr"
        save_feature_table(ft, path, name=key)
        back = load_feature_table(path, name=key)
        np.testing.assert_allclose(before, back.features(scaled=True))


def test_an_unrecognised_scaler_factory_refuses_to_save(tmp_path) -> None:
    """It used to be recorded as "custom" and reload as plain standardisation."""
    from sklearn.preprocessing import MinMaxScaler

    from cellpax import save_feature_table

    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 21), dtype=pl.Int64),
            **{f"m{i}": rng.normal(size=20) for i in range(3)},
        }
    )
    ft = FeatureTable(
        df, features=[f"m{i}" for i in range(3)], scaler_factory=MinMaxScaler
    )
    with pytest.raises(TypeError, match="silently changing every scaled value"):
        save_feature_table(ft, tmp_path / "f.zarr", name="t")


def test_an_unknown_scaler_tag_refuses_to_load(tmp_path) -> None:
    from datafolio import DataFolio

    from cellpax import load_feature_table, save_feature_table

    ft = _run_table()
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    folio = DataFolio(path, allow_existing=True)
    manifest = folio.get("t/manifest")
    manifest["scaler"] = "quantile"
    folio.add("t/manifest", manifest, overwrite=True)

    with pytest.raises(ValueError, match="unknown scaler tag"):
        load_feature_table(path, name="t")
