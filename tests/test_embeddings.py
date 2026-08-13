"""Step 5: embeddings — PCA-native, UMAP optional, dataframe join."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable


def _table(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 4)), rng.normal(8, 0.3, (n // 2, 4))]
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
            "region": ["L"] * (n // 2) + ["R"] * (n - n // 2),
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(4)])


def test_pca_embedding_store_and_join() -> None:
    ft = _table(60)
    coords = ft.embed(method="pca", n_components=2, name="pca")
    assert coords.columns == ["cell_id", "pca0", "pca1"]
    assert coords.height == 60
    assert ft.embedding(name="pca").equals(coords)

    joined = ft.dataframe(embedding="pca")
    assert "pca0" in joined.columns and "region" in joined.columns
    assert joined.height == 60

    with pytest.raises(KeyError, match="No embedding"):
        ft.embedding(name="missing")
    with pytest.raises(ValueError, match="Unknown embedding method"):
        ft.embed(method="tsne")


def test_embedding_respects_mask() -> None:
    ft = _table(60)
    ft.add_mask("left", pl.col("region") == "L")
    coords = ft.embed("left", method="pca", n_components=2)
    assert coords.height == 30
    # joining that mask's embedding into the mask's dataframe lines up
    df = ft.dataframe("left", embedding="pca")
    assert df.height == 30 and df["pca0"].null_count() == 0


def test_features_pca_reduces_correlated_features() -> None:
    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(size=(n, 3))
    # 7 more columns as noisy linear combinations of the same 3 latent axes --
    # correlated, so PCA compresses them even after per-column standardizing.
    weights = rng.normal(size=(3, 7))
    derived = base @ weights + rng.normal(scale=0.05, size=(n, 7))
    coords = np.hstack([base, derived])
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(10)},
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(10)])

    reduced = ft.features_pca(explained_variance=0.95)
    assert reduced.shape[0] == n
    assert reduced.shape[1] < 10  # captured by far fewer than the 10 raw columns

    # a near-1.0 target keeps (almost) everything
    full = ft.features_pca(explained_variance=0.999999)
    assert full.shape[1] > reduced.shape[1]


def test_features_pca_respects_mask_and_columns() -> None:
    ft = _table(60)
    ft.add_mask("left", pl.col("region") == "L")
    reduced = ft.features_pca("left", columns=["m0", "m1"], explained_variance=0.95)
    assert reduced.shape[0] == 30
    assert reduced.shape[1] <= 2


def test_umap_is_optional() -> None:
    ft = _table(40)
    if importlib.util.find_spec("umap") is None:
        with pytest.raises(ImportError, match="umap-learn"):
            ft.embed(method="umap")
    else:  # pragma: no cover - depends on optional dep
        assert ft.embed(method="umap", n_components=2).height == 40


def test_pacmap_is_optional() -> None:
    ft = _table(40)
    if importlib.util.find_spec("pacmap") is None:
        for method in ("pacmap", "localmap"):
            with pytest.raises(ImportError, match="cellpax\\[pacmap\\]"):
                ft.embed(method=method)
    else:  # pragma: no cover - depends on optional dep
        assert ft.embed(method="pacmap", n_components=2, seed=0).height == 40


# -- EmbeddingView: coordinates and the columns that address them ----------------


def _two_embeddings(n: int = 60) -> FeatureTable:
    ft = _table(n)
    ft.add_mask("left", pl.col("region") == "L")
    ft.embed("left", method="pca", name="umap_core")
    return ft


def test_view_carries_the_frame_and_its_coordinate_columns() -> None:
    """The point: the embedding's name is written once rather than three times."""
    ft = _two_embeddings()
    view = ft.embedding_view("left", name="umap_core")

    assert view.name == "umap_core"
    assert view.mask == "left"
    assert view.coords == ("umap_core0", "umap_core1")
    assert view.frame.height == 30
    assert all(column in view.frame.columns for column in view.coords)


def test_coords_come_from_the_stored_frame() -> None:
    """Not rebuilt as f"{name}{i}" — the frame cannot drift from what embed wrote."""
    ft = _two_embeddings()
    stored = ft.embedding("left", name="umap_core")
    view = ft.embedding_view("left", name="umap_core")
    assert list(view.coords) == [c for c in stored.columns if c != ft.id_column]


def test_indexing_iteration_and_length_agree() -> None:
    ft = _two_embeddings()
    view = ft.embedding_view("left", name="umap_core")

    assert view.n_components == len(view) == 2
    assert view[0] == view.x == "umap_core0"
    assert view[1] == view.y == "umap_core1"
    assert view[-1] == "umap_core1"
    assert list(view) == list(view.coords)


def test_xy_splats_into_a_plotting_call() -> None:
    ft = _two_embeddings()
    view = ft.embedding_view("left", name="umap_core")

    def fake_scatter(*, x, y, data):
        return x, y, data.height

    assert fake_scatter(**view.xy, data=view.frame) == (
        "umap_core0",
        "umap_core1",
        30,
    )


def test_coordinates_work_as_polars_predicates() -> None:
    """Several call sites filter on coordinates rather than plotting them."""
    ft = _two_embeddings()
    view = ft.embedding_view("left", name="umap_core")
    kept = view.frame.filter(pl.col(view.x) > view.frame[view.x].median())
    assert 0 < kept.height < view.frame.height


# -- more or fewer than two components -------------------------------------------


def test_pair_selects_any_two_axes() -> None:
    ft = _table(60)
    ft.embed(method="pca", name="e", n_components=3)
    view = ft.embedding_view(name="e")

    assert view.n_components == 3
    assert view.coords == ("e0", "e1", "e2")
    # xy names its own two-dimensionality; pair makes any other projection explicit
    assert view.xy == {"x": "e0", "y": "e1"}
    assert view.pair(0, 2) == {"x": "e0", "y": "e2"}
    assert view.pair(2, 1) == {"x": "e2", "y": "e1"}


def test_a_missing_axis_says_how_many_there_are() -> None:
    ft = _table(60)
    ft.embed(method="pca", name="e", n_components=3)
    view = ft.embedding_view(name="e")
    with pytest.raises(IndexError, match="has 3 component"):
        view.pair(0, 5)
    with pytest.raises(IndexError, match="has 3 component"):
        view[7]


def test_one_component_gives_x_but_not_y() -> None:
    """The case that makes x/y aliases rather than a promise the data is 2-D."""
    ft = _table(60)
    ft.embed(method="pca", name="flat", n_components=1)
    view = ft.embedding_view(name="flat")

    assert view.n_components == 1
    assert view.x == "flat0"
    with pytest.raises(IndexError, match="has 1 component"):
        view.y
    with pytest.raises(IndexError, match="has 1 component"):
        view.xy


# -- resolving the name ----------------------------------------------------------


def test_a_single_embedding_needs_no_name() -> None:
    ft = _two_embeddings()
    assert ft.embedding_view("left").name == "umap_core"


def test_several_embeddings_raise_listing_them() -> None:
    """That message is the whole point of not defaulting silently."""
    ft = _two_embeddings()
    ft.embed("left", method="pca", name="pacmap_like")

    with pytest.raises(ValueError, match="has 2 embeddings") as caught:
        ft.embedding_view("left")
    message = str(caught.value)
    assert "pacmap_like" in message and "umap_core" in message
    assert "name=" in message


def test_no_embedding_names_the_mask_and_points_elsewhere() -> None:
    ft = _two_embeddings()
    with pytest.raises(ValueError, match="has no embeddings") as caught:
        ft.embedding_view()  # the implicit "all" mask, which has none
    message = str(caught.value)
    assert "ft.embed" in message
    assert "left" in message, "should say which masks do have embeddings"


def test_an_explicit_wrong_name_still_raises_from_embedding() -> None:
    ft = _two_embeddings()
    with pytest.raises(KeyError, match="No embedding"):
        ft.embedding_view("left", name="left")  # the mask name, a real mistake


# -- what the view forwards ------------------------------------------------------


def test_labels_join_onto_the_frame() -> None:
    from cellpax.labels import LabelSet

    ft = _two_embeddings()
    ids = ft.embedding("left", name="umap_core")[ft.id_column].to_numpy()
    labels = LabelSet.from_labels(ids, ["a"] * 15 + ["b"] * 15, name="grp", mask="left")
    view = ft.embedding_view("left", labels=labels)
    assert "grp" in view.frame.columns
    assert view.frame["grp"].null_count() == 0


def test_scaled_and_columns_are_forwarded() -> None:
    """``columns`` picks what ``scaled`` applies to, not what the frame carries."""
    ft = _two_embeddings()
    raw = ft.embedding_view("left")
    partial = ft.embedding_view("left", scaled=True, columns=["m0", "m1"])

    # every feature column is present either way
    assert {"m0", "m1", "m2", "m3"}.issubset(partial.frame.columns)
    # m0 was scaled, m3 was left alone
    assert partial.frame["m0"].to_list() != raw.frame["m0"].to_list()
    assert partial.frame["m3"].to_list() == raw.frame["m3"].to_list()
    # and scaling never touches the coordinates
    assert partial.coords == raw.coords


def test_a_redefined_mask_cannot_serve_stale_coordinates_through_the_view() -> None:
    """A mask redefined after embed drops the embedding rather than serving it."""
    ft = _two_embeddings()
    with pytest.warns(UserWarning, match="'left' was dropped"):
        ft.drop_mask("left")
    ft.add_mask("left", pl.col("cell_id") > 0)  # now covers every cell

    with pytest.raises(KeyError, match="No embedding"):
        ft.embedding_view("left", name="umap_core")


# -- discovery and persistence ---------------------------------------------------


def test_embeddings_lists_mask_name_pairs_including_derived_names() -> None:
    ft = _table(60)
    ft.add_mask("left", pl.col("region") == "L")
    ft.define_features("pair", columns=["m0", "m1"])
    ft.embed("left", method="pca", name="explicit")
    ft.embed(method="pca", columns="pair")  # name derived from the collection

    assert ft.embeddings == [("all", "pca_pair"), ("left", "explicit")]
    assert ft.embedding_view(name="pca_pair").coords == ("pca_pair0", "pca_pair1")


def test_the_view_survives_a_save_and_reload(tmp_path) -> None:
    """Coordinates persist while fitted models do not.

    This is the case that forces deriving ``coords`` from the stored frame: rebuilding
    them from a ``FittedEmbedding`` would work in-session and break after a reload.
    """
    from cellpax import load_feature_table, save_feature_table

    ft = _two_embeddings()
    before = ft.embedding_view("left", name="umap_core")
    save_feature_table(ft, tmp_path / "f.zarr", name="t")
    back = load_feature_table(tmp_path / "f.zarr", name="t")

    after = back.embedding_view("left", name="umap_core")
    assert after.coords == before.coords
    assert after.frame.height == before.frame.height
    assert back.embeddings == ft.embeddings
    # the fit itself is gone, as documented
    with pytest.raises(KeyError, match="re-run embed"):
        back.embedding_model("left", name="umap_core")


def test_graph_provenance_lists_embeddings_after_a_reload(tmp_path) -> None:
    """It used to read the session-only models, so a reloaded table showed none."""
    from cellpax import load_feature_table, save_feature_table

    ft = _two_embeddings()
    live = ft.graph_provenance().filter(pl.col("consumer") == "embedding")
    assert live.height == 1
    assert live["space"][0] == "scaled"

    save_feature_table(ft, tmp_path / "f.zarr", name="t")
    back = load_feature_table(tmp_path / "f.zarr", name="t")

    reloaded = back.graph_provenance().filter(pl.col("consumer") == "embedding")
    assert reloaded.height == 1, "coordinates persisted, so the row must survive"
    assert reloaded["name"][0] == "umap_core"
    # null rather than a guess: how it was made did not survive
    assert reloaded["space"][0] is None
    assert reloaded["graph_type"][0] is None


# -- registering coordinates computed elsewhere ----------------------------------


def _spatial(n: int = 60):
    ft = _table(n)
    ft.add_mask("left", pl.col("region") == "L")
    xy = np.column_stack(
        [
            np.linspace(0.0, 400.0, int(ft.mask_series("left").sum())),
            np.linspace(900.0, 0.0, int(ft.mask_series("left").sum())),
        ]
    )
    return ft, xy


def test_external_coordinates_behave_like_any_other_embedding() -> None:
    """Soma position makes a perfectly good 'embedding' — same joins, same views."""
    ft, xy = _spatial()
    frame = ft.add_embedding(xy, "left", name="soma_xz", space="anatomical")

    assert frame.columns == ["cell_id", "soma_xz0", "soma_xz1"]
    v = ft.embedding_view("left", name="soma_xz")
    assert v.coords == ("soma_xz0", "soma_xz1")
    assert v.xy == {"x": "soma_xz0", "y": "soma_xz1"}
    assert v.frame.height == 30
    assert "region" in v.frame.columns  # metadata joins as usual
    np.testing.assert_allclose(v.frame["soma_xz0"].to_numpy(), xy[:, 0])


def test_a_frame_is_matched_on_id_rather_than_row_order() -> None:
    ft, xy = _spatial()
    ids = ft.embedding("left", name="soma_xz") if False else ft._cell_ids("left")
    order = np.random.default_rng(0).permutation(len(ids))
    shuffled = pl.DataFrame(
        {"cell_id": ids[order], "a": xy[order, 0], "b": xy[order, 1]}
    )
    ft.add_embedding(shuffled, "left", name="by_id")
    np.testing.assert_allclose(
        ft.embedding("left", name="by_id").drop("cell_id").to_numpy(), xy
    )


def test_coverage_is_checked_when_added_not_when_read() -> None:
    """A mismatch should fail here rather than surface later as null coordinates."""
    ft, xy = _spatial()
    with pytest.raises(ValueError, match="pass a frame with the id column"):
        ft.add_embedding(xy[:5], "left", name="short")

    ids = ft._cell_ids("left")
    partial = pl.DataFrame({"cell_id": ids[:10], "a": xy[:10, 0]})
    with pytest.raises(ValueError, match="absent"):
        ft.add_embedding(partial, "left", name="partial")


def test_adding_twice_needs_overwrite() -> None:
    ft, xy = _spatial()
    ft.add_embedding(xy, "left", name="soma_xz")
    with pytest.raises(ValueError, match="already exists"):
        ft.add_embedding(xy, "left", name="soma_xz")
    ft.add_embedding(xy * 2, "left", name="soma_xz", overwrite=True)
    np.testing.assert_allclose(
        ft.embedding("left", name="soma_xz")["soma_xz0"].to_numpy(), xy[:, 0] * 2
    )


def test_there_is_no_model_behind_external_coordinates() -> None:
    ft, xy = _spatial()
    ft.add_embedding(xy, "left", name="soma_xz")
    with pytest.raises(KeyError, match="re-run embed"):
        ft.embedding_model("left", name="soma_xz")


def test_provenance_records_where_they_came_from() -> None:
    ft, xy = _spatial()
    ft.add_embedding(xy, "left", name="soma_xz", space="anatomical")
    row = ft.graph_provenance().filter(pl.col("name") == "soma_xz").row(0, named=True)
    assert row["space"] == "anatomical"
    assert row["graph_type"] == "external"


def test_external_provenance_survives_a_reload(tmp_path) -> None:
    """Unlike a fitted model, the label is metadata the caller supplied."""
    from cellpax import load_feature_table, save_feature_table

    ft, xy = _spatial()
    ft.add_embedding(xy, "left", name="soma_xz", space="anatomical")
    save_feature_table(ft, tmp_path / "f.zarr", name="t")
    back = load_feature_table(tmp_path / "f.zarr", name="t")

    assert back.embedding_view("left", name="soma_xz").coords == (
        "soma_xz0",
        "soma_xz1",
    )
    row = back.graph_provenance().filter(pl.col("name") == "soma_xz").row(0, named=True)
    assert row["space"] == "anatomical"


def test_higher_dimensional_external_coordinates_work() -> None:
    ft, xy = _spatial()
    xyz = np.column_stack([xy, xy[:, :1] * 0.5])
    ft.add_embedding(xyz, "left", name="soma_xyz")
    v = ft.embedding_view("left", name="soma_xyz")
    assert v.n_components == 3
    assert v.pair(0, 2) == {"x": "soma_xyz0", "y": "soma_xyz2"}


# -- weighted spaces reach embed and project -------------------------------------


def _wide_table(n: int = 120, dim: int = 10) -> FeatureTable:
    """Enough independent variance that pca=0.95 keeps several components.

    ``_table`` is deliberately degenerate under the default PCA — two tight blobs in four
    features collapse to one component — which is fine for its own tests but leaves
    nothing for a 2-D embedding to work with.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, dim - 3))
    block = base[:, :1] + rng.normal(scale=0.05, size=(n, 3))
    coords = np.column_stack([base, block])
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


def test_embed_applies_a_weighted_space_rather_than_skipping_it() -> None:
    """The whole reason feature_weights belongs inside FittedSpace."""
    from sklearn.decomposition import PCA

    from cellpax.diagnostics import block_weights

    ft = _wide_table()
    names = list(ft.feature_columns)
    scaled = ft.features(scaled=True)
    space = ft.space(feature_weights=block_weights(scaled, names))
    assert space.feature_weights is not None
    assert space.n_components >= 2

    ft.embed(method="pca", name="weighted", space=space, seed=0)
    stored = ft.embedding(name="weighted").drop("cell_id").to_numpy()
    expected = PCA(n_components=2, random_state=0).fit_transform(
        space.transform_scaled(scaled)
    )
    np.testing.assert_allclose(np.abs(stored), np.abs(expected))
    assert ft.embedding_model(name="weighted").space == space.label


def test_a_weighted_space_caches_separately_from_an_unweighted_one() -> None:
    from cellpax.diagnostics import block_weights

    ft = _wide_table()
    w = block_weights(ft.features(scaled=True), list(ft.feature_columns))
    plain, weighted = ft.space(), ft.space(feature_weights=w)
    assert plain is not weighted
    assert ft.space(feature_weights=w) is weighted  # still cached
    assert plain.label == "pca(0.95)"
    assert weighted.label == "pca(0.95, weighted)"
