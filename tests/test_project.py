"""Projecting new cells through a mask's existing fits, without re-fitting."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable, FittedEmbedding, FittedScaler


def _table(n: int = 120) -> FeatureTable:
    """Two well-separated blobs, half of them in a ``core`` mask.

    ``skewed`` is heavy-tailed on purpose so ``preprocess`` picks a real per-feature
    transform and the projection chain has to reproduce it, not just a scaler.
    """
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 3)), rng.normal(8, 0.3, (n - n // 2, 3))]
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(3)},
            "skewed": rng.lognormal(3, 1.4, n),
            "is_core": [True, False] * (n // 2),
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(3)] + ["skewed"])
    ft.add_mask("core", pl.col("is_core"))
    return ft


# -- the scaler accessor ---------------------------------------------------------


def test_scaler_returns_the_same_cached_fit_features_used() -> None:
    ft = _table()
    ft.features("core", scaled=True)
    fitted = ft.scaler("core")

    assert isinstance(fitted, FittedScaler)
    # the cached object itself, so inspecting it describes the real scaling
    assert fitted is ft._scaler("core", ft.feature_columns)
    assert hasattr(fitted.scaler, "mean_")


def test_scaler_fits_on_demand_and_is_per_mask() -> None:
    ft = _table()
    core = ft.scaler("core")
    everything = ft.scaler()

    assert core is not everything
    # mask-relative: the core is half the cells, so its centering differs
    assert not np.allclose(core.scaler.mean_, everything.scaler.mean_)


def test_scaler_records_preprocess_transforms() -> None:
    ft = _table()
    ft.preprocess(skew_screen=True)
    assert ft.transforms["skewed"] == "ihs"

    fitted = ft.scaler("core")
    assert fitted.transforms == [None, None, None, "ihs"]


def test_scaler_narrowed_by_columns() -> None:
    ft = _table()
    ft.define_features("motion", columns=["m0", "m1"])
    fitted = ft.scaler("core", columns="motion")

    assert fitted.scaler.mean_.shape == (2,)
    assert fitted is not ft.scaler("core")


# -- project -------------------------------------------------------------------


def test_project_reproduces_scaled_features_for_the_masks_own_rows() -> None:
    ft = _table()
    ft.preprocess(skew_screen=True)
    raw = ft.features("core", scaled=False)

    np.testing.assert_allclose(
        ft.project(raw, "core"), ft.features("core", scaled=True)
    )


def test_project_freezes_the_masks_scaling_for_outside_cells() -> None:
    ft = _table()
    outside = ft.dataframe().filter(~pl.col("is_core"))

    in_core_space = ft.project(outside, "core")
    own_rows = ~ft.dataframe()["is_core"].to_numpy()
    in_own_space = ft.features(scaled=True)[own_rows]

    # the whole point: these cells are placed relative to core, not re-centered
    assert not np.allclose(in_core_space, in_own_space)
    assert in_core_space.shape == in_own_space.shape


def test_project_matches_frame_columns_by_name_ignoring_order_and_extras() -> None:
    ft = _table()
    outside = ft.dataframe().filter(~pl.col("is_core"))
    expected = ft.project(outside, "core")

    shuffled = outside.select(["skewed", "cell_id", "m2", "m0", "m1"])
    np.testing.assert_allclose(ft.project(shuffled, "core"), expected)


def test_project_accepts_a_raw_array_in_resolved_column_order() -> None:
    ft = _table()
    raw = ft.features("core", scaled=False)
    np.testing.assert_allclose(
        ft.project(raw, "core"), ft.project(ft.dataframe("core"), "core")
    )


def test_project_rejects_misshapen_input() -> None:
    ft = _table()
    raw = ft.features("core", scaled=False)

    with pytest.raises(ValueError, match="2-dimensional"):
        ft.project(raw[:, 0], "core")
    with pytest.raises(ValueError, match="but the fit covers"):
        ft.project(raw[:, :2], "core")
    with pytest.raises(ValueError, match="missing feature columns"):
        ft.project(ft.dataframe("core").select(["m0", "m1"]), "core")


# -- embedding models ----------------------------------------------------------


def test_embed_retains_its_estimator_and_reprojects_to_stored_coordinates() -> None:
    ft = _table()
    ft.preprocess(skew_screen=True)
    coords = ft.embed("core", method="pca", n_components=2, seed=0)

    model = ft.embedding_model("core", name="pca")
    assert isinstance(model, FittedEmbedding)
    assert model.method == "pca"
    assert model.columns == tuple(ft.feature_columns)
    assert model.n_components == 2

    stored = coords.select(["pca0", "pca1"]).to_numpy()
    raw = ft.features("core", scaled=False)
    np.testing.assert_allclose(ft.project(raw, "core", embedding="pca"), stored)


def test_project_places_new_cells_in_an_existing_embedding() -> None:
    ft = _table()
    ft.embed("core", method="pca", n_components=2, seed=0)
    outside = ft.dataframe().filter(~pl.col("is_core"))

    placed = ft.project(outside, "core", embedding="pca")
    assert placed.shape == (outside.height, 2)
    # the two blobs stay separated in the core's basis
    left = placed[outside["m0"].to_numpy() < 4]
    right = placed[outside["m0"].to_numpy() >= 4]
    assert abs(left[:, 0].mean() - right[:, 0].mean()) > 1.0


def test_embedding_model_records_the_columns_it_was_fit_on() -> None:
    ft = _table()
    ft.define_features("motion", columns=["m0", "m1"])
    ft.embed("core", method="pca", n_components=2, columns="motion", seed=0)

    model = ft.embedding_model("core", name="pca_motion")
    assert model.columns == ("m0", "m1")


def test_project_refuses_columns_the_embedding_was_not_fit_on() -> None:
    ft = _table()
    ft.embed("core", method="pca", n_components=2, seed=0)
    raw = ft.features("core", scaled=False)

    with pytest.raises(ValueError, match="cannot accept"):
        ft.project(raw, "core", columns=["m0", "m1"], embedding="pca")


def test_embedding_model_raises_for_an_embedding_that_was_never_computed() -> None:
    ft = _table()
    with pytest.raises(KeyError, match="nor its coordinates"):
        ft.embedding_model("core", name="pca")


# -- invalidation --------------------------------------------------------------


def test_preprocess_drops_embeddings_computed_under_the_old_transforms() -> None:
    """Coordinates from a scaling that no longer exists must not be served as current."""
    ft = _table()
    ft.embed("core", method="pca", n_components=2, seed=0)
    with pytest.warns(UserWarning, match="preprocess"):
        ft.preprocess(skew_screen=True)

    with pytest.raises(KeyError, match="No embedding"):
        ft.embedding("core", name="pca")
    with pytest.raises(KeyError, match="re-run embed"):
        ft.embedding_model("core", name="pca")


def test_redefining_a_mask_drops_its_embedding_model() -> None:
    ft = _table()
    ft.embed("core", method="pca", n_components=2, seed=0)
    ft.add_mask("core", pl.col("cell_id") <= 10)

    with pytest.raises(KeyError, match="re-run embed"):
        ft.embedding_model("core", name="pca")


def test_redefining_a_mask_leaves_another_masks_model_alone() -> None:
    ft = _table()
    ft.add_mask("low", pl.col("cell_id") <= 60)
    ft.embed("core", method="pca", n_components=2, seed=0)
    ft.embed("low", method="pca", n_components=2, seed=0)

    ft.add_mask("core", pl.col("cell_id") <= 10)
    assert ft.embedding_model("low", name="pca").method == "pca"


def test_adding_features_drops_embedding_models() -> None:
    ft = _table()
    ft.embed("core", method="pca", n_components=2, seed=0)
    ft.add_features(
        pl.DataFrame(
            {
                "cell_id": ft.dataframe()["cell_id"],
                "extra": np.zeros(ft.n_cells),
            }
        ),
        on="cell_id",
    )

    with pytest.raises(KeyError, match="re-run embed"):
        ft.embedding_model("core", name="pca")


def test_a_cross_mask_space_applies_its_own_scaler_not_the_targets() -> None:
    """A parent space handed to a child mask must not mix the two scalings."""
    from cellpax.clustering import fauxnograph_coclustering

    ft = _table()
    ft.add_mask("child", pl.col("cell_id") <= 30)
    parent_space = ft.space(explained_variance=2)  # fit on "all"

    # the coherent projection: raw child rows through the parent's frozen fits
    expected = parent_space.transform(ft.features("child", scaled=False))

    clus = ft.cluster(
        "child", space=parent_space, n_neighbors=10, n_times=2, seed=3, n_jobs=1
    )
    manual = fauxnograph_coclustering(
        expected, n_neighbors=10, n_times=2, seed=3, n_jobs=1
    )
    assert (clus.similarity_matrix != manual).nnz == 0


def test_project_through_a_cross_mask_space_embedding_is_coherent() -> None:
    ft = _table()
    ft.add_mask("child", pl.col("cell_id") <= 30)
    parent_space = ft.space(explained_variance=2)

    coords = ft.embed(
        "child", method="pca", n_components=2, name="e", space=parent_space, seed=0
    )
    # projecting the very rows the embedding was fit on must reproduce the
    # stored coordinates — the round trip that a mixed scaling breaks
    raw = ft.dataframe("child")
    projected = ft.project(raw, "child", embedding="e")
    stored = coords.drop(ft.id_column).to_numpy()
    assert np.allclose(projected, stored, atol=1e-8)
