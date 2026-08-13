"""PaCMAP and LocalMAP embedding backends (optional ``pacmap`` extra).

Split from ``test_embeddings.py`` deliberately: ``pytest.importorskip`` at module level
aborts collection of the whole module, so keeping these beside the PCA and UMAP tests would
skip those too whenever the extra is absent.

pacmap prints an unconditional notice when ``random_state`` is set and warns about
"high-dimensional" input on matrices that are nothing of the sort, hence the filters.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable

pytest.importorskip("pacmap", reason="optional 'pacmap' extra not installed")

_BACKENDS = ("pacmap", "localmap")


def _grouped(n: int = 160, dim: int = 20) -> FeatureTable:
    """Four groups separated across *every* feature, so they are separable as given.

    Offsetting only a few of many columns does not do that: with three signal dimensions
    against seventeen of noise, Euclidean neighbourhoods in the input space are already
    ~50% pure, and an embedding faithfully representing that space scores badly through no
    fault of its own. Separability has to hold in the space being embedded for a
    structure-preservation check to mean anything.
    """
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1.0, (n, dim))
    coords += np.repeat([0, 1, 2, 3], n // 4)[:, None] * 3.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


def _wide(n: int = 120, dim: int = 130) -> FeatureTable:
    """Above pacmap's 100-feature threshold, where its own reduction actually fires."""
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1.0, (n, dim))
    coords[:, :3] += np.repeat([0, 1], n // 2)[:, None] * 5.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_backend_embeds_and_stores_coordinates(method: str) -> None:
    ft = _grouped()
    frame = ft.embed(method=method, name=method, seed=0)
    assert frame.height == ft.n_cells
    assert [c for c in frame.columns if c != "cell_id"] == [f"{method}0", f"{method}1"]
    assert np.isfinite(frame[f"{method}0"].to_numpy()).all()
    assert ft.embedding(name=method).equals(frame)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_seed_makes_the_embedding_reproducible(method: str) -> None:
    """Which the notebooks currently lack, and which matters once settings are compared."""
    ft = _grouped()
    first = ft.embed(method=method, name="a", seed=0)
    second = ft.embed(method=method, name="b", seed=0)
    np.testing.assert_allclose(
        first[["a0", "a1"]].to_numpy(), second[["b0", "b1"]].to_numpy()
    )
    different = ft.embed(method=method, name="c", seed=7)
    assert not np.allclose(
        first[["a0", "a1"]].to_numpy(), different[["c0", "c1"]].to_numpy()
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_backend_keyword_arguments_pass_through(method: str) -> None:
    ft = _grouped()
    ft.embed(method=method, n_components=3, name="e", n_neighbors=6, seed=0)
    model = ft.embedding_model(name="e")
    assert model.model.n_neighbors == 6
    assert model.n_components == 3
    assert ft.embedding(name="e").columns == ["cell_id", "e0", "e1", "e2"]


@pytest.mark.filterwarnings("ignore")
def test_localmap_takes_its_own_parameter() -> None:
    ft = _grouped()
    ft.embed(method="localmap", name="e", low_dist_thres=5, seed=0)
    assert ft.embedding_model(name="e").model.low_dist_thres == 5


def test_an_unknown_method_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="'pca', 'umap', 'pacmap', 'localmap'"):
        _grouped().embed(method="trimap")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_planted_groups_stay_together_in_two_dimensions(method: str) -> None:
    """Weak but real: an embedding that scrambles separable groups is broken.

    Asserts the input space is separable first, so a failure points at the embedding
    rather than at a fixture whose groups were never separable to begin with.
    """
    from cellpax.clustering import neighborhood_purity

    ft = _grouped()
    truth = np.repeat([0, 1, 2, 3], ft.n_cells // 4)
    in_space = neighborhood_purity(
        ft.features(scaled=True), truth, n_neighbors=10
    ).mean()
    assert in_space > 0.99, "fixture is not separable as given; the check is vacuous"

    coords = ft.embed(method=method, name="e", seed=0)[["e0", "e1"]].to_numpy()
    assert neighborhood_purity(coords, truth, n_neighbors=10).mean() > 0.9


# -- what the backend does to its input on its own ------------------------------


def test_reduction_predicate_matches_pacmaps_own_condition() -> None:
    """It fires only above 100 features, so below that the flag is not a knob at all."""
    from cellpax.featuretable import _pacmap_reduction

    assert _pacmap_reduction(130, {}) == "tsvd(100)"
    assert _pacmap_reduction(130, {"apply_pca": True}) == "tsvd(100)"
    assert _pacmap_reduction(130, {"apply_pca": False}) is None
    assert _pacmap_reduction(130, {"distance": "hamming"}) is None
    assert _pacmap_reduction(81, {}) is None
    assert _pacmap_reduction(100, {}) is None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_a_wide_input_records_the_backends_own_reduction(method: str) -> None:
    ft = _wide()
    ft.embed(method=method, name="wide", seed=0)
    assert ft.embedding_model(name="wide").internal_reduction == "tsvd(100)"
    row = ft.graph_provenance().filter(pl.col("name") == "wide").row(0, named=True)
    # the space it built pairs in is not the space it was handed, and says so
    assert row["space"] == "scaled -> tsvd(100)"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_a_narrow_input_records_no_reduction(method: str) -> None:
    ft = _grouped()
    ft.embed(method=method, name="narrow", seed=0)
    assert ft.embedding_model(name="narrow").internal_reduction is None
    row = ft.graph_provenance().filter(pl.col("name") == "narrow").row(0, named=True)
    assert row["space"] == "scaled"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_passing_a_space_turns_off_the_backends_own_reduction(method: str) -> None:
    """Reducing an already-reduced, partly-whitened space would undo the weighting."""
    ft = _wide()
    ft.embed(method=method, name="in_space", space=ft.space(alpha=0.5), seed=0)
    model = ft.embedding_model(name="in_space")
    assert model.model.apply_pca is False
    assert model.internal_reduction is None
    assert model.space == "pca(0.95, alpha=0.5)"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_an_explicit_apply_pca_still_wins(method: str) -> None:
    """The automatic ``apply_pca=False`` is a default, not an override.

    Needs a space that stays above 100 components, since below that the flag has no
    effect at all — ``space()`` at 0.95 keeps 76 of 130 here, which is why the reduction
    would not fire regardless.
    """
    ft = _wide()
    space = ft.space(explained_variance=1.0)
    assert space.n_components > 100, "otherwise apply_pca could not fire either way"

    ft.embed(method=method, name="forced", space=space, apply_pca=True, seed=0)
    assert ft.embedding_model(name="forced").internal_reduction == "tsvd(100)"

    ft.embed(method=method, name="auto", space=space, seed=0)
    assert ft.embedding_model(name="auto").internal_reduction is None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_the_input_matrix_is_not_mutated(method: str) -> None:
    """pacmap min-max normalises internally; confirm it works on its own copy."""
    ft = _grouped()
    before = ft.features(scaled=True).copy()
    ft.embed(method=method, name="e", apply_pca=False, seed=0)
    np.testing.assert_array_equal(ft.features(scaled=True), before)


# -- projecting new cells in ----------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_new_cells_project_into_an_existing_embedding(method: str) -> None:
    """Needs save_tree=True: transform otherwise wants the training matrix handed back."""
    ft = _grouped()
    ft.embed(method=method, name="e", seed=0)
    assert ft.embedding_model(name="e").model.save_tree is True
    projected = ft.project(ft.features()[:5], embedding="e")
    assert projected.shape == (5, 2)
    assert np.isfinite(projected).all()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_projection_runs_through_a_stored_space(method: str) -> None:
    ft = _grouped()
    ft.embed(method=method, name="e", space=ft.space(alpha=0.5), seed=0)
    assert ft.project(ft.features()[:4], embedding="e").shape == (4, 2)


# -- interaction with the rest of the library -----------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_an_embedding_is_never_confused_with_a_leiden_graph(method: str) -> None:
    """It builds its own neighbour structure, which is what keeps the comparison honest."""
    ft = _grouped()
    ft.cluster(n_neighbors=15, n_times=2, seed=0, n_jobs=1, name="run")
    ft.embed(method=method, name="e", seed=0)
    row = ft.graph_provenance().filter(pl.col("name") == "e").row(0, named=True)
    assert row["graph_type"] == f"internal:{method}"
    assert row["n_neighbors"] is None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_sharing_a_space_with_a_clustering_still_warns(method: str) -> None:
    import warnings as _warnings

    ft = _grouped()
    ft.cluster(n_neighbors=15, n_times=2, seed=0, n_jobs=1, name="run")
    ft.embed(method=method, name="same", space=ft.space(), seed=0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ft.graph_provenance()
    assert any("close to circular" in str(w.message) for w in caught)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_the_model_is_dropped_when_preprocessing_changes(method: str) -> None:
    ft = _grouped()
    ft.embed(method=method, name="e", seed=0)
    ft.preprocess()
    with pytest.raises(KeyError, match="re-run embed"):
        ft.embedding_model(name="e")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_coordinates_survive_a_reload_but_the_fit_does_not(method, tmp_path) -> None:
    """Documented for every backend: coordinates persist, fits are session-only."""
    from cellpax import load_feature_table, save_feature_table

    ft = _grouped()
    coords = ft.embed(method=method, name="e", seed=0)
    save_feature_table(ft, tmp_path / "f.zarr", name="t")
    back = load_feature_table(tmp_path / "f.zarr", name="t")
    assert back.embedding(name="e").equals(coords)
    with pytest.raises(KeyError, match="re-run embed"):
        back.embedding_model(name="e")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", _BACKENDS)
def test_faiss_is_pinned_during_the_call_and_restored_after(method: str) -> None:
    """The deadlock guard, and the half of it that is easy to lose.

    Pinning faiss to one thread is what stops a cross-``libomp`` barrier from hanging the
    process; *restoring* the count is what stops cellpax from leaving the whole session
    single-threaded, which is the bug pacmap itself has when given a ``random_state``.
    """
    faiss = pytest.importorskip("faiss", reason="faiss ships with the pacmap extra")

    before = faiss.omp_get_max_threads()
    ft = _grouped()

    ft.embed(method=method, name="e", seed=0)
    assert faiss.omp_get_max_threads() == before

    ft.project(ft.dataframe().head(8), embedding="e")
    assert faiss.omp_get_max_threads() == before


@pytest.mark.filterwarnings("ignore")
def test_the_guard_pins_to_one_thread_and_survives_an_exception() -> None:
    faiss = pytest.importorskip("faiss", reason="faiss ships with the pacmap extra")
    from cellpax.featuretable import _faiss_single_thread

    before = faiss.omp_get_max_threads()
    with _faiss_single_thread():
        assert faiss.omp_get_max_threads() == 1
    assert faiss.omp_get_max_threads() == before

    with pytest.raises(RuntimeError):
        with _faiss_single_thread():
            raise RuntimeError("boom")
    assert faiss.omp_get_max_threads() == before
