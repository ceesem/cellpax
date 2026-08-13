"""Table-level seeds, recorded call parameters, and what a reload re-derives."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import fauxnograph_coclustering
from cellpax.featuretable import FeatureTable
from cellpax.persist import load_feature_table


def _blobs(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 4)), rng.normal(8, 0.3, (n // 2, 4))]
    )
    return pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
        }
    )


def _table(seed: int = 0) -> FeatureTable:
    return FeatureTable(_blobs(), features=[f"m{i}" for i in range(4)], seed=seed)


# -- reproducibility ----------------------------------------------------------


def test_the_same_seed_gives_the_same_consensus_across_n_jobs() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (50, 5))
    serial = fauxnograph_coclustering(data, n_neighbors=10, n_times=4, seed=7, n_jobs=1)
    parallel = fauxnograph_coclustering(
        data, n_neighbors=10, n_times=4, seed=7, n_jobs=2
    )
    assert (serial != parallel).nnz == 0


def test_an_unseeded_cluster_call_is_deterministic_by_default() -> None:
    """seed=None now means "derive from the table", not "different every run"."""
    a = _table().cluster(n_neighbors=10, n_times=3, n_jobs=1, name="run")
    b = _table().cluster(n_neighbors=10, n_times=3, n_jobs=1, name="run")
    assert (a.similarity_matrix != b.similarity_matrix).nnz == 0


def test_the_derived_seed_follows_the_call_identity_not_the_data() -> None:
    ft = _table()
    by_name = ft._derive_seed("cluster", None, "run")
    assert by_name == ft._derive_seed("cluster", None, "run")
    assert by_name != ft._derive_seed("cluster", None, "other_run")
    assert by_name != ft._derive_seed("embed", None, "run")
    assert by_name != _table(seed=1)._derive_seed("cluster", None, "run")


def test_an_explicit_seed_still_wins() -> None:
    ft = _table()
    clus = ft.cluster(n_neighbors=10, n_times=2, seed=123, n_jobs=1, name="run")
    assert clus.params["seed"] == 123


# -- recorded parameters ------------------------------------------------------


def test_cluster_records_the_call_that_produced_it() -> None:
    ft = _table()
    clus = ft.cluster(n_neighbors=10, n_times=2, resolution=0.5, n_jobs=1, name="run")
    params = clus.params
    assert params["n_neighbors"] == [10]
    assert params["resolution"] == [0.5]
    assert params["n_times"] == 2
    assert isinstance(params["seed"], int)


def test_replaying_recorded_params_reproduces_the_ensemble() -> None:
    ft = _table()
    first = ft.cluster(n_neighbors=10, n_times=3, n_jobs=1, name="run")
    replay = ft.cluster(
        n_neighbors=first.params["n_neighbors"],
        resolution=first.params["resolution"],
        n_times=first.params["n_times"],
        min_cluster_size=first.params["min_cluster_size"],
        seed=first.params["seed"],
        n_jobs=1,
        name="replay",
    )
    assert (first.similarity_matrix != replay.similarity_matrix).nnz == 0


def test_embedding_params_survive_a_reload(tmp_path) -> None:
    ft = _table()
    ft.embed(method="pca", n_components=2, name="p")
    ft.save(tmp_path / "folio", "run")
    back = load_feature_table(tmp_path / "folio", "run")
    assert back._embedding_params[("all", "p")]["method"] == "pca"
    assert back._embedding_params[("all", "p")]["n_components"] == 2


def test_clustering_params_and_table_seed_survive_a_reload(tmp_path) -> None:
    ft = _table(seed=11)
    ft.cluster(n_neighbors=10, n_times=2, n_jobs=1, name="run")
    ft.save(tmp_path / "folio", "a")
    back = load_feature_table(tmp_path / "folio", "a")
    assert back.seed == 11
    assert back.clustering("run").params["n_neighbors"] == [10]


# -- persistence correctness --------------------------------------------------


def test_two_spaces_on_one_mask_with_different_columns_both_survive(tmp_path) -> None:
    ft = _table()
    narrow = ft.space(columns=["m0", "m1"], explained_variance=2)
    wide = ft.space(explained_variance=2)
    assert narrow is not wide
    ft.save(tmp_path / "folio", "run")
    back = load_feature_table(tmp_path / "folio", "run")
    assert len(back._space_cache) == 2


def test_component_count_and_variance_fraction_are_different_spaces() -> None:
    ft = _table()
    one_component = ft.space(explained_variance=1)
    all_variance = ft.space(explained_variance=1.0)
    assert one_component.n_components == 1
    assert all_variance.n_components == 4


def test_stored_cell_ids_survive_a_same_size_mask_redefinition(tmp_path) -> None:
    """Re-deriving ids from the current mask silently re-pairs matrix rows."""
    ft = _table()
    ft.add_mask("half", pl.col("cell_id") <= 30)
    clus = ft.cluster("half", n_neighbors=10, n_times=2, n_jobs=1, name="run")
    original_ids = clus.cell_ids

    # same size, different membership — the trap re-derivation cannot detect
    with pytest.warns(UserWarning, match="'half' was redefined"):
        ft.add_mask("half", pl.col("cell_id") > 30)
    ft._clusterings["run"] = clus  # keep the pre-redefinition clustering

    ft.save(tmp_path / "folio", "a")
    back = load_feature_table(tmp_path / "folio", "a")
    assert np.array_equal(back.clustering("run").cell_ids, original_ids)


def test_an_unknown_manifest_version_refuses_to_load(tmp_path) -> None:
    from datafolio import DataFolio

    ft = _table()
    ft.save(tmp_path / "folio", "run")
    folio = DataFolio(tmp_path / "folio", allow_existing=True)
    manifest = folio.get("run/manifest")
    manifest["version"] = "99"
    folio.add("run/manifest", manifest, overwrite=True)
    with pytest.raises(ValueError, match="manifest version 99"):
        load_feature_table(folio, "run")
