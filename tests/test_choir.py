"""CHOIR-style statistically-validated cluster resolution."""

from __future__ import annotations

import numpy as np
import polars as pl

from cellpax.choir import _distinguishable, choir_labels
from cellpax.featuretable import FeatureTable


def _blobs(centers, per=40, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.vstack([rng.normal(c, 0.35, (per, dim)) for c in centers])
    n = coords.shape[0]
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


_PARAMS = dict(n_neighbors=15, n_times=5, seed=0, n_jobs=1)
# CHOIR uses a fixed alpha (no Bonferroni); modest iterations keep tests fast
_CHOIR = dict(min_cluster_size=8, n_iterations=40, n_estimators=25, seed=0, n_jobs=1)


def test_distinguishable_decision() -> None:
    """The core CHOIR test: distinct groups split, one population merges."""
    rng = np.random.default_rng(0)
    opts = dict(
        alpha=0.05,
        n_iterations=40,
        n_estimators=25,
        sample_max=1000,
        min_accuracy=0.5,
        use_variance=True,
        n_jobs=1,
    )
    far_a = rng.normal(0, 0.5, (60, 5))
    far_b = rng.normal(6, 0.5, (60, 5))
    assert _distinguishable(far_a, far_b, rng=np.random.default_rng(1), **opts) is True

    # two random halves of ONE population (random labels, no real boundary) -> merge
    pop = rng.normal(0, 1, (120, 5))
    assert (
        _distinguishable(pop[:60], pop[60:], rng=np.random.default_rng(2), **opts)
        is False
    )


def test_choir_finds_two_separated_groups() -> None:
    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10]], per=40)
    ft.cluster(name="run", **_PARAMS)
    labels = ft.cluster_choir("run", name="subclass", **_CHOIR)
    assert len(labels.ids) == 2


def test_choir_resolves_three_groups() -> None:
    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10], [0, 10, 0, 10]], per=40)
    ft.cluster(name="run", **_PARAMS)
    labels = ft.cluster_choir("run", name="subclass", **_CHOIR)
    assert len(labels.ids) == 3
    ft.attach(labels)
    assert "subclass" in ft.dataframe().columns


def test_choir_labels_accepts_similarity_matrix_directly() -> None:
    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10]], per=40)
    sim = ft.cluster(name="run", **_PARAMS)
    labels = choir_labels(sim.linkage, ft.features(scaled=True), **_CHOIR)
    assert set(np.unique(labels)) == {0, 1}
    assert len(ft.cluster_choir(sim, **_CHOIR).ids) == 2


def test_choir_reselect_recovers_groups_amid_noise() -> None:
    # separation lives in 3 of 20 features; the rest are pure noise
    rng = np.random.default_rng(0)
    per, noise_dim = 40, 17
    signal = np.vstack([rng.normal(0, 0.3, (per, 3)), rng.normal(10, 0.3, (per, 3))])
    noise = rng.normal(0, 1, (2 * per, noise_dim))
    coords = np.hstack([signal, noise])
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 2 * per + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(coords.shape[1])},
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(coords.shape[1])])
    ft.cluster(name="run", **_PARAMS)
    labels = ft.cluster_choir(
        "run",
        reselect=True,
        n_features=5,
        min_cluster_size=8,
        n_iterations=40,
        n_estimators=25,
        seed=0,
        n_jobs=1,
    )
    assert len(labels.ids) == 2


def test_choir_reselect_with_pca_runs() -> None:
    ft = _blobs([[0, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 10]], per=40, dim=6)
    ft.cluster(name="run", **_PARAMS)
    labels = ft.cluster_choir("run", reselect=True, n_features=5, n_pcs=3, **_CHOIR)
    assert len(labels.ids) == 2


def test_choir_prunes_high_res_leiden_overclustering() -> None:
    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10], [0, 10, 0, 10]], per=40)
    over = ft.overcluster(
        resolution=4.0, n_neighbors=10, seed=0
    )  # deliberately over-split
    n_over = len(over.ids)
    assert n_over > 3  # high-res Leiden over-splits the 3 groups
    labels = ft.cluster_choir(
        over_clustering=over,
        min_cluster_size=8,
        n_iterations=40,
        n_estimators=25,
        seed=0,
        n_jobs=1,
    )
    # CHOIR prunes the over-clustering back down while keeping the true structure
    assert 3 <= len(labels.ids) < n_over


def test_choir_prunes_kmeans_overclustering() -> None:
    from sklearn.cluster import KMeans

    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10]], per=40)
    km = KMeans(n_clusters=8, random_state=0, n_init=10).fit(ft.features(scaled=True))
    labels = ft.cluster_choir(
        over_clustering=km.labels_,
        min_cluster_size=8,
        n_iterations=40,
        n_estimators=25,
        seed=0,
        n_jobs=1,
    )
    assert len(labels.ids) == 2


def test_cluster_choir_requires_exactly_one_source() -> None:
    import pytest

    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10]], per=20)
    ft.cluster(name="run", **_PARAMS)
    with pytest.raises(ValueError, match="exactly one"):
        ft.cluster_choir("run", over_clustering=np.zeros(40))
    with pytest.raises(ValueError, match="exactly one"):
        ft.cluster_choir()


def test_choir_respects_min_cluster_size() -> None:
    ft = _blobs([[0, 0, 0, 0], [10, 10, 10, 10]], per=40)
    ft.cluster(name="run", **_PARAMS)
    # min_cluster_size larger than either group -> top split can't be tested -> 1 cluster
    labels = ft.cluster_choir(
        "run", min_cluster_size=100, n_iterations=40, n_estimators=25, seed=0, n_jobs=1
    )
    assert len(labels.ids) == 1
