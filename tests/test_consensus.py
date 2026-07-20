"""Step 3: consensus clustering library + ft.cluster wiring."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.consensus import (
    SimilarityMatrix,
    fauxnograph_coclustering,
    make_clipped_scaler,
)
from cellpax.featuretable import FeatureTable


def _two_blobs(n: int = 60):
    rng = np.random.default_rng(0)
    a = rng.normal([0.0, 0.0, 0.0], 0.3, size=(n // 2, 3))
    b = rng.normal([8.0, 8.0, 8.0], 0.3, size=(n // 2, 3))
    coords = np.vstack([a, b])
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": coords[:, 0],
            "m1": coords[:, 1],
            "m2": coords[:, 2],
        }
    )
    return df


def test_fauxnograph_coclustering_and_similarity_matrix() -> None:
    data = _two_blobs(60).select("m0", "m1", "m2").to_numpy()
    M = fauxnograph_coclustering(
        data, n_neighbors=15, resolution_parameter=1.0, n_times=3, seed=0, n_jobs=1
    )
    assert M.shape == (60, 60)
    sim = SimilarityMatrix(M, normalized=True)
    # two well-separated blobs -> a mid threshold yields exactly 2 clusters
    labels = sim.cluster_labels(0.5)
    assert len(np.unique(labels)) == 2
    x, counts = sim.cluster_count_curve(n_points=50)
    assert counts.min() >= 1 and counts.max() >= 2


def test_ft_cluster_recovers_two_groups() -> None:
    ft = FeatureTable(_two_blobs(60), features=["m0", "m1", "m2"])
    sim = ft.cluster(
        n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1, name="run1"
    )
    labels = sim.cluster_labels(0.5)
    assert len(np.unique(labels)) == 2
    # stored and retrievable
    assert ft.clustering("run1") is sim
    with pytest.raises(KeyError, match="Unknown clustering"):
        ft.clustering("missing")


def test_clipped_scaler_as_feature_table_scaler() -> None:
    ft = FeatureTable(
        _two_blobs(60),
        features=["m0", "m1", "m2"],
        scaler_factory=make_clipped_scaler,
    )
    # scaling works end to end with the clipped (RobustScaler+clip) pipeline
    scaled = ft.features(scaled=True)
    assert scaled.shape == (60, 3)
    assert np.isfinite(scaled).all()


def test_similarity_matrix_validation() -> None:
    with pytest.raises(ValueError, match="square"):
        SimilarityMatrix(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        SimilarityMatrix(np.array([[1.0, 0.2], [0.9, 1.0]]))


def test_consensus_branches() -> None:
    from cellpax.consensus import clipped_scaler_factory

    data = _two_blobs(40).select("m0", "m1", "m2").to_numpy()
    # normalize=False, mutual_only=True, sweep lists
    M = fauxnograph_coclustering(
        data,
        n_neighbors=[10],
        resolution_parameter=[1.0],
        n_times=2,
        mutual_only=True,
        normalize=False,
        seed=1,
        n_jobs=1,
    )
    sim = SimilarityMatrix(M, normalized=False)
    # count curve with a minimum cluster size takes the per-threshold path
    x, counts = sim.cluster_count_curve(
        np.linspace(0, sim.max_value, 20), min_cluster_size=5
    )
    assert len(x) == len(counts) == 20

    factory = clipped_scaler_factory(lower=1.0, upper=99.0)
    scaler = factory()
    scaler.fit(data)
    assert scaler.transform(data).shape == data.shape
