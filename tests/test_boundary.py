"""The boundary report: gaps versus cuts, judged by evidence that can disagree."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.boundary import boundary_report
from cellpax.featuretable import FeatureTable


def _two_gaps(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated blobs — every leg should call this discrete."""
    rng = np.random.default_rng(seed)
    half = n // 2
    features = np.vstack(
        [rng.normal(0, 0.5, (half, 3)), rng.normal(8, 0.5, (n - half, 3))]
    )
    codes = np.array([0] * half + [1] * (n - half))
    return features, codes


def _sliced_continuum(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One elongated Gaussian cut at its median — the northwest-corner case."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1.0, (n, 3))
    features[:, 0] *= 6.0  # elongate so the cut axis has room
    codes = (features[:, 0] > np.median(features[:, 0])).astype(np.int64)
    return features, codes


def test_a_real_gap_reads_discrete_on_every_leg() -> None:
    features, codes = _two_gaps()
    report = boundary_report(features, codes)

    row = report.row(0, named=True)
    assert row["verdict"] == "discrete"
    assert row["dip_p"] < 0.05
    assert row["valley_ratio"] < 0.2
    assert row["connectivity_ratio"] < 0.1
    assert row["vote_dip"] == row["vote_valley"] == "discrete"


def test_a_sliced_continuum_reads_continuous() -> None:
    features, codes = _sliced_continuum()
    report = boundary_report(features, codes)

    row = report.row(0, named=True)
    assert row["verdict"] == "continuous"
    assert row["dip_p"] > 0.05  # a cut Gaussian projects unimodally
    assert row["valley_ratio"] > 0.5  # the boundary is as dense as the peaks
    assert row["vote_valley"] == "continuous"


def test_the_consensus_leg_separates_gap_from_cut() -> None:
    """Cross co-clustering mass: pinned near zero over a gap, banded over a cut."""
    from cellpax.clustering import fauxnograph_coclustering

    gap_features, gap_codes = _two_gaps(120)
    cut_features, cut_codes = _sliced_continuum(120)
    sweep = dict(
        n_neighbors=15,
        resolution_parameter=list(np.geomspace(0.1, 2.0, 6)),
        n_times=2,
        seed=0,
        n_jobs=1,
    )
    gap_sim = fauxnograph_coclustering(gap_features, **sweep)
    cut_sim = fauxnograph_coclustering(cut_features, **sweep)

    gap = boundary_report(gap_features, gap_codes, similarity=gap_sim).row(
        0, named=True
    )
    cut = boundary_report(cut_features, cut_codes, similarity=cut_sim).row(
        0, named=True
    )
    assert gap["cocluster_cross_mean"] < 0.05
    assert cut["cocluster_cross_mean"] > gap["cocluster_cross_mean"]
    assert cut["cocluster_band"] > gap["cocluster_band"]


def test_small_clusters_are_reported_not_dropped() -> None:
    features, codes = _two_gaps(60)
    codes = codes.copy()
    codes[:5] = 2  # a 5-cell cluster below min_cells
    report = boundary_report(features, codes, min_cells=10)

    small = report.filter(pl.col("cluster_a") == "2").vstack(
        report.filter(pl.col("cluster_b") == "2")
    )
    assert small.height == 2
    assert set(small["verdict"].to_list()) == {"too_small"}
    assert small["dip"].null_count() == 2


def test_names_and_unassigned_cells_are_respected() -> None:
    features, codes = _two_gaps()
    codes = codes.copy()
    codes[:10] = -1
    report = boundary_report(features, codes, names={0: "L2", 1: "L5"})

    row = report.row(0, named=True)
    assert {row["cluster_a"], row["cluster_b"]} == {"L2", "L5"}
    assert row["n_a"] + row["n_b"] == len(codes) - 10


def test_thresholds_are_parameters_not_truths() -> None:
    features, codes = _sliced_continuum()
    # an absurd threshold flips the valley vote: the report's opinions are
    # parameterized, the evidence columns are not
    strict = boundary_report(features, codes, valley_deep=10.0)
    assert strict.row(0, named=True)["vote_valley"] == "discrete"


def test_it_refuses_a_single_cluster() -> None:
    features, _ = _two_gaps()
    with pytest.raises(ValueError, match="at least two clusters"):
        boundary_report(features, np.zeros(features.shape[0], dtype=int))


def test_pak_density_needs_the_extra_or_a_clear_error() -> None:
    pytest.importorskip("dadapy")
    features, codes = _two_gaps(120)
    report = boundary_report(features, codes, density="pak")
    assert report.row(0, named=True)["verdict"] == "discrete"


# -- the table wrapper ------------------------------------------------------------


def _table(n: int = 160, dim: int = 12) -> FeatureTable:
    """Two groups separated in a few features, the rest independent noise.

    Independent variance keeps ``pca=0.95`` from collapsing to one component
    (the degenerate over-splitting case the guide documents).
    """
    rng = np.random.default_rng(1)
    coords = rng.normal(0, 1.0, (n, dim))
    coords[n // 2 :, :3] += 8.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


def test_ft_boundary_report_rebuilds_the_clustering_space() -> None:
    ft = _table()
    clus = ft.cluster(n_neighbors=15, resolution=0.2, n_times=3, n_jobs=1, name="run")
    labels = clus.label(distance_threshold=0.5, name="kind")
    labels.rename(["left", "right"])

    report = ft.boundary_report("run", labels=labels)
    row = report.row(0, named=True)
    assert {row["cluster_a"], row["cluster_b"]} == {"left", "right"}
    assert row["verdict"] == "discrete"
    assert row["cocluster_cross_mean"] is not None  # the consensus rode along


def test_ft_boundary_report_warns_without_recorded_params() -> None:
    from cellpax.clustering import Clustering, fauxnograph_coclustering

    ft = _table()
    data = ft.features(scaled=True)
    bare = Clustering(
        fauxnograph_coclustering(data, n_neighbors=15, n_times=2, seed=0, n_jobs=1),
        cell_ids=ft._cell_ids(),
        mask="all",
        columns=tuple(ft.feature_columns),
        normalized=True,
    )
    with pytest.warns(UserWarning, match="no recorded parameters"):
        report = ft.boundary_report(bare, distance_threshold=0.5)
    assert report.height == 1
