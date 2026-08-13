"""Step 3: consensus clustering library + ft.cluster wiring."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import (
    SimilarityMatrix,
    coclustering_matrix,
    fauxnograph_coclustering,
    make_clipped_scaler,
    neighborhood_purity,
    neighborhood_self_predictions,
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


def _many_features(n: int = 60, dim: int = 20, sep: int = 3, seed: int = 0):
    """Two groups separated in ``sep`` of ``dim`` features, the rest independent noise.

    Realistic for the ``pca=0.95`` default in a way ``_two_blobs`` is not: the
    features carry independent variance, so the reduction keeps many components
    instead of collapsing to one.
    """
    rng = np.random.default_rng(seed)
    coords = rng.normal(0, 1.0, (n, dim))
    coords[n // 2 :, :sep] += 8.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(dim)])


def test_ft_cluster_recovers_two_groups() -> None:
    """The default (``pca=0.95``) path, on features PCA can meaningfully reduce."""
    ft = _many_features()
    sim = ft.cluster(
        n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1, name="run1"
    )

    assert 1 < ft.features_pca(explained_variance=0.95, seed=0).shape[1] < ft.n_features
    assert len(np.unique(sim.cluster_labels(0.5))) == 2
    # stored and retrievable
    assert ft.clustering("run1") is sim
    with pytest.raises(KeyError, match="Unknown clustering"):
        ft.clustering("missing")


def test_ft_cluster_recovers_two_groups_in_the_full_space_too() -> None:
    ft = FeatureTable(_two_blobs(60), features=["m0", "m1", "m2"])
    sim = ft.cluster(
        n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1, pca=False
    )
    assert len(np.unique(sim.cluster_labels(0.5))) == 2


def test_ft_cluster_defaults_to_pca_ninety_five() -> None:
    ft = _many_features()
    params = dict(n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1)

    default = ft.cluster(**params)
    for equivalent in (0.95, True):
        np.testing.assert_array_equal(
            default.similarity_matrix.toarray(),
            ft.cluster(**params, pca=equivalent).similarity_matrix.toarray(),
        )


def _splits_the_blobs(labels: np.ndarray, n: int = 60) -> bool:
    """Every cluster drawn from one blob only (it may split a blob further)."""
    blob = np.arange(n) >= n // 2
    return all(len(np.unique(blob[labels == c])) == 1 for c in np.unique(labels))


def test_ft_cluster_pca_on_perfectly_correlated_features_splits_but_stays_pure() -> (
    None
):
    """The degenerate case: one component, so Leiden cuts finer — but never across.

    Documents why the small fixtures elsewhere pin ``pca=False``. When PCA can
    reduce meaningfully (``_many_features``) both spaces agree, so this fixture is
    also where the choice of space demonstrably changes the consensus.
    """
    ft = FeatureTable(_two_blobs(60), features=["m0", "m1", "m2"])
    params = dict(n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1)
    assert ft.features_pca(explained_variance=0.95, seed=0).shape[1] == 1

    sim = ft.cluster(**params)
    labels = sim.cluster_labels(0.5)
    assert len(np.unique(labels)) > 2
    assert _splits_the_blobs(labels)
    assert not np.array_equal(
        sim.similarity_matrix.toarray(),
        ft.cluster(**params, pca=False).similarity_matrix.toarray(),
    )


def test_ft_cluster_pca_stores_and_labels_like_any_clustering() -> None:
    ft = _many_features()
    sim = ft.cluster(
        n_neighbors=15, resolution=1.0, n_times=3, seed=0, n_jobs=1, pca=0.95, name="p"
    )

    assert ft.clustering("p") is sim
    labels = ft.label("p", distance_threshold=0.5, name="sub")
    assert labels.n_clusters == 2
    assert _splits_the_blobs(labels.codes)


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
    from cellpax.clustering import clipped_scaler_factory

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


def test_neighborhood_self_predictions_excludes_self() -> None:
    data = _two_blobs(60).select("m0", "m1", "m2").to_numpy()
    labels = np.array([0] * 30 + [1] * 30)
    neighbor_labels = neighborhood_self_predictions(data, labels, n_neighbors=5)
    assert neighbor_labels.shape == (60, 5)
    # two well-separated blobs -> every neighbor agrees with its own point's label
    assert np.all(neighbor_labels == labels[:, None])


def test_neighborhood_purity_separated_blobs_is_perfect() -> None:
    data = _two_blobs(60).select("m0", "m1", "m2").to_numpy()
    labels = np.array([0] * 30 + [1] * 30)
    purity = neighborhood_purity(data, labels, n_neighbors=5)
    assert purity.shape == (60,)
    assert np.allclose(purity, 1.0)


def test_neighborhood_purity_scrambled_labels_drop_below_perfect() -> None:
    data = _two_blobs(60).select("m0", "m1", "m2").to_numpy()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=60)
    purity = neighborhood_purity(data, labels, n_neighbors=5)
    assert purity.mean() < 1.0


def test_ft_neighborhood_purity_with_labelset_and_column_name() -> None:
    ft = FeatureTable(_two_blobs(60), features=["m0", "m1", "m2"])
    from cellpax.labels import LabelSet

    labels = LabelSet(np.arange(1, 61), np.array([0] * 30 + [1] * 30), name="blob")
    result = ft.neighborhood_purity(labels, n_neighbors=5)
    assert result.columns == ["cell_id", "purity"]
    assert np.allclose(result["purity"].to_numpy(), 1.0)

    # same result when the labels are already attached and referenced by name
    ft.attach(labels)
    by_column = ft.neighborhood_purity("blob", n_neighbors=5)
    assert np.allclose(by_column["purity"].to_numpy(), 1.0)


# -- clipping rules -------------------------------------------------------------


def _one_outlier(n: int) -> np.ndarray:
    """A clean Gaussian column plus one genuinely extreme cell."""
    rng = np.random.default_rng(0)
    column = rng.normal(size=(n, 1))
    column[0, 0] = 40.0
    return column


def test_percentile_clip_is_still_the_default() -> None:
    from cellpax.clustering import PercentileClipper

    clipper = make_clipped_scaler().named_steps["clipper"]
    assert isinstance(clipper, PercentileClipper)
    assert (clipper.lower, clipper.upper) == (0.1, 99.9)


def test_sigma_mode_swaps_in_the_sample_size_independent_clipper() -> None:
    from cellpax.clustering import SigmaClipper

    clipper = make_clipped_scaler(mode="sigma", n_sigma=4.0).named_steps["clipper"]
    assert isinstance(clipper, SigmaClipper)
    assert clipper.n_sigma == 4.0


def test_unknown_clip_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        make_clipped_scaler(mode="winsorize")  # type: ignore[arg-type]


def test_sigma_clipper_leaves_well_behaved_data_alone() -> None:
    """It clips only what is extreme — possibly nothing, which a percentile cannot do."""
    from cellpax.clustering import SigmaClipper

    rng = np.random.default_rng(0)
    inside = rng.normal(size=(200, 3)) * 0.5
    clipper = SigmaClipper(5.0).fit(inside)
    np.testing.assert_array_equal(clipper.transform(inside), inside)

    with_outliers = inside.copy()
    with_outliers[0, 0] = 9.0
    clipped = clipper.transform(with_outliers)
    assert clipped[0, 0] == 5.0
    assert np.abs(clipped).max() <= 5.0


def test_sigma_clipper_has_no_fitted_bounds() -> None:
    """Which is why a frozen transform carrying it cannot absorb distribution shift."""
    from cellpax.clustering import SigmaClipper

    clipper = SigmaClipper(5.0).fit(np.zeros((10, 4)))
    assert not hasattr(clipper, "lower_bounds_")
    assert not hasattr(clipper, "upper_bounds_")
    assert clipper.n_features_in_ == 4
    with pytest.raises(ValueError, match="n_sigma must be positive"):
        SigmaClipper(0.0).fit(np.zeros((10, 4)))


def test_a_small_cohort_lets_an_outlier_set_its_own_percentile_bound() -> None:
    """The breakdown-point failure, and the reason the sigma rule exists.

    At n=500 the 99.9th percentile is interpolated between the top two order statistics,
    so a single extreme cell partly determines the bound meant to clip it: make the cell
    ten times more extreme and the bound moves out with it, leaving the clipping about as
    weak in relative terms. The sigma bound comes from a median and an IQR, which one cell
    cannot move.
    """
    from sklearn.preprocessing import RobustScaler

    from cellpax.clustering import PercentileClipper, SigmaClipper

    bounds = {}
    for outlier in (40.0, 400.0):
        column = _one_outlier(500)
        column[0, 0] = outlier
        scaled = RobustScaler().fit_transform(column)
        bounds[outlier] = (
            PercentileClipper(0.1, 99.9).fit(scaled).upper_bounds_[0],
            SigmaClipper(5.0).fit(scaled).n_sigma,
        )

    # the percentile bound is dragged outward by the outlier itself
    assert bounds[400.0][0] > 5 * bounds[40.0][0]
    # the sigma bound does not budge
    assert bounds[400.0][1] == bounds[40.0][1]


def test_a_large_cohort_gives_the_percentile_bound_a_stable_tail() -> None:
    """By n≈5000 the tail holds enough points that no single cell moves the bound."""
    from sklearn.preprocessing import RobustScaler

    from cellpax.clustering import PercentileClipper

    bounds = []
    for outlier in (40.0, 400.0):
        column = _one_outlier(5000)
        column[0, 0] = outlier
        scaled = RobustScaler().fit_transform(column)
        bounds.append(PercentileClipper(0.1, 99.9).fit(scaled).upper_bounds_[0])
    assert bounds[1] == pytest.approx(bounds[0], rel=0.05)


def test_the_percentile_clip_bites_on_a_large_cohort() -> None:
    """The same rule, the same data shape, a completely different amount of clipping."""
    from sklearn.preprocessing import RobustScaler

    from cellpax.clustering import PercentileClipper

    counts = []
    for n in (500, 20000):
        scaled = RobustScaler().fit_transform(_one_outlier(n))
        clipper = PercentileClipper(0.1, 99.9).fit(scaled)
        counts.append(int((clipper.transform(scaled) != scaled).sum()))
    assert counts[0] < counts[1] / 5


def test_clipped_scaler_factory_carries_its_configuration() -> None:
    """So persistence records the actual bounds instead of just "some clipped scaler"."""
    from cellpax.clustering import clipped_scaler_factory

    factory = clipped_scaler_factory(1.0, 99.0, mode="sigma", n_sigma=3.0)
    params = factory._cellpax_scaler_params
    assert params == {
        "kind": "clipped",
        "lower": 1.0,
        "upper": 99.0,
        "mode": "sigma",
        "n_sigma": 3.0,
    }
    assert factory().named_steps["clipper"].n_sigma == 3.0


def test_sigma_clipped_scaler_works_as_a_feature_table_scaler() -> None:
    from cellpax.clustering import clipped_scaler_factory

    ft = FeatureTable(
        _two_blobs(60),
        features=["m0", "m1", "m2"],
        scaler_factory=clipped_scaler_factory(mode="sigma", n_sigma=4.0),
    )
    scaled = ft.features(scaled=True)
    assert scaled.shape == (60, 3)
    assert np.isfinite(scaled).all()
    assert np.abs(scaled).max() <= 4.0


# -- graph construction as a third consensus axis --------------------------------


def test_graph_type_sweeps_into_one_consensus() -> None:
    from cellpax.clustering import axis_stability

    data = _two_blobs(80).select("m0", "m1", "m2").to_numpy()
    matrix, partitions = fauxnograph_coclustering(
        data,
        graph_type=["knn", "snn_jaccard", "umap_fuzzy"],
        n_neighbors=[10, 20],
        resolution_parameter=[0.5, 1.0],
        n_times=2,
        seed=0,
        n_jobs=1,
        return_partitions=True,
    )
    assert partitions.n_runs == 3 * 2 * 2 * 2
    assert sorted(set(partitions.graph_type.tolist())) == [
        "knn",
        "snn_jaccard",
        "umap_fuzzy",
    ]
    assert "graph_type" in partitions.summary().columns
    assert "graph_type" in partitions.by_setting().columns

    reference = SimilarityMatrix(matrix, normalized=True).cluster_labels(0.5)
    stability = axis_stability(partitions, reference)
    assert stability.height == 3
    assert set(stability.columns) >= {"graph_type", "mean_ari", "min_ari", "n_runs"}
    assert stability["mean_ari"].min() > 0.5


def test_partitions_default_to_plain_knn() -> None:
    """So runs and reloads from before graph_type was swept still describe themselves."""
    from cellpax.clustering import Partitions

    partitions = Partitions(
        labels=np.zeros((5, 2), dtype=np.int64),
        n_neighbors=np.array([10, 10]),
        resolution=np.array([1.0, 1.0]),
    )
    assert partitions.graph_type.tolist() == ["knn", "knn"]


def test_filter_selects_on_graph_type_and_realised_grain() -> None:
    data = _two_blobs(80).select("m0", "m1", "m2").to_numpy()
    _matrix, partitions = fauxnograph_coclustering(
        data,
        graph_type=["knn", "umap_fuzzy"],
        n_neighbors=10,
        resolution_parameter=[0.2, 1.0, 4.0],
        n_times=2,
        seed=0,
        n_jobs=1,
        return_partitions=True,
    )
    only_fuzzy = partitions.filter(graph_type="umap_fuzzy")
    assert only_fuzzy.n_runs == 6
    assert set(only_fuzzy.graph_type.tolist()) == {"umap_fuzzy"}

    counts = partitions.cluster_counts()
    coarse = partitions.filter(n_clusters_max=int(np.median(counts)))
    assert 0 < coarse.n_runs < partitions.n_runs
    assert coarse.cluster_counts().max() <= int(np.median(counts))

    with pytest.raises(ValueError, match="no runs match"):
        partitions.filter(n_clusters_min=10_000)


def test_axis_stability_rejects_unknown_grouping_and_length_mismatch() -> None:
    from cellpax.clustering import axis_stability

    data = _two_blobs(40).select("m0", "m1", "m2").to_numpy()
    _matrix, partitions = fauxnograph_coclustering(
        data, n_neighbors=10, n_times=2, seed=0, n_jobs=1, return_partitions=True
    )
    with pytest.raises(ValueError, match="by must name settings"):
        axis_stability(partitions, np.zeros(40), by=["alpha"])
    with pytest.raises(ValueError, match="same cells|partitions cover"):
        axis_stability(partitions, np.zeros(7))


def test_a_distance_matrix_input_works_sparse_or_dense() -> None:
    from scipy.sparse import csr_matrix

    distances = np.array(
        [[0.0, 0.1, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]], dtype=float
    )
    dense = SimilarityMatrix(distances, similarity=False)
    sparse = SimilarityMatrix(csr_matrix(distances), similarity=False)
    assert np.allclose(
        dense.similarity_matrix.toarray(), sparse.similarity_matrix.toarray()
    )


def test_cluster_labels_are_zero_based_and_contiguous() -> None:
    coords = _two_blobs().select("m0", "m1", "m2").to_numpy()
    matrix = fauxnograph_coclustering(coords, n_neighbors=10, n_times=2, seed=0)
    sim = SimilarityMatrix(matrix, normalized=True)
    labels = sim.cluster_labels(0.5)
    assigned = np.unique(labels[labels >= 0])
    assert assigned.min() == 0
    assert np.array_equal(assigned, np.arange(assigned.size))


def test_linkage_method_is_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="centroid"):
        SimilarityMatrix(np.eye(3), method="centroid")


def test_normalization_uses_the_shared_run_count_not_the_minimum() -> None:
    """Two cells dropped in *different* runs must not be biased low."""
    # 3 runs; cells 0 and 1 co-cluster in every run that kept both.
    # cell 0 is dropped in run 2, cell 1 is dropped in run 1: both were kept
    # together only in run 0, where they share a group.
    groups = np.array(
        [
            [0, 0, -1],
            [0, -1, 0],
            [1, 1, 1],
        ]
    )
    matrix = coclustering_matrix(groups, normalize=True)
    # |A_0 ∩ A_1| = 1 (run 0) and they co-clustered in it -> similarity 1.0;
    # min(|A_0|, |A_1|) = 2 would have reported 0.5
    assert matrix[0, 1] == pytest.approx(1.0)
