"""Consensus clustering — the fauxnograph kNN/Leiden library, ported from dfc.

Step 3 of the redesign (see DESIGN_PROPOSAL.md). A functional library the
FeatureTable drives via ``ft.cluster(...)``:

- ``fauxnograph_coclustering`` — repeated kNN/Leiden clustering into a sparse
  co-clustering (consensus) matrix.
- ``SimilarityMatrix`` — the consensus wrapper: hierarchical linkage, cluster
  labels at a distance threshold, and the cluster-count curve.
- clipped/standard/robust scaler factories.

Ported largely as-is from ``dendritic_feature_clustering`` (the current CellPax
fauxnograph is itself a port), trimmed to numpy/scipy (no pandas/tqdm coupling)
to stay polars-native.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import igraph as ig
import leidenalg as la
import numpy as np
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse import hstack as sparse_hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# --------------------------------------------------------------------------- #
# scalers
# --------------------------------------------------------------------------- #


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to per-feature percentile bounds fit on the data."""

    def __init__(self, lower: float = 0.1, upper: float = 99.9) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def make_clipped_scaler(lower: float = 0.1, upper: float = 99.9) -> Pipeline:
    """RobustScaler followed by percentile clipping (dfc's default scaler)."""
    return Pipeline(
        [("scaler", RobustScaler()), ("clipper", PercentileClipper(lower, upper))]
    )


def clipped_scaler_factory(lower: float = 0.1, upper: float = 99.9):
    """Return a zero-argument factory producing a fresh clipped scaler."""

    def factory() -> Pipeline:
        return make_clipped_scaler(lower=lower, upper=upper)

    return factory


# --------------------------------------------------------------------------- #
# kNN / Leiden consensus
# --------------------------------------------------------------------------- #


def kneighbor_graph(
    data: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
) -> ig.Graph:
    """Build an undirected kNN graph from a feature matrix."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(data)
    _, indices = nn.kneighbors(data)
    edges = [
        (i, neighbor)
        for i, neighbors in enumerate(indices)
        for neighbor in neighbors
        if i != neighbor
    ]
    graph = ig.Graph(edges=edges, directed=True)
    mode = "mutual" if mutual_only else "collapse"
    return graph.as_undirected(mode=mode)


def cluster_leiden(
    graph: ig.Graph,
    resolution_parameter: float = 1.0,
    seed: int | None = None,
    min_cluster_size: int = 1,
    partition_type: Any = la.RBConfigurationVertexPartition,
) -> np.ndarray:
    """One Leiden partition of a graph; small clusters become label -1."""
    partition = la.find_partition(
        graph, partition_type, resolution_parameter=resolution_parameter, seed=seed
    )
    labels = np.array(partition.membership)
    if min_cluster_size > 1:
        too_small = np.flatnonzero(np.array(partition.sizes()) < min_cluster_size)
        for index in too_small:
            labels[labels == index] = -1
    return labels


def _run_single(graph, resolution, min_cluster_size, seed) -> np.ndarray:
    return cluster_leiden(
        graph,
        resolution_parameter=resolution,
        seed=seed,
        min_cluster_size=min_cluster_size,
    )


def coclustering_matrix(groups: np.ndarray, normalize: bool = False) -> csr_matrix:
    """Sparse co-clustering counts from an ``(n_cells, n_runs)`` label array.

    Builds one sparse indicator block per run and computes ``A @ A.T`` in a single
    sparse matmul — peak memory O(n × total_clusters), never a dense n×n array.
    Label ``-1`` (dropped small clusters) never co-clusters.
    """
    n, n_runs = groups.shape
    blocks = []
    for run in range(n_runs):
        column = groups[:, run]
        keep = column >= 0
        if not keep.any():
            continue
        cell_idx = np.flatnonzero(keep)
        _, inverse = np.unique(column[keep], return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        blocks.append(
            csr_matrix(
                (np.ones(len(cell_idx), dtype=np.float32), (cell_idx, inverse)),
                shape=(n, n_clusters),
            )
        )
    if not blocks:
        running = csr_matrix((n, n), dtype=np.float32)
    else:
        full = sparse_hstack(blocks, format="csr")
        running = full @ full.T

    if normalize:
        counts = (groups >= 0).sum(axis=1).astype(np.float32)
        cx = coo_matrix(running)
        denom = np.minimum(counts[cx.row], counts[cx.col])
        data = np.where(denom > 0, cx.data / denom, np.float32(0))
        running = coo_matrix((data, (cx.row, cx.col)), shape=(n, n)).tocsr()
        running.eliminate_zeros()
    return running


def fauxnograph_coclustering(
    data: np.ndarray,
    *,
    n_neighbors: int | list[int] = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    n_times: int = 1,
    resolution_parameter: float | list[float] = 1.0,
    min_cluster_size: int = 1,
    normalize: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
) -> csr_matrix:
    """Repeated kNN/Leiden clustering into a sparse consensus matrix.

    ``n_neighbors`` and ``resolution_parameter`` may be lists to sweep several
    settings; every (neighbor, resolution) combination is run ``n_times``.
    """
    neighbors = [n_neighbors] if isinstance(n_neighbors, int) else list(n_neighbors)
    resolutions = (
        [resolution_parameter]
        if isinstance(resolution_parameter, (int, float))
        else list(resolution_parameter)
    )
    total = len(neighbors) * len(resolutions) * n_times
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=total)

    columns: list[np.ndarray] = []
    seed_idx = 0
    for n in neighbors:
        graph = kneighbor_graph(
            data, n_neighbors=n, metric=metric, mutual_only=mutual_only
        )
        jobs = []
        for resolution in resolutions:
            for _ in range(n_times):
                jobs.append((resolution, int(seeds[seed_idx])))
                seed_idx += 1
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single)(graph, resolution, min_cluster_size, run_seed)
            for resolution, run_seed in jobs
        )
        columns.extend(results)

    groups = np.column_stack(columns)
    return coclustering_matrix(groups, normalize=normalize)


# --------------------------------------------------------------------------- #
# SimilarityMatrix
# --------------------------------------------------------------------------- #


class SimilarityMatrix:
    """A consensus/similarity matrix with hierarchical clustering utilities.

    Wraps a (sparse) square similarity matrix and lazily computes the linkage,
    leaf order, cluster labels at a distance threshold, and the cluster-count
    curve. Caches expensive computations.
    """

    def __init__(
        self,
        matrix: np.ndarray | csr_matrix,
        *,
        similarity: bool = True,
        normalized: bool = False,
        method: Literal["average", "single", "complete"] = "average",
    ) -> None:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("similarity matrix must be square")
        if matrix.shape[0] == 0:
            raise ValueError("similarity matrix cannot be empty")
        if not issparse(matrix):
            if not np.allclose(matrix, matrix.T):
                raise ValueError("similarity matrix must be symmetric")
            if normalized and not np.allclose(np.diag(matrix), 1.0, atol=0.01):
                warnings.warn("diagonal is not 1.0; matrix may not be normalized")

        self.max_value = (
            1.0
            if normalized
            else float(matrix.max() if issparse(matrix) else np.max(matrix))
        )
        if not similarity:
            matrix = self.max_value - matrix
        self.similarity_matrix = (
            matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        )
        self.method = method
        self._linkage: np.ndarray | None = None
        self._leaf_order: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.similarity_matrix.shape

    @property
    def linkage(self) -> np.ndarray:
        """Scipy linkage matrix (cached), built from the sparse similarity."""
        if self._linkage is None:
            n = self.similarity_matrix.shape[0]
            condensed = np.full(n * (n - 1) // 2, self.max_value)
            cx = coo_matrix(self.similarity_matrix)
            upper = cx.row < cx.col
            rows, cols, vals = cx.row[upper], cx.col[upper], cx.data[upper]
            condensed_idx = n * rows - rows * (rows + 1) // 2 + cols - rows - 1
            condensed[condensed_idx] = self.max_value - vals
            self._linkage = linkage(condensed, method=self.method)
        return self._linkage

    @property
    def leaf_order(self) -> np.ndarray:
        if self._leaf_order is None:
            self._leaf_order = leaves_list(self.linkage)
        return self._leaf_order

    def cluster_labels(
        self, distance_threshold: float, *, min_cluster_size: int = 1
    ) -> np.ndarray:
        """Cut the dendrogram at ``distance_threshold`` into integer labels.

        Clusters smaller than ``min_cluster_size`` are relabeled ``-1``.
        """
        labels = fcluster(self.linkage, t=distance_threshold, criterion="distance")
        if min_cluster_size > 1:
            values, counts = np.unique(labels, return_counts=True)
            for value in values[counts < min_cluster_size]:
                labels[labels == value] = -1
        return labels

    def cluster_count_curve(
        self,
        distance_range: np.ndarray | None = None,
        *,
        n_points: int = 100,
        min_cluster_size: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Number of clusters as a function of distance threshold."""
        if distance_range is None:
            distance_range = np.linspace(0.0, float(self.linkage[:, 2].max()), n_points)
        if min_cluster_size == 1:
            n_samples = self.shape[0]
            counts = n_samples - np.searchsorted(
                self.linkage[:, 2], distance_range, side="right"
            )
        else:
            n_clusters = []
            for d in distance_range:
                labels = self.cluster_labels(d, min_cluster_size=min_cluster_size)
                n_clusters.append(len(np.unique(labels[labels >= 0])))
            counts = np.array(n_clusters)
        return distance_range, counts
