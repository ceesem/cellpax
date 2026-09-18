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

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import igraph as ig
import leidenalg as la
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse import hstack as sparse_hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

if TYPE_CHECKING:
    from cellpax.labels import LabelSet

# --------------------------------------------------------------------------- #
# scalers
# --------------------------------------------------------------------------- #


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to per-feature percentile bounds fit on the data.

    Defines the bound by *rank*, which has two consequences worth knowing before
    choosing it over :class:`SigmaClipper`.

    Parameters
    ----------
    lower : float, default 0.1
        Lower percentile fitted independently for each feature.
    upper : float, default 99.9
        Upper percentile fitted independently for each feature.

    Notes
    -----
    **The bound stops being robust on small cohorts.** An extreme percentile has a
    breakdown point of roughly ``f/100``, so at the default ``99.9`` it takes only
    ``0.001 × n`` contaminated points to move it — below one cell when ``n < 1000``.
    Concretely, at n=500 ``np.percentile`` interpolates the bound between the top two
    order statistics, so *the outlier sets its own clip bound*: one cell at 40 robust
    units yields a bound near 15, and the same cell at 400 yields a bound near 165. The
    more extreme the cell, the less it is clipped, which inverts the intent. By n≈5000
    the tail holds enough points that the bound settles and stops depending on any one
    of them.

    **And it pulls rare populations toward the bulk by construction.** ``f × n`` cells
    are clipped however clean the data is, so a population that is both rare and
    genuinely extreme is partly clipped automatically. No choice of ``f`` removes that;
    it follows from ranking rather than from anything about the data.
    """

    def __init__(self, lower: float = 0.1, upper: float = 99.9) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PercentileClipper":
        """Fit the per-feature percentile bounds.

        Parameters
        ----------
        X : numpy.ndarray
            ``(n_samples, n_features)`` matrix used to estimate the bounds.
        y : numpy.ndarray, optional
            Ignored. Accepted for scikit-learn transformer compatibility.

        Returns
        -------
        PercentileClipper
            This fitted transformer.
        """
        # nan-aware, so a feature with missing values still gets real bounds rather
        # than NaN ones that would turn every clipped value into NaN
        self.lower_bounds_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip rows to the fitted per-feature percentile bounds.

        Parameters
        ----------
        X : numpy.ndarray
            ``(n_samples, n_features)`` matrix to transform.

        Returns
        -------
        numpy.ndarray
            Clipped matrix with the same shape as ``X``.
        """
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` unchanged: clipping discards information and cannot be undone.

        Values inside the bounds are what ``transform`` left alone, so identity is the
        exact inverse there; a clipped value stays at its bound. Defined so a pipeline
        containing a clipper can still be inverted — which is what mapping one
        dataset's distribution onto another's (``join_datasets``) needs.

        Parameters
        ----------
        X : numpy.ndarray
            ``(n_samples, n_features)`` matrix in clipped units.

        Returns
        -------
        numpy.ndarray
            The same matrix.
        """
        return np.asarray(X)


class SigmaClipper(BaseEstimator, TransformerMixin):
    """Clip every feature at ``±n_sigma``, in the units of the preceding scaler.

    The sample-size-independent alternative to :class:`PercentileClipper`. The bound is
    a fixed value rather than an order statistic, so it means the same thing at n=500 and
    at n=21000, it cannot be moved by the cells it is meant to clip, and it clips *only*
    what is actually extreme — possibly nothing. That makes it the right choice for small
    cohorts, where a percentile bound is set by one or two observations, and for rare
    populations, which a percentile rule pulls toward the bulk by construction.

    ``n_sigma`` is in the units of whatever scaled the data, and in
    :func:`make_clipped_scaler` that is :class:`~sklearn.preprocessing.RobustScaler`,
    which divides by the **IQR, not the standard deviation**. For a Gaussian
    IQR ≈ 1.349σ, so ``n_sigma=5`` is roughly ±6.7 Gaussian σ — do not read ``5.0``
    as five standard deviations. Useful sweep range is nearer 3–5 than 5–10.

    Has no fitted parameters at all: ``fit`` records the feature count and nothing
    else. That is a feature rather than an omission — a frozen transform carries no
    clip bounds, so reapplying it to a future dataset cannot silently absorb that
    dataset's distribution shift into the clipping step.

    Parameters
    ----------
    n_sigma : float, default 5.0
        Symmetric clip bound in the units produced by the preceding scaler.
    """

    def __init__(self, n_sigma: float = 5.0) -> None:
        self.n_sigma = n_sigma

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SigmaClipper":
        """Validate the clip threshold and record the feature count.

        Parameters
        ----------
        X : numpy.ndarray
            ``(n_samples, n_features)`` matrix defining the input width.
        y : numpy.ndarray, optional
            Ignored. Accepted for scikit-learn transformer compatibility.

        Returns
        -------
        SigmaClipper
            This fitted transformer.

        Raises
        ------
        ValueError
            If ``n_sigma`` is not positive.
        """
        if self.n_sigma <= 0:
            raise ValueError(f"n_sigma must be positive, got {self.n_sigma}")
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to ``[-n_sigma, n_sigma]``.

        Parameters
        ----------
        X : numpy.ndarray
            Matrix in the units of the preceding scaler.

        Returns
        -------
        numpy.ndarray
            Clipped matrix with the same shape as ``X``.
        """
        return np.clip(X, -self.n_sigma, self.n_sigma)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` unchanged: clipping discards information and cannot be undone.

        Identity is exact inside ``±n_sigma``, and a clipped value stays at the bound.
        Defined so a pipeline containing the clipper can be inverted.

        Parameters
        ----------
        X : numpy.ndarray
            Matrix in the units of the preceding scaler.

        Returns
        -------
        numpy.ndarray
            The same matrix.
        """
        return np.asarray(X)


def make_clipped_scaler(
    lower: float = 0.1,
    upper: float = 99.9,
    *,
    mode: Literal["percentile", "sigma"] = "percentile",
    n_sigma: float = 5.0,
) -> Pipeline:
    """RobustScaler followed by clipping (dfc's default scaler).

    Parameters
    ----------
    lower, upper : float, default 0.1 and 99.9
        Percentile bounds, used by ``mode="percentile"`` only.
    mode : {'percentile', 'sigma'}, default 'percentile'
        Which clipping rule to use. ``'percentile'`` is the historical behaviour.
    n_sigma : float, default 5.0
        Bound for ``mode="sigma"``, in the units of the preceding ``RobustScaler`` — that
        is **IQR units, not standard deviations**. Ignored by ``mode="percentile"``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``RobustScaler`` then the chosen clipper. Pass the *factory*
        (:func:`clipped_scaler_factory`) rather than this to
        ``FeatureTable(scaler_factory=...)``, since each ``(mask, columns)`` pair needs its
        own fit.

    Notes
    -----
    ``mode="percentile"`` clips at the ``lower``/``upper`` percentiles of the scaled data;
    ``mode="sigma"`` clips at ``±n_sigma`` and ignores ``lower``/``upper``.

    Prefer ``mode="sigma"`` for cohorts of a few hundred cells, where a percentile bound
    is set by one or two observations and an outlier ends up choosing its own bound, and
    wherever a rare-but-real population must not be dragged toward the bulk. See
    :class:`SigmaClipper` for what ``n_sigma`` actually measures — it is IQR units, not
    standard deviations.

    Clipping runs *after* scaling in both modes, which is what gives ``n_sigma``
    its units.
    """
    if mode == "percentile":
        clipper: Any = PercentileClipper(lower, upper)
    elif mode == "sigma":
        clipper = SigmaClipper(n_sigma)
    else:
        raise ValueError(
            f"mode must be 'percentile' or 'sigma', got {mode!r}; "
            "'sigma' clips at ±n_sigma in RobustScaler (IQR) units and is the "
            "sample-size-independent choice"
        )
    return Pipeline([("scaler", RobustScaler()), ("clipper", clipper)])


def clipped_scaler_factory(
    lower: float = 0.1,
    upper: float = 99.9,
    *,
    mode: Literal["percentile", "sigma"] = "percentile",
    n_sigma: float = 5.0,
) -> Callable[[], Pipeline]:
    """Return a zero-argument factory producing a fresh clipped scaler.

    Parameters
    ----------
    lower, upper : float, default 0.1 and 99.9
        Percentile bounds, used by ``mode="percentile"`` only.
    mode : {'percentile', 'sigma'}, default 'percentile'
        Which clipping rule the produced scalers use.
    n_sigma : float, default 5.0
        Bound for ``mode="sigma"``, in IQR units — see :class:`SigmaClipper`.

    Returns
    -------
    callable
        A zero-argument callable returning a fresh pipeline. This is the form
        ``FeatureTable(scaler_factory=...)`` wants, since each ``(mask, columns)`` pair gets
        its own fit.

    Notes
    -----
    The returned factory carries its configuration, so persistence records the actual
    percentiles or ``n_sigma`` rather than just "some clipped scaler" — reloading at the
    wrong bounds would change every scaled value silently.
    """

    def factory() -> Pipeline:
        return make_clipped_scaler(lower=lower, upper=upper, mode=mode, n_sigma=n_sigma)

    factory._cellpax_scaler_params = {  # type: ignore[attr-defined]
        "kind": "clipped",
        "lower": float(lower),
        "upper": float(upper),
        "mode": mode,
        "n_sigma": float(n_sigma),
    }
    return factory


def quantile_scaler_factory(
    *,
    clip: tuple[float, float] | None = (1.0, 99.0),
    output_distribution: Literal["normal", "uniform"] = "normal",
    n_quantiles: int = 1000,
    subsample: int = 100_000,
    random_state: int = 0,
) -> Callable[[], Any]:
    """Return a factory producing a percentile clip followed by a quantile transform.

    A quantile transform maps each feature to its rank within the fit population and
    then onto a fixed reference distribution, so it removes *any* monotone difference
    in a feature's marginal — offset, scale, units, and the nonlinear warps a different
    detection threshold or extraction pipeline introduces — not just location and
    scale. That is what makes it the harmonizer of choice for
    :func:`~cellpax.datasets.join_datasets`: fit once per dataset (or per dataset ×
    subclass), then map through one dataset's transform and back out through the
    reference's inverse.

    Parameters
    ----------
    clip : tuple of float or None, default (1.0, 99.0)
        Percentile bounds applied *before* the rank transform, fit on the same cells.
        Clipping collapses each tail onto its bound, so a handful of extreme cells
        cannot claim the outermost quantiles. ``None`` skips the clip.
    output_distribution : {'normal', 'uniform'}, default 'normal'
        Reference distribution of the transformed values. ``'normal'`` suits Euclidean
        neighbour graphs when the scaler is used directly on a table.
    n_quantiles : int, default 1000
        Number of quantile landmarks; capped by scikit-learn at the number of fit cells.
    subsample : int, default 100000
        Maximum number of cells used to estimate the quantiles.
    random_state : int, default 0
        Seed for that subsample, fixed so a refit reproduces the same transform.

    Returns
    -------
    callable
        A zero-argument callable returning a fresh unfitted scaler: a
        ``Pipeline([("clipper", PercentileClipper), ("scaler", QuantileTransformer)])``,
        or a bare ``QuantileTransformer`` when ``clip`` is ``None``.

    Notes
    -----
    A rank transform discards within-population scale by construction: after it, every
    feature has the same marginal. Used as a table's own ``scaler_factory`` that is a
    strong normalization; used through ``join_datasets``, the reference dataset's
    inverse restores real units, so only the *difference* between datasets is removed.
    """
    from sklearn.preprocessing import QuantileTransformer

    if clip is not None:
        lower, upper = (float(clip[0]), float(clip[1]))
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError(f"clip must satisfy 0 <= lower < upper <= 100, got {clip}")
    if output_distribution not in ("normal", "uniform"):
        raise ValueError(
            f"output_distribution must be 'normal' or 'uniform', got "
            f"{output_distribution!r}"
        )

    def factory() -> Any:
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=random_state,
        )
        if clip is None:
            return transformer
        return Pipeline(
            [("clipper", PercentileClipper(lower, upper)), ("scaler", transformer)]
        )

    factory._cellpax_scaler_params = {  # type: ignore[attr-defined]
        "kind": "quantile",
        "clip": None if clip is None else [lower, upper],
        "output_distribution": output_distribution,
        "n_quantiles": int(n_quantiles),
        "subsample": int(subsample),
        "random_state": int(random_state),
    }
    return factory


# --------------------------------------------------------------------------- #
# kNN / Leiden consensus
# --------------------------------------------------------------------------- #


GraphType = Literal["knn", "knn_distance", "snn_jaccard", "umap_fuzzy"]

GRAPH_TYPES: tuple[str, ...] = ("knn", "knn_distance", "snn_jaccard", "umap_fuzzy")

_logger = logging.getLogger("cellpax.clustering")


def _neighbor_arrays(
    data: np.ndarray, n_neighbors: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(distances, indices)`` of each row's ``n_neighbors`` neighbours, self removed.

    Self is dropped by position rather than by assuming it comes first: with duplicate
    rows scikit-learn is free to return some other tied row as the nearest match, and
    slicing ``[:, 1:]`` would then silently drop a real neighbour and keep the self-edge.
    A stable argsort over the is-self mask moves self to the end while preserving
    distance order, so exactly ``n_neighbors`` genuine neighbours survive per row.
    """
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    finder.fit(data)
    distances, indices = finder.kneighbors(data)
    is_self = indices == np.arange(indices.shape[0])[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")[:, :n_neighbors]
    rows = np.arange(indices.shape[0])[:, None]
    return distances[rows, order], indices[rows, order]


def _directed_from_neighbors(
    indices: np.ndarray, weights: np.ndarray, n: int
) -> csr_matrix:
    """Sparse ``(n, n)`` matrix with ``weights[i, j]`` on each observed neighbour edge."""
    rows = np.repeat(np.arange(n), indices.shape[1])
    return coo_matrix((weights.ravel(), (rows, indices.ravel())), shape=(n, n)).tocsr()


def _distance_weights(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locally-scaled kernel weights ``exp(-d / mean_i(d))``, plus a degenerate mask.

    The bandwidth is per-cell, so a weight means "near for this cell" rather than "near
    in absolute units" — necessary when density varies across the population, which for
    morphological features it does.
    """
    scale = distances.mean(axis=1, keepdims=True)
    degenerate = (scale <= 0).ravel()
    safe = np.where(scale > 0, scale, 1.0)
    weights = np.exp(-distances / safe)
    weights[degenerate] = 1.0
    return weights, degenerate


def _smooth_knn_weights(
    distances: np.ndarray, *, n_iter: int = 64, tolerance: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """UMAP's fuzzy simplicial set membership strengths, plus a degenerate mask.

    For each cell, subtract ``rho_i`` (its distance to the nearest *distinct* neighbour)
    and pick ``sigma_i`` by bisection so the membership strengths sum to ``log2(k)``:

    .. math:: w_{ij} = \\exp(-\\max(0, d_{ij} - \\rho_i) / \\sigma_i)

    Subtracting ``rho_i`` is the whole point of this weighting: the nearest neighbour
    lands at ``d = rho`` and therefore keeps weight exactly ``1.0``, so no cell is ever
    fully disconnected however sparse its region. That is what tends to keep rare and
    peripheral populations attached, and it is exactly the guarantee SNN/Jaccard lacks.

    Implemented here rather than imported from ``umap-learn`` for two reasons: it keeps
    the dependency optional, and it makes the graph a separate object from any UMAP
    *embedding*'s graph by construction, so comparing feature-space neighbourhoods with
    embedding-space neighbourhoods cannot become circular.
    """
    n, k = distances.shape
    target = np.log2(k)
    rho = np.zeros(n, dtype=float)
    positive = distances > 0
    has_positive = positive.any(axis=1)
    if has_positive.any():
        masked = np.where(positive, distances, np.inf)
        rho[has_positive] = masked[has_positive].min(axis=1)

    offsets = np.clip(distances - rho[:, None], 0.0, None)
    lo = np.zeros(n, dtype=float)
    hi = np.full(n, np.inf, dtype=float)
    sigma = np.ones(n, dtype=float)
    for _ in range(n_iter):
        strengths = np.exp(-offsets / sigma[:, None]).sum(axis=1)
        if np.all(np.abs(strengths - target) < tolerance):
            break
        too_wide = strengths > target
        hi = np.where(too_wide, sigma, hi)
        lo = np.where(too_wide, lo, sigma)
        unbounded = ~too_wide & np.isinf(hi)
        sigma = np.where(
            unbounded, sigma * 2.0, (lo + np.where(np.isinf(hi), lo, hi)) / 2.0
        )

    # UMAP's floor on sigma: without it, a cell whose neighbours are all nearly
    # equidistant drives sigma toward zero and every weight but the first collapses.
    mean_distance = distances.mean(axis=1)
    sigma = np.maximum(sigma, 1e-3 * np.where(mean_distance > 0, mean_distance, 1.0))

    weights = np.exp(-offsets / sigma[:, None])
    degenerate = ~has_positive
    weights[degenerate] = 1.0
    return weights, degenerate


def _jaccard_weights(indices: np.ndarray, n: int, prune: float) -> csr_matrix:
    """Shared-nearest-neighbour Jaccard overlap on the kNN edges.

    ``|N(i) ∩ N(j)| / |N(i) ∪ N(j)|`` over closed neighbourhoods, computed as one sparse
    boolean matmul and kept only on pairs that are already kNN-adjacent.

    Unlike the fuzzy set this offers no guarantee that a cell keeps any edge at all: two
    cells in a sparse region can be mutual neighbours and still share almost no
    neighbourhood, so their edge is weak and ``prune`` may remove it. That denoises dense
    regions harder and fragments sparse ones, which is the tradeoff to be aware of when
    it disagrees with the other weightings about a rare population. Cells stranded that
    way are counted as ``graph["n_isolated"]`` and warned about.
    """
    k = indices.shape[1]
    rows = np.repeat(np.arange(n), k)
    closed = coo_matrix(
        (
            np.ones(rows.size + n, dtype=np.float32),
            (
                np.concatenate([rows, np.arange(n)]),
                np.concatenate([indices.ravel(), np.arange(n)]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    closed.data[:] = 1.0  # collapse any duplicate neighbour entries to a set
    adjacency = _directed_from_neighbors(indices, np.ones_like(indices, dtype=float), n)
    adjacency = adjacency.maximum(adjacency.T)
    intersection = (closed @ closed.T).multiply(adjacency.astype(bool))
    intersection = intersection.tocoo()
    sizes = np.asarray(closed.sum(axis=1)).ravel()
    union = sizes[intersection.row] + sizes[intersection.col] - intersection.data
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection.data / union, 0.0)
    keep = (jaccard > prune) & (intersection.row != intersection.col)
    return coo_matrix(
        (jaccard[keep], (intersection.row[keep], intersection.col[keep])), shape=(n, n)
    ).tocsr()


def _graph_from_matrix(matrix: csr_matrix, n: int) -> ig.Graph:
    """Undirected igraph from the upper triangle of a symmetric weight matrix."""
    upper = coo_matrix(matrix)
    keep = upper.row < upper.col
    graph = ig.Graph(n=n, edges=list(zip(upper.row[keep], upper.col[keep])))
    graph.es["weight"] = upper.data[keep].astype(float).tolist()
    return graph


def kneighbor_graph(
    data: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    graph_type: GraphType = "knn",
    *,
    prune: float = 0.0,
) -> ig.Graph:
    """Build an undirected neighbour graph from a feature matrix.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_cells, n_features)`` coordinates. Already scaled and, usually, reduced.
    n_neighbors : int, default 30
        Neighbours per cell, self excluded.
    metric : str, default 'minkowski'
        Distance metric, passed to :class:`sklearn.neighbors.NearestNeighbors`.
    mutual_only : bool, default False
        Keep only reciprocated edges. **``graph_type="knn"`` only** — the weighted forms
        are symmetrised by their own rule, which is part of what defines them.
    graph_type : {'knn', 'knn_distance', 'snn_jaccard', 'umap_fuzzy'}, default 'knn'
        How edges are weighted. See the notes; the choice is not neutral.
    prune : float, default 0.0
        Jaccard floor for ``'snn_jaccard'``: edges at or below it are dropped. Ignored by
        every other weighting. Seurat uses ``1/15``.

    Returns
    -------
    igraph.Graph
        Undirected. Weighted types carry ``es["weight"]``; ``'knn'`` does not, which is how
        :func:`cluster_leiden` decides whether to pass weights. Graph attributes record
        ``graph_type``, ``n_neighbors``, ``n_degenerate`` (cells whose whole neighbourhood
        sits at distance 0, i.e. duplicate rows) and ``n_isolated`` (cells left with no
        edges at all — distinct, and the cost of aggressive Jaccard pruning).

    Notes
    -----
    The three weightings have known and *different* failure modes, which is why this
    belongs in the consensus ensemble rather than being fixed upstream:

    ``"knn"``
        Unweighted, the historical default. Every edge counts the same, so this is the
        only option where Leiden's RBConfiguration null is the plain configuration model
        on unit weights.
    ``"knn_distance"``
        ``exp(-d / mean_i(d))`` with a per-cell bandwidth. The distance-aware baseline.
    ``"snn_jaccard"``
        Neighbourhood overlap. Denoises dense regions hardest and fragments sparse ones;
        offers no guarantee a cell keeps any edge. ``prune`` drops edges below a Jaccard
        floor (Seurat uses ``1/15``).
    ``"umap_fuzzy"``
        UMAP's fuzzy simplicial set. Every cell keeps at least one full-weight edge, so
        it tends to hold rare and peripheral populations together.

    Weighted graphs carry ``es["weight"]`` and ``"knn"`` does not, which is how
    :func:`cluster_leiden` decides whether to pass weights.

    Two things to watch. Resolution is **not comparable across graph types**: fuzzy
    weights, Jaccard values in ``[0, 1]``, and unit weights put RBConfiguration on three
    different scales, so one geomspaced sweep lands at very different granularities per
    type. Pool on realised cluster count (``Partitions.filter(n_clusters_min=...)``)
    rather than on nominal resolution. And ``mutual_only`` applies to ``"knn"`` only;
    the weighted forms are symmetrised by their own rule (max for the distance kernel,
    the probabilistic t-conorm for the fuzzy set), which is part of what defines them.
    """
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"graph_type must be one of {GRAPH_TYPES}, got {graph_type!r}")
    data = np.asarray(data)
    n = data.shape[0]

    if graph_type == "knn":
        # Kept verbatim: this is the historical path and its output is pinned by tests.
        finder = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        finder.fit(data)
        _, indices = finder.kneighbors(data)
        edges = [
            (i, neighbor)
            for i, neighbors in enumerate(indices)
            for neighbor in neighbors
            if i != neighbor
        ]
        graph = ig.Graph(edges=edges, directed=True)
        mode = "mutual" if mutual_only else "collapse"
        graph = graph.as_undirected(mode=mode)
        n_degenerate = 0
    else:
        distances, indices = _neighbor_arrays(data, n_neighbors, metric)
        # Independent of the weighting: a cell whose whole neighbourhood sits at
        # distance 0 has duplicate feature rows, and neither rho_i nor a local bandwidth
        # means anything for it.
        n_degenerate = int((~(distances > 0).any(axis=1)).sum())
        if graph_type == "knn_distance":
            weights, _ = _distance_weights(distances)
            directed = _directed_from_neighbors(indices, weights, n)
            matrix = directed.maximum(directed.T)
        elif graph_type == "umap_fuzzy":
            weights, _ = _smooth_knn_weights(distances)
            directed = _directed_from_neighbors(indices, weights, n)
            transpose = directed.T.tocsr()
            # Probabilistic t-conorm: a ∪ b = a + b - a·b, UMAP's symmetrisation.
            matrix = directed + transpose - directed.multiply(transpose)
        else:
            matrix = _jaccard_weights(indices, n, prune)
        graph = _graph_from_matrix(matrix.tocsr(), n)

    n_isolated = int(np.asarray(graph.degree()).__eq__(0).sum())
    graph["n_degenerate"] = n_degenerate
    graph["n_isolated"] = n_isolated
    graph["graph_type"] = graph_type
    graph["n_neighbors"] = int(n_neighbors)
    if n_degenerate:
        warnings.warn(
            f"{n_degenerate} cells have a degenerate neighbourhood (all {n_neighbors} "
            f"neighbours at distance 0) in the {graph_type!r} graph, so their edge "
            "weights carry no local scale; check for duplicate feature rows with "
            "cellpax.diagnostics.duplicate_rows",
            stacklevel=2,
        )
    if n_isolated:
        # Distinct from the above, and the documented cost of Jaccard weighting: two
        # cells in a sparse region can be mutual neighbours and still share almost no
        # neighbourhood, so pruning can strand them. An isolated cell cannot join any
        # community, so it is dropped from every partition rather than mis-assigned.
        warnings.warn(
            f"{n_isolated} cells have no edges in the {graph_type!r} graph at "
            f"n_neighbors={n_neighbors}"
            + (f", prune={prune:g}" if graph_type == "snn_jaccard" else "")
            + "; they cannot join any community. Sparse regions fragment under Jaccard "
            "weighting in particular — lower prune, raise n_neighbors, or compare "
            "against graph_type='umap_fuzzy', which leaves every cell one full-weight "
            "edge",
            stacklevel=2,
        )
    _logger.info(
        "graph built: type=%s n_neighbors=%d nodes=%d edges=%d degenerate=%d isolated=%d",
        graph_type,
        n_neighbors,
        graph.vcount(),
        graph.ecount(),
        n_degenerate,
        n_isolated,
    )
    return graph


def cluster_leiden(
    graph: ig.Graph,
    resolution_parameter: float = 1.0,
    seed: int | None = None,
    min_cluster_size: int = 1,
    partition_type: Any = la.RBConfigurationVertexPartition,
) -> np.ndarray:
    """One Leiden partition of a graph; small clusters become label -1.

    Edge weights are honoured when the graph carries them, which the weighted
    ``graph_type`` builders set and the plain kNN builder does not. Ignoring them would
    quietly turn every weighting back into an unweighted graph.
    """
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    partition = la.find_partition(
        graph,
        partition_type,
        weights=weights,
        resolution_parameter=resolution_parameter,
        seed=seed,
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


def consensus_density(groups: np.ndarray) -> dict[str, float]:
    """Predict how dense ``A @ A.T`` will be, before paying for it.

    The consensus matrix is sparse only when the runs are *fine*. Each run
    contributes a ``size**2`` block per cluster, so the fraction of pairs a run
    co-clusters is ``Σ size**2 / n**2``, and a pair is nonzero in the pool when
    any run co-clusters it. Treating runs as independent gives

        density ≈ 1 - Π_runs (1 - p_run)

    which is accurate to a tenth of a percent against measured nnz, and errs
    high on real data (correlated runs overlap, so the union is smaller than the
    independent estimate) — the safe direction for a warning.

    This is the number to look at before ``Clustering.restrict``. Narrowing to a
    coarse grain window selects the runs with the largest clusters, so the
    restricted consensus can be an order of magnitude denser than the pool it
    came from: at 20 runs, 40 clusters each gives ~40% density, while 4 clusters
    each gives ~99.7%.

    Returns ``density``, predicted ``nnz``, ``gb`` for the CSR matrix, and
    ``peak_gb`` — the normalization pass runs about 1.5x the matrix.

    Parameters
    ----------
    groups : numpy.ndarray
        Integer ``(n_cells, n_runs)`` membership matrix; negative values mark
        cells omitted from a run.

    Returns
    -------
    dict of str to float
        Predicted density, nonzero count, matrix size, peak working size, and
        cell count.
    """
    n, n_runs = groups.shape
    if n == 0 or n_runs == 0:
        return {"density": 0.0, "nnz": 0.0, "gb": 0.0, "peak_gb": 0.0, "n_cells": n}
    survives = 1.0
    for run in range(n_runs):
        column = groups[:, run]
        column = column[column >= 0]
        if column.size == 0:
            continue
        _, counts = np.unique(column, return_counts=True)
        p_run = float((counts.astype(np.float64) ** 2).sum()) / float(n) ** 2
        survives *= max(0.0, 1.0 - p_run)
    density = 1.0 - survives
    nnz = density * float(n) ** 2
    gb = nnz * 8 / 1e9  # float32 data + int32 indices
    return {
        "density": density,
        "nnz": nnz,
        "gb": gb,
        "peak_gb": gb * 1.5,
        "n_cells": n,
    }


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
        _normalize_inplace(running, groups)
    return running


def _normalize_inplace(running: csr_matrix, groups: np.ndarray) -> None:
    """Divide co-clustering counts by the runs that kept both cells, in place.

    The denominator is the documented one: ``|A_i ∩ A_j|``, the number of runs
    that kept *both* cells. ``min(|A_i|, |A_j|)`` only agrees when every drop is
    shared; two cells dropped in different runs would otherwise be biased low.

    Done row-chunked and in place because this is where consensus memory
    actually goes. The obvious version round-trips through COO — ``coo_matrix``,
    a row array, a column array, a denominator, a divided copy, then a fresh
    ``coo_matrix(...).tocsr()`` — which measures at ~4x the matrix in peak RSS.
    That is affordable on a sparse consensus and ruinous on a dense one, and
    ``restrict`` produces exactly the dense case: narrowing to coarse runs
    selects the runs whose clusters are largest, and ``A @ A.T`` gains a
    ``size**2`` block per cluster, so a pool that was 10% dense can come back at
    99%. At 34.6k cells that is the difference between ~10 GB and ~40 GB.

    Chunking keeps the temporaries proportional to a slice rather than to nnz,
    so peak stays just above the matrix itself.
    """
    n, n_runs = groups.shape
    if running.nnz == 0:
        return
    running.sort_indices()
    indptr, indices, data = running.indptr, running.indices, running.data
    missing = groups < 0
    miss_counts = missing.sum(axis=1).astype(np.float32)
    any_missing = bool(miss_counts.any())

    shared: csr_matrix | None = None
    if any_missing:
        # Pairs where BOTH cells were ever dropped are the only ones the
        # correction can touch. M·Mᵀ counts the runs that dropped both, read
        # back per chunk rather than as one nnz-sized gather.
        miss_sparse = csr_matrix(missing.astype(np.float32))
        shared = (miss_sparse @ miss_sparse.T).tocsr()
        shared.sort_indices()

    # ~8M nnz per chunk keeps the working set in the low hundreds of MB
    # regardless of how dense the consensus turned out to be.
    per_row = max(running.nnz / max(n, 1), 1.0)
    chunk = max(1, min(n, int(8_000_000 / per_row)))

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        lo, hi = int(indptr[start]), int(indptr[stop])
        if hi == lo:
            continue
        cols = indices[lo:hi]
        rows = np.repeat(
            np.arange(start, stop, dtype=np.int64),
            np.diff(indptr[start : stop + 1]).astype(np.int64),
        )
        denom = np.float32(n_runs) - miss_counts[rows] - miss_counts[cols]
        if shared is not None:
            sub = shared[start:stop]
            if sub.nnz:
                # both index spaces are row-sorted, so one key search per chunk
                # recovers the shared-miss count for exactly these pairs
                sub_rows = np.repeat(
                    np.arange(start, stop, dtype=np.int64),
                    np.diff(sub.indptr).astype(np.int64),
                )
                sub_key = sub_rows * n + sub.indices.astype(np.int64)
                key = rows * n + cols.astype(np.int64)
                position = np.searchsorted(sub_key, key)
                np.clip(position, 0, sub_key.size - 1, out=position)
                hit = sub_key[position] == key
                denom += np.where(hit, sub.data[position], np.float32(0)).astype(
                    np.float32
                )
        block = data[lo:hi]
        np.divide(block, denom, out=block, where=denom > 0)
        block[denom <= 0] = np.float32(0)

    running.eliminate_zeros()


@dataclass(frozen=True)
class Partitions:
    """The individual kNN/Leiden runs behind a consensus matrix.

    The consensus matrix is a summary, and summaries hide their inputs. This is the
    input: ``labels`` is ``(n_cells, n_runs)`` with the raw Leiden membership of every
    run (``-1`` for a cell the run dropped), and ``n_neighbors`` / ``resolution``
    record the setting each column came from.

    Worth reading whenever a consensus comes out sparser or more finely split than
    expected, because those are properties of the runs, not of the consensus: cells
    can only co-cluster as often as the runs put them together, so runs that each
    split into thirty communities cannot produce a dense matrix or a handful of
    clusters no matter how the dendrogram is cut. ``by_setting()`` is usually the
    fastest look.

    ``graph_type`` records which weighting each run's graph used, and defaults to
    ``"knn"`` throughout for runs (or reloads) from before that was a swept axis.

    Attributes
    ----------
    labels : numpy.ndarray
        Integer ``(n_cells, n_runs)`` membership matrix; ``-1`` is unassigned.
    n_neighbors : numpy.ndarray
        Neighbour count for each run, in matrix-column order.
    resolution : numpy.ndarray
        Leiden resolution for each run.
    graph_type : numpy.ndarray
        Graph weighting name for each run.
    """

    labels: np.ndarray
    n_neighbors: np.ndarray
    resolution: np.ndarray
    graph_type: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.graph_type is None:
            filled = np.full(self.labels.shape[1], "knn", dtype="<U16")
            object.__setattr__(self, "graph_type", filled)
        else:
            object.__setattr__(
                self, "graph_type", np.asarray(self.graph_type, dtype="<U16")
            )

    @property
    def n_runs(self) -> int:
        """Number of clustering runs."""
        return int(self.labels.shape[1])

    @property
    def n_cells(self) -> int:
        """Number of cells represented by each run."""
        return int(self.labels.shape[0])

    def cluster_counts(self) -> np.ndarray:
        """Count clusters produced by each run.

        Returns
        -------
        numpy.ndarray
            One cluster count per run.
        """
        return np.array(
            [np.unique(col[col >= 0]).size for col in self.labels.T], dtype=np.int64
        )

    def summary(self) -> pl.DataFrame:
        """Summarize each ensemble run.

        Returns
        -------
        polars.DataFrame
            Settings, cluster count, and dropped-cell count per run.
        """
        return pl.DataFrame(
            {
                "run": np.arange(self.n_runs, dtype=np.int64),
                "graph_type": self.graph_type,
                "n_neighbors": self.n_neighbors,
                "resolution": self.resolution,
                "n_clusters": self.cluster_counts(),
                "n_unassigned": (self.labels < 0).sum(axis=0).astype(np.int64),
                "largest_cluster": np.array(
                    [
                        int(np.bincount(col[col >= 0]).max()) if (col >= 0).any() else 0
                        for col in self.labels.T
                    ],
                    dtype=np.int64,
                ),
            }
        )

    def by_setting(self) -> pl.DataFrame:
        """Cluster counts aggregated per ``(graph_type, n_neighbors, resolution)``.

        Read this against the number of groups you expect. A resolution whose runs
        each produce thirty communities is not going to consense into five.

        Read it *across* graph types too, because the resolution column does not mean
        the same thing in each: weighted graphs put Leiden's resolution on different
        scales, so the same nominal value can produce five communities under one
        weighting and eighty under another. The ``median_clusters`` column is the
        comparable one, and :meth:`filter` takes ``n_clusters_min``/``n_clusters_max``
        so the pool can be selected on it.

        Returns
        -------
        polars.DataFrame
            Cluster-count summaries grouped by graph settings.
        """
        return (
            self.summary()
            .group_by("graph_type", "n_neighbors", "resolution")
            .agg(
                pl.len().alias("n_runs"),
                pl.col("n_clusters").min().alias("min_clusters"),
                pl.col("n_clusters").median().alias("median_clusters"),
                pl.col("n_clusters").max().alias("max_clusters"),
                pl.col("n_unassigned").mean().alias("mean_unassigned"),
            )
            .sort("graph_type", "n_neighbors", "resolution")
        )

    def filter(
        self,
        *,
        resolution_min: float | None = None,
        resolution_max: float | None = None,
        n_neighbors: int | Sequence[int] | None = None,
        graph_type: str | Sequence[str] | None = None,
        n_clusters_min: int | None = None,
        n_clusters_max: int | None = None,
    ) -> "Partitions":
        """The subset of runs matching these settings.

        Parameters
        ----------
        resolution_min, resolution_max : float, optional
            Keep runs whose *nominal* resolution falls in this range.
        n_neighbors : int or sequence of int, optional
            Keep runs using these neighbourhood sizes.
        graph_type : str or sequence of str, optional
            Keep runs using these edge weightings — the way to see what one weighting says
            before it is pooled with the others.
        n_clusters_min, n_clusters_max : int, optional
            Keep runs whose *realised* cluster count falls in this range. This is the
            selection to use when pooling across graph types: a nominal resolution is not
            comparable between weightings but a cluster count is, so it puts every
            weighting's contribution in the same band instead of letting whichever one
            happened to land mid-range dominate.

        Returns
        -------
        Partitions
            The matching runs. Raises if none match, listing what was actually swept.

        Notes
        -----
        Averaging runs of different grain does not give you both grains — it gives a
        weighted blend, in which the coarse structure is diluted in proportion to how many
        of the runs split it. Sweeping ``0.1`` to ``1.5`` and keeping only the low end
        recovers the broad answer that sweep buried, and since the runs are already
        computed it costs nothing to look.
        """
        keep = np.ones(self.n_runs, dtype=bool)
        if resolution_min is not None:
            keep &= self.resolution >= resolution_min
        if resolution_max is not None:
            keep &= self.resolution <= resolution_max
        if n_neighbors is not None:
            wanted = (
                [n_neighbors] if isinstance(n_neighbors, int) else list(n_neighbors)
            )
            keep &= np.isin(self.n_neighbors, wanted)
        if graph_type is not None:
            types = [graph_type] if isinstance(graph_type, str) else list(graph_type)
            keep &= np.isin(self.graph_type, types)
        if n_clusters_min is not None or n_clusters_max is not None:
            counts = self.cluster_counts()
            if n_clusters_min is not None:
                keep &= counts >= n_clusters_min
            if n_clusters_max is not None:
                keep &= counts <= n_clusters_max
        if not keep.any():
            counts = self.cluster_counts()
            raise ValueError(
                f"no runs match; this clustering swept graph types "
                f"{sorted(set(self.graph_type.tolist()))}, n_neighbors "
                f"{sorted(set(self.n_neighbors.tolist()))}, resolutions "
                f"{sorted(set(np.round(self.resolution, 4).tolist()))}, and realised "
                f"{counts.min()}-{counts.max()} clusters per run"
            )
        return Partitions(
            labels=self.labels[:, keep],
            n_neighbors=self.n_neighbors[keep],
            resolution=self.resolution[keep],
            graph_type=self.graph_type[keep],
        )

    def coclustering(self, *, normalize: bool = True) -> csr_matrix:
        """Re-consense just these runs into a co-clustering matrix.

        Parameters
        ----------
        normalize : bool, default True
            Divide counts by runs that retained both cells.

        Returns
        -------
        scipy.sparse.csr_matrix
            Square co-clustering matrix over the partition rows.
        """
        return coclustering_matrix(self.labels, normalize=normalize)

    def __repr__(self) -> str:
        counts = self.cluster_counts()
        types = sorted(set(self.graph_type.tolist()))
        return (
            f"Partitions(n_cells={self.n_cells}, n_runs={self.n_runs}, "
            f"graph_types={types}, clusters_per_run={counts.min()}-{counts.max()})"
        )


def _check_nesting(codes: Sequence[np.ndarray], min_cluster_size: int) -> None:
    """Verify each finer level refines the one above it.

    A monotone dendrogram guarantees this, and average/complete/single linkage are all
    monotone, so at ``min_cluster_size=1`` a violation is a bug rather than a tradeoff
    and raises. Above 1 the guarantee genuinely cannot hold — dropping clusters below a
    size floor removes cells at some levels and not others — so cells unassigned in
    either level are excluded and anything left over is warned about with a count.
    """
    violations = 0
    for level in range(1, len(codes)):
        coarse, fine = codes[level - 1], codes[level]
        usable = (coarse >= 0) & (fine >= 0)
        if not usable.any():
            continue
        pairs = np.unique(np.stack([fine[usable], coarse[usable]], axis=1), axis=0)
        _, per_fine = np.unique(pairs[:, 0], return_counts=True)
        violations += int((per_fine > 1).sum())
    if not violations:
        return
    if min_cluster_size == 1:
        raise ValueError(
            f"{violations} clusters split across levels, so the emitted levels are not "
            "nested. Average, complete and single linkage are monotone and cannot do "
            f"this — check the linkage method ({violations} inversions suggest a "
            "centroid or median linkage)"
        )
    warnings.warn(
        f"{violations} finer clusters span more than one coarser cluster; with "
        f"min_cluster_size={min_cluster_size} the levels are not strictly nested "
        "because a cell dropped at one level can be assigned at another. Cut at "
        "min_cluster_size=1 for strict nesting, or treat the levels as independent",
        stacklevel=3,
    )


def _modal_agreement(left: np.ndarray, right: np.ndarray) -> bool:
    """Did one run put these two groups in the same community?

    Compares each group's modal label, ignoring cells the run dropped. Cheaper than
    counting cross-group pairs and answers the same question at the granularity that
    matters here — whether the run treated the two groups as one community.
    """
    left, right = left[left >= 0], right[right >= 0]
    if left.size == 0 or right.size == 0:
        return False
    return bool(np.bincount(left).argmax() == np.bincount(right).argmax())


def _leaf_members(
    link: np.ndarray, n: int, *, max_block: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Leaf indices under every dendrogram node, capped at ``max_block`` per node.

    Capping keeps this O(n × max_block) rather than O(n²): a caterpillar tree would
    otherwise accumulate the whole leaf set at every node on the spine. Exact sizes are
    still available from the linkage itself, so the cap only affects the *sample* a
    downstream statistic is estimated from, never a reported size.
    """
    members: list[np.ndarray] = [np.array([i], dtype=np.int64) for i in range(n)]
    for left, right, _height, _size in link:
        joined = np.concatenate([members[int(left)], members[int(right)]])
        if joined.size > max_block:
            joined = rng.choice(joined, size=max_block, replace=False)
        members.append(joined)
    return members


def _coverage_verdict(n_runs: int, n_axes_inside: int, n_axes_total: int) -> str:
    """Plain reading of what a grain window kept. The check people skip."""
    if n_runs == 0:
        return "empty: the window contains no runs at all — widen it or sweep finer"
    if n_runs < 10:
        return (
            f"thin: only {n_runs} runs survive; the consensus rests on very few votes"
        )
    if n_axes_total > 1 and n_axes_inside <= 1:
        return (
            "un-marginalised: every surviving run comes from one "
            "(graph_type, n_neighbors) — the graph axis is no longer averaged over"
        )
    if n_axes_total > 1 and n_axes_inside < n_axes_total / 2:
        return f"lopsided: {n_axes_inside} of {n_axes_total} graph settings represented"
    return f"ok: {n_runs} runs across {n_axes_inside} of {n_axes_total} graph settings"


def _plateau_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Group a threshold scan into runs of constant cluster count."""
    if frame.height == 0:
        return frame.clear()
    return (
        frame.with_columns(
            (pl.col("n_clusters") != pl.col("n_clusters").shift())
            .fill_null(True)
            .cum_sum()
            .alias("_block")
        )
        .group_by("_block")
        .agg(
            pl.col("distance_threshold").min().alias("lo"),
            pl.col("distance_threshold").max().alias("hi"),
            pl.col("n_clusters").first().alias("n_clusters"),
            pl.col("n_unassigned").max().alias("max_unassigned"),
            pl.col("largest_cluster").max().alias("largest_cluster"),
            pl.col("median_cluster_size").median().alias("median_cluster_size"),
            pl.len().alias("n_points"),
        )
        .with_columns(
            (pl.col("hi") - pl.col("lo")).alias("width"),
            ((pl.col("hi") + pl.col("lo")) / 2).alias("midpoint"),
        )
        .filter(pl.col("n_clusters") > 1)
        .sort("width", descending=True)
        .drop("_block")
        .select(
            "n_clusters",
            "midpoint",
            "width",
            "lo",
            "hi",
            "max_unassigned",
            "largest_cluster",
            "median_cluster_size",
            "n_points",
        )
    )


@dataclass(frozen=True)
class CutSuggestion:
    """A defensible threshold, the evidence for it, and what it competed with.

    Deliberately not just a number. The threshold alone is unreproducible — it
    means nothing without the size floor and grain window it was chosen under —
    and a suggestion with no visible runner-up invites more confidence than the
    evidence carries.

    ``threshold`` is ``None`` when nothing was defensible; ``reason`` says why.

    Attributes
    ----------
    threshold : float, optional
        Suggested dendrogram cut height, or ``None`` when none qualifies.
    n_clusters : int, optional
        Cluster count produced by ``threshold``.
    width : float, optional
        Width of the selected cluster-count plateau.
    min_cluster_size : int
        Size floor used while evaluating candidate cuts.
    n_unassigned : int, optional
        Cells excluded at the selected cut.
    largest_cluster : int, optional
        Size of the largest selected cluster.
    support_ceiling : float, optional
        Lowest merge height unsupported by every resolution band.
    above_ceiling : bool, optional
        Whether the suggestion exceeds ``support_ceiling``.
    reason : str
        Human-readable explanation of the result.
    alternatives : polars.DataFrame
        Leading candidate plateaus, best first.
    """

    threshold: float | None
    n_clusters: int | None
    width: float | None
    min_cluster_size: int
    n_unassigned: int | None
    largest_cluster: int | None
    support_ceiling: float | None
    above_ceiling: bool | None
    reason: str
    alternatives: pl.DataFrame

    @property
    def ok(self) -> bool:
        """Whether a usable threshold was found below the support ceiling."""
        return self.threshold is not None and not self.above_ceiling

    def __repr__(self) -> str:
        if self.threshold is None:
            return f"CutSuggestion(none — {self.reason})"
        flag = " ABOVE SUPPORT CEILING" if self.above_ceiling else ""
        return (
            f"CutSuggestion(threshold={self.threshold:.3f}, "
            f"n_clusters={self.n_clusters}, width={self.width:.3f}, "
            f"min_cluster_size={self.min_cluster_size}{flag})"
        )


@dataclass(frozen=True)
class ConsensusHierarchy:
    """The consensus read as a tree rather than as one cut through it.

    What :meth:`Clustering.hierarchy` returns: the ``linkage`` itself, the per-merge
    stability table, the nested label ladder and what each of its levels means, and the
    per-cell stability score — everything needed to say which merges the ensemble
    actually supports and which cells sit on a boundary.

    Two figures come out of this, and both are assembled from these frames rather than
    drawn here, since the package ships no plotting layer:

    >>> h = clus.hierarchy()                                   # doctest: +SKIP
    >>> seg = h.dendrogram_frame()                             # doctest: +SKIP
    >>> for row in seg.iter_rows(named=True):                  # doctest: +SKIP
    ...     ax.plot([row["x0"], row["x1"]], [row["y0"], row["y1"]],
    ...             color=cmap(row["coclustering_frequency"]))
    >>> frame = ft.dataframe(mask, embedding="umap").join(     # doctest: +SKIP
    ...     h.cell_stability_frame(), on="cell_id")
    >>> sns.scatterplot(data=frame, x="umap0", y="umap1",      # doctest: +SKIP
    ...                 hue="stability", palette="magma", s=1)

    The expectation to check on that second figure: low-stability cells should
    concentrate along interdigitated cluster boundaries and along continuous streaks
    between clusters. Scattered uniformly instead, the instability is not about boundaries
    and the cut is not the thing to adjust.

    Attributes
    ----------
    linkage : numpy.ndarray
        SciPy linkage matrix defining the hierarchy.
    merge_table : polars.DataFrame
        One row per dendrogram merge with stability evidence.
    nested_labels : polars.DataFrame
        Per-cell integer labels at each selected level.
    nested_levels : polars.DataFrame
        Metadata describing the selected hierarchy levels.
    cell_stability : numpy.ndarray
        Per-cell maximum co-clustering support.
    leaf_order : numpy.ndarray
        Dendrogram leaf ordering.
    cell_ids : numpy.ndarray, optional
        Cell identifiers aligned with the per-cell arrays.
    max_value : float, default 1.0
        Similarity value representing full agreement.
    """

    linkage: np.ndarray
    merge_table: pl.DataFrame
    nested_labels: pl.DataFrame
    nested_levels: pl.DataFrame
    cell_stability: np.ndarray
    leaf_order: np.ndarray
    cell_ids: np.ndarray | None = None
    max_value: float = 1.0

    @property
    def n_cells(self) -> int:
        """Number of cells represented by the hierarchy."""
        return int(self.linkage.shape[0] + 1)

    @property
    def n_levels(self) -> int:
        """Number of nested label levels."""
        return int(self.nested_levels.height)

    def cell_stability_frame(self) -> pl.DataFrame:
        """Per-cell stability alongside its label at every level.

        The frame the stability figure joins against an embedding: ``cell_id`` (when the
        clustering carried ids), ``stability``, and one ``level_*`` column per level.

        Returns
        -------
        polars.DataFrame
            Per-cell stability and labels at each hierarchy level.
        """
        data: dict[str, Any] = {}
        if self.cell_ids is not None:
            data["cell_id"] = np.asarray(self.cell_ids)
        data["stability"] = self.cell_stability
        for column in self.nested_labels.columns:
            if column.startswith("level_"):
                data[column] = self.nested_labels[column].to_numpy()
        return pl.DataFrame(data)

    def dendrogram_frame(self) -> pl.DataFrame:
        """The dendrogram as line segments, each carrying its merge's stability.

        Three rows per merge — left riser, top bar, right riser — with ``x0/y0/x1/y1``,
        ``merge``, ``height`` and ``coclustering_frequency``, so a merge can be coloured
        by how much of the ensemble supported it. Leaf positions follow
        :attr:`leaf_order`, the same order :meth:`Clustering.sorted_matrix` uses, so the
        dendrogram lines up with the consensus block image.

        Returned rather than plotted because ``scipy.cluster.hierarchy.dendrogram``
        computes the layout and draws it in one step, which makes per-merge colouring
        awkward and couples the figure to whatever axes are current.

        Returns
        -------
        polars.DataFrame
            Three line-segment rows per dendrogram merge.
        """
        link = self.linkage
        n = self.n_cells
        position = np.empty(n + link.shape[0], dtype=float)
        position[self.leaf_order] = np.arange(n, dtype=float)
        node_height = np.concatenate([np.zeros(n), link[:, 2]])

        rows: list[dict[str, Any]] = []
        for merge in range(link.shape[0]):
            left, right = int(link[merge, 0]), int(link[merge, 1])
            height = float(link[merge, 2])
            x_left, x_right = position[left], position[right]
            position[n + merge] = (x_left + x_right) / 2.0
            frequency = self.max_value - height
            for x0, y0, x1, y1 in (
                (x_left, node_height[left], x_left, height),
                (x_left, height, x_right, height),
                (x_right, height, x_right, node_height[right]),
            ):
                rows.append(
                    {
                        "merge": merge,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "height": height,
                        "coclustering_frequency": frequency,
                    }
                )
        return pl.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"ConsensusHierarchy(n_cells={self.n_cells}, n_levels={self.n_levels}, "
            f"clusters_per_level={self.nested_levels['n_clusters'].to_list()})"
        )


def axis_stability(
    partitions: Partitions,
    reference: np.ndarray,
    *,
    by: str | Sequence[str] = ("graph_type",),
) -> pl.DataFrame:
    """How well each slice of the ensemble agrees with the pooled consensus.

    Parameters
    ----------
    partitions : Partitions
        The individual runs, from ``Clustering.partitions``.
    reference : numpy.ndarray
        Labels to score each run against, row-aligned to the partitions — normally the
        pooled consensus cut into labels, i.e. ``clus.cluster_labels(threshold)``.
    by : str or sequence of str, default ('graph_type',)
        Which settings to group by: ``'graph_type'``, ``'n_neighbors'``, ``'resolution'``,
        or several.

    Returns
    -------
    pl.DataFrame
        One row per group with ``n_runs``, ``median_clusters``, and ``mean_ari`` /
        ``median_ari`` / ``min_ari`` / ``max_ari``. Cells unassigned (``-1``) in either
        labelling are excluded pairwise, so a run that dropped many small clusters is
        scored on what it did assign.

    Notes
    -----
    The point is to keep a discordant slice from being averaged in silently. Adding
    graph construction to the ensemble converts it from a choice that has to be
    defended into one that is marginalised over, but marginalising is only honest if
    the contributions are comparable: a weighting that systematically disagrees with
    the pool is evidence about that weighting, not noise to be diluted. One graph type
    at half the ARI of the others should be looked at rather than absorbed.

    Cells unassigned (``-1``) in either labelling are excluded pairwise, so a run that
    dropped many small clusters is scored on what it did assign.
    """
    from sklearn.metrics import adjusted_rand_score

    keys = [by] if isinstance(by, str) else list(by)
    known = {"graph_type", "n_neighbors", "resolution"}
    unknown = set(keys) - known
    if unknown:
        raise ValueError(
            f"by must name settings from {sorted(known)}, got {sorted(unknown)}"
        )

    reference = np.asarray(reference).reshape(-1)
    if reference.shape[0] != partitions.n_cells:
        raise ValueError(
            f"reference has {reference.shape[0]} labels but the partitions cover "
            f"{partitions.n_cells} cells"
        )

    scores = np.empty(partitions.n_runs, dtype=float)
    for run in range(partitions.n_runs):
        column = partitions.labels[:, run]
        usable = (column >= 0) & (reference >= 0)
        scores[run] = (
            float(adjusted_rand_score(reference[usable], column[usable]))
            if usable.sum() > 1
            else float("nan")
        )

    frame = partitions.summary().with_columns(pl.Series("ari", scores))
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("n_runs"),
            pl.col("n_clusters").median().alias("median_clusters"),
            pl.col("ari").mean().alias("mean_ari"),
            pl.col("ari").median().alias("median_ari"),
            pl.col("ari").min().alias("min_ari"),
            pl.col("ari").max().alias("max_ari"),
        )
        .sort(keys)
    )


def fauxnograph_coclustering(
    data: np.ndarray,
    *,
    n_neighbors: int | list[int] = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    graph_type: str | Sequence[str] = "knn",
    prune: float = 0.0,
    n_times: int = 1,
    resolution_parameter: float | list[float] = 1.0,
    min_cluster_size: int = 1,
    normalize: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
    return_partitions: bool = False,
) -> csr_matrix | tuple[csr_matrix, Partitions]:
    """Repeated kNN/Leiden clustering into a sparse consensus matrix.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_cells, n_features)`` coordinates to build graphs in.
    n_neighbors : int or list of int, default 30
        Neighbourhood size(s). A list sweeps it.
    metric : str, default 'minkowski'
        Distance metric for the neighbour search.
    mutual_only : bool, default False
        Keep only reciprocated edges; ``graph_type="knn"`` only.
    graph_type : str or sequence of str, default 'knn'
        Edge weighting(s) — see :func:`kneighbor_graph`. A sequence sweeps it as a third
        consensus axis.
    prune : float, default 0.0
        Jaccard floor for ``'snn_jaccard'``; ignored otherwise.
    n_times : int, default 1
        Repeats per ``(graph_type, n_neighbors, resolution)`` combination.
    resolution_parameter : float or list of float, default 1.0
        Leiden resolution(s). A list sweeps it; prefer a *geometric* range.
    min_cluster_size : int, default 1
        Clusters smaller than this become ``-1`` in a run, and ``-1`` never co-clusters.
    normalize : bool, default True
        Divide counts by the runs that kept both cells, giving similarities in ``[0, 1]``.
    seed : int, optional
        Seeds every run reproducibly.
    n_jobs : int, default -1
        Parallel workers.
    return_partitions : bool, default False
        Also return the individual runs, which is what makes the consensus checkable
        rather than guessed at.

    Returns
    -------
    scipy.sparse.csr_matrix or tuple of (csr_matrix, Partitions)
        The consensus matrix, and the runs behind it when ``return_partitions``.

    Notes
    -----
    One graph is built per ``(graph_type, n_neighbors)`` pair and reused across every
    resolution and repeat.

    Sweeping ``graph_type`` marginalises over a decision the consensus otherwise takes
    on faith, and flags the cells whose assignment depends on it. But the three
    weightings put Leiden's resolution on **different scales**, so a single resolution
    grid does not sample the same grain in each — see
    :meth:`Partitions.filter`'s ``n_clusters_min``/``n_clusters_max`` for pooling on
    realised grain instead, and :func:`axis_stability` for checking that no one
    weighting is being averaged in against the rest.

    ``resolution_parameter`` sets how finely each run splits, and the consensus can
    never be coarser than its runs. ``1.0`` is the Leiden default and gives many
    communities; dfc swept ``np.linspace(0.025, 0.2, 5)``, which sits entirely in the
    range yielding a handful of broad classes.

    Sweeping a range does not give you every grain in it — pooled runs of different
    grain average into a blend, in which a pair that only coarse runs group together
    lands at the fraction of runs that were coarse enough. Prefer a *geometric* sweep
    (``np.geomspace(0.02, 1.5, 12)``) and read ``Partitions.by_setting()``: cluster
    count plateaus against resolution, and each plateau is a grain the data actually
    supports. Linear spacing undersamples the low end, where the coarse plateau lives,
    so it shows up as one sample and reads as noise. Cut a band rather than the pool
    (``Clustering.restrict``) and the distance threshold stops being load-bearing.
    ``return_partitions`` returns the runs alongside the matrix so this is checkable
    rather than guessed at.
    """
    neighbors = [n_neighbors] if isinstance(n_neighbors, int) else list(n_neighbors)
    resolutions = (
        [resolution_parameter]
        if isinstance(resolution_parameter, (int, float))
        else list(resolution_parameter)
    )
    graph_types = [graph_type] if isinstance(graph_type, str) else list(graph_type)
    total = len(graph_types) * len(neighbors) * len(resolutions) * n_times
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=total)

    columns: list[np.ndarray] = []
    settings: list[tuple[str, int, float]] = []
    seed_idx = 0
    for kind in graph_types:
        for n in neighbors:
            graph = kneighbor_graph(
                data,
                n_neighbors=n,
                metric=metric,
                mutual_only=mutual_only,
                graph_type=kind,  # type: ignore[arg-type]
                prune=prune,
            )
            jobs = []
            for resolution in resolutions:
                for _ in range(n_times):
                    jobs.append((resolution, int(seeds[seed_idx])))
                    seed_idx += 1
                    settings.append((str(kind), int(n), float(resolution)))
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_single)(graph, resolution, min_cluster_size, run_seed)
                for resolution, run_seed in jobs
            )
            columns.extend(results)

    groups = np.column_stack(columns)
    matrix = coclustering_matrix(groups, normalize=normalize)
    if not return_partitions:
        return matrix
    return matrix, Partitions(
        labels=groups,
        n_neighbors=np.array([s[1] for s in settings], dtype=np.int64),
        resolution=np.array([s[2] for s in settings], dtype=np.float64),
        graph_type=np.array([s[0] for s in settings], dtype="<U16"),
    )


# --------------------------------------------------------------------------- #
# SimilarityMatrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SortedMatrix:
    """A co-clustering matrix permuted into cluster blocks, ready to image.

    What :meth:`SimilarityMatrix.sorted_matrix` returns. ``matrix`` is dense and
    symmetric with both axes in ``order``; cells of one cluster are contiguous, so
    a good clustering shows bright squares on the diagonal and dark off it. Within
    each block rows keep their dendrogram leaf order, so substructure a coarser cut
    merged is still visible as brighter sub-squares.

    ``boundaries`` are the ``k + 1`` block edges along either axis — the lines to
    draw between clusters — and ``centers`` the midpoints to hang ``names`` on:

    >>> sm = clus.sorted_matrix(distance_threshold=0.6)     # doctest: +SKIP
    >>> ax.imshow(sm.matrix, vmin=0, vmax=1, cmap="magma")  # doctest: +SKIP
    >>> ax.set_xticks(sm.centers, sm.names)                 # doctest: +SKIP
    >>> ax.hlines(sm.boundaries[1:-1], *ax.get_xlim())      # doctest: +SKIP

    Attributes
    ----------
    matrix : numpy.ndarray
        Dense similarity matrix in ``order``.
    order : numpy.ndarray
        Original row indices in displayed order.
    codes : numpy.ndarray
        Cluster code for each displayed row.
    names : list of str
        Display names in block order.
    boundaries : numpy.ndarray
        ``k + 1`` block-edge positions.
    cell_ids : numpy.ndarray, optional
        Cell identifiers in displayed order.
    sizes : numpy.ndarray
        Number of cells in each block.
    """

    matrix: np.ndarray
    order: np.ndarray
    codes: np.ndarray
    names: list[str]
    boundaries: np.ndarray
    cell_ids: np.ndarray | None = None
    sizes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def centers(self) -> np.ndarray:
        """Midpoint of each block, for tick placement."""
        return (self.boundaries[:-1] + self.boundaries[1:]) / 2

    @property
    def n_cells(self) -> int:
        """Number of cells along either matrix axis."""
        return int(self.matrix.shape[0])

    def block(self, index: int) -> np.ndarray:
        """Return one cluster's within-block submatrix.

        Parameters
        ----------
        index : int
            Block index in :attr:`names` order.

        Returns
        -------
        numpy.ndarray
            Dense within-cluster similarity block.
        """
        start, stop = self.boundaries[index], self.boundaries[index + 1]
        return self.matrix[start:stop, start:stop]

    def block_means(self) -> np.ndarray:
        """``(k, k)`` mean co-clustering between every pair of blocks.

        The whole matrix boiled down to one number per cluster pair: the diagonal
        is each cluster's internal cohesion, the off-diagonal how much two clusters
        still get confused for each other. Two blocks with a high off-diagonal mean
        are the pair a higher ``distance_threshold`` would merge first.

        Returns
        -------
        numpy.ndarray
            Square block-by-block mean similarity matrix.
        """
        k = len(self.names)
        edges = self.boundaries
        out = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                sub = self.matrix[edges[i] : edges[i + 1], edges[j] : edges[j + 1]]
                out[i, j] = float(sub.mean()) if sub.size else 0.0
        return out

    def __repr__(self) -> str:
        return (
            f"SortedMatrix(n_cells={self.n_cells}, n_blocks={len(self.names)}, "
            f"sizes={self.sizes.tolist()})"
        )


class SimilarityMatrix:
    """A consensus/similarity matrix with hierarchical clustering utilities.

    Wraps a (sparse) square similarity matrix and lazily computes the linkage,
    leaf order, cluster labels at a distance threshold, and the cluster-count
    curve. Caches expensive computations.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix, optional
        Square similarity or distance matrix. ``None`` creates a deferred matrix
        for subclasses that provide its materializer.
    similarity : bool, default True
        Interpret ``matrix`` as similarity when true, distance when false.
    normalized : bool, default False
        Whether similarity values are normalized to ``[0, 1]``.
    method : {'average', 'single', 'complete'}, default 'average'
        Hierarchical linkage method.
    _n : int, optional
        Deferred matrix size. Reserved for subclasses.
    """

    def __init__(
        self,
        matrix: np.ndarray | csr_matrix | None,
        *,
        similarity: bool = True,
        normalized: bool = False,
        method: Literal["average", "single", "complete"] = "average",
        _n: int | None = None,
    ) -> None:
        if method not in {"average", "single", "complete"}:
            # centroid/median linkage is non-monotone, which surfaces later as a
            # baffling nesting-violation raise; refuse it here with its own name
            raise ValueError(
                f"method must be 'average', 'single', or 'complete', got {method!r}"
            )
        self.normalized = bool(normalized)
        self.method = method
        self._linkage: np.ndarray | None = None
        self._leaf_order: np.ndarray | None = None
        self._deferred: Any = None
        if matrix is None:
            # deferred materialization: a subclass supplies the thunk (a reload
            # stores the runs, not the n×n matrix they imply — deriving it here
            # would make every load pay for consensus nobody may read)
            if _n is None:
                raise ValueError("a deferred matrix needs _n (the cell count)")
            self._n = int(_n)
            self._matrix: csr_matrix | None = None
            self._max_value: float | None = 1.0 if normalized else None
            return
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("similarity matrix must be square")
        if matrix.shape[0] == 0:
            raise ValueError("similarity matrix cannot be empty")
        if not issparse(matrix):
            if not np.allclose(matrix, matrix.T):
                raise ValueError("similarity matrix must be symmetric")
            if normalized and not np.allclose(np.diag(matrix), 1.0, atol=0.01):
                warnings.warn("diagonal is not 1.0; matrix may not be normalized")

        self._max_value = (
            1.0
            if normalized
            else float(matrix.max() if issparse(matrix) else np.max(matrix))
        )
        if not similarity:
            # distances are dense by nature (missing entries would mean distance
            # zero, not "far"), and scalar-minus-sparse is not defined anyway
            dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
            matrix = self._max_value - dense
        self._matrix = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        self._n = int(self._matrix.shape[0])

    @property
    def similarity_matrix(self) -> csr_matrix:
        """The consensus matrix, derived from stored runs on first use if deferred."""
        if self._matrix is None:
            if self._deferred is None:
                raise ValueError("no matrix and no way to derive one")
            derived = self._deferred()
            self._matrix = derived.tocsr() if issparse(derived) else csr_matrix(derived)
            self._deferred = None
            if self._max_value is None:
                self._max_value = float(self._matrix.max())
        return self._matrix

    @property
    def max_value(self) -> float:
        """Full-agreement similarity (1.0 when normalized; else the matrix max)."""
        if self._max_value is None:
            self.similarity_matrix  # materializes and sets it
        return self._max_value  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int]:
        """Square matrix shape as ``(n_cells, n_cells)``."""
        return (self._n, self._n)

    @property
    def linkage(self) -> np.ndarray:
        """Scipy linkage matrix (cached).

        Built from the sparse similarity, but the condensed distance matrix it
        feeds scipy is dense — ``n(n-1)/2`` float64, about 1.6 GB at 20k cells
        and growing quadratically, which is why large tables get a warning
        rather than a silent allocation.
        """
        if self._linkage is None:
            n = self.similarity_matrix.shape[0]
            if n > 25_000:
                gb = n * (n - 1) / 2 * 8 / 1e9
                warnings.warn(
                    f"building the linkage for {n} cells allocates a ~{gb:.1f} GB "
                    f"condensed distance matrix (plus scipy's working copy)"
                )
            condensed = np.full(n * (n - 1) // 2, self.max_value)
            cx = coo_matrix(self.similarity_matrix)
            upper = cx.row < cx.col
            # int64 throughout: scipy hands back int32 indices, and n * row
            # overflows int32 past ~46k cells, which would silently scatter
            # distances onto the wrong pairs rather than raising.
            rows = cx.row[upper].astype(np.int64)
            cols = cx.col[upper].astype(np.int64)
            vals = cx.data[upper]
            condensed_idx = n * rows - rows * (rows + 1) // 2 + cols - rows - 1
            condensed[condensed_idx] = self.max_value - vals
            self._linkage = linkage(condensed, method=self.method)
        return self._linkage

    @property
    def leaf_order(self) -> np.ndarray:
        """Dendrogram leaf order, computed lazily and returned as a copy."""
        if self._leaf_order is None:
            self._leaf_order = leaves_list(self.linkage)
        return self._leaf_order

    def cluster_labels(
        self, distance_threshold: float, *, min_cluster_size: int = 1
    ) -> np.ndarray:
        """Cut the dendrogram at ``distance_threshold`` into integer labels.

        Labels are contiguous ``0..k-1`` (``-1`` for clusters smaller than
        ``min_cluster_size``), the same numbering every other cut in the
        library uses — ``fcluster``'s raw 1-based codes never escape.

        Parameters
        ----------
        distance_threshold : float
            Maximum linkage distance within a cluster.
        min_cluster_size : int, default 1
            Replace smaller clusters with ``-1``.

        Returns
        -------
        numpy.ndarray
            Integer code for each matrix row.
        """
        labels = fcluster(self.linkage, t=distance_threshold, criterion="distance")
        labels = labels.astype(np.int64)
        if min_cluster_size > 1:
            values, counts = np.unique(labels, return_counts=True)
            for value in values[counts < min_cluster_size]:
                labels[labels == value] = -1
        keep = sorted({int(v) for v in np.unique(labels) if int(v) != -1})
        remap = {old: new for new, old in enumerate(keep)}
        return np.array([remap.get(int(v), -1) for v in labels], dtype=np.int64)

    # -- inspection ------------------------------------------------------------

    def _row_codes(
        self, labels: Any, distance_threshold: float | None, min_cluster_size: int
    ) -> np.ndarray:
        """Per-matrix-row integer cluster codes from labels or a fresh cut."""
        n = self.shape[0]
        if labels is None:
            if distance_threshold is None:
                raise ValueError("pass either labels= or distance_threshold=")
            # cluster_labels already renumbers to contiguous 0..k-1
            return self.cluster_labels(
                distance_threshold, min_cluster_size=min_cluster_size
            )
        if hasattr(labels, "codes_for"):  # a LabelSet
            ids = getattr(self, "_cell_ids", None)
            if ids is None:
                raise ValueError(
                    "a bare SimilarityMatrix has no cell ids to align a LabelSet "
                    "against; pass a plain array of codes in matrix-row order"
                )
            return labels.codes_for(ids)
        codes = np.asarray(labels, dtype=np.int64)
        if codes.shape[0] != n:
            raise ValueError(
                f"labels has {codes.shape[0]} entries but the matrix is {n}x{n}"
            )
        return codes

    def sorted_matrix(
        self,
        labels: Any = None,
        *,
        distance_threshold: float | None = None,
        min_cluster_size: int = 1,
        include_unassigned: bool = True,
        subsample: int | None = None,
        max_cells: int = 8000,
        seed: int | None = 0,
    ) -> SortedMatrix:
        """The co-clustering matrix permuted into cluster blocks (see :class:`SortedMatrix`).

        The picture the consensus matrix is *for*: cells of a cluster sit
        contiguously, so structure reads off the diagonal directly — crisp bright
        squares mean the cut found real groups, a bright haze spanning two blocks
        means they are one group the threshold split, and a block with no internal
        brightness is cells the cut grouped but the runs never actually agreed on.

        Blocks come from ``labels`` (a ``LabelSet`` or an array of per-row codes)
        or, with ``distance_threshold``, from a fresh cut of this dendrogram at that
        threshold. Within a block rows stay in dendrogram leaf order so substructure
        stays visible. Unassigned cells (``-1``) form a trailing ``"unassigned"``
        block — worth keeping: that block is exactly the cells ``min_cluster_size``
        threw away, and whether it has any internal structure at all is the fastest
        way to tell whether they were dropped fairly.

        The result is dense ``(n, n)``, so ``max_cells`` refuses to silently
        allocate a huge array; pass ``subsample`` to take that many cells spread
        proportionally across blocks instead.

        Parameters
        ----------
        labels : LabelSet or numpy.ndarray, optional
            Cluster definition, aligned by id when possible.
        distance_threshold : float, optional
            Cut height used when ``labels`` is omitted.
        min_cluster_size : int, default 1
            Size floor for a newly computed cut.
        include_unassigned : bool, default True
            Include ``-1`` rows as a trailing block.
        subsample : int, optional
            Approximate total cells sampled proportionally across blocks.
        max_cells : int, default 8000
            Refuse a larger dense result unless subsampling is requested.
        seed : int, optional
            Random seed for block-proportional subsampling.

        Returns
        -------
        SortedMatrix
            Dense matrix and the ordering metadata needed to display it.

        Raises
        ------
        ValueError
            If neither labels nor a threshold is supplied, alignment fails, or
            the dense result would exceed ``max_cells``.
        """
        codes = self._row_codes(labels, distance_threshold, min_cluster_size)
        n = self.shape[0]

        names_by_code: dict[int, str] = {}
        if hasattr(labels, "meta"):
            names_by_code = {i: m.name for i, m in labels.meta.items()}

        keep_codes = [int(c) for c in np.unique(codes) if int(c) != -1]
        rank = np.empty(n, dtype=np.int64)
        rank[self.leaf_order] = np.arange(n)

        blocks: list[np.ndarray] = []
        names: list[str] = []
        rng = np.random.default_rng(seed)
        groups = [(c, np.flatnonzero(codes == c)) for c in keep_codes]
        if include_unassigned and (codes == -1).any():
            groups.append((-1, np.flatnonzero(codes == -1)))

        total = sum(len(m) for _, m in groups)
        if subsample is not None and subsample < total:
            share = subsample / total
            groups = [
                (
                    c,
                    np.sort(
                        rng.choice(
                            m, size=max(1, int(round(len(m) * share))), replace=False
                        )
                    ),
                )
                for c, m in groups
            ]
            total = sum(len(m) for _, m in groups)
        if total > max_cells:
            raise ValueError(
                f"sorting {total} cells would build a dense {total}x{total} matrix "
                f"(~{total * total * 4 / 1e9:.1f} GB). Pass subsample=… to sample "
                f"cells across blocks, or raise max_cells if you have the memory."
            )

        for code, members in groups:
            blocks.append(members[np.argsort(rank[members], kind="stable")])
            names.append(
                "unassigned" if code == -1 else names_by_code.get(code, str(code))
            )

        order = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        sizes = np.array([len(b) for b in blocks], dtype=np.int64)
        boundaries = np.concatenate([[0], np.cumsum(sizes)])
        dense = np.asarray(
            self.similarity_matrix[order, :][:, order].todense(), dtype=np.float32
        )
        cell_ids = getattr(self, "_cell_ids", None)
        return SortedMatrix(
            matrix=dense,
            order=order,
            codes=codes[order],
            names=names,
            boundaries=boundaries,
            cell_ids=None if cell_ids is None else np.asarray(cell_ids)[order],
            sizes=sizes,
        )

    def consensus_strength(self) -> np.ndarray:
        """Each cell's strongest co-clustering with any *other* cell.

        The per-cell diagnostic behind cells that never join a cluster no matter how
        far the dendrogram is cut. A cell scoring 0 never landed in a group with
        anything across every kNN/Leiden run, so its distance to all other cells is
        the maximum and it can only merge in the final collapse of the tree — no
        ``distance_threshold`` short of that will fold it into a neighbour. Low
        scorers are the cells ``min_cluster_size`` drops.

        The value is a fraction of runs only when the matrix is ``normalized``;
        otherwise it is a raw co-assignment count out of ``max_value``.

        Returns
        -------
        numpy.ndarray
            Strongest non-self similarity for every cell.
        """
        # Drop the diagonal on a CSR copy, not via LIL: LIL stores every nonzero
        # as Python objects, ~80 bytes against CSR's 8, and this matrix is the
        # largest thing in the session. `setdiag(0)` is structure-preserving here
        # (it only clears entries), so CSR takes it without complaint.
        matrix = self.similarity_matrix.copy()
        matrix.setdiag(0)
        matrix.eliminate_zeros()
        if matrix.nnz == 0:
            return np.zeros(self.shape[0], dtype=np.float64)
        return np.asarray(matrix.max(axis=1).todense(), dtype=np.float64).ravel()

    def threshold_scan(
        self,
        distance_range: np.ndarray | None = None,
        *,
        n_points: int = 25,
        min_cluster_size: int = 1,
    ) -> pl.DataFrame:
        """What each candidate ``distance_threshold`` actually yields, as a table.

        ``cluster_count_curve`` answers "how many clusters", which is only half of
        choosing a cut — the other half is how many cells survive it, and how lopsided
        the result is. One row per threshold with ``n_clusters``, ``n_assigned`` /
        ``n_unassigned``, and the largest and median cluster size.

        Read it at the ``min_cluster_size`` you intend to cut at. At the default of 1
        nothing is ever dropped and every singleton counts as a cluster, so a curve
        read at 1 and a cut made at 10 disagree — usually by a lot, since it is the
        singletons that the count is mostly made of.

        Parameters
        ----------
        distance_range : numpy.ndarray, optional
            Candidate thresholds. Defaults to an evenly spaced linkage range.
        n_points : int, default 25
            Number of default thresholds.
        min_cluster_size : int, default 1
            Size floor applied at every cut.

        Returns
        -------
        polars.DataFrame
            Per-threshold cluster, assignment, and size statistics.
        """
        if distance_range is None:
            distance_range = np.linspace(0.0, float(self.linkage[:, 2].max()), n_points)
        rows = []
        for d in np.asarray(distance_range, dtype=float):
            labels = self.cluster_labels(float(d), min_cluster_size=min_cluster_size)
            assigned = labels[labels >= 0]
            _, counts = np.unique(assigned, return_counts=True)
            rows.append(
                {
                    "distance_threshold": float(d),
                    "n_clusters": int(counts.size),
                    "n_assigned": int(assigned.size),
                    "n_unassigned": int(labels.size - assigned.size),
                    "largest_cluster": int(counts.max()) if counts.size else 0,
                    "median_cluster_size": float(np.median(counts))
                    if counts.size
                    else 0.0,
                }
            )
        return pl.DataFrame(rows)

    def plateaus(
        self,
        distance_range: np.ndarray | None = None,
        *,
        n_points: int = 60,
        min_cluster_size: int = 1,
    ) -> pl.DataFrame:
        """Runs of thresholds giving the same cluster count, widest first.

        The decision :meth:`threshold_scan` supports, taken out of the table. A
        plateau is a gap in the dendrogram — a band the structure is indifferent
        to — so its *midpoint* is the defensible place to cut and its edges are
        not. Scanning by eye for where a count stops changing is exactly the
        step that is tedious by hand and trivial to get subtly wrong.

        One row per plateau: ``lo`` / ``hi`` / ``width`` / ``midpoint``, the
        ``n_clusters`` it holds, and the worst ``max_unassigned`` and
        ``largest_cluster`` anywhere inside it — because a plateau that is
        stable in count while bleeding cells, or while one cluster swallows the
        cohort, is not the stable structure it looks like.

        Single-cluster plateaus are dropped: the tree collapsing is not a grain.
        **An empty frame is the informative result** — it says the count changes
        at every threshold, so this cohort has no scale the ensemble agrees on
        and no cut here is defensible.

        Parameters
        ----------
        distance_range : numpy.ndarray, optional
            Candidate thresholds passed to :meth:`threshold_scan`.
        n_points : int, default 60
            Number of default thresholds.
        min_cluster_size : int, default 1
            Size floor applied at every cut.

        Returns
        -------
        polars.DataFrame
            Constant-cluster-count threshold bands, widest first.
        """
        frame = self.threshold_scan(
            distance_range, n_points=n_points, min_cluster_size=min_cluster_size
        )
        return _plateau_frame(frame)

    def cluster_count_curve(
        self,
        distance_range: np.ndarray | None = None,
        *,
        n_points: int = 100,
        min_cluster_size: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Number of clusters as a function of distance threshold.

        Counted at ``min_cluster_size``, which defaults to 1 — so by default every
        singleton counts. Pass the same ``min_cluster_size`` you will cut at, or the
        curve will promise far more clusters than ``label`` gives back. See
        :meth:`threshold_scan` for the fuller picture, cell counts included.

        Parameters
        ----------
        distance_range : numpy.ndarray, optional
            Candidate thresholds. Defaults to an evenly spaced linkage range.
        n_points : int, default 100
            Number of default thresholds.
        min_cluster_size : int, default 1
            Size floor applied at every cut.

        Returns
        -------
        distance_range : numpy.ndarray
            Evaluated thresholds.
        n_clusters : numpy.ndarray
            Cluster count at each threshold.
        """
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

    # -- the hierarchy, rather than one cut through it -------------------------

    def merge_table(self) -> pl.DataFrame:
        """Every merge in the dendrogram, annotated with how stable it is.

        One row per linkage merge, coarsest last: ``merge`` (the row index),
        ``cluster_id`` (scipy's ``n + merge`` label for the group it forms), ``left`` and
        ``right`` (the two groups joined, in the same labelling), ``left_size`` /
        ``right_size`` / ``n_cells``, ``height``, and ``coclustering_frequency``.

        The stability annotation costs nothing, and it is worth being explicit about why.
        Distance here *is* ``max_value - similarity``, and similarity is the fraction of
        ensemble runs that put a pair together, so under average linkage a merge's height
        is one minus the mean co-clustering frequency between the two groups. The
        frequency is therefore just ``max_value - height`` — not a separate measurement,
        and not something to recompute from the runs.

        Read it as: merges at high frequency held across most of the ensemble and are
        types; merges at low frequency only appeared in the runs fine enough to make
        them, and are subtypes or noise. :meth:`Clustering.merge_support` splits that
        apart by resolution when the runs are still around.

        Returns
        -------
        polars.DataFrame
            Linkage merges with sizes, heights, and support frequency.
        """
        link = self.linkage
        n = self.shape[0]
        left = link[:, 0].astype(np.int64)
        right = link[:, 1].astype(np.int64)
        counts = np.concatenate(
            [np.ones(n, dtype=np.int64), link[:, 3].astype(np.int64)]
        )
        return pl.DataFrame(
            {
                "merge": np.arange(link.shape[0], dtype=np.int64),
                "cluster_id": np.arange(n, n + link.shape[0], dtype=np.int64),
                "left": left,
                "right": right,
                "left_size": counts[left],
                "right_size": counts[right],
                "n_cells": link[:, 3].astype(np.int64),
                "height": link[:, 2],
                "coclustering_frequency": self.max_value - link[:, 2],
            }
        )

    def nested_labels(
        self,
        heights: Sequence[float] | np.ndarray | None = None,
        *,
        n_levels: int = 6,
        min_cluster_size: int = 1,
        cell_ids: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Labels at several cut heights at once, one column per level.

        Parameters
        ----------
        heights : sequence of float, optional
            Distance thresholds to cut at, in any order — they are sorted descending so
            ``level_0`` is always the tallest cut and therefore the coarsest. Defaults to
            ``n_levels`` values evenly spaced over ``0.1`` to ``0.9`` of ``max_value``,
            i.e. co-clustering frequencies from 0.9 down to 0.1.
        n_levels : int, default 6
            How many default heights to use. Ignored when ``heights`` is given.
        min_cluster_size : int, default 1
            Clusters smaller than this become ``-1`` at that level. Above 1 the levels are
            no longer strictly nested — see the notes.
        cell_ids : numpy.ndarray, optional
            Ids for the ``cell_id`` column. Defaults to the clustering's own.

        Returns
        -------
        pl.DataFrame
            One row per cell: ``cell_id`` (when available) plus ``level_0`` … ``level_k``,
            coarsest to finest. Ready to hand to
            :meth:`cellpax.LabelSet.from_labels` a level at a time, which is how a level
            becomes an attachable, renameable ``LabelSet``. See :meth:`nested_levels` for
            what each column's cut actually was.

        Notes
        -----
        The resolution sweep is geomspaced because structure exists at more than one
        scale; collapsing the consensus to a single flat vector throws that away again.
        This emits the ladder instead, so downstream code selects granularity rather than
        having it chosen for it.

        Levels that give the same cluster count as the previous one are dropped, and so is
        a level where ``min_cluster_size`` removed every cluster; both are logged rather
        than left to be noticed.

        Nesting is checked, not assumed. Average, complete and single linkage all give
        monotone dendrograms, so cuts at decreasing thresholds *are* nested and a violation
        means something is wrong — at ``min_cluster_size=1`` that raises. Above 1 it cannot
        hold strictly: a cell in a cluster too small to keep is ``-1`` at that level while
        remaining assigned at others, so nesting is checked over cells assigned in both
        levels and any residual violation is warned about with a count.
        """
        codes, _ = self._nested_codes(
            heights, n_levels=n_levels, min_cluster_size=min_cluster_size
        )
        ids = cell_ids if cell_ids is not None else getattr(self, "_cell_ids", None)
        frame_data: dict[str, Any] = {}
        if ids is not None:
            frame_data["cell_id"] = np.asarray(ids)
        for level, column in enumerate(codes):
            frame_data[f"level_{level}"] = column.astype(np.int64)
        return pl.DataFrame(frame_data)

    def nested_levels(
        self,
        heights: Sequence[float] | np.ndarray | None = None,
        *,
        n_levels: int = 6,
        min_cluster_size: int = 1,
    ) -> pl.DataFrame:
        """What each level of :meth:`nested_labels` is: height, frequency, size.

        Parameters
        ----------
        heights : sequence of float, optional
            The same thresholds passed to :meth:`nested_labels`, so the two line up.
        n_levels : int, default 6
            How many default heights to use. Ignored when ``heights`` is given.
        min_cluster_size : int, default 1
            Must match what :meth:`nested_labels` was called with, since it changes which
            levels survive.

        Returns
        -------
        pl.DataFrame
            One row per level: ``level``, ``height``, ``coclustering_frequency``
            (``max_value - height``, the consensus fraction that level is cut at),
            ``n_clusters`` and ``n_unassigned``. The companion frame, so a ``level_*``
            column can be read back to the cut it came from.
        """
        codes, kept = self._nested_codes(
            heights, n_levels=n_levels, min_cluster_size=min_cluster_size
        )
        return pl.DataFrame(
            {
                "level": np.arange(len(codes), dtype=np.int64),
                "height": np.asarray(kept, dtype=float),
                "coclustering_frequency": self.max_value
                - np.asarray(kept, dtype=float),
                "n_clusters": np.array(
                    [int(np.unique(c[c >= 0]).size) for c in codes], dtype=np.int64
                ),
                "n_unassigned": np.array(
                    [int((c < 0).sum()) for c in codes], dtype=np.int64
                ),
            }
        )

    def _nested_codes(
        self,
        heights: Sequence[float] | np.ndarray | None,
        *,
        n_levels: int,
        min_cluster_size: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Code arrays and the heights they were cut at, coarsest first.

        Memoized per argument set: ``hierarchy()`` reads the same ladder three
        ways (codes, labels, levels), and each cut is a full ``fcluster`` pass.
        """
        memo_key = (
            None if heights is None else tuple(float(h) for h in heights),
            n_levels,
            min_cluster_size,
        )
        cache = getattr(self, "_nested_cache", None)
        if cache is None:
            cache = self._nested_cache = {}
        if memo_key in cache:
            codes, kept = cache[memo_key]
            return [c.copy() for c in codes], list(kept)
        if heights is None:
            heights = np.linspace(0.1, 0.9, n_levels) * self.max_value
        # Descending, so level_0 is the tallest cut and therefore the coarsest, and each
        # subsequent level refines it. The nesting check below depends on this order.
        heights = np.sort(np.asarray(heights, dtype=float))[::-1]

        codes: list[np.ndarray] = []
        kept: list[float] = []
        previous_count = -1
        skipped = 0
        for height in heights:
            column = self.cluster_labels(
                float(height), min_cluster_size=min_cluster_size
            )
            count = int(np.unique(column[column >= 0]).size)
            # A cut where min_cluster_size dropped every cluster is not a level — it is
            # an all-unassigned column, and emitting it would look like a granularity.
            if count == 0 or count == previous_count:
                skipped += 1
                continue
            codes.append(column)
            kept.append(float(height))
            previous_count = count
        if skipped:
            _logger.info(
                "nested_labels: dropped %d of %d requested heights that repeated the "
                "previous cluster count",
                skipped,
                len(heights),
            )
        if not codes:
            raise ValueError(
                "no cut height produced any clusters; check max_value against the "
                f"heights requested ({np.round(heights, 4).tolist()})"
            )
        _check_nesting(codes, min_cluster_size)
        cache[memo_key] = ([c.copy() for c in codes], list(kept))
        return codes, kept

    def cell_stability(self) -> np.ndarray:
        """Per-cell consensus stability — an alias for :meth:`consensus_strength`.

        The fraction of runs in which a cell landed with whichever cell it most often
        landed with, i.e. its maximum co-clustering frequency. Named for how it reads on
        a figure: low scorers concentrate along interdigitated cluster boundaries and
        along continuous streaks between clusters, so a UMAP coloured by this is the
        companion to one coloured by hard labels, and the two together say which
        boundaries the labels are actually confident about.

        Returns
        -------
        numpy.ndarray
            Per-cell maximum consensus similarity.
        """
        return self.consensus_strength()


class Clustering(SimilarityMatrix):
    """A consensus clustering that remembers which cells it was computed over.

    Everything a :class:`SimilarityMatrix` does, plus the provenance needed to cut
    it into labels without restating it: the ``mask``, that mask's ``cell_ids`` in
    matrix-row order, the feature ``columns``, and the ``space`` they were compared
    in. ``ft.cluster`` returns one of these.

    That provenance is the point. ``label`` pairs row *i* of the matrix with cell
    *i* of the mask, so cutting a clustering against the wrong mask silently
    mislabels every cell. Carrying the mask on the object makes that impossible —
    and, given ``order_by``, so is forgetting to order the clusters:

    >>> clus = ft.cluster("l23", n_times=20, name="run",
    ...                   order_by="soma_depth_um")        # doctest: +SKIP
    >>> labels = clus.label(distance_threshold=0.6)        # already depth-ordered
    >>> ft.attach(labels)                                  # doctest: +SKIP

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix, optional
        Consensus matrix, or ``None`` when it should be derived from partitions.
    cell_ids : numpy.ndarray
        Cell identifiers in matrix-row order.
    mask : str
        Source mask name.
    columns : tuple of str, optional
        Features used to construct the representation.
    space : str, optional
        Human-readable representation label.
    similarity : bool, default True
        Whether ``matrix`` already contains similarities.
    normalized : bool, default False
        Whether similarities are already normalized to ``[0, 1]``.
    method : {'average', 'single', 'complete'}, default 'average'
        Hierarchical linkage method.
    order_by : str, optional
        Column used to order cluster identifiers.
    order_values : numpy.ndarray, optional
        Values of ``order_by`` in matrix-row order.
    order_agg : {'mean', 'median'}, default 'mean'
        Per-cluster aggregation used for ordering.
    order_ascending : bool, default True
        Sort direction for cluster ordering.
    partitions : Partitions, optional
        Individual ensemble runs.
    params : dict, optional
        JSON-safe provenance for the originating call.
    """

    def __init__(
        self,
        matrix: np.ndarray | csr_matrix | None,
        *,
        cell_ids: np.ndarray,
        mask: str,
        columns: tuple[str, ...] = (),
        space: str = "",
        similarity: bool = True,
        normalized: bool = False,
        method: Literal["average", "single", "complete"] = "average",
        order_by: str | None = None,
        order_values: np.ndarray | None = None,
        order_agg: Literal["mean", "median"] = "mean",
        order_ascending: bool = True,
        partitions: Partitions | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        cells = np.asarray(cell_ids)
        if matrix is None:
            # derive the consensus from the runs on first use. A reload takes
            # this path: the runs are what persists, and the n×n matrix they
            # imply is minutes of work at scale that most loads never need.
            if partitions is None:
                raise ValueError(
                    "matrix=None needs partitions to derive the consensus from"
                )
            if partitions.n_cells != cells.shape[0]:
                raise ValueError(
                    f"cell_ids has {cells.shape[0]} entries but the partitions "
                    f"cover {partitions.n_cells} cells"
                )
            super().__init__(
                None,
                similarity=similarity,
                normalized=normalized,
                method=method,
                _n=cells.shape[0],
            )
            self._deferred = lambda: partitions.coclustering(normalize=normalized)
        else:
            super().__init__(
                matrix, similarity=similarity, normalized=normalized, method=method
            )
        if cells.shape[0] != self.shape[0]:
            raise ValueError(
                f"cell_ids has {cells.shape[0]} entries but the matrix is "
                f"{self.shape[0]}x{self.shape[0]}; they must describe the same cells "
                f"in the same order"
            )
        self._cell_ids = cells
        self._mask = mask
        self._columns = tuple(columns)
        self._space = space
        values = None
        if order_values is not None:
            values = np.asarray(order_values, dtype=float)
            if values.shape[0] != self.shape[0]:
                raise ValueError(
                    f"order_values has {values.shape[0]} entries but the matrix is "
                    f"{self.shape[0]}x{self.shape[0]}"
                )
        self._order_by = order_by
        self._order_values = values
        self._order_agg = order_agg
        self._order_ascending = order_ascending
        self._partitions = partitions
        self._params = dict(params) if params else None

    @property
    def mask(self) -> str:
        """The mask this was clustered over."""
        return self._mask

    @property
    def params(self) -> dict[str, Any] | None:
        """The ``ft.cluster`` call that produced this, seed included.

        What makes a stored clustering re-derivable rather than merely present:
        replaying ``ft.cluster(mask, **{k: v for k, v in params.items()})`` on
        the same table reproduces the ensemble bit for bit. ``None`` on
        clusterings built by hand or loaded from an analysis saved before
        parameters were recorded.
        """
        return dict(self._params) if self._params else None

    @property
    def cell_ids(self) -> np.ndarray:
        """The mask's cell ids, in matrix-row order. A copy."""
        return self._cell_ids.copy()

    @property
    def columns(self) -> tuple[str, ...]:
        """The feature columns compared."""
        return self._columns

    @property
    def space(self) -> str:
        """How those features were reduced, e.g. ``'pca(0.95)'`` or ``'scaled'``."""
        return self._space

    def _row_codes(
        self, labels: Any, distance_threshold: float | None, min_cluster_size: int
    ) -> np.ndarray:
        """Cut through ``label`` so inspection sees the same ids labelling does."""
        if labels is None and distance_threshold is not None:
            labels = self.label(
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
            )
        return super()._row_codes(labels, distance_threshold, min_cluster_size)

    def restrict(
        self,
        *,
        resolution_min: float | None = None,
        resolution_max: float | None = None,
        n_neighbors: int | Sequence[int] | None = None,
        graph_type: str | Sequence[str] | None = None,
        n_clusters_min: int | None = None,
        n_clusters_max: int | None = None,
    ) -> "Clustering":
        """A new ``Clustering`` re-consensed from only the runs matching these settings.

        Parameters
        ----------
        resolution_min, resolution_max : float, optional
            Keep runs whose nominal resolution falls in this range.
        n_neighbors : int or sequence of int, optional
            Keep runs using these neighbourhood sizes.
        graph_type : str or sequence of str, optional
            Keep runs using these edge weightings — how to see what one weighting says
            before it is pooled with the others.
        n_clusters_min, n_clusters_max : int, optional
            Keep runs by *realised* cluster count rather than nominal resolution, which is
            the selection that makes a pool across graph types comparable.

        Returns
        -------
        Clustering
            Re-consensed over the narrowed runs. Mask, cell ids, columns, space and any
            ``order_by`` carry over; the linkage is rebuilt. Raises when this clustering
            has no partitions to narrow.

        Notes
        -----
        The Leiden runs are what clustering actually costs, and they are already done,
        so changing your mind about the grain is free. A sweep spanning fine and coarse
        resolutions blends them — cells of one broad group co-cluster only in the runs
        coarse enough to keep them together, so the broad blocks arrive at that
        fraction rather than at 1.0, and the threshold that recovers them sits right up
        against the point where the whole dendrogram collapses. Narrowing to the coarse
        runs puts that structure back at full strength:

        >>> broad = clus.restrict(resolution_max=0.3)   # doctest: +SKIP
        >>> broad.label(distance_threshold=0.5)         # doctest: +SKIP

        ``graph_type`` isolates one weighting — the way to see what a single graph
        construction says before it is pooled with the others.
        ``n_clusters_min``/``n_clusters_max`` select on realised grain instead of nominal
        resolution, which is the selection that makes a pool across graph types
        comparable:

        >>> mid = clus.restrict(n_clusters_min=8, n_clusters_max=40)   # doctest: +SKIP

        Mask, cell ids, columns, space and any ``order_by`` carry over; the linkage is
        rebuilt from the narrowed consensus.
        """
        if self._partitions is None:
            raise ValueError(
                "this Clustering has no partitions to restrict; partitions persist "
                "with a saved analysis (manifest v2+), so this is either a matrix "
                "built by hand or an analysis saved before they were stored — "
                "re-run ft.cluster"
            )
        subset = self._partitions.filter(
            resolution_min=resolution_min,
            resolution_max=resolution_max,
            n_neighbors=n_neighbors,
            graph_type=graph_type,
            n_clusters_min=n_clusters_min,
            n_clusters_max=n_clusters_max,
        )
        filters = {
            "resolution_min": resolution_min,
            "resolution_max": resolution_max,
            "n_neighbors": n_neighbors,
            "graph_type": graph_type,
            "n_clusters_min": n_clusters_min,
            "n_clusters_max": n_clusters_max,
        }
        return Clustering(
            subset.coclustering(normalize=self.normalized),
            cell_ids=self._cell_ids,
            mask=self._mask,
            columns=self._columns,
            space=self._space,
            normalized=self.normalized,
            method=self.method,
            order_by=self._order_by,
            order_values=self._order_values,
            order_agg=self._order_agg,
            order_ascending=self._order_ascending,
            partitions=subset,
            params={
                **(self._params or {}),
                "restrict": {k: v for k, v in filters.items() if v is not None},
            },
        )

    def hierarchy(
        self,
        heights: Sequence[float] | np.ndarray | None = None,
        *,
        n_levels: int = 6,
        min_cluster_size: int = 1,
    ) -> ConsensusHierarchy:
        """The whole tree, annotated — see :class:`ConsensusHierarchy`.

        Parameters
        ----------
        heights : sequence of float, optional
            Cut heights for the label ladder — see :meth:`nested_labels`.
        n_levels : int, default 6
            How many default heights to use.
        min_cluster_size : int, default 1
            Clusters smaller than this become ``-1`` at a level.

        Returns
        -------
        ConsensusHierarchy
            The linkage, the per-merge stability table, the nested label ladder and what
            each of its levels means, plus the per-cell stability score.

        Notes
        -----
        The alternative to picking one ``distance_threshold``: keep the linkage, keep the
        stability of every merge, and emit labels at several heights so granularity is
        chosen downstream rather than baked in here.
        """
        codes, kept = self._nested_codes(
            heights, n_levels=n_levels, min_cluster_size=min_cluster_size
        )
        return ConsensusHierarchy(
            linkage=self.linkage,
            merge_table=self.merge_table(),
            nested_labels=self.nested_labels(
                kept, min_cluster_size=min_cluster_size, cell_ids=self._cell_ids
            ),
            nested_levels=self.nested_levels(kept, min_cluster_size=min_cluster_size),
            cell_stability=self.cell_stability(),
            leaf_order=self.leaf_order,
            cell_ids=self._cell_ids,
            max_value=self.max_value,
        )

    def cell_stability_frame(self) -> pl.DataFrame:
        """``cell_id`` and per-cell consensus stability, ready to join to an embedding.

        The companion to a UMAP coloured by hard labels. See :meth:`cell_stability` for
        what the number is, and :class:`ConsensusHierarchy` for the version that also
        carries the label at every level.

        Returns
        -------
        polars.DataFrame
            Cell identifiers and consensus stability.
        """
        return pl.DataFrame(
            {
                "cell_id": np.asarray(self._cell_ids),
                "stability": self.cell_stability(),
            }
        )

    def soft_labels(
        self,
        labels: Any = None,
        *,
        distance_threshold: float | None = None,
        min_cluster_size: int = 1,
    ) -> pl.DataFrame:
        """Each cell's mean co-clustering frequency with every cluster's members.

        The ensemble-derived soft assignment behind a hard cut: entry ``(i, k)``
        is the average fraction of runs that put cell *i* with a member of
        cluster *k* (cell *i* itself excluded). For a cell inside a tight
        cluster the own-cluster column approaches 1.0 and the rest approach
        0.0; a cell on a boundary reads as a genuine split. Derived entirely
        from the consensus matrix, so it costs nothing beyond a column sum.

        ``labels`` is a ``LabelSet`` (or per-row codes) defining the clusters;
        or pass ``distance_threshold`` to cut one here. Columns are named
        ``p_<cluster>`` using the label set's names when it has them.

        Read it as an honest ensemble frequency, not a calibrated probability:
        rows need not sum to one, and a cell far from everything can still
        score high against its nearest cluster. Requires a normalized matrix
        for the fraction-of-runs reading; on an unnormalized one the values
        are scaled by ``max_value`` instead.

        Parameters
        ----------
        labels : LabelSet or numpy.ndarray, optional
            Cluster definition, aligned by id when possible.
        distance_threshold : float, optional
            Cut height used when ``labels`` is omitted.
        min_cluster_size : int, default 1
            Size floor for a newly computed cut.

        Returns
        -------
        polars.DataFrame
            Cell identifiers and one ``p_<cluster>`` support column per cluster.
        """
        codes = self._row_codes(labels, distance_threshold, min_cluster_size)
        matrix = self.similarity_matrix
        names: dict[int, str] = {}
        if labels is not None and hasattr(labels, "ids"):
            names = {
                int(i): str(n)
                for i, n in zip(labels.ids, labels.names)
                if n is not None
            }
        columns: dict[str, Any] = {"cell_id": np.asarray(self._cell_ids)}
        n = self.shape[0]
        values = sorted({int(v) for v in np.unique(codes) if int(v) != -1})

        # Every cluster's column sums in one pass: right-multiplying by the
        # one-hot cluster indicator sums each cell's row within each cluster.
        # Keep the indicator sparse and in the matrix's own dtype -- it has one
        # entry per assigned cell, so the product costs O(nnz) regardless of how
        # many clusters there are, and neither operand gets copied or upcast.
        # The obvious versions are both traps at consensus-matrix scale: a
        # per-cluster `matrix[:, members]` slice is O(nnz) *per cluster*, and
        # zeroing the diagonal via `tolil(copy=True)` costs ~80 bytes a nonzero
        # against CSR's 8 -- a 10x blow-up of a matrix that is already the
        # largest thing in the session. Subtract the diagonal afterwards instead.
        column_of = {value: i for i, value in enumerate(values)}
        assigned = np.flatnonzero(codes != -1)
        indicator = csr_matrix(
            (
                np.ones(assigned.size, dtype=matrix.dtype),
                (assigned, [column_of[int(v)] for v in codes[assigned]]),
            ),
            shape=(n, len(values)),
        )
        sums = np.asarray((matrix @ indicator).todense(), dtype=float)
        diagonal = matrix.diagonal()

        for value in values:
            members = np.flatnonzero(codes == value)
            # A member doesn't vouch for itself: drop it from both sum and count.
            own = sums[:, column_of[value]].copy()
            own[members] -= diagonal[members]
            counts = np.full(n, float(members.size))
            counts[members] -= 1.0
            mean = own / np.maximum(counts, 1.0) / self.max_value
            columns[f"p_{names.get(value, value)}"] = mean
        return pl.DataFrame(columns)

    def merge_support(
        self,
        *,
        n_bands: int = 4,
        max_merges: int = 200,
        max_block: int = 200,
        seed: int | None = 0,
    ) -> pl.DataFrame:
        """Which resolutions supported each merge — the type-versus-subtype signal.

        Parameters
        ----------
        n_bands : int, default 4
            How many quantile bands to split the swept resolutions into. More bands give
            finer resolution on *where* support lives, but fewer runs per band.
        max_merges : int, default 200
            Annotate only the top this many merges — the coarse end of the tree, where the
            type/subtype question lives. Logged when it truncates.
        max_block : int, default 200
            Sample each merged group down to this many cells, so the reported fractions are
            estimates rather than exact. Keeps the walk O(n × max_block) instead of O(n²).
            Logged when it bites; exact group sizes still come from :meth:`merge_table`.
        seed : int, optional
            Seeds that sampling.

        Returns
        -------
        pl.DataFrame
            One row per annotated merge: ``merge``, ``height``,
            ``coclustering_frequency``, ``n_cells``, ``support_band_0 …
            support_band_{n_bands-1}``, and ``resolution_min_supporting`` — the lowest band
            centre whose support passes 0.5, or null when none does.

        Notes
        -----
        A merge's :meth:`merge_table` frequency says *how much* of the ensemble kept the two
        groups together, but not *which part*. That distinction is the one that matters: a
        merge holding across coarse and fine runs alike is a type, while one appearing only
        in the runs fine enough to create it is a subtype. Pooling loses the difference;
        this recovers it, because the runs are still around.

        Support is measured by comparing each group's *modal* label per run — cheaper than
        counting cross-group pairs, and it answers the question at the granularity that
        matters: whether the run treated the two groups as one community.
        """
        if self._partitions is None:
            raise ValueError(
                "merge_support needs the individual runs; they persist with a saved "
                "analysis (manifest v2+), so this clustering was either built by "
                "hand or saved before runs were stored — re-run ft.cluster"
            )
        link = self.linkage
        n = self.shape[0]
        n_merges = link.shape[0]
        first = max(0, n_merges - max_merges)
        if first:
            _logger.info(
                "merge_support: annotating the top %d of %d merges (max_merges=%d)",
                n_merges - first,
                n_merges,
                max_merges,
            )

        rng = np.random.default_rng(seed)
        members = _leaf_members(link, n, max_block=max_block, rng=rng)
        capped = sum(1 for m in members[n:] if m.size == max_block)
        if capped:
            _logger.info(
                "merge_support: %d merge groups sampled down to max_block=%d cells",
                capped,
                max_block,
            )

        labels = self._partitions.labels
        resolutions = self._partitions.resolution
        edges = np.quantile(resolutions, np.linspace(0, 1, n_bands + 1))
        band = np.clip(
            np.searchsorted(edges, resolutions, side="right") - 1, 0, n_bands - 1
        )
        centres = np.array(
            [
                float(np.median(resolutions[band == b]))
                if (band == b).any()
                else np.nan
                for b in range(n_bands)
            ]
        )

        rows: list[dict[str, Any]] = []
        for merge in range(first, n_merges):
            left_rows = members[int(link[merge, 0])]
            right_rows = members[int(link[merge, 1])]
            agree = np.array(
                [
                    _modal_agreement(labels[left_rows, run], labels[right_rows, run])
                    for run in range(labels.shape[1])
                ]
            )
            supports = [
                float(agree[band == b].mean()) if (band == b).any() else float("nan")
                for b in range(n_bands)
            ]
            passing = [centres[b] for b in range(n_bands) if supports[b] > 0.5]
            row: dict[str, Any] = {
                "merge": merge,
                "height": float(link[merge, 2]),
                "coclustering_frequency": float(self.max_value - link[merge, 2]),
                "n_cells": int(link[merge, 3]),
                "resolution_min_supporting": min(passing) if passing else None,
            }
            for b in range(n_bands):
                row[f"support_band_{b}"] = supports[b]
            rows.append(row)

        frame = pl.DataFrame(rows)
        band_meta = ", ".join(
            f"band_{b}≈{centres[b]:.3g}"
            for b in range(n_bands)
            if np.isfinite(centres[b])
        )
        _logger.info("merge_support: resolution band centres %s", band_meta)
        return frame

    @property
    def partitions(self) -> Partitions | None:
        """The individual Leiden runs behind this consensus (see :class:`Partitions`).

        Persisted with a saved analysis (manifest v2+) and restored on load, so
        ``restrict``/``merge_support``/``axis_stability`` keep working after a
        reload. ``None`` only for a matrix built by hand or an analysis saved
        before the runs were stored.
        """
        return self._partitions

    @property
    def order_by(self) -> str | None:
        """Column whose per-cluster aggregate orders the ids, if one was given."""
        return self._order_by

    @property
    def order_values(self) -> np.ndarray | None:
        """The ``order_by`` values in matrix-row order. A copy."""
        return None if self._order_values is None else self._order_values.copy()

    def merge_verdicts(self, *, n_bands: int = 4, **kwargs: Any) -> pl.DataFrame:
        """:meth:`merge_support` with the reading attached, coarsest first.

        Adds ``verdict`` to the per-band support table so the three cases can be
        filtered rather than eyeballed across ``n_bands`` float columns:

        ``all-band``
            every resolution band kept the two groups together — a type.
        ``fine-only``
            some band did, not all — a subtype split that exists only because
            the fine runs could make it.
        ``unsupported``
            no band did. A threshold above this merge is joining groups nothing
            in the ensemble ever put together.

        The supports are sampled estimates (``max_block`` cells per group), so
        0.48 against 0.52 is a tie and the boundaries between these three
        categories are soft.

        Parameters
        ----------
        n_bands : int, default 4
            Number of resolution bands used to assess support.
        **kwargs
            Additional arguments forwarded to :meth:`merge_support`.

        Returns
        -------
        polars.DataFrame
            Merge-support table with a categorical ``verdict`` column.
        """
        frame = self.merge_support(n_bands=n_bands, **kwargs)
        bands = [c for c in frame.columns if c.startswith("support_band_")]
        if not bands:
            return frame.with_columns(pl.lit("unknown").alias("verdict"))
        return frame.with_columns(
            pl.when(pl.max_horizontal(bands) <= 0.5)
            .then(pl.lit("unsupported"))
            .when(pl.min_horizontal(bands) > 0.5)
            .then(pl.lit("all-band"))
            .otherwise(pl.lit("fine-only"))
            .alias("verdict")
        ).sort("height", descending=True)

    def support_ceiling(self, *, n_bands: int = 4, **kwargs: Any) -> float | None:
        """The lowest merge height no resolution band supports, or ``None``.

        The one number to read off :meth:`merge_verdicts`: cut above it and you
        have accepted a merge the ensemble never made. ``None`` means every
        annotated merge is backed by at least one band, which is the all-clear.

        Expect it to sit near the top of the tree in the healthy case — the root
        merge joins everything, and no run puts every cell in one community, so
        it is essentially always unsupported. A ceiling *well below* the tallest
        merge is the informative case.

        Parameters
        ----------
        n_bands : int, default 4
            Number of resolution bands used to assess support.
        **kwargs
            Additional arguments forwarded to :meth:`merge_verdicts`.

        Returns
        -------
        float or None
            Lowest unsupported merge height, or ``None`` when all are supported.
        """
        frame = self.merge_verdicts(n_bands=n_bands, **kwargs)
        unsupported = frame.filter(pl.col("verdict") == "unsupported")
        if unsupported.height == 0:
            return None
        return float(unsupported.select(pl.col("height").min()).item())

    def grain_coverage(self, **window: Any) -> pl.DataFrame:
        """Which runs a ``restrict`` window would keep, and whether that is enough.

        The check people skip before trusting a restriction. Seeds at one
        setting reproduce each other, so realised cluster counts arrive in
        knots; a window can fall between two of them, or catch only one graph
        type — in which case the pool has stopped marginalising over graph
        construction whatever the sweep nominally covered.

        One row, with a ``verdict`` string that says which of those happened.

        Parameters
        ----------
        **window
            Run filters accepted by :meth:`Partitions.filter`.

        Returns
        -------
        polars.DataFrame
            One-row summary of retained runs and graph-setting coverage.

        Raises
        ------
        ValueError
            If the individual partitions were not retained.
        """
        if self._partitions is None:
            raise ValueError("grain coverage needs the individual runs")
        summary = self._partitions.summary()
        inside = self._partitions.filter(**window).summary() if window else summary

        def axes(frame: pl.DataFrame) -> int:
            if frame.height == 0:
                return 0
            return int(
                frame.select(pl.struct("graph_type", "n_neighbors").n_unique()).item()
            )

        n_in, n_axes_in, n_axes_all = inside.height, axes(inside), axes(summary)
        return pl.DataFrame(
            {
                "n_runs_total": [summary.height],
                "n_runs_in_window": [n_in],
                "n_axes_total": [n_axes_all],
                "n_axes_in_window": [n_axes_in],
                "verdict": [_coverage_verdict(n_in, n_axes_in, n_axes_all)],
            }
        )

    def suggest_cut(
        self,
        *,
        min_cluster_size: int = 1,
        n_points: int = 60,
        n_bands: int = 4,
        max_unassigned: float | None = None,
        check_support: bool = True,
    ) -> CutSuggestion:
        """The widest plateau that survives the checks, with its evidence.

        A convenience over :meth:`plateaus` and :meth:`support_ceiling`, not a
        replacement for reading them: it applies the rule you would apply by
        hand — widest plateau, at the size floor you intend, below the support
        ceiling — and hands back the runner-up so the margin is visible.

        ``max_unassigned`` rejects plateaus that hold their cluster count while
        discarding more than that fraction of cells; a count that is stable only
        because the size floor is eating the cohort is not stable structure.

        Returns a :class:`CutSuggestion` whose ``threshold`` is ``None`` when
        nothing qualifies. That is a real answer about the data, not a failure:
        a cohort with no plateau has no scale the ensemble agrees on.

        Parameters
        ----------
        min_cluster_size : int, default 1
            Size floor used at every candidate cut.
        n_points : int, default 60
            Number of thresholds used to find plateaus.
        n_bands : int, default 4
            Resolution bands used for merge support.
        max_unassigned : float, optional
            Reject candidates dropping more than this fraction of cells.
        check_support : bool, default True
            Require the chosen plateau to respect the support ceiling.

        Returns
        -------
        CutSuggestion
            Best qualifying plateau and its visible alternatives.
        """
        table = self.plateaus(n_points=n_points, min_cluster_size=min_cluster_size)
        empty = table.clear()
        if table.height == 0:
            return CutSuggestion(
                None,
                None,
                None,
                min_cluster_size,
                None,
                None,
                None,
                None,
                "no plateau: the cluster count changes at every threshold, so no "
                "cut here is defensible — revisit the grain window or the space",
                empty,
            )

        n_cells = self.shape[0]
        kept = table
        if max_unassigned is not None:
            kept = table.filter(pl.col("max_unassigned") <= max_unassigned * n_cells)
            if kept.height == 0:
                return CutSuggestion(
                    None,
                    None,
                    None,
                    min_cluster_size,
                    None,
                    None,
                    None,
                    None,
                    f"every plateau discards more than {max_unassigned:.0%} of cells "
                    f"at min_cluster_size={min_cluster_size}",
                    table,
                )

        ceiling = self.support_ceiling(n_bands=n_bands) if check_support else None
        if ceiling is not None:
            below = kept.filter(pl.col("midpoint") <= ceiling)
            if below.height:
                kept = below

        best = kept.row(0, named=True)
        above = None if ceiling is None else bool(best["midpoint"] > ceiling)
        reason = (
            f"widest plateau at min_cluster_size={min_cluster_size}: "
            f"{best['n_clusters']} clusters across {best['width']:.3f} of threshold"
        )
        if above:
            reason += (
                f" — but it sits above the support ceiling ({ceiling:.3f}), so the "
                f"merges it accepts are ones no resolution band backs"
            )
        return CutSuggestion(
            threshold=float(best["midpoint"]),
            n_clusters=int(best["n_clusters"]),
            width=float(best["width"]),
            min_cluster_size=min_cluster_size,
            n_unassigned=int(best["max_unassigned"]),
            largest_cluster=int(best["largest_cluster"]),
            support_ceiling=ceiling,
            above_ceiling=above,
            reason=reason,
            alternatives=kept.head(6),
        )

    def label(
        self,
        *,
        distance_threshold: float,
        min_cluster_size: int = 1,
        name: str = "label",
        order: bool = True,
    ) -> "LabelSet":
        """Cut the dendrogram at ``distance_threshold`` into a ``LabelSet``.

        The mask-safe counterpart of ``FeatureTable.label``: cell ids and mask come
        from this clustering, so there is nothing to line up by hand. Cluster ids are
        renumbered to a contiguous ``0..k-1``; cells dropped by ``min_cluster_size``
        stay unassigned. Read ``threshold_scan()`` first to choose a threshold.

        When the clustering was built with ``order_by``, ids come out ordered by that
        column's per-cluster aggregate — cluster 0 is the shallowest, and every cut of
        this clustering numbers its clusters the same way, so two thresholds stay
        comparable. Pass ``order=False`` for the raw dendrogram order.

        Parameters
        ----------
        distance_threshold : float
            Maximum linkage distance within a cluster.
        min_cluster_size : int, default 1
            Leave smaller clusters unassigned.
        name : str, default 'label'
            Name of the returned label set.
        order : bool, default True
            Apply the recorded ``order_by`` ordering when available.

        Returns
        -------
        LabelSet
            Cluster labels aligned to this clustering's cell identifiers.
        """
        from cellpax.labels import LabelSet

        labels = LabelSet.from_clustering(
            self,
            self._cell_ids,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            name=name,
            mask=self._mask,
        )
        if order and self._order_values is not None:
            labels = labels.reorder_by(
                self._order_values,
                cell_ids=self._cell_ids,
                agg=self._order_agg,
                ascending=self._order_ascending,
            )
            # reorder keeps names pinned to their cluster, which is right for named
            # clusters but leaves a fresh cut's id-derived names off by a permutation
            # ("cluster 4" sitting at id 0). Nothing has been named yet here, so
            # restate them as the new ids.
            labels.rename({i: str(i) for i in labels.ids})
        return labels

    def __repr__(self) -> str:
        space = f", space={self._space!r}" if self._space else ""
        order = f", order_by={self._order_by!r}" if self._order_by else ""
        return (
            f"Clustering(mask={self._mask!r}, n_cells={self.shape[0]}, "
            f"n_features={len(self._columns)}{space}{order}, method={self.method!r})"
        )


# --------------------------------------------------------------------------- #
# neighborhood prediction / purity
# --------------------------------------------------------------------------- #


def neighborhood_self_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    """For each row, the labels of its ``n_neighbors`` nearest *other* rows.

    Ported from dfc's ``neighborhood_self_predictions``. Fits a kNN index on
    ``features``, queries each point against itself, and drops the self-match
    *by position* — with duplicate rows the self-edge is not guaranteed to come
    back first, so the naive ``[:, 1:]`` slice could keep it and drop a real
    neighbour (see :func:`_neighbor_arrays`, which this shares) — giving a
    leave-one-out view of local label agreement.

    Returns an ``(n_rows, n_neighbors)`` array of neighbor label values.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_rows, n_features)`` coordinate matrix.
    labels : numpy.ndarray
        Label value for each row.
    n_neighbors : int, default 20
        Number of self-excluded neighbours to query.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_neighbors)`` neighbour-label matrix.
    """
    _, indices = _neighbor_arrays(features, n_neighbors, "minkowski")
    return np.asarray(labels)[indices]


def neighborhood_purity(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    """Fraction of each point's self-excluded nearest neighbors sharing its label.

    Ported from dfc's ``compute_neighborhood_purity``: a per-cell diagnostic
    for how well a clustering's labels respect local structure in feature
    space — 1.0 means every one of a cell's ``n_neighbors`` nearest other
    cells carries the same label, 0.0 means none do.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_rows, n_features)`` coordinate matrix.
    labels : numpy.ndarray
        Label value for each row.
    n_neighbors : int, default 20
        Number of self-excluded neighbours to compare.

    Returns
    -------
    numpy.ndarray
        Per-row fraction of neighbours sharing its label.
    """
    labels = np.asarray(labels)
    neighbor_labels = neighborhood_self_predictions(
        features, labels, n_neighbors=n_neighbors
    )
    return (neighbor_labels == labels[:, None]).mean(axis=1)


def neighbor_label_composition(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_neighbors: int = 30,
    majority: float = 0.5,
    cell_ids: np.ndarray | None = None,
    names: dict[int, str | None] | None = None,
    id_column: str = "cell_id",
) -> pl.DataFrame:
    """Triage each cell by what its nearest neighbours in feature space are labelled.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` coordinates neighbours are found in — normally the
        clustering space, so the verdict is about the space the labels came from.
    labels : numpy.ndarray
        Row-aligned label codes, ``-1`` for unassigned.
    n_neighbors : int, default 30
        How many nearest neighbours each verdict is based on. Small values are noisy per
        cell; large ones blur genuinely local disagreement.
    majority : float, default 0.5
        Fraction of *assigned* neighbours a label must reach to decide the verdict. Below
        it on every label, the cell is ``"mixed"``. Raising it demands more agreement
        before calling a cell explained, moving cells into ``"mixed"``.
    cell_ids : numpy.ndarray, optional
        Ids for the identifier column. Omitted from the frame when not given.
    names : dict of int to str, optional
        Code-to-name mapping, adding readable ``label_name`` / ``top_other_name`` columns.
    id_column : str, default 'cell_id'
        Name for the identifier column.

    Returns
    -------
    pl.DataFrame
        One row per cell with ``own_fraction``, ``top_other_label``,
        ``top_other_fraction``, ``verdict`` and ``n_neighbors``.

    Notes
    -----
    The follow-up to a cell that looks misplaced on an embedding. A UMAP is a lossy
    2-D summary, so a dot in the wrong-coloured cloud has two very different
    explanations and the embedding cannot tell them apart — but the feature-space
    neighbourhood can:

    ``verdict="own"``
        Most neighbours share its label. The label is fine and the *embedding* put it
        in the wrong place; nothing to fix in the clustering.
    ``verdict="other"``
        Most neighbours carry one other label. Either the label is wrong, or this is a
        genuine rare cell that sits inside another population in feature space too.
    ``verdict="mixed"``
        No label reaches ``majority``. The consensus had nothing clear to say here —
        cross-check ``Clustering.cell_stability`` for that cell.

    One row per cell with ``own_fraction``, ``top_other_label`` and
    ``top_other_fraction``, so ``.group_by("verdict").len()`` turns "some dots
    scattered here and there" into three countable groups.

    Unassigned cells (``-1``) get ``verdict="unassigned"`` and are left out of the
    three-way split rather than being counted as a category of their own.
    """
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != np.asarray(features).shape[0]:
        raise ValueError(
            f"labels cover {labels.shape[0]} cells but features has "
            f"{np.asarray(features).shape[0]} rows"
        )
    neighbor_labels = neighborhood_self_predictions(
        features, labels, n_neighbors=n_neighbors
    )

    assigned = neighbor_labels >= 0
    n_assigned = assigned.sum(axis=1)
    own = ((neighbor_labels == labels[:, None]) & assigned).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        own_fraction = np.where(n_assigned > 0, own / n_assigned, 0.0)

    codes = np.unique(labels[labels >= 0])
    top_other = np.full(labels.shape[0], -1, dtype=np.int64)
    top_other_fraction = np.zeros(labels.shape[0], dtype=float)
    for code in codes:
        matches = ((neighbor_labels == code) & assigned).sum(axis=1)
        fraction = np.where(n_assigned > 0, matches / n_assigned, 0.0)
        # "other" excludes a cell's own label, which own_fraction already reports
        candidate = (labels != code) & (fraction > top_other_fraction)
        top_other = np.where(candidate, code, top_other)
        top_other_fraction = np.where(candidate, fraction, top_other_fraction)

    verdict = np.where(
        own_fraction >= majority,
        "own",
        np.where(top_other_fraction >= majority, "other", "mixed"),
    )
    verdict = np.where(labels < 0, "unassigned", verdict)

    frame_data: dict[str, Any] = {}
    if cell_ids is not None:
        frame_data[id_column] = np.asarray(cell_ids)
    frame_data.update(
        {
            "label": labels,
            "own_fraction": own_fraction,
            "top_other_label": top_other,
            "top_other_fraction": top_other_fraction,
            "verdict": verdict,
            "n_neighbors": np.full(labels.shape[0], n_neighbors, dtype=np.int64),
        }
    )
    frame = pl.DataFrame(frame_data)
    if names:
        lookup = {int(k): v for k, v in names.items()}
        frame = frame.with_columns(
            pl.col("label").replace_strict(lookup, default=None).alias("label_name"),
            pl.col("top_other_label")
            .replace_strict(lookup, default=None)
            .alias("top_other_name"),
        )
    return frame
