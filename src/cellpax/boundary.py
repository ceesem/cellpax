"""Is the boundary between two clusters a gap or a cut through one thing?

Separability cannot answer this: any slice through a continuum is stably
distinguishable at its ends, which is why a classifier test certifies every
split it is shown (and why CHOIR left the library). The question that matters
is whether a *density valley*, a *connectivity break*, or a *multimodal
profile* actually exists at the boundary — three different measurements that
fail differently, reported side by side per cluster pair:

``dip`` / ``dip_p``
    Hartigan's dip test on the pair's cells projected onto the axis joining
    the cluster centroids (via the ``diptest`` package). Two genuine modes
    give a significant dip; a sliced continuum projects to one mode. Blind to
    curved boundaries — a banana cut lengthwise can project unimodally.

``connectivity_ratio``
    Observed kNN cross-edges between the pair against the configuration-model
    expectation ``d_a * d_b / 2m`` (the modularity null; the statistic PAGA
    built cluster-graph abstraction on — Wolf et al., Genome Biology 2019).
    Near 0 means the graph itself separates the pair; the absolute scale
    depends on ``n_neighbors`` and geometry, so read it comparatively across
    the pairs of one report rather than against a universal constant.

``valley_ratio``
    Saddle-to-peak density along the boundary: the densest kNN cross-edge
    (each cell's density proxied by its inverse k-th neighbour distance)
    against the pair's weaker density peak. Near 0 is a deep valley or no
    contact at all; near 1 means the boundary runs through terrain as dense
    as the clusters themselves — the "northwest corner vs southwest corner"
    signature. With the optional ``dadapy`` extra, ``density="pak"`` swaps
    the proxy for the point-adaptive kNN estimator (Rodriguez et al.), the
    density that Advanced Density Peaks clustering merges on.

``cocluster_cross_mean`` / ``cocluster_band``
    From the consensus matrix, when one is supplied: the mean co-clustering
    frequency across the pair, and the fraction of cross pairs at
    *intermediate* frequency (0.2–0.8). A real gap has its cross mass pinned
    near 0 — runs either keep the pair apart or the threshold was silly —
    while a cut continuum shows a band of cells that co-cluster in some runs
    and not others. This is the one leg no other package can provide, because
    it is a property of the ensemble, not of the coordinates.

``verdict``
    ``"discrete"`` / ``"continuous"`` / ``"ambiguous"`` by majority of the
    legs that expressed an opinion. The verdict is a summary, not a result:
    the columns are the result, and the thresholds behind the votes are
    parameters with documented defaults, not truths.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, issparse

__all__ = ["boundary_report"]


def _knn_density(features: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Inverse k-th-neighbour distance — a monotone, scale-free density proxy."""
    from sklearn.neighbors import NearestNeighbors

    finder = NearestNeighbors(n_neighbors=n_neighbors + 1)
    finder.fit(features)
    distances, _ = finder.kneighbors(features)
    radius = distances[:, -1]
    return 1.0 / np.maximum(radius, np.finfo(float).tiny)


def _pak_density(features: np.ndarray) -> np.ndarray:
    """Point-adaptive kNN log-density via the optional dadapy extra."""
    try:
        from dadapy import Data
    except ImportError as error:
        raise ImportError(
            "density='pak' needs the optional 'dadapy' package; install the "
            "commit-pinned extra: pip install 'cellpax[dadapy]'"
        ) from error
    data = Data(np.ascontiguousarray(features, dtype=float), verbose=False)
    data.compute_density_PAk()
    log_den = np.asarray(data.log_den, dtype=float)
    # densities enter the valley statistic only through ratios of exp(log_den);
    # subtracting the max keeps the exponentials finite
    return np.exp(log_den - log_den.max())


def _knn_graph_edges(features: np.ndarray, n_neighbors: int) -> csr_matrix:
    """Symmetrised unweighted kNN adjacency (union rule), no self edges."""
    from sklearn.neighbors import kneighbors_graph

    directed = kneighbors_graph(features, n_neighbors=n_neighbors, mode="connectivity")
    adjacency = directed.maximum(directed.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency.tocsr()


def boundary_report(
    features: np.ndarray,
    codes: np.ndarray,
    *,
    similarity: Any = None,
    max_value: float = 1.0,
    names: dict[int, str] | None = None,
    n_neighbors: int = 15,
    density: str = "knn",
    min_cells: int = 10,
    dip_alpha: float = 0.05,
    valley_deep: float = 0.5,
    connectivity_low: float = 0.1,
    connectivity_high: float = 0.5,
    band_wide: float = 0.25,
    cross_sparse: float = 0.05,
) -> pl.DataFrame:
    """Per-cluster-pair evidence that a boundary is a gap versus a cut.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_dims)`` coordinates in the space the clustering was
        computed in — the space the boundary claim is *about*.
        ``FeatureTable.boundary_report`` reconstructs this from a stored
        clustering's recorded parameters.
    codes : numpy.ndarray
        Row-aligned cluster codes, ``-1`` unassigned (excluded throughout).
    similarity : scipy sparse or numpy array, optional
        The consensus similarity matrix, row-aligned with ``features``.
        Enables the co-clustering columns — the ensemble's own read on the
        boundary — and comes along automatically through the table method.
    max_value : float, default 1.0
        The similarity's full-agreement value (``Clustering.max_value``).
    names : dict of int to str, optional
        Code-to-name mapping for readable ``cluster_a`` / ``cluster_b``.
    n_neighbors : int, default 15
        Neighbourhood size for both the density proxy and the cross-edge
        graph. The verdict should be checked at more than one value before
        being believed; it is a resolution parameter like any other.
    density : {'knn', 'pak'}, default 'knn'
        ``'knn'`` is the in-library inverse k-th-neighbour-distance proxy;
        ``'pak'`` is dadapy's point-adaptive estimator (optional extra),
        better calibrated when density varies across the population.
    min_cells : int, default 10
        Pairs where either cluster is smaller than this are reported with
        null statistics rather than silently dropped.
    dip_alpha, valley_deep, connectivity_low, connectivity_high, band_wide, cross_sparse : float
        The vote thresholds — see the module docstring for what each leg
        votes on. Defaults are starting points, and the verdict column is a
        summary of the evidence columns, never a substitute for them.

    Returns
    -------
    pl.DataFrame
        One row per unordered cluster pair, worst-separated first (highest
        ``valley_ratio``): sizes, the evidence columns, per-leg votes, and
        ``verdict``.
    """
    if density not in {"knn", "pak"}:
        raise ValueError(f"density must be 'knn' or 'pak', got {density!r}")
    import diptest as _diptest

    features = np.asarray(features, dtype=float)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if features.shape[0] != codes.shape[0]:
        raise ValueError("features and codes must have the same number of rows")
    clusters = sorted(int(v) for v in np.unique(codes) if int(v) != -1)
    if len(clusters) < 2:
        raise ValueError("boundary_report needs at least two clusters")

    rho = (
        _knn_density(features, n_neighbors)
        if density == "knn"
        else _pak_density(features)
    )
    adjacency = _knn_graph_edges(features, n_neighbors)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    total_edges = adjacency.nnz / 2.0

    sim = None
    if similarity is not None:
        sim = similarity.tocsr() if issparse(similarity) else csr_matrix(similarity)

    members = {value: np.flatnonzero(codes == value) for value in clusters}
    label = {
        value: str(names.get(value, value)) if names else str(value)
        for value in clusters
    }
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(clusters):
        for b in clusters[i + 1 :]:
            rows.append(
                _pair_row(
                    a,
                    b,
                    members[a],
                    members[b],
                    label,
                    features=features,
                    rho=rho,
                    adjacency=adjacency,
                    degrees=degrees,
                    total_edges=total_edges,
                    sim=sim,
                    max_value=max_value,
                    diptest_module=_diptest,
                    min_cells=min_cells,
                    thresholds=dict(
                        dip_alpha=dip_alpha,
                        valley_deep=valley_deep,
                        connectivity_low=connectivity_low,
                        connectivity_high=connectivity_high,
                        band_wide=band_wide,
                        cross_sparse=cross_sparse,
                    ),
                )
            )
    frame = pl.DataFrame(rows)
    return frame.sort("valley_ratio", descending=True, nulls_last=True)


def _pair_row(
    a: int,
    b: int,
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    label: dict[int, str],
    *,
    features: np.ndarray,
    rho: np.ndarray,
    adjacency: csr_matrix,
    degrees: np.ndarray,
    total_edges: float,
    sim: csr_matrix | None,
    max_value: float,
    diptest_module: Any,
    min_cells: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cluster_a": label[a],
        "cluster_b": label[b],
        "n_a": int(rows_a.size),
        "n_b": int(rows_b.size),
    }
    columns = [
        "dip",
        "dip_p",
        "connectivity_ratio",
        "valley_ratio",
        "cocluster_cross_mean",
        "cocluster_band",
        "vote_dip",
        "vote_connectivity",
        "vote_valley",
        "vote_cocluster",
        "verdict",
    ]
    if rows_a.size < min_cells or rows_b.size < min_cells:
        base.update({c: None for c in columns})
        base["verdict"] = "too_small"
        return base

    # dip on the centroid-difference axis
    axis = features[rows_b].mean(axis=0) - features[rows_a].mean(axis=0)
    norm = float(np.linalg.norm(axis))
    pair_rows = np.concatenate([rows_a, rows_b])
    if norm == 0:
        dip, dip_p = None, None
    else:
        projected = features[pair_rows] @ (axis / norm)
        dip, dip_p = diptest_module.diptest(projected)
        dip, dip_p = float(dip), float(dip_p)

    # cross edges vs the configuration null
    cross = adjacency[rows_a][:, rows_b]
    observed = float(cross.nnz)
    expected = (
        float(degrees[rows_a].sum())
        * float(degrees[rows_b].sum())
        / max(2.0 * total_edges, np.finfo(float).tiny)
    )
    connectivity = observed / max(expected, np.finfo(float).tiny)

    # saddle-to-peak density along the boundary
    peak = min(
        float(np.quantile(rho[rows_a], 0.9)), float(np.quantile(rho[rows_b], 0.9))
    )
    if observed:
        cx = cross.tocoo()
        saddle = float(np.max(np.minimum(rho[rows_a[cx.row]], rho[rows_b[cx.col]])))
    else:
        saddle = 0.0
    valley = saddle / max(peak, np.finfo(float).tiny)

    cross_mean = band = None
    if sim is not None:
        block = sim[rows_a][:, rows_b]
        n_pairs = rows_a.size * rows_b.size
        cross_mean = float(block.sum()) / (n_pairs * max_value)
        if block.nnz:
            values = np.asarray(block.tocoo().data) / max_value
            band = float(((values > 0.2) & (values < 0.8)).sum()) / n_pairs
        else:
            band = 0.0

    t = thresholds
    votes: dict[str, str | None] = {
        "vote_dip": None
        if dip_p is None
        else ("discrete" if dip_p < t["dip_alpha"] else "continuous"),
        "vote_valley": "discrete" if valley < t["valley_deep"] else "continuous",
        "vote_connectivity": (
            "discrete"
            if connectivity < t["connectivity_low"]
            else "continuous"
            if connectivity > t["connectivity_high"]
            else None
        ),
        "vote_cocluster": None
        if cross_mean is None
        else (
            "discrete"
            if cross_mean < t["cross_sparse"]
            else "continuous"
            if band is not None and band > t["band_wide"]
            else None
        ),
    }
    cast = [v for v in votes.values() if v is not None]
    discrete = cast.count("discrete")
    continuous = cast.count("continuous")
    if len(cast) >= 2 and discrete > continuous:
        verdict = "discrete"
    elif len(cast) >= 2 and continuous > discrete:
        verdict = "continuous"
    else:
        verdict = "ambiguous"

    base.update(
        {
            "dip": dip,
            "dip_p": dip_p,
            "connectivity_ratio": float(connectivity),
            "valley_ratio": float(valley),
            "cocluster_cross_mean": cross_mean,
            "cocluster_band": band,
            **votes,
            "verdict": verdict,
        }
    )
    return base
