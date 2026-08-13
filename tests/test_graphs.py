"""Graph construction as a swept axis: the three weightings and their failure modes."""

from __future__ import annotations

import warnings

import igraph as ig
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from cellpax.clustering import (
    GRAPH_TYPES,
    _smooth_knn_weights,
    cluster_leiden,
    kneighbor_graph,
)


def _three_blobs(n: int = 90, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, 0.4, (n // 3, 5)) for c in (0.0, 6.0, 12.0)])


def _dense_core_sparse_tail(seed: int = 1) -> np.ndarray:
    """A tight core plus a diffuse halo — where the weightings visibly disagree."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.4, (200, 5)), rng.normal(5, 2.5, (40, 5))])


def _old_kneighbor_graph(data, k, metric="minkowski", mutual_only=False):
    """The pre-``graph_type`` implementation, kept here to pin the default path."""
    finder = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    finder.fit(data)
    _, indices = finder.kneighbors(data)
    edges = [
        (i, neighbor)
        for i, neighbors in enumerate(indices)
        for neighbor in neighbors
        if i != neighbor
    ]
    graph = ig.Graph(edges=edges, directed=True)
    return graph.as_undirected(mode="mutual" if mutual_only else "collapse")


# -- the default path is unchanged -----------------------------------------------


def test_plain_knn_is_identical_to_the_old_implementation() -> None:
    data = _three_blobs()
    for mutual_only in (False, True):
        new = kneighbor_graph(data, 15, mutual_only=mutual_only)
        old = _old_kneighbor_graph(data, 15, mutual_only=mutual_only)
        assert new.ecount() == old.ecount()
        assert sorted(new.get_edgelist()) == sorted(old.get_edgelist())


def test_plain_knn_carries_no_weights() -> None:
    """Which is how ``cluster_leiden`` knows not to pass any."""
    assert "weight" not in kneighbor_graph(_three_blobs(), 15).es.attributes()


def test_every_graph_type_builds_and_records_its_provenance() -> None:
    data = _three_blobs()
    for graph_type in GRAPH_TYPES:
        graph = kneighbor_graph(data, 15, graph_type=graph_type)
        assert graph.vcount() == data.shape[0]
        assert graph.ecount() > 0
        assert graph["graph_type"] == graph_type
        assert graph["n_neighbors"] == 15
        assert graph["n_degenerate"] == 0


def test_unknown_graph_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="graph_type must be one of"):
        kneighbor_graph(_three_blobs(), 15, graph_type="jaccard")  # type: ignore[arg-type]


def test_weighted_types_carry_weights() -> None:
    data = _three_blobs()
    for graph_type in ("knn_distance", "snn_jaccard", "umap_fuzzy"):
        graph = kneighbor_graph(data, 15, graph_type=graph_type)
        weights = np.asarray(graph.es["weight"])
        assert weights.shape[0] == graph.ecount()
        assert np.all(weights > 0)


# -- the fuzzy set's defining guarantee ------------------------------------------


def test_umap_fuzzy_leaves_every_cell_one_full_weight_edge() -> None:
    """Subtracting rho_i is the whole point: nothing is ever fully disconnected.

    This is the property that tends to hold rare and peripheral populations together, and
    the one SNN/Jaccard lacks.
    """
    graph = kneighbor_graph(_dense_core_sparse_tail(), 15, graph_type="umap_fuzzy")
    strongest = np.zeros(graph.vcount())
    for edge in graph.es:
        strongest[edge.source] = max(strongest[edge.source], edge["weight"])
        strongest[edge.target] = max(strongest[edge.target], edge["weight"])
    np.testing.assert_allclose(strongest.min(), 1.0)


def test_smooth_knn_weights_hit_the_log2k_target() -> None:
    data = _dense_core_sparse_tail()
    finder = NearestNeighbors(n_neighbors=16).fit(data)
    distances, _ = finder.kneighbors(data)
    weights, degenerate = _smooth_knn_weights(distances[:, 1:])
    np.testing.assert_allclose(weights.sum(axis=1), np.log2(15), atol=1e-4)
    assert not degenerate.any()


def test_snn_jaccard_offers_no_such_guarantee() -> None:
    graph = kneighbor_graph(
        _dense_core_sparse_tail(), 15, graph_type="snn_jaccard", prune=1 / 15
    )
    strongest = np.zeros(graph.vcount())
    for edge in graph.es:
        strongest[edge.source] = max(strongest[edge.source], edge["weight"])
        strongest[edge.target] = max(strongest[edge.target], edge["weight"])
    assert strongest.min() < 1.0


def test_snn_jaccard_prunes_harder_than_the_fuzzy_set() -> None:
    """The documented difference in failure mode, pinned rather than asserted in prose."""
    data = _dense_core_sparse_tail()
    fuzzy = kneighbor_graph(data, 15, graph_type="umap_fuzzy")
    jaccard = kneighbor_graph(data, 15, graph_type="snn_jaccard", prune=1 / 15)
    assert jaccard.ecount() < fuzzy.ecount()
    tail = np.arange(200, 240)
    assert min(jaccard.degree(tail)) < min(fuzzy.degree(tail))


def test_prune_removes_weak_jaccard_edges_monotonically() -> None:
    data = _dense_core_sparse_tail()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = [
            kneighbor_graph(data, 15, graph_type="snn_jaccard", prune=p).ecount()
            for p in (0.0, 0.1, 0.25, 0.4)
        ]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_pruning_to_isolation_is_reported_apart_from_duplicate_rows() -> None:
    """Two different broken neighbourhoods, so two different counts and warnings.

    A stranded cell is Jaccard's documented cost; a zero-distance neighbourhood is a
    duplicate-row problem. Reporting the second when the first happened would send you
    looking for ties that are not there.
    """
    data = _dense_core_sparse_tail()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = kneighbor_graph(data, 15, graph_type="snn_jaccard", prune=0.4)
    assert graph["n_degenerate"] == 0
    assert graph["n_isolated"] > 0
    messages = [str(w.message) for w in caught]
    assert any("no edges" in m for m in messages)
    assert not any("distance 0" in m for m in messages)


# -- Leiden must actually use the weights ----------------------------------------


def test_leiden_uses_edge_weights_when_present() -> None:
    """Ignoring them would quietly turn every weighting back into an unweighted graph."""
    graph = ig.Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    graph.es["weight"] = [10.0, 0.01, 10.0, 0.01, 0.01]
    weighted = cluster_leiden(graph, resolution_parameter=1.0, seed=0)

    unweighted = graph.copy()
    del unweighted.es["weight"]
    plain = cluster_leiden(unweighted, resolution_parameter=1.0, seed=0)

    # the strong edges pair (0,1) and (2,3); the unweighted graph has no reason to
    assert weighted[0] == weighted[1] and weighted[2] == weighted[3]
    assert len(set(weighted)) != len(set(plain)) or list(weighted) != list(plain)


def test_min_cluster_size_still_drops_small_clusters_on_weighted_graphs() -> None:
    data = _three_blobs()
    graph = kneighbor_graph(data, 15, graph_type="umap_fuzzy")
    labels = cluster_leiden(
        graph, resolution_parameter=8.0, seed=0, min_cluster_size=20
    )
    assert (labels == -1).any()


# -- ties and duplicate rows -----------------------------------------------------


def test_duplicate_rows_are_reported_as_degenerate_and_warned_about() -> None:
    """A whole neighbourhood at distance 0 leaves rho_i and the bandwidth meaningless."""
    rng = np.random.default_rng(0)
    data = np.vstack([np.zeros((20, 4)), rng.normal(5, 0.3, (60, 4))])
    for graph_type in ("knn_distance", "umap_fuzzy"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = kneighbor_graph(data, 15, graph_type=graph_type)
        assert graph["n_degenerate"] == 20
        assert any("degenerate neighbourhood" in str(w.message) for w in caught)


def test_degenerate_cells_still_get_usable_edges() -> None:
    rng = np.random.default_rng(0)
    data = np.vstack([np.zeros((20, 4)), rng.normal(5, 0.3, (60, 4))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = kneighbor_graph(data, 15, graph_type="umap_fuzzy")
    weights = np.asarray(graph.es["weight"])
    assert np.all(np.isfinite(weights))
    assert min(graph.degree(range(20))) > 0


def test_self_edges_are_dropped_even_when_ties_reorder_the_neighbours() -> None:
    """Slicing [:, 1:] would keep a self-edge when a tied row sorts first instead."""
    rng = np.random.default_rng(0)
    data = np.vstack([np.zeros((10, 3)), rng.normal(4, 0.3, (40, 3))])
    for graph_type in GRAPH_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = kneighbor_graph(data, 8, graph_type=graph_type)
        assert not any(edge.source == edge.target for edge in graph.es)
