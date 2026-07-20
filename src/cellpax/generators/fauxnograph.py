"""Self-contained fauxnograph consensus clustering backend."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import leidenalg as la
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

from cellpax.config import CandidateCutConfig, ClusteringConfig
from cellpax.generators.base import CandidatePartition, GeneratorArtifacts

DEFAULT_HIERARCHY_MEMORY_LIMIT_BYTES = 1024**3


class HierarchyPreflightError(ValueError):
    """Raised before an obviously oversized quadratic hierarchy materialization."""


def estimate_hierarchy_memory(n_cells: int) -> dict[str, int]:
    """Conservatively estimate dense/linkage bytes required by the hierarchy path."""
    if n_cells < 0:
        raise ValueError("n_cells must be non-negative")
    square_float32 = n_cells * n_cells * 4
    condensed_float64 = n_cells * (n_cells - 1) // 2 * 8
    linkage_float64 = max(n_cells - 1, 0) * 4 * 8
    return {
        "consensus_dense": square_float32,
        "distance_dense": square_float32,
        "condensed_working": condensed_float64,
        "linkage": linkage_float64,
        "estimated_peak": (square_float32 * 2 + condensed_float64 + linkage_float64),
    }


@dataclass(frozen=True, slots=True)
class FauxnographState:
    """Portable output of the expensive fauxnograph consensus computation."""

    coclustering: csr_matrix
    cell_ids: np.ndarray
    label_runs: np.ndarray
    runs: tuple["FauxnographRun", ...]
    linkage_matrix: np.ndarray | None

    def run_metadata(self) -> pl.DataFrame:
        """Return one inspectable provenance row for each retained label column."""
        return pl.DataFrame(
            [run.row() for run in self.runs],
            schema={
                "run_index": pl.Int32,
                "n_neighbors": pl.Int32,
                "resolution_parameter": pl.Float64,
                "repeat": pl.Int32,
                "seed": pl.Int64,
                "n_candidates": pl.Int32,
                "n_noise": pl.Int64,
            },
        )

    def labels(self, *, long: bool = False) -> pl.DataFrame:
        """Materialize retained labels in compact wide or tidy long form."""
        wide = pl.DataFrame(
            {
                "cell_id": pl.Series(self.cell_ids, dtype=pl.Int64),
                **{
                    f"run_{index:05d}": pl.Series(labels, dtype=pl.Int32)
                    for index, labels in enumerate(self.label_runs.T)
                },
            }
        )
        if not long:
            return wide
        return (
            wide.unpivot(
                index="cell_id", variable_name="run", value_name="candidate_id"
            )
            .with_columns(
                pl.col("run").str.strip_prefix("run_").cast(pl.Int32).alias("run_index")
            )
            .select("cell_id", "run_index", "candidate_id")
        )


@dataclass(frozen=True, slots=True)
class FauxnographRun:
    """Parameters and summary statistics for one retained Leiden partition."""

    run_index: int
    n_neighbors: int
    resolution_parameter: float
    repeat: int
    seed: int
    n_candidates: int
    n_noise: int

    def row(self) -> dict[str, int | float]:
        return {
            "run_index": self.run_index,
            "n_neighbors": self.n_neighbors,
            "resolution_parameter": self.resolution_parameter,
            "repeat": self.repeat,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "n_noise": self.n_noise,
        }


def kneighbor_graph(
    data: np.ndarray,
    n_neighbors: int = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    neighbor_weighting: str = "unweighted",
) -> ig.Graph:
    """Construct the undirected k-nearest-neighbor graph used by fauxnograph."""
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("data must be a two-dimensional array with at least two rows")
    if not 0 < n_neighbors < data.shape[0]:
        raise ValueError("n_neighbors must be positive and smaller than the row count")
    nearest = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nearest.fit(data)
    _, indices = nearest.kneighbors(data)
    if neighbor_weighting not in {"unweighted", "jaccard"}:
        raise ValueError("neighbor_weighting must be 'unweighted' or 'jaccard'")
    jaccard_sets = [set(map(int, neighbors)) for neighbors in indices]
    neighbor_sets = [
        neighbors - {row_index} for row_index, neighbors in enumerate(jaccard_sets)
    ]
    edges = [
        (row_index, int(neighbor))
        for row_index, neighbors in enumerate(neighbor_sets)
        for neighbor in neighbors
    ]
    graph = ig.Graph(n=data.shape[0], edges=edges, directed=True)
    graph = graph.as_undirected(mode="mutual" if mutual_only else "collapse")
    if neighbor_weighting == "jaccard":
        graph.es["weight"] = [
            len(jaccard_sets[left] & jaccard_sets[right])
            / len(jaccard_sets[left] | jaccard_sets[right])
            for left, right in graph.get_edgelist()
        ]
    return graph


def cluster_leiden(
    graph: ig.Graph,
    *,
    resolution_parameter: float = 1.0,
    min_cluster_size: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Run one RB-configuration Leiden partition on a prepared graph."""
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution_parameter,
        seed=seed,
        weights=graph.es["weight"] if "weight" in graph.es.attributes() else None,
    )
    labels = np.asarray(partition.membership, dtype=np.int32)
    if min_cluster_size > 1:
        small = np.flatnonzero(np.asarray(partition.sizes()) < min_cluster_size)
        for candidate_id in small:
            labels[labels == candidate_id] = -1
    return labels


def fauxnograph_clustering(
    data: np.ndarray | None = None,
    *,
    graph: ig.Graph | None = None,
    n_neighbors: int = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    neighbor_weighting: str = "unweighted",
    resolution_parameter: float = 1.0,
    min_cluster_size: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Run a single Phenograph-style kNN/Leiden clustering."""
    if graph is None:
        if data is None:
            raise ValueError("Either data or graph must be provided")
        graph = kneighbor_graph(
            data, n_neighbors, metric, mutual_only, neighbor_weighting
        )
    return cluster_leiden(
        graph,
        resolution_parameter=resolution_parameter,
        min_cluster_size=min_cluster_size,
        seed=seed,
    )


def coclustering_matrix(
    label_runs: np.ndarray,
    *,
    normalize: bool = False,
    opportunity_normalize: bool = False,
) -> csr_matrix:
    """Build a sparse cell-by-cell co-clustering matrix from repeated labels."""
    if label_runs.ndim != 2 or label_runs.shape[1] == 0:
        raise ValueError("label_runs must contain at least one clustering run")
    blocks: list[csr_matrix] = []
    for labels in label_runs.T:
        valid = labels >= 0
        rows = np.flatnonzero(valid)
        values = labels[valid]
        if len(rows) == 0:
            continue
        _, columns = np.unique(values, return_inverse=True)
        blocks.append(
            csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, columns)),
                shape=(len(labels), int(columns.max()) + 1),
            )
        )
    if not blocks:
        return csr_matrix((label_runs.shape[0], label_runs.shape[0]), dtype=np.float32)
    indicator = hstack(blocks, format="csr")
    matrix = (indicator @ indicator.T).tocsr()
    if opportunity_normalize:
        matrix = matrix.astype(np.float32).tocoo()
        valid = (label_runs >= 0).astype(np.int32)
        chunk_size = 250_000
        for start in range(0, matrix.nnz, chunk_size):
            stop = min(start + chunk_size, matrix.nnz)
            denominators = np.einsum(
                "ij,ij->i",
                valid[matrix.row[start:stop]],
                valid[matrix.col[start:stop]],
            )
            matrix.data[start:stop] /= denominators
        matrix = matrix.tocsr()
    elif normalize:
        matrix = matrix.astype(np.float32) / label_runs.shape[1]
    return matrix


def fauxnograph_coclustering(
    data: np.ndarray,
    *,
    n_neighbors: int | list[int] = 30,
    metric: str = "minkowski",
    mutual_only: bool = False,
    neighbor_weighting: str = "unweighted",
    n_times: int = 1,
    resolution_parameter: float | list[float] = 1.0,
    min_cluster_size: int = 1,
    normalize: bool = False,
    opportunity_normalize: bool = False,
    seed: int | None = None,
    n_jobs: int = -1,
    return_runs: bool = False,
) -> csr_matrix | tuple[csr_matrix, np.ndarray, tuple[FauxnographRun, ...]]:
    """Run repeated Leiden partitions and return their co-clustering matrix."""
    neighbors = [n_neighbors] if isinstance(n_neighbors, int) else list(n_neighbors)
    resolutions = (
        [resolution_parameter]
        if isinstance(resolution_parameter, (int, float))
        else list(resolution_parameter)
    )
    total_runs = len(neighbors) * len(resolutions) * n_times
    random = np.random.default_rng(seed)
    seeds = iter(random.integers(0, 2**31, size=total_runs))
    runs: list[np.ndarray] = []
    metadata: list[FauxnographRun] = []
    run_index = 0
    for neighbor_count in neighbors:
        graph = kneighbor_graph(
            data, neighbor_count, metric, mutual_only, neighbor_weighting
        )
        jobs = [
            (resolution, repeat, int(next(seeds)))
            for resolution in resolutions
            for repeat in range(n_times)
        ]
        results = Parallel(n_jobs=n_jobs)(
            delayed(cluster_leiden)(
                graph,
                resolution_parameter=resolution,
                min_cluster_size=min_cluster_size,
                seed=run_seed,
            )
            for resolution, _, run_seed in jobs
        )
        for (resolution, repeat, run_seed), labels in zip(jobs, results, strict=True):
            runs.append(labels)
            retained = labels[labels >= 0]
            metadata.append(
                FauxnographRun(
                    run_index=run_index,
                    n_neighbors=neighbor_count,
                    resolution_parameter=float(resolution),
                    repeat=repeat,
                    seed=run_seed,
                    n_candidates=len(np.unique(retained)),
                    n_noise=int(np.sum(labels < 0)),
                )
            )
            run_index += 1
    label_matrix = np.column_stack(runs).astype(np.int32, copy=False)
    matrix = coclustering_matrix(
        label_matrix,
        normalize=normalize,
        opportunity_normalize=opportunity_normalize,
    )
    if return_runs:
        return matrix, label_matrix, tuple(metadata)
    return matrix


def _generic_hierarchy(
    linkage_matrix: np.ndarray, cell_ids: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_cells = len(cell_ids)
    parent: dict[int, int] = {}
    children: dict[int, tuple[int, int]] = {}
    sizes = {index: 1 for index in range(n_cells)}
    scores: dict[int, float | None] = {index: None for index in range(n_cells)}
    for offset, row in enumerate(linkage_matrix):
        node = n_cells + offset
        left, right = int(row[0]), int(row[1])
        parent[left] = node
        parent[right] = node
        children[node] = (left, right)
        sizes[node] = int(row[3])
        scores[node] = float(row[2])
    root = 2 * n_cells - 2
    levels = {root: 0}
    pending = [root]
    while pending:
        node = pending.pop()
        for child in children.get(node, ()):
            levels[child] = levels[node] + 1
            pending.append(child)

    nodes = pl.DataFrame(
        [
            {
                "node_id": f"node-{node}",
                "parent_node_id": None if node == root else f"node-{parent[node]}",
                "level": levels[node],
                "merge_score": scores[node],
                "n_cells": sizes[node],
                "metadata_json": "{}",
            }
            for node in range(2 * n_cells - 1)
        ],
        schema={
            "node_id": pl.String,
            "parent_node_id": pl.String,
            "level": pl.Int32,
            "merge_score": pl.Float64,
            "n_cells": pl.Int64,
            "metadata_json": pl.String,
        },
    )
    members = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "leaf_node_id": [f"node-{index}" for index in range(n_cells)],
        }
    )
    return nodes, members


@dataclass(slots=True)
class FauxnographGenerator:
    """Generator using CellPax's owned fauxnograph consensus implementation."""

    method = "fauxnograph"
    hierarchy_memory_limit_bytes: int = DEFAULT_HIERARCHY_MEMORY_LIMIT_BYTES

    def __post_init__(self) -> None:
        if self.hierarchy_memory_limit_bytes < 1:
            raise ValueError("hierarchy_memory_limit_bytes must be positive")

    def compute(
        self,
        coordinates: np.ndarray,
        cell_ids: np.ndarray,
        config: ClusteringConfig,
    ) -> GeneratorArtifacts:
        params = dict(config.compute_params)
        method = params.pop("linkage_method")
        build_hierarchy = params.pop("build_hierarchy")
        if max(params["n_neighbors"]) >= len(cell_ids):
            raise ValueError("Every n_neighbors value must be smaller than the scope")
        estimate = estimate_hierarchy_memory(len(cell_ids))
        if (
            build_hierarchy
            and estimate["estimated_peak"] > self.hierarchy_memory_limit_bytes
        ):
            gib = estimate["estimated_peak"] / 1024**3
            limit_gib = self.hierarchy_memory_limit_bytes / 1024**3
            raise HierarchyPreflightError(
                f"Hierarchy for {len(cell_ids):,} cells is estimated to require "
                f"{gib:.2f} GiB of quadratic working memory, above the backend "
                f"limit of {limit_gib:.2f} GiB; set build_hierarchy=False for "
                "the sparse/native path or explicitly raise the generator limit"
            )
        matrix, labels, runs = fauxnograph_coclustering(
            coordinates, seed=config.seed, return_runs=True, **params
        )
        linkage_matrix: np.ndarray | None = None
        nodes: pl.DataFrame | None = None
        members: pl.DataFrame | None = None
        if build_hierarchy:
            dense = matrix.toarray()
            distances = float(np.max(dense)) - dense
            np.fill_diagonal(distances, 0.0)
            linkage_matrix = linkage(squareform(distances, checks=True), method=method)
            nodes, members = _generic_hierarchy(linkage_matrix, cell_ids)
        return GeneratorArtifacts(
            payload=FauxnographState(
                coclustering=matrix,
                cell_ids=cell_ids.astype(np.int64, copy=True),
                label_runs=labels,
                runs=runs,
                linkage_matrix=linkage_matrix,
            ),
            hierarchy_nodes=nodes,
            hierarchy_members=members,
            structural_summary={
                "consensus_shape": [len(cell_ids), len(cell_ids)],
                "consensus_nnz": matrix.nnz,
                "label_runs_shape": list(labels.shape),
                "n_runs": len(runs),
            },
        )

    def cut(self, payload: object, config: CandidateCutConfig) -> CandidatePartition:
        if not isinstance(payload, FauxnographState):
            raise TypeError("Fauxnograph generator payload has the wrong type")
        params = config.cut_params
        if config.cut_method == "distance":
            if payload.linkage_matrix is None:
                raise ValueError("Distance cuts require build_hierarchy=True")
            labels = fcluster(
                payload.linkage_matrix,
                t=params["distance_threshold"],
                criterion="distance",
            ).astype(np.int32)
            labels -= labels.min()
        elif config.cut_method == "native":
            run_index = params["run_index"]
            if run_index >= payload.label_runs.shape[1]:
                raise ValueError("run_index is outside the retained fauxnograph runs")
            labels = payload.label_runs[:, run_index].copy()
        else:
            raise ValueError("Fauxnograph supports distance and native cuts")
        minimum = params["min_cluster_size"]
        if minimum > 1:
            values, counts = np.unique(labels, return_counts=True)
            for value in values[counts < minimum]:
                labels[labels == value] = -1
        return CandidatePartition(candidate_ids=labels)
