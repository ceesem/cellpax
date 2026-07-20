"""Clustering-generator interfaces and built-in implementations."""

from cellpax.generators.base import (
    CandidateGenerator,
    CandidatePartition,
    GeneratorArtifacts,
    builtin_generator,
)
from cellpax.generators.fauxnograph import (
    FauxnographGenerator,
    FauxnographRun,
    FauxnographState,
    HierarchyPreflightError,
    cluster_leiden,
    coclustering_matrix,
    estimate_hierarchy_memory,
    fauxnograph_clustering,
    fauxnograph_coclustering,
    kneighbor_graph,
)

__all__ = [
    "CandidateGenerator",
    "CandidatePartition",
    "FauxnographGenerator",
    "FauxnographRun",
    "FauxnographState",
    "HierarchyPreflightError",
    "GeneratorArtifacts",
    "builtin_generator",
    "cluster_leiden",
    "coclustering_matrix",
    "fauxnograph_clustering",
    "fauxnograph_coclustering",
    "estimate_hierarchy_memory",
    "kneighbor_graph",
]
