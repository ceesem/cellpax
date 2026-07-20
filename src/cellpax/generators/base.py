"""Method-independent contracts for clustering generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

from cellpax.config import CandidateCutConfig, ClusteringConfig


@dataclass(frozen=True, slots=True)
class GeneratorArtifacts:
    """Serializable method payload plus optional generic hierarchy artifacts."""

    payload: object
    hierarchy_nodes: pl.DataFrame | None = None
    hierarchy_members: pl.DataFrame | None = None
    structural_summary: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidatePartition:
    """One flat partition returned across the generator seam.

    When supplied, ``hierarchy_node_ids`` has one entry for each distinct
    non-negative raw label in ``candidate_ids``, ordered by ascending raw label.
    """

    candidate_ids: np.ndarray
    membership_strengths: np.ndarray | None = None
    hierarchy_node_ids: tuple[str | None, ...] | None = None
    boundary_evidence: pl.DataFrame | None = None


@runtime_checkable
class CandidateGenerator(Protocol):
    """Backend contract; implementations may call an external process."""

    method: str

    def compute(
        self,
        coordinates: np.ndarray,
        cell_ids: np.ndarray,
        config: ClusteringConfig,
    ) -> GeneratorArtifacts: ...

    def cut(
        self, payload: object, config: CandidateCutConfig
    ) -> CandidatePartition: ...


def builtin_generator(method: str) -> CandidateGenerator:
    """Resolve a built-in generator by stable method name."""
    if method == "fauxnograph":
        from cellpax.generators.fauxnograph import FauxnographGenerator

        return FauxnographGenerator()
    raise ValueError(f"No built-in clustering generator for {method!r}")
