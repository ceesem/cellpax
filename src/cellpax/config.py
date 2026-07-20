"""Strict, fully resolved configuration records for CellPax operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from cellpax.contracts import SCHEMA_VERSION
from cellpax.identity import canonical_hash, canonical_json_bytes

_TRANSFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "raw_join": {},
    "standard_scaler": {"with_mean": True, "with_std": True},
    "robust_scaler": {
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": [25.0, 75.0],
    },
    "clipped_scaler": {"lower_percentile": 0.5, "upper_percentile": 99.5},
}

_REPRESENTATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "pca": {"svd_solver": "auto", "whiten": False},
    "scaled_passthrough": {},
}

_CLUSTERING_DEFAULTS: dict[str, dict[str, Any]] = {
    "fauxnograph": {
        "n_neighbors": [30],
        "metric": "minkowski",
        "mutual_only": False,
        "neighbor_weighting": "unweighted",
        "n_times": 1,
        "resolution_parameter": [1.0],
        "min_cluster_size": 1,
        "normalize": True,
        "opportunity_normalize": False,
        "n_jobs": -1,
        "linkage_method": "average",
        "build_hierarchy": True,
    }
}

_CUT_DEFAULTS: dict[str, dict[str, Any]] = {
    "distance": {"distance_threshold": 0.5, "min_cluster_size": 1},
    "resolution": {"resolution": 1.0, "min_cluster_size": 1},
    "tree_prune": {"min_cluster_size": 1},
    "native": {"run_index": 0, "min_cluster_size": 1},
}

_PROPAGATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "knn": {
        "n_neighbors": 5,
        "weights": "distance",
        "metric": "minkowski",
    }
}


def _resolved_params(
    method: str,
    params: Mapping[str, Any] | None,
    defaults: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        resolved = dict(defaults[method])
    except KeyError as error:
        raise ValueError(f"Unsupported method {method!r}") from error
    supplied = dict(params or {})
    unknown = set(supplied) - set(resolved)
    if unknown:
        raise ValueError(f"Unknown {method} parameters: {sorted(unknown)}")
    resolved.update(supplied)
    return resolved


@dataclass(frozen=True, slots=True)
class KeepConfig:
    """Resolved configuration for a Slice 1 scope-only keep operation."""

    config_schema_version: str
    operation: str
    parent_revision_id: str | None
    scope_id: str
    feature_selection_id: str | None
    feature_space_id: str | None
    clustering_representation_id: str | None
    visualization_representation_id: str | None
    candidate_set_id: str | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "config_schema_version",
            "operation",
            "parent_revision_id",
            "scope_id",
            "feature_selection_id",
            "feature_space_id",
            "clustering_representation_id",
            "visualization_representation_id",
            "candidate_set_id",
        }
    )

    def __post_init__(self) -> None:
        if self.config_schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema version {self.config_schema_version!r}"
            )
        if self.operation != "keep_scope_revision":
            raise ValueError(f"Unsupported keep operation {self.operation!r}")
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise TypeError("scope_id must be a non-empty string")
        for name in (
            "parent_revision_id",
            "feature_selection_id",
            "feature_space_id",
            "clustering_representation_id",
            "visualization_representation_id",
            "candidate_set_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{name} must be null or a non-empty string")

    @classmethod
    def resolve(
        cls,
        *,
        scope_id: str,
        parent_revision_id: str | None = None,
        feature_selection_id: str | None = None,
        feature_space_id: str | None = None,
        clustering_representation_id: str | None = None,
        visualization_representation_id: str | None = None,
        candidate_set_id: str | None = None,
    ) -> "KeepConfig":
        """Resolve defaults into the one canonical current config shape."""
        return cls(
            config_schema_version=SCHEMA_VERSION,
            operation="keep_scope_revision",
            parent_revision_id=parent_revision_id,
            scope_id=scope_id,
            feature_selection_id=feature_selection_id,
            feature_space_id=feature_space_id,
            clustering_representation_id=clustering_representation_id,
            visualization_representation_id=visualization_representation_id,
            candidate_set_id=candidate_set_id,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "KeepConfig":
        """Parse a mapping strictly, rejecting missing and unknown keys."""
        if not isinstance(value, Mapping):
            raise TypeError("KeepConfig input must be a mapping")
        unknown = set(value) - cls._FIELDS
        missing = cls._FIELDS - set(value)
        if unknown or missing:
            raise ValueError(
                f"KeepConfig keys mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**dict(value))

    def resolved(self) -> dict[str, Any]:
        """Return the canonical JSON-ready mapping used for hashing."""
        return {
            "config_schema_version": self.config_schema_version,
            "operation": self.operation,
            "parent_revision_id": self.parent_revision_id,
            "scope_id": self.scope_id,
            "feature_selection_id": self.feature_selection_id,
            "feature_space_id": self.feature_space_id,
            "clustering_representation_id": self.clustering_representation_id,
            "visualization_representation_id": self.visualization_representation_id,
            "candidate_set_id": self.candidate_set_id,
        }

    @property
    def config_hash(self) -> str:
        """Hash the fully resolved, environment-independent configuration."""
        return canonical_hash(self.resolved())


@dataclass(frozen=True, slots=True)
class FeatureSpaceConfig:
    """Strict configuration for one fitted feature-space stage."""

    transform: str
    params_json: str
    missing_policy: str
    seed: int | None

    @classmethod
    def resolve(
        cls,
        *,
        transform: str,
        params: Mapping[str, Any] | None = None,
        missing_policy: str = "error",
        seed: int | None = None,
    ) -> "FeatureSpaceConfig":
        resolved = _resolved_params(transform, params, _TRANSFORM_DEFAULTS)
        if missing_policy not in {"error", "median", "drop"}:
            raise ValueError(f"Unsupported missing policy {missing_policy!r}")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or null")
        if transform == "clipped_scaler":
            lower = resolved["lower_percentile"]
            upper = resolved["upper_percentile"]
            if not (0 <= lower < upper <= 100):
                raise ValueError(
                    "Clipped-scaler percentiles must satisfy 0 <= lower < upper <= 100"
                )
        if transform == "robust_scaler":
            quantiles = resolved["quantile_range"]
            if (
                not isinstance(quantiles, (list, tuple))
                or len(quantiles) != 2
                or not (0 <= quantiles[0] < quantiles[1] <= 100)
            ):
                raise ValueError(
                    "quantile_range must contain two increasing percentiles"
                )
            resolved["quantile_range"] = list(quantiles)
        return cls(
            transform=transform,
            params_json=canonical_json_text(resolved),
            missing_policy=missing_policy,
            seed=seed,
        )

    @classmethod
    def raw_join(
        cls, *, missing_policy: str = "error", seed: int | None = None
    ) -> "FeatureSpaceConfig":
        """Pass selected values through unchanged."""
        return cls.resolve(
            transform="raw_join", missing_policy=missing_policy, seed=seed
        )

    @classmethod
    def standard_scaler(
        cls,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        missing_policy: str = "error",
        seed: int | None = None,
    ) -> "FeatureSpaceConfig":
        """Zero-mean/unit-variance scaling."""
        return cls.resolve(
            transform="standard_scaler",
            params={"with_mean": with_mean, "with_std": with_std},
            missing_policy=missing_policy,
            seed=seed,
        )

    @classmethod
    def robust_scaler(
        cls,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Sequence[float] = (25.0, 75.0),
        missing_policy: str = "error",
        seed: int | None = None,
    ) -> "FeatureSpaceConfig":
        """Median/IQR scaling that tolerates outliers."""
        return cls.resolve(
            transform="robust_scaler",
            params={
                "with_centering": with_centering,
                "with_scaling": with_scaling,
                "quantile_range": list(quantile_range),
            },
            missing_policy=missing_policy,
            seed=seed,
        )

    @classmethod
    def clipped_scaler(
        cls,
        *,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        missing_policy: str = "error",
        seed: int | None = None,
    ) -> "FeatureSpaceConfig":
        """Percentile-clipped min-max scaling."""
        return cls.resolve(
            transform="clipped_scaler",
            params={
                "lower_percentile": lower_percentile,
                "upper_percentile": upper_percentile,
            },
            missing_policy=missing_policy,
            seed=seed,
        )

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Strict configuration for one versioned coordinate representation."""

    method: str
    n_components: int | None
    params_json: str
    seed: int | None
    spatial_input_json: str | None
    recompute_deterministic: bool

    @classmethod
    def resolve(
        cls,
        *,
        method: str,
        n_components: int | None = None,
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
        spatial_input: Mapping[str, Any] | None = None,
    ) -> "RepresentationConfig":
        resolved = _resolved_params(method, params, _REPRESENTATION_DEFAULTS)
        if spatial_input is not None:
            raise ValueError("Spatial representations are not implemented in Slice 2")
        if method == "pca":
            if (
                not isinstance(n_components, int)
                or isinstance(n_components, bool)
                or n_components < 1
            ):
                raise ValueError("PCA n_components must be a positive integer")
        elif n_components is not None:
            raise ValueError("scaled_passthrough does not accept n_components")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or null")
        return cls(
            method=method,
            n_components=n_components,
            params_json=canonical_json_text(resolved),
            seed=seed,
            spatial_input_json=None,
            recompute_deterministic=True,
        )

    @classmethod
    def pca(
        cls,
        *,
        n_components: int,
        svd_solver: str = "auto",
        whiten: bool = False,
        seed: int | None = None,
    ) -> "RepresentationConfig":
        """Principal-component coordinates."""
        return cls.resolve(
            method="pca",
            n_components=n_components,
            params={"svd_solver": svd_solver, "whiten": whiten},
            seed=seed,
        )

    @classmethod
    def scaled_passthrough(cls, *, seed: int | None = None) -> "RepresentationConfig":
        """Use the feature space's own columns as coordinates."""
        return cls.resolve(method="scaled_passthrough", seed=seed)

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    """Strict, resolved configuration for one expensive clustering run."""

    method: str
    compute_params_json: str
    seed: int | None
    spatial_input_json: str | None
    recompute_deterministic: bool

    @classmethod
    def resolve(
        cls,
        *,
        method: str,
        compute_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
        spatial_input: Mapping[str, Any] | None = None,
        recompute_deterministic: bool = False,
    ) -> "ClusteringConfig":
        resolved = _resolved_params(method, compute_params, _CLUSTERING_DEFAULTS)
        if spatial_input is not None:
            raise ValueError("Spatial clustering inputs are not implemented in Slice 3")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or null")
        if not isinstance(recompute_deterministic, bool):
            raise TypeError("recompute_deterministic must be boolean")
        neighbors = resolved.get("n_neighbors")
        resolutions = resolved.get("resolution_parameter")
        for name, values, predicate in (
            (
                "n_neighbors",
                neighbors,
                lambda value: isinstance(value, int) and value > 0,
            ),
            (
                "resolution_parameter",
                resolutions,
                lambda value: isinstance(value, (int, float)) and value > 0,
            ),
        ):
            if (
                not isinstance(values, (list, tuple))
                or not values
                or not all(predicate(value) for value in values)
            ):
                raise ValueError(f"{name} must be a non-empty list of positive values")
            resolved[name] = list(values)
        if resolved["n_times"] < 1 or resolved["min_cluster_size"] < 1:
            raise ValueError("n_times and min_cluster_size must be positive")
        if resolved["linkage_method"] not in {"average", "single", "complete"}:
            raise ValueError("Unsupported linkage_method")
        if resolved["neighbor_weighting"] not in {"unweighted", "jaccard"}:
            raise ValueError("Unsupported neighbor_weighting")
        if resolved["opportunity_normalize"] and not resolved["normalize"]:
            raise ValueError("opportunity_normalize requires normalize=True")
        if not isinstance(resolved["build_hierarchy"], bool):
            raise TypeError("build_hierarchy must be boolean")
        return cls(
            method=method,
            compute_params_json=canonical_json_text(resolved),
            seed=seed,
            spatial_input_json=None,
            recompute_deterministic=recompute_deterministic,
        )

    @classmethod
    def external(
        cls,
        *,
        method: str,
        compute_params: Mapping[str, Any],
        seed: int | None = None,
        recompute_deterministic: bool = False,
    ) -> "ClusteringConfig":
        """Resolve an adapter-owned config without assuming an in-process backend."""
        if not method:
            raise ValueError("method is required")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or null")
        return cls(
            method=method,
            compute_params_json=canonical_json_text(dict(compute_params)),
            seed=seed,
            spatial_input_json=None,
            recompute_deterministic=recompute_deterministic,
        )

    @classmethod
    def fauxnograph(
        cls,
        *,
        n_neighbors: Sequence[int] = (30,),
        metric: str = "minkowski",
        mutual_only: bool = False,
        neighbor_weighting: str = "unweighted",
        n_times: int = 1,
        resolution_parameter: Sequence[float] = (1.0,),
        min_cluster_size: int = 1,
        normalize: bool = True,
        opportunity_normalize: bool = False,
        n_jobs: int = -1,
        linkage_method: str = "average",
        build_hierarchy: bool = True,
        seed: int | None = None,
        recompute_deterministic: bool = False,
    ) -> "ClusteringConfig":
        """The built-in kNN/Leiden consensus backend.

        ``n_neighbors`` and ``resolution_parameter`` are swept lists: pass one
        value each for a single run, or several to combine multiple settings.
        """
        return cls.resolve(
            method="fauxnograph",
            compute_params={
                "n_neighbors": list(n_neighbors),
                "metric": metric,
                "mutual_only": mutual_only,
                "neighbor_weighting": neighbor_weighting,
                "n_times": n_times,
                "resolution_parameter": list(resolution_parameter),
                "min_cluster_size": min_cluster_size,
                "normalize": normalize,
                "opportunity_normalize": opportunity_normalize,
                "n_jobs": n_jobs,
                "linkage_method": linkage_method,
                "build_hierarchy": build_hierarchy,
            },
            seed=seed,
            recompute_deterministic=recompute_deterministic,
        )

    @property
    def compute_params(self) -> dict[str, Any]:
        return json.loads(self.compute_params_json)


@dataclass(frozen=True, slots=True)
class CandidateCutConfig:
    """Strict, resolved configuration for a cheap candidate-set derivation."""

    cut_method: str
    cut_params_json: str

    @classmethod
    def resolve(
        cls,
        *,
        cut_method: str,
        cut_params: Mapping[str, Any] | None = None,
    ) -> "CandidateCutConfig":
        resolved = _resolved_params(cut_method, cut_params, _CUT_DEFAULTS)
        if resolved["min_cluster_size"] < 1:
            raise ValueError("min_cluster_size must be positive")
        if cut_method == "distance" and resolved["distance_threshold"] < 0:
            raise ValueError("distance_threshold must be non-negative")
        if cut_method == "native" and resolved["run_index"] < 0:
            raise ValueError("run_index must be non-negative")
        return cls(cut_method=cut_method, cut_params_json=canonical_json_text(resolved))

    @classmethod
    def external(
        cls, *, cut_method: str, cut_params: Mapping[str, Any]
    ) -> "CandidateCutConfig":
        if cut_method not in _CUT_DEFAULTS:
            raise ValueError(f"Unsupported cut method {cut_method!r}")
        return cls(
            cut_method=cut_method,
            cut_params_json=canonical_json_text(dict(cut_params)),
        )

    @classmethod
    def distance(
        cls, *, threshold: float = 0.5, min_cluster_size: int = 1
    ) -> "CandidateCutConfig":
        """Cut the consensus hierarchy at a distance threshold."""
        return cls.resolve(
            cut_method="distance",
            cut_params={
                "distance_threshold": threshold,
                "min_cluster_size": min_cluster_size,
            },
        )

    @classmethod
    def resolution(
        cls, *, resolution: float = 1.0, min_cluster_size: int = 1
    ) -> "CandidateCutConfig":
        """Cut by a resolution parameter."""
        return cls.resolve(
            cut_method="resolution",
            cut_params={
                "resolution": resolution,
                "min_cluster_size": min_cluster_size,
            },
        )

    @classmethod
    def tree_prune(cls, *, min_cluster_size: int = 1) -> "CandidateCutConfig":
        """Prune the hierarchy by minimum cluster size."""
        return cls.resolve(
            cut_method="tree_prune", cut_params={"min_cluster_size": min_cluster_size}
        )

    @classmethod
    def native(
        cls, *, run_index: int = 0, min_cluster_size: int = 1
    ) -> "CandidateCutConfig":
        """Use a backend-native partition by run index."""
        return cls.resolve(
            cut_method="native",
            cut_params={"run_index": run_index, "min_cluster_size": min_cluster_size},
        )

    @property
    def cut_params(self) -> dict[str, Any]:
        return json.loads(self.cut_params_json)


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    """Strict configuration for a fitted label-propagation computation."""

    method: str
    params_json: str
    seed: int | None
    recompute_deterministic: bool

    @classmethod
    def resolve(
        cls,
        *,
        method: str = "knn",
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> "PropagationConfig":
        resolved = _resolved_params(method, params, _PROPAGATION_DEFAULTS)
        if resolved["n_neighbors"] < 1:
            raise ValueError("n_neighbors must be positive")
        if resolved["weights"] not in {"uniform", "distance"}:
            raise ValueError("KNN weights must be 'uniform' or 'distance'")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or null")
        return cls(
            method=method,
            params_json=canonical_json_text(resolved),
            seed=seed,
            recompute_deterministic=True,
        )

    @classmethod
    def knn(
        cls,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "minkowski",
        seed: int | None = None,
    ) -> "PropagationConfig":
        """k-nearest-neighbor label propagation."""
        return cls.resolve(
            method="knn",
            params={
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
            },
            seed=seed,
        )

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)


def canonical_json_text(value: object) -> str:
    """Encode config JSON with CellPax's canonical rules."""
    return canonical_json_bytes(value).decode("utf-8")
