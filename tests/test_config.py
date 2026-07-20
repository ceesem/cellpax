from __future__ import annotations

import pytest

from cellpax import (
    SCHEMA_VERSION,
    CandidateCutConfig,
    ClusteringConfig,
    FeatureSpaceConfig,
    KeepConfig,
    RepresentationConfig,
)


def keep_mapping(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "config_schema_version": SCHEMA_VERSION,
        "operation": "keep_scope_revision",
        "parent_revision_id": None,
        "scope_id": "scope",
        "feature_selection_id": None,
        "feature_space_id": None,
        "clustering_representation_id": None,
        "visualization_representation_id": None,
        "candidate_set_id": None,
    }
    value.update(overrides)
    return value


def test_keep_config_resolves_and_round_trips_strictly() -> None:
    config = KeepConfig.resolve(
        scope_id="scope",
        parent_revision_id="parent",
        feature_selection_id="selection",
    )
    assert config.resolved() == {
        "config_schema_version": SCHEMA_VERSION,
        "operation": "keep_scope_revision",
        "parent_revision_id": "parent",
        "scope_id": "scope",
        "feature_selection_id": "selection",
        "feature_space_id": None,
        "clustering_representation_id": None,
        "visualization_representation_id": None,
        "candidate_set_id": None,
    }
    assert KeepConfig.from_mapping(config.resolved()) == config
    assert KeepConfig.from_mapping(
        dict(reversed(config.resolved().items()))
    ).config_hash == (config.config_hash)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], TypeError, "must be a mapping"),
        (
            {**keep_mapping(), "scpoe_id": "scope"},
            ValueError,
            "unknown=.*scpoe_id",
        ),
        (
            keep_mapping(config_schema_version="old"),
            ValueError,
            "Unsupported config schema",
        ),
        (
            keep_mapping(operation="other"),
            ValueError,
            "Unsupported keep operation",
        ),
        (
            keep_mapping(scope_id=""),
            TypeError,
            "scope_id",
        ),
        (
            keep_mapping(parent_revision_id=1),
            TypeError,
            "parent_revision_id",
        ),
    ],
)
def test_keep_config_rejects_invalid_input(
    value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        KeepConfig.from_mapping(value)


def test_slice_two_configs_resolve_defaults_strictly() -> None:
    robust = FeatureSpaceConfig.resolve(transform="robust_scaler")
    assert robust.params == {
        "quantile_range": [25.0, 75.0],
        "with_centering": True,
        "with_scaling": True,
    }
    assert robust.missing_policy == "error"
    pca = RepresentationConfig.resolve(method="pca", n_components=2, seed=7)
    assert pca.params == {"svd_solver": "auto", "whiten": False}
    assert pca.recompute_deterministic is True


def test_slice_two_configs_reject_unknown_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unknown robust_scaler"):
        FeatureSpaceConfig.resolve(
            transform="robust_scaler", params={"quantlie_range": [1, 99]}
        )
    with pytest.raises(ValueError, match="missing policy"):
        FeatureSpaceConfig.resolve(transform="raw_join", missing_policy="ignore")
    with pytest.raises(ValueError, match="percentiles"):
        FeatureSpaceConfig.resolve(
            transform="clipped_scaler",
            params={"lower_percentile": 99, "upper_percentile": 1},
        )
    with pytest.raises(ValueError, match="quantile_range"):
        FeatureSpaceConfig.resolve(
            transform="robust_scaler", params={"quantile_range": [90, 10]}
        )
    with pytest.raises(TypeError, match="seed"):
        FeatureSpaceConfig.resolve(transform="raw_join", seed=True)
    with pytest.raises(ValueError, match="positive integer"):
        RepresentationConfig.resolve(method="pca", n_components=0)


def test_slice_three_configs_are_strict_and_fully_resolved() -> None:
    clustering = ClusteringConfig.resolve(
        method="fauxnograph",
        compute_params={"n_neighbors": [2], "n_times": 3},
        seed=42,
    )
    assert clustering.compute_params["normalize"] is True
    assert clustering.compute_params["opportunity_normalize"] is False
    assert clustering.compute_params["neighbor_weighting"] == "unweighted"
    assert clustering.compute_params["build_hierarchy"] is True
    assert clustering.recompute_deterministic is False
    assert clustering.compute_params["resolution_parameter"] == [1.0]
    cut = CandidateCutConfig.resolve(
        cut_method="distance", cut_params={"distance_threshold": 0.25}
    )
    assert cut.cut_params == {"distance_threshold": 0.25, "min_cluster_size": 1}

    with pytest.raises(ValueError, match="Unknown fauxnograph"):
        ClusteringConfig.resolve(
            method="fauxnograph", compute_params={"neighborz": [2]}
        )
    with pytest.raises(ValueError, match="n_neighbors"):
        ClusteringConfig.resolve(
            method="fauxnograph", compute_params={"n_neighbors": []}
        )
    with pytest.raises(ValueError, match="non-negative"):
        CandidateCutConfig.resolve(
            cut_method="distance", cut_params={"distance_threshold": -1}
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"spatial_input": {"x": "soma_x"}}, ValueError, "Spatial"),
        ({"seed": True}, TypeError, "seed"),
        ({"recompute_deterministic": 1}, TypeError, "boolean"),
        ({"compute_params": {"n_times": 0}}, ValueError, "positive"),
        (
            {"compute_params": {"linkage_method": "ward"}},
            ValueError,
            "linkage_method",
        ),
        (
            {"compute_params": {"neighbor_weighting": "magic"}},
            ValueError,
            "neighbor_weighting",
        ),
        (
            {
                "compute_params": {
                    "normalize": False,
                    "opportunity_normalize": True,
                }
            },
            ValueError,
            "requires normalize",
        ),
        (
            {"compute_params": {"build_hierarchy": 1}},
            TypeError,
            "build_hierarchy",
        ),
    ],
)
def test_clustering_config_rejects_invalid_values(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        ClusteringConfig.resolve(method="fauxnograph", **kwargs)


def test_external_clustering_and_cut_config_error_paths() -> None:
    with pytest.raises(ValueError, match="method is required"):
        ClusteringConfig.external(method="", compute_params={})
    with pytest.raises(TypeError, match="seed"):
        ClusteringConfig.external(method="stub", compute_params={}, seed=True)
    with pytest.raises(ValueError, match="min_cluster_size"):
        CandidateCutConfig.resolve(
            cut_method="native", cut_params={"min_cluster_size": 0}
        )
    with pytest.raises(ValueError, match="run_index"):
        CandidateCutConfig.resolve(cut_method="native", cut_params={"run_index": -1})
    with pytest.raises(ValueError, match="Unsupported cut"):
        CandidateCutConfig.external(cut_method="unknown", cut_params={})
    with pytest.raises(ValueError, match="does not accept"):
        RepresentationConfig.resolve(method="scaled_passthrough", n_components=2)
    with pytest.raises(ValueError, match="Spatial representations"):
        RepresentationConfig.resolve(
            method="pca", n_components=2, spatial_input={"position_role": "soma"}
        )
    with pytest.raises(TypeError, match="seed"):
        RepresentationConfig.resolve(method="pca", n_components=2, seed="bad")
