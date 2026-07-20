from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.sparse import issparse

from cellpax import (
    CandidateCutConfig,
    CandidatePartition,
    ClusteringConfig,
    FeatureSpaceConfig,
    GeneratorArtifacts,
    HierarchyPreflightError,
    RepresentationConfig,
    Study,
)
from cellpax.generators.base import builtin_generator
from cellpax.generators.fauxnograph import (
    FauxnographGenerator,
    cluster_leiden,
    coclustering_matrix,
    estimate_hierarchy_memory,
    fauxnograph_clustering,
    fauxnograph_coclustering,
    kneighbor_graph,
)


def build_clustering_inputs(path: Path):
    study = Study.create(path, created_by="slice-three")
    cell_ids = list(range(1, 9))
    study.register_universe(
        pl.DataFrame({"cell_id": pl.Series(cell_ids, dtype=pl.Int64)}),
        semantic_roles={},
    )
    values = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "x": [-2.2, -2.0, -1.8, -2.1, 1.8, 2.0, 2.2, 2.1],
            "y": [-0.1, 0.1, 0.0, 0.2, 0.1, -0.1, 0.0, -0.2],
        }
    )
    catalog = pl.DataFrame(
        {
            "feature_id": ["x", "y"],
            "column_name": ["x", "y"],
            "modality": ["synthetic", "synthetic"],
            "family": ["position", "position"],
            "units": pl.Series([None, None], dtype=pl.String),
            "description": pl.Series([None, None], dtype=pl.String),
            "raw_or_derived": ["raw", "raw"],
        }
    )
    block = study.register_feature_block(values, catalog)
    selection = study.preview_feature_selection(
        pl.DataFrame(
            {
                "feature_block_id": [block.feature_block_id] * 2,
                "feature_id": ["x", "y"],
            }
        ),
        derivation_text="all",
    )
    scope = study.preview_scope(cell_ids, derivation_text="all cells")
    inputs = study.keep("inputs", scope=scope, feature_selection=selection)
    space = study.preview_feature_space(
        scope=scope,
        fit_scope=scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(transform="raw_join"),
    )
    representation = study.preview_representation(
        scope=scope,
        fit_scope=scope,
        feature_space=space,
        config=RepresentationConfig.resolve(method="scaled_passthrough"),
    )
    represented = study.keep(
        "represented",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        parent_revision=inputs,
    )
    return study, scope, space, representation, represented


def test_owned_fauxnograph_primitives() -> None:
    data = np.array([[-2.0, 0.0], [-1.9, 0.1], [1.9, 0.0], [2.0, 0.1]])
    graph = kneighbor_graph(data, n_neighbors=1, neighbor_weighting="jaccard")
    assert graph.vcount() == 4
    assert "weight" in graph.es.attributes()
    assert all(weight > 0 for weight in graph.es["weight"])
    matrix = fauxnograph_coclustering(
        data,
        n_neighbors=[1],
        resolution_parameter=[0.5, 1.0],
        n_times=2,
        normalize=True,
        opportunity_normalize=True,
        neighbor_weighting="jaccard",
        seed=7,
        n_jobs=1,
    )
    dense = matrix.toarray()
    assert issparse(matrix)
    assert dense.shape == (4, 4)
    assert np.allclose(dense, dense.T)
    assert np.allclose(np.diag(dense), 1.0)

    noise = coclustering_matrix(
        np.array([[-1, -1], [-1, -1]], dtype=np.int32), normalize=True
    )
    assert noise.nnz == 0
    partially_observed = np.array([[0, -1], [0, -1], [1, 1]], dtype=np.int32)
    total_normalized = coclustering_matrix(partially_observed, normalize=True)
    opportunity_normalized = coclustering_matrix(
        partially_observed, normalize=True, opportunity_normalize=True
    )
    assert total_normalized[0, 1] == 0.5
    assert opportunity_normalized[0, 1] == 1.0


def test_sparse_fauxnograph_path_can_skip_hierarchy_and_use_native_run() -> None:
    data = np.array([[-2.0, 0.0], [-1.9, 0.1], [1.9, 0.0], [2.0, 0.1]])
    generator = FauxnographGenerator()
    artifacts = generator.compute(
        data,
        np.array([1, 2, 3, 4]),
        ClusteringConfig.resolve(
            method="fauxnograph",
            compute_params={
                "n_neighbors": [1],
                "n_times": 2,
                "n_jobs": 1,
                "build_hierarchy": False,
            },
            seed=9,
        ),
    )
    assert artifacts.hierarchy_nodes is None
    assert artifacts.hierarchy_members is None
    assert artifacts.payload.linkage_matrix is None
    partition = generator.cut(
        artifacts.payload,
        CandidateCutConfig.resolve(cut_method="native", cut_params={"run_index": 1}),
    )
    assert partition.candidate_ids.shape == (4,)

    estimate = estimate_hierarchy_memory(10_000)
    assert estimate["estimated_peak"] > 1024**3
    with pytest.raises(HierarchyPreflightError, match="build_hierarchy=False"):
        FauxnographGenerator(hierarchy_memory_limit_bytes=1).compute(
            data,
            np.array([1, 2, 3, 4]),
            ClusteringConfig.resolve(
                method="fauxnograph",
                compute_params={"n_neighbors": [1], "n_jobs": 1},
            ),
        )


def test_fauxnograph_guard_and_alternate_paths() -> None:
    data = np.array([[-2.0, 0.0], [-1.9, 0.1], [1.9, 0.0], [2.0, 0.1]])
    with pytest.raises(ValueError, match="non-negative"):
        estimate_hierarchy_memory(-1)
    with pytest.raises(ValueError, match="two-dimensional"):
        kneighbor_graph(np.array([1.0, 2.0]), n_neighbors=1)
    with pytest.raises(ValueError, match="n_neighbors"):
        kneighbor_graph(data, n_neighbors=4)
    with pytest.raises(ValueError, match="neighbor_weighting"):
        kneighbor_graph(data, n_neighbors=1, neighbor_weighting="magic")
    with pytest.raises(ValueError, match="Either data or graph"):
        fauxnograph_clustering()

    graph = kneighbor_graph(data, n_neighbors=1, mutual_only=True)
    assert fauxnograph_clustering(graph=graph, seed=2).shape == (4,)
    assert fauxnograph_clustering(data, n_neighbors=1, seed=2).shape == (4,)
    assert np.all(cluster_leiden(graph, min_cluster_size=5, seed=2) == -1)
    with pytest.raises(ValueError, match="at least one"):
        coclustering_matrix(np.empty((4, 0), dtype=np.int32))
    raw = coclustering_matrix(np.array([[0], [0], [1], [1]], dtype=np.int32))
    assert raw[0, 1] == 1
    returned = fauxnograph_coclustering(
        data,
        n_neighbors=1,
        resolution_parameter=1.0,
        n_times=1,
        seed=2,
        n_jobs=1,
        return_runs=True,
    )
    assert len(returned) == 3

    with pytest.raises(ValueError, match="positive"):
        FauxnographGenerator(hierarchy_memory_limit_bytes=0)
    with pytest.raises(ValueError, match="No built-in"):
        builtin_generator("unknown")
    generator = FauxnographGenerator()
    with pytest.raises(ValueError, match="smaller than the scope"):
        generator.compute(
            data,
            np.arange(4),
            ClusteringConfig.resolve(
                method="fauxnograph",
                compute_params={"n_neighbors": [4], "n_jobs": 1},
            ),
        )
    config = ClusteringConfig.resolve(
        method="fauxnograph",
        compute_params={
            "n_neighbors": [1],
            "n_jobs": 1,
            "build_hierarchy": False,
        },
        seed=2,
    )
    artifacts = generator.compute(data, np.arange(4), config)
    with pytest.raises(TypeError, match="wrong type"):
        generator.cut({}, CandidateCutConfig.resolve(cut_method="native"))
    with pytest.raises(ValueError, match="require build_hierarchy"):
        generator.cut(
            artifacts.payload,
            CandidateCutConfig.resolve(cut_method="distance"),
        )
    with pytest.raises(ValueError, match="outside"):
        generator.cut(
            artifacts.payload,
            CandidateCutConfig.resolve(
                cut_method="native", cut_params={"run_index": 99}
            ),
        )
    with pytest.raises(ValueError, match="distance and native"):
        generator.cut(
            artifacts.payload,
            CandidateCutConfig.external(cut_method="resolution", cut_params={}),
        )
    small = generator.cut(
        artifacts.payload,
        CandidateCutConfig.resolve(
            cut_method="native", cut_params={"min_cluster_size": 5}
        ),
    )
    assert np.all(small.candidate_ids == -1)


def test_fauxnograph_run_multiple_cuts_keep_reopen_and_compare(tmp_path: Path) -> None:
    study, scope, space, representation, represented = build_clustering_inputs(
        tmp_path / "faux"
    )
    config = ClusteringConfig.resolve(
        method="fauxnograph",
        compute_params={
            "n_neighbors": [2],
            "n_times": 2,
            "resolution_parameter": [0.5, 1.0],
            "n_jobs": 1,
            "neighbor_weighting": "jaccard",
            "opportunity_normalize": True,
        },
        seed=11,
    )
    run = study.preview_clustering_run(
        scope=scope, representation=representation, config=config
    )
    strict = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.resolve(
            cut_method="distance", cut_params={"distance_threshold": 0.2}
        ),
    )
    broad = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.resolve(
            cut_method="distance", cut_params={"distance_threshold": 0.8}
        ),
    )
    assert strict.candidate_set_id != broad.candidate_set_id
    assert strict.clustering_run_id == broad.clustering_run_id == run.clustering_run_id
    assert study.candidate_membership(strict)["membership_strength"].null_count() == 8
    hierarchy = study.candidate_hierarchy(run)
    assert hierarchy is not None
    assert hierarchy[1].height == 8
    metadata = study.clustering_run_metadata(run)
    labels = study.clustering_run_labels(run)
    assert metadata.height == 4
    assert labels.shape == (8, 5)
    assert study.clustering_run_labels(run, long=True).height == 32
    state = study.folio.get_model(run.generator_ref, trusted=True)
    assert issparse(state.coclustering)
    comparison = study.compare_candidate_sets(strict, broad)
    assert comparison["n_cells"].sum() == 8

    revision = study.keep(
        "strict candidates",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        candidate_set=strict,
        parent_revision=represented,
    )
    assert revision.candidate_set_id == strict.candidate_set_id
    assert study.registry("clustering_run").height == 1
    assert study.registry("candidate_set").height == 1
    assert (
        study.registry("candidate_set")["candidate_set_id"][0] != broad.candidate_set_id
    )
    broad_revision = study.keep(
        "broad candidates",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        candidate_set=broad,
        parent_revision=revision,
    )
    assert study.compare_revisions(revision, broad_revision)["n_cells"].sum() == 8
    study.validate()

    contents = study.folio.list_contents()
    reused_run = study.preview_clustering_run(
        scope=scope, representation=representation, config=config
    )
    reused_set = study.preview_candidate_set(
        clustering_run=reused_run,
        config=CandidateCutConfig.resolve(
            cut_method="distance", cut_params={"distance_threshold": 0.2}
        ),
    )
    assert reused_run.clustering_run_id == run.clustering_run_id
    assert reused_set.candidate_set_id == strict.candidate_set_id
    assert study.folio.list_contents() == contents

    reopened = Study.open(tmp_path / "faux", read_only=True)
    reopened.validate()
    assert reopened.candidate_membership(strict.candidate_set_id).height == 8


class StubGenerator:
    method = "synthetic_stub"

    def compute(self, coordinates, cell_ids, config):
        return GeneratorArtifacts(payload={"n_cells": len(cell_ids)})

    def cut(self, payload, config):
        labels = np.arange(payload["n_cells"], dtype=np.int32) % 2
        labels[-1] = -1
        evidence = pl.DataFrame(
            {
                "candidate_id_a": pl.Series([0], dtype=pl.Int32),
                "candidate_id_b": pl.Series([1], dtype=pl.Int32),
                "metric": ["stub_pvalue"],
                "value": [0.04],
                "n_permutations": pl.Series([100], dtype=pl.Int32),
                "evidence_ref": pl.Series([None], dtype=pl.String),
            }
        )
        return CandidatePartition(candidate_ids=labels, boundary_evidence=evidence)


class HierarchyOrderStub(StubGenerator):
    method = "hierarchy_order_stub"

    def __init__(self) -> None:
        self.hierarchy_node_ids = ("node-ten", "node-twenty")

    def cut(self, payload, config):
        labels = np.array([20, 10, 20, 10, -1, 20, 10, -1], dtype=np.int32)
        return CandidatePartition(
            candidate_ids=labels,
            hierarchy_node_ids=self.hierarchy_node_ids,
        )


def test_external_capable_stub_uses_the_same_downstream_contract(
    tmp_path: Path,
) -> None:
    study, scope, space, representation, represented = build_clustering_inputs(
        tmp_path / "stub"
    )
    stub = StubGenerator()
    run = study.preview_clustering_run(
        scope=scope,
        representation=representation,
        config=ClusteringConfig.external(
            method=stub.method, compute_params={"command": "synthetic"}, seed=3
        ),
        generator=stub,
    )
    assert (
        set(json.loads(representation.software_versions))
        == set(json.loads(run.software_versions))
        == set(json.loads(represented.software_versions))
    )
    candidate_set = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.external(
            cut_method="native", cut_params={"source": "stub"}
        ),
        generator=stub,
    )
    membership = study.candidate_membership(candidate_set)
    assert membership["candidate_id"].null_count() == 1
    revision = study.keep(
        "stub candidates",
        scope=scope,
        feature_space=space,
        clustering_representation=representation,
        candidate_set=candidate_set,
        parent_revision=represented,
    )
    assert revision.candidate_set_id == candidate_set.candidate_set_id
    assert study.registry("candidate_boundary_evidence").height == 1
    assert study.candidate_boundary_evidence(candidate_set).height == 1
    assert study.candidate_hierarchy(run) is None
    study.validate()


def test_hierarchy_node_ids_align_to_sorted_raw_labels(tmp_path: Path) -> None:
    study, scope, _, representation, _ = build_clustering_inputs(
        tmp_path / "hierarchy-order"
    )
    stub = HierarchyOrderStub()
    run = study.preview_clustering_run(
        scope=scope,
        representation=representation,
        config=ClusteringConfig.external(method=stub.method, compute_params={}),
        generator=stub,
    )
    candidate_set = study.preview_candidate_set(
        clustering_run=run,
        config=CandidateCutConfig.external(cut_method="native", cut_params={}),
        generator=stub,
    )
    definitions = study.candidate_definitions(candidate_set)
    assert definitions["hierarchy_node_id"].to_list() == ["node-ten", "node-twenty"]

    stub.hierarchy_node_ids = ("too-short",)
    with pytest.raises(ValueError, match="sorted distinct non-negative"):
        study.preview_candidate_set(
            clustering_run=run,
            config=CandidateCutConfig.external(cut_method="native", cut_params={}),
            generator=stub,
        )
