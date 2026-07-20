"""Ergonomic helpers: config factories, builders, and the fluent RevisionBuilder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cellpax import (
    CandidateCutConfig,
    ClusteringConfig,
    FeatureDefinition,
    FeatureSpaceConfig,
    PropagationConfig,
    RepresentationConfig,
    Study,
    feature_catalog,
)


def _blobs(n: int = 40) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.4, size=(n // 2, 3))
    b = rng.normal(loc=[6.0, 6.0, 6.0], scale=0.4, size=(n // 2, 3))
    return np.vstack([a, b]), list(range(1, n + 1))


def test_config_factories_match_resolve() -> None:
    assert FeatureSpaceConfig.standard_scaler() == FeatureSpaceConfig.resolve(
        transform="standard_scaler"
    )
    assert FeatureSpaceConfig.raw_join() == FeatureSpaceConfig.resolve(
        transform="raw_join"
    )
    assert FeatureSpaceConfig.robust_scaler() == FeatureSpaceConfig.resolve(
        transform="robust_scaler"
    )
    assert FeatureSpaceConfig.clipped_scaler() == FeatureSpaceConfig.resolve(
        transform="clipped_scaler"
    )
    assert RepresentationConfig.pca(n_components=2) == RepresentationConfig.resolve(
        method="pca", n_components=2
    )
    assert RepresentationConfig.scaled_passthrough() == RepresentationConfig.resolve(
        method="scaled_passthrough"
    )
    assert ClusteringConfig.fauxnograph(n_neighbors=(10,)) == ClusteringConfig.resolve(
        method="fauxnograph", compute_params={"n_neighbors": [10]}
    )
    assert CandidateCutConfig.distance(threshold=0.4) == CandidateCutConfig.resolve(
        cut_method="distance", cut_params={"distance_threshold": 0.4}
    )
    assert CandidateCutConfig.native() == CandidateCutConfig.resolve(
        cut_method="native"
    )
    assert PropagationConfig.knn(n_neighbors=7) == PropagationConfig.resolve(
        params={"n_neighbors": 7}
    )


def test_feature_catalog_builder() -> None:
    built = feature_catalog(
        [
            FeatureDefinition("x", modality="synthetic", family="position", units="um"),
            FeatureDefinition("y", modality="synthetic", family="position"),
        ]
    )
    assert built["column_name"].to_list() == ["x", "y"]  # defaults to feature_id
    assert built["units"].to_list() == ["um", None]
    with pytest.raises(ValueError, match="at least one"):
        feature_catalog([])


def test_builder_and_ergonomic_inputs_run_end_to_end(tmp_path: Path) -> None:
    coords, cell_ids = _blobs()
    study = Study.create(tmp_path / "ergo", created_by="tester")

    # universe accepts a bare iterable; semantic_roles defaults to {}
    study.register_universe(cell_ids)

    values = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "morph_0": coords[:, 0],
            "morph_1": coords[:, 1],
            "morph_2": coords[:, 2],
        }
    )
    # register a block from FeatureDefinition objects (no hand-built catalog frame)
    block = study.register_feature_block(
        values,
        [
            FeatureDefinition(f"morph_{i}", modality="morphology", family="shape")
            for i in range(3)
        ],
    )

    build = study.build()
    build.scope(cell_ids).select(block)  # select-all, default derivation text
    inputs = build.keep("inputs")
    build.feature_space(FeatureSpaceConfig.standard_scaler())  # fit_scope defaults
    build.representation(RepresentationConfig.pca(n_components=2))
    build.clustering(ClusteringConfig.fauxnograph(n_neighbors=(10,), n_jobs=1))
    build.candidates(CandidateCutConfig.distance(threshold=0.5))
    revision = build.keep("candidates")

    assert inputs.parent_revision_id is None
    assert revision.parent_revision_id == inputs.revision_id  # builder chained parents
    definitions = study.candidate_definitions(revision.candidate_set_id)
    assert definitions.height == 2

    # view dispatcher matches the free function
    from cellpax.views import cells as cells_view

    assert study.view("cells", revision).equals(cells_view(study, revision))
    with pytest.raises(KeyError, match="Unknown view"):
        study.view("nope", revision)

    study.validate()


def test_select_all_and_explicit_ids_and_empty_derivation(tmp_path: Path) -> None:
    coords, cell_ids = _blobs(6)
    study = Study.create(tmp_path / "sel", created_by="tester")
    study.register_universe(cell_ids)
    values = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "a": coords[:, 0],
            "b": coords[:, 1],
        }
    )
    block = study.register_feature_block(
        values,
        [
            FeatureDefinition("a", modality="m", family="f"),
            FeatureDefinition("b", modality="m", family="f"),
        ],
    )
    all_features = study.select_all_features(block)
    assert all_features.n_features == 2
    one = study.preview_feature_selection(block, ["a"])
    assert one.n_features == 1
    # a block handle and its id select identically (same content id)
    by_id = study.preview_feature_selection(block.feature_block_id, ["a"])
    assert by_id.feature_selection_id == one.feature_selection_id
    with pytest.raises(ValueError, match="Unknown feature block"):
        study.preview_feature_selection("missing-block", ["a"])
    with pytest.raises(TypeError, match="feature_ids is only valid"):
        study.preview_feature_selection(values, ["a"])
    with pytest.raises(ValueError, match="non-empty string"):
        study.preview_feature_selection(block, derivation_text="")


def test_builder_adopts_existing_and_reports_state(tmp_path: Path) -> None:
    coords, cell_ids = _blobs(6)
    study = Study.create(tmp_path / "adopt", created_by="tester")
    study.register_universe(cell_ids)
    values = pl.DataFrame(
        {"cell_id": pl.Series(cell_ids, dtype=pl.Int64), "a": coords[:, 0]}
    )
    block = study.register_feature_block(
        values, [FeatureDefinition("a", modality="m", family="f")]
    )
    scope = study.preview_scope(cell_ids)
    selection = study.select_all_features(block)

    build = study.build()
    with pytest.raises(ValueError, match="requires a scope"):
        build.keep("too-early")
    with pytest.raises(ValueError, match="feature_space requires a scope"):
        build.feature_space(FeatureSpaceConfig.raw_join())

    build.use_scope(scope).use_selection(selection)
    assert build.current_scope == scope
    assert build.current_candidate_set is None
    revision = build.keep("adopted")
    assert revision.scope_id == scope.scope_id
    # accepts ids as well as records
    assert study.build().use_scope(scope.scope_id).current_scope == scope


def _clustered_study(path: Path):
    coords, cell_ids = _blobs()
    study = Study.create(path, created_by="tester")
    study.register_universe(
        pl.DataFrame(
            {
                "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
                "region": ["left"] * 20 + ["right"] * 20,
            }
        ),
        semantic_roles={"location": ["region"]},
    )
    values = pl.DataFrame(
        {
            "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
            "m0": coords[:, 0],
            "m1": coords[:, 1],
            "m2": coords[:, 2],
        }
    )
    block = study.register_feature_block(
        values,
        [
            FeatureDefinition(f"m{i}", modality="morphology", family="shape")
            for i in range(3)
        ],
    )
    b = study.build()
    b.scope(cell_ids).select(block)
    b.keep("inputs")
    b.feature_space(FeatureSpaceConfig.standard_scaler())
    b.representation(RepresentationConfig.pca(n_components=2))
    b.clustering(ClusteringConfig.fauxnograph(n_neighbors=(10,), n_jobs=1))
    b.candidates(CandidateCutConfig.distance(threshold=0.5))
    revision = b.keep("candidates")
    return study, revision


def test_feature_values_on_preview_and_kept(tmp_path: Path) -> None:
    study, revision = _clustered_study(tmp_path / "fv")
    space = study.get_feature_space(revision.feature_space_id)  # kept
    kept = study.feature_values(space.feature_space_id)
    assert kept.columns == ["cell_id", "m0", "m1", "m2"]  # readable names
    physical = study.feature_values(space, semantic_columns=False)
    assert physical.columns == [
        "cell_id",
        "feature_00000",
        "feature_00001",
        "feature_00002",
    ]
    # works on a bare preview record too
    raw_preview = study.preview_feature_space(
        scope=revision.scope_id,
        feature_selection=space.feature_selection_id,
        config=FeatureSpaceConfig.raw_join(),
    )
    assert study.feature_values(raw_preview).columns == ["cell_id", "m0", "m1", "m2"]


def test_feature_table_raw_and_normalized(tmp_path: Path) -> None:
    study, revision = _clustered_study(tmp_path / "ft")

    raw = study.feature_table(revision)  # normalized=False by default
    norm = study.feature_table(revision, normalized=True)

    # tidy frame: cell_id + tracking labels + universe metadata + feature columns
    for column in ("cell_id", "candidate_id", "taxon_id", "region", "m0", "m1", "m2"):
        assert column in raw.columns
    assert "revision_id" not in raw.columns
    assert raw.height == 40

    # labels/metadata identical; only the feature columns change with the flag
    tracking = ["cell_id", "candidate_id", "taxon_id", "region"]
    assert raw.select(tracking).equals(norm.select(tracking))
    assert not raw.select("m0", "m1", "m2").equals(norm.select("m0", "m1", "m2"))

    # facet/axis/color roles are all just columns
    assert set(raw["region"].to_list()) == {"left", "right"}
    assert raw.filter(pl.col("candidate_id") == 0).height > 0

    # opt out of universe metadata
    assert "region" not in study.feature_table(revision, include_metadata=False).columns

    # a revision without a feature space is rejected
    inputs = study.get_revision(
        study.registry("kept_revision").filter(pl.col("name") == "inputs")[
            "revision_id"
        ][0]
    )
    with pytest.raises(ValueError, match="requires a revision with a feature space"):
        study.feature_table(inputs)


def _skewed_block(path: Path):
    rng = np.random.default_rng(0)
    n = 300
    ids = list(range(1, n + 1))
    study = Study.create(path, created_by="tester")
    study.register_universe(ids)
    src = pl.DataFrame(
        {
            "cell_id": pl.Series(ids, dtype=pl.Int64),
            "path_length": rng.lognormal(3, 1.2, n),  # wide, positive -> log
            "branch_vol": rng.lognormal(1, 1.8, n),  # very wide -> log
            "radius": rng.normal(5, 0.6, n).clip(0.1),  # tight -> leave
            "signed_bias": rng.normal(0, 2, n),  # negatives -> skip
        }
    )
    block = study.register_feature_block(
        src,
        [
            FeatureDefinition("path_length", modality="morph", family="length"),
            FeatureDefinition("branch_vol", modality="morph", family="volume"),
            FeatureDefinition("radius", modality="morph", family="length"),
            FeatureDefinition("signed_bias", modality="morph", family="misc"),
        ],
    )
    return study, block, src


def test_derive_feature_block_explicit(tmp_path: Path) -> None:
    study, block, src = _skewed_block(tmp_path / "derive")
    derived = study.derive_feature_block(
        block,
        {"log_path": ("path_length", "log1p"), "sqrt_vol": ("branch_vol", "sqrt")},
    )
    cat = study.feature_catalog().filter(
        pl.col("feature_block_id") == derived.feature_block_id
    )
    assert set(cat["feature_id"].to_list()) == {"log_path", "sqrt_vol"}
    assert set(cat["raw_or_derived"].to_list()) == {"derived"}
    assert cat.filter(pl.col("feature_id") == "log_path")["description"][0] == (
        "log1p(path_length)"
    )
    values = study.folio.get(derived.values_ref, frame="polars")
    assert np.allclose(
        values["log_path"].to_numpy(), np.log1p(src["path_length"].to_numpy())
    )
    # provenance: derived block records the source values as lineage
    assert block.values_ref in study.folio.get_inputs(derived.values_ref)
    with pytest.raises(ValueError, match="Unknown source feature"):
        study.derive_feature_block(block, {"x": ("missing", "log1p")})
    with pytest.raises(ValueError, match="Unsupported derivation op"):
        study.derive_feature_block(block, {"x": ("radius", "cube")})


def test_auto_log_features_widths_and_reproducibility(tmp_path: Path) -> None:
    study, block, src = _skewed_block(tmp_path / "auto")
    auto = study.auto_log_features(block)
    values = study.folio.get(auto.values_ref, frame="polars")

    # wide, positive features are log-scaled
    assert np.allclose(
        values["path_length"].to_numpy(), np.log1p(src["path_length"].to_numpy())
    )
    assert np.allclose(
        values["branch_vol"].to_numpy(), np.log1p(src["branch_vol"].to_numpy())
    )
    # tight and negative features pass through unchanged
    assert np.allclose(values["radius"].to_numpy(), src["radius"].to_numpy())
    assert np.allclose(values["signed_bias"].to_numpy(), src["signed_bias"].to_numpy())

    cat = study.feature_catalog().filter(
        pl.col("feature_block_id") == auto.feature_block_id
    )
    desc = dict(zip(cat["feature_id"].to_list(), cat["description"].to_list()))
    assert desc["path_length"].startswith("log1p (auto:")
    assert desc["radius"].startswith("identity (auto:")
    assert "has negatives" in desc["signed_bias"]

    # deterministic: same source + thresholds -> same content-addressed block
    assert study.auto_log_features(block).feature_block_id == auto.feature_block_id
    # a single-metric run is accepted too
    assert study.auto_log_features(block, metric="skew").feature_block_id is not None

    with pytest.raises(ValueError, match="metric must be"):
        study.auto_log_features(block, metric="variance")
    with pytest.raises(ValueError, match="method must be"):
        study.auto_log_features(block, method="ln")
    with pytest.raises(KeyError, match="Unknown feature block"):
        study.get_feature_block("missing")
