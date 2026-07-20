from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from datafolio import DataFolio

from cellpax import (
    FeatureSpaceConfig,
    MissingValuesError,
    RepresentationConfig,
    ScopeReductionRequiredError,
    Study,
)
from cellpax.spaces import (
    assemble_selected_values,
    fit_feature_space,
    fit_representation,
)


def build_inputs(path: Path):
    study = Study.create(path, created_by="slice-two")
    cells = pl.DataFrame(
        {
            "cell_id": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
            "root_id": pl.Series([11, 12, 13, 14, 15, 16], dtype=pl.Int64),
        }
    )
    study.register_universe(cells, semantic_roles={"alias": ["root_id"]})
    values = pl.DataFrame(
        {
            "cell_id": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
            "size": pl.Series([1.0, 2.0, None, 4.0, 5.0, 6.0], dtype=pl.Float64),
            "branches": pl.Series([2, 3, 5, 7, 11, 13], dtype=pl.Int32),
        }
    )
    catalog = pl.DataFrame(
        {
            "feature_id": ["size", "branches"],
            "column_name": ["size", "branches"],
            "modality": ["morphology", "morphology"],
            "family": ["shape", "shape"],
            "units": pl.Series(["um", None], dtype=pl.String),
            "description": pl.Series([None, None], dtype=pl.String),
            "raw_or_derived": ["raw", "raw"],
        }
    )
    block = study.register_feature_block(values, catalog, nullable_columns=["size"])
    selected = pl.DataFrame(
        {
            "feature_block_id": pl.Series(
                [block.feature_block_id, block.feature_block_id], dtype=pl.String
            ),
            "feature_id": pl.Series(["size", "branches"], dtype=pl.String),
        }
    )
    selection = study.preview_feature_selection(
        selected, derivation_text="all features"
    )
    global_scope = study.preview_scope(range(1, 7), derivation_text="global")
    input_revision = study.keep(
        "inputs", scope=global_scope, feature_selection=selection
    )
    return study, selection, global_scope, input_revision


def test_global_then_branch_local_feature_spaces_and_pca_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "study"
    study, selection, global_scope, input_revision = build_inputs(path)

    global_space = study.preview_feature_space(
        scope=global_scope,
        fit_scope=global_scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(
            transform="robust_scaler", missing_policy="median"
        ),
    )
    global_pca = study.preview_representation(
        scope=global_scope,
        fit_scope=global_scope,
        feature_space=global_space,
        config=RepresentationConfig.resolve(method="pca", n_components=2, seed=17),
    )
    global_revision = study.keep(
        "global-pca",
        scope=global_scope,
        feature_space=global_space,
        clustering_representation=global_pca,
        visualization_representation=global_pca,
        parent_revision=input_revision,
    )
    contents = study.folio.list_contents()
    reused_space = study.preview_feature_space(
        scope=global_scope,
        fit_scope=global_scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(
            transform="robust_scaler", missing_policy="median"
        ),
    )
    reused_pca = study.preview_representation(
        scope=global_scope,
        fit_scope=global_scope,
        feature_space=global_space,
        config=RepresentationConfig.resolve(method="pca", n_components=2, seed=17),
    )
    assert reused_space == global_space
    assert reused_pca == global_pca
    assert study.folio.list_contents() == contents

    branch_scope = study.preview_scope(
        [1, 2, 3], derivation_text="CGE branch", parent=global_scope
    )
    branch_space = study.preview_feature_space(
        scope=branch_scope,
        fit_scope=branch_scope,
        feature_selection=selection,
        parent=global_space,
        config=FeatureSpaceConfig.resolve(
            transform="clipped_scaler",
            params={"lower_percentile": 0.5, "upper_percentile": 99.5},
        ),
    )
    branch_pca = study.preview_representation(
        scope=branch_scope,
        fit_scope=branch_scope,
        feature_space=branch_space,
        config=RepresentationConfig.resolve(method="pca", n_components=2, seed=23),
    )
    branch_revision = study.keep(
        "branch-pca",
        scope=branch_scope,
        feature_space=branch_space,
        clustering_representation=branch_pca,
        visualization_representation=branch_pca,
        parent_revision=global_revision,
    )

    assert branch_space.parent_feature_space_id == global_space.feature_space_id
    assert global_space.fit_scope_id == global_scope.scope_id
    assert branch_space.fit_scope_id == branch_scope.scope_id
    assert branch_revision.feature_space_id == branch_space.feature_space_id
    assert branch_revision.clustering_representation_id == branch_pca.representation_id
    assert (
        branch_revision.visualization_representation_id == branch_pca.representation_id
    )
    assert study.registry("feature_space").height == 2
    assert study.registry("representation").height == 2

    report = study.feature_space_missingness(global_space)
    assert report.filter(pl.col("feature_id") == "size")["missing_scope"][0] == 1
    global_values = study.folio.get(global_space.values_ref, frame="polars")
    branch_values = study.folio.get(branch_space.values_ref, frame="polars")
    coords = study.folio.get(branch_pca.coords_ref, frame="polars")
    assert global_values.null_count().sum_horizontal()[0] == 0
    assert branch_values["cell_id"].to_list() == [1, 2, 3]
    assert coords.schema == pl.Schema(
        {"cell_id": pl.Int64, "dim_0": pl.Float64, "dim_1": pl.Float64}
    )
    assert study.folio.get_model(global_space.fitted_state_ref, trusted=True)
    assert study.folio.get_model(branch_pca.fitted_state_ref, trusted=True)
    original_get_model = DataFolio.get_model

    def reject_pickle_load(*args, **kwargs):
        raise RuntimeError("pickle loading disabled")

    monkeypatch.setattr(DataFolio, "get_model", reject_pickle_load)
    study.validate()
    with pytest.raises(RuntimeError, match="pickle loading disabled"):
        study.validate(trusted_models=True)
    monkeypatch.setattr(DataFolio, "get_model", original_get_model)
    study.validate(trusted_models=True)

    reopened = Study.open(path, read_only=True)
    reopened.validate()
    assert reopened.get_feature_space(branch_space.feature_space_id).transform == (
        "clipped_scaler"
    )
    assert reopened.get_representation(branch_pca.representation_id).method == "pca"

    code = """
import json, sys
from cellpax import Study
study = Study.open(sys.argv[1], read_only=True)
study.validate()
revision = study.registry("kept_revision").filter(
    __import__('polars').col('name') == 'branch-pca'
).row(0, named=True)
print(json.dumps({
    'space': revision['feature_space_id'],
    'representation': revision['clustering_representation_id'],
}))
"""
    output = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.stdout) == {
        "space": branch_space.feature_space_id,
        "representation": branch_pca.representation_id,
    }


def test_missingness_policies_and_scaled_passthrough(tmp_path: Path) -> None:
    study, selection, scope, _ = build_inputs(tmp_path / "study")
    with pytest.raises(MissingValuesError) as caught:
        study.preview_feature_space(
            scope=scope,
            fit_scope=scope,
            feature_selection=selection,
            config=FeatureSpaceConfig.resolve(transform="raw_join"),
        )
    assert (
        caught.value.report.filter(pl.col("feature_id") == "size")["missing_scope"][0]
        == 1
    )

    with pytest.raises(ScopeReductionRequiredError) as reduction:
        study.preview_feature_space(
            scope=scope,
            fit_scope=scope,
            feature_selection=selection,
            config=FeatureSpaceConfig.resolve(
                transform="raw_join", missing_policy="drop"
            ),
        )
    assert reduction.value.retained_members["cell_id"].to_list() == [1, 2, 4, 5, 6]
    reduced_scope = study.preview_scope(
        reduction.value.retained_members,
        derivation_text="complete-case cells",
        parent=scope,
    )
    dropped = study.preview_feature_space(
        scope=reduced_scope,
        fit_scope=reduced_scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(transform="raw_join", missing_policy="drop"),
    )
    dropped_values = study.folio.get(dropped.values_ref, frame="polars")
    assert dropped.fitted_state_ref is None
    assert dropped_values["cell_id"].to_list() == [1, 2, 4, 5, 6]
    assert dropped.scope_id == reduced_scope.scope_id

    imputed = study.preview_feature_space(
        scope=scope,
        fit_scope=scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(
            transform="standard_scaler", missing_policy="median"
        ),
    )
    passthrough = study.preview_representation(
        scope=scope,
        fit_scope=scope,
        feature_space=imputed,
        config=RepresentationConfig.resolve(method="scaled_passthrough"),
    )
    coords = study.folio.get(passthrough.coords_ref, frame="polars")
    values = study.folio.get(imputed.values_ref, frame="polars")
    assert passthrough.fitted_state_ref is None
    assert (
        coords.drop("cell_id").to_numpy().tolist()
        == values.drop("cell_id").to_numpy().tolist()
    )
    revision = study.keep(
        "passthrough",
        scope=scope,
        feature_space=imputed,
        clustering_representation=passthrough,
    )
    assert revision.feature_space_id == imputed.feature_space_id
    study.validate()


def test_slice_two_preflight_rejects_invalid_dimensions_and_lineage(
    tmp_path: Path,
) -> None:
    study, selection, scope, _ = build_inputs(tmp_path / "study")
    space = study.preview_feature_space(
        scope=scope,
        fit_scope=scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(
            transform="raw_join", missing_policy="median"
        ),
    )
    with pytest.raises(ValueError, match="exceeds fit data"):
        study.preview_representation(
            scope=scope,
            fit_scope=scope,
            feature_space=space,
            config=RepresentationConfig.resolve(method="pca", n_components=3),
        )
    representation = study.preview_representation(
        scope=scope,
        fit_scope=scope,
        feature_space=space,
        config=RepresentationConfig.resolve(method="scaled_passthrough"),
    )
    with pytest.raises(ValueError, match="requires a feature space"):
        study.keep(
            "invalid representation",
            scope=scope,
            clustering_representation=representation,
        )


def test_slice_two_manifest_corruption_is_detected(tmp_path: Path) -> None:
    study, selection, scope, _ = build_inputs(tmp_path / "study")
    space = study.preview_feature_space(
        scope=scope,
        fit_scope=scope,
        feature_selection=selection,
        config=FeatureSpaceConfig.resolve(
            transform="raw_join", missing_policy="median"
        ),
    )
    study.keep("space", scope=scope, feature_space=space)
    ref = f"cellpax/manifests/feature-space/{space.feature_space_id}"
    manifest = study.folio.get(ref)
    manifest["missing_policy"] = "drop"
    study.folio.add(ref, manifest, overwrite=True)
    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        study.validate()


def test_space_execution_guards_are_enforced() -> None:
    members = pl.DataFrame({"cell_id": pl.Series([1], dtype=pl.Int64)})
    selection = pl.DataFrame(
        {
            "position": pl.Series([0], dtype=pl.Int32),
            "feature_block_id": ["block"],
            "feature_id": ["text"],
        }
    )
    catalog = pl.DataFrame(
        {
            "feature_block_id": ["block"],
            "feature_id": ["text"],
            "column_name": ["text"],
        }
    )
    blocks = {
        "block": pl.DataFrame(
            {
                "cell_id": pl.Series([1], dtype=pl.Int64),
                "text": pl.Series(["value"], dtype=pl.String),
            }
        )
    }
    with pytest.raises(TypeError, match="must be numeric"):
        assemble_selected_values(members, selection, catalog, blocks)

    passthrough = RepresentationConfig.resolve(method="scaled_passthrough")
    scope_values = pl.DataFrame(
        {
            "cell_id": pl.Series([1], dtype=pl.Int64),
            "feature_00000": pl.Series([1.0], dtype=pl.Float64),
        }
    )
    wrong_schema = scope_values.rename({"feature_00000": "other"})
    with pytest.raises(ValueError, match="schemas differ"):
        fit_representation(scope_values, wrong_schema, passthrough)
    null_values = scope_values.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("feature_00000")
    )
    with pytest.raises(ValueError, match="input contains null"):
        fit_representation(null_values, scope_values, passthrough)
    with pytest.raises(ValueError, match="fit input contains null"):
        fit_representation(scope_values, null_values, passthrough)

    feature_config = FeatureSpaceConfig.resolve(
        transform="raw_join", missing_policy="drop"
    )
    with pytest.raises(ValueError, match="left no cells"):
        fit_feature_space(scope_values, null_values, selection, feature_config)
