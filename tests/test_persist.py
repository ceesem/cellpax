"""Step 6: first-class DataFolio save/load (many-per-folio, namespaced)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cellpax.consensus import make_clipped_scaler
from cellpax.featuretable import FeatureTable
from cellpax.persist import list_analyses


def _built_table(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 4)), rng.normal(8, 0.3, (n // 2, 4))]
    )
    meta = pl.DataFrame(
        {
            "feature_id": [f"m{i}" for i in range(4)],
            "family": ["axon", "axon", "dend", "dend"],
        }
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
            "region": ["L"] * (n // 2) + ["R"] * (n - n // 2),
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(4)], feature_metadata=meta)
    ft.add_mask("left", pl.col("region") == "L")
    ft.define_features("axon", family="axon")
    ft.preprocess()
    ft.embed(method="pca", n_components=2)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    ft.attach(labels)
    return ft


def _assert_round_trip(a: FeatureTable, b: FeatureTable) -> None:
    assert b.feature_columns == a.feature_columns
    assert b.id_column == a.id_column
    assert set(b.masks) == set(a.masks)
    assert b.transforms == a.transforms
    assert set(b.collections) == set(a.collections)
    assert b.collections["axon"].columns == a.collections["axon"].columns
    # full table (features, metadata, masks, attached labels) preserved
    assert b.dataframe().sort("cell_id").equals(a.dataframe().sort("cell_id"))
    # scaled features reproduce (transforms + scaler restored)
    assert np.allclose(b.features(scaled=True), a.features(scaled=True))
    # embedding + clustering restored
    assert b.embedding(name="pca").equals(a.embedding(name="pca"))
    la = a.label("run", distance_threshold=0.5, name="subclass")
    lb = b.label("run", distance_threshold=0.5, name="subclass")
    assert np.array_equal(
        np.sort(lb.to_frame()["subclass_id"].to_numpy()),
        np.sort(la.to_frame()["subclass_id"].to_numpy()),
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    ft = _built_table()
    folio = tmp_path / "study_folio"
    ft.save(folio, "l23it")
    reloaded = FeatureTable.load(folio, "l23it")
    _assert_round_trip(ft, reloaded)


def test_many_analyses_and_user_content_coexist(tmp_path: Path) -> None:
    from datafolio import DataFolio

    folio = DataFolio(tmp_path / "shared")
    # user-added content in its own namespace
    folio.add("notes/readme", {"author": "casey", "note": "L2/3 IT analysis"})

    _built_table().save(folio, "analysis_a")
    _built_table(40).save(folio, "analysis_b")

    assert list_analyses(folio) == ["analysis_a", "analysis_b"]
    # user content is untouched and not mistaken for an analysis
    assert folio.get("notes/readme")["author"] == "casey"

    a = FeatureTable.load(folio, "analysis_a")
    b = FeatureTable.load(folio, "analysis_b")
    assert a.n_cells == 60 and b.n_cells == 40


def test_clipped_scaler_tag_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, 31), dtype=pl.Int64),
            "x": rng.normal(size=30),
            "y": rng.normal(size=30),
        }
    )
    ft = FeatureTable(df, features=["x", "y"], scaler_factory=make_clipped_scaler)
    ft.save(tmp_path / "f", "clip")
    reloaded = FeatureTable.load(tmp_path / "f", "clip")
    assert np.allclose(reloaded.features(scaled=True), ft.features(scaled=True))
