"""Step 5: embeddings — PCA-native, UMAP optional, dataframe join."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable


def _table(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 4)), rng.normal(8, 0.3, (n // 2, 4))]
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
            "region": ["L"] * (n // 2) + ["R"] * (n - n // 2),
        }
    )
    return FeatureTable(df, features=[f"m{i}" for i in range(4)])


def test_pca_embedding_store_and_join() -> None:
    ft = _table(60)
    coords = ft.embed(method="pca", n_components=2, name="pca")
    assert coords.columns == ["cell_id", "pca0", "pca1"]
    assert coords.height == 60
    assert ft.embedding(name="pca").equals(coords)

    joined = ft.dataframe(embedding="pca")
    assert "pca0" in joined.columns and "region" in joined.columns
    assert joined.height == 60

    with pytest.raises(KeyError, match="No embedding"):
        ft.embedding(name="missing")
    with pytest.raises(ValueError, match="Unknown embedding method"):
        ft.embed(method="tsne")


def test_embedding_respects_mask() -> None:
    ft = _table(60)
    ft.add_mask("left", pl.col("region") == "L")
    coords = ft.embed("left", method="pca", n_components=2)
    assert coords.height == 30
    # joining that mask's embedding into the mask's dataframe lines up
    df = ft.dataframe("left", embedding="pca")
    assert df.height == 30 and df["pca0"].null_count() == 0


def test_features_pca_reduces_correlated_features() -> None:
    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(size=(n, 3))
    # 7 more columns as noisy linear combinations of the same 3 latent axes --
    # correlated, so PCA compresses them even after per-column standardizing.
    weights = rng.normal(size=(3, 7))
    derived = base @ weights + rng.normal(scale=0.05, size=(n, 7))
    coords = np.hstack([base, derived])
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(10)},
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(10)])

    reduced = ft.features_pca(explained_variance=0.95)
    assert reduced.shape[0] == n
    assert reduced.shape[1] < 10  # captured by far fewer than the 10 raw columns

    # a near-1.0 target keeps (almost) everything
    full = ft.features_pca(explained_variance=0.999999)
    assert full.shape[1] > reduced.shape[1]


def test_features_pca_respects_mask_and_columns() -> None:
    ft = _table(60)
    ft.add_mask("left", pl.col("region") == "L")
    reduced = ft.features_pca("left", columns=["m0", "m1"], explained_variance=0.95)
    assert reduced.shape[0] == 30
    assert reduced.shape[1] <= 2


def test_umap_is_optional() -> None:
    ft = _table(40)
    if importlib.util.find_spec("umap") is None:
        with pytest.raises(ImportError, match="umap-learn"):
            ft.embed(method="umap")
    else:  # pragma: no cover - depends on optional dep
        assert ft.embed(method="umap", n_components=2).height == 40
