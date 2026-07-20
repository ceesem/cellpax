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


def test_umap_is_optional() -> None:
    ft = _table(40)
    if importlib.util.find_spec("umap") is None:
        with pytest.raises(ImportError, match="umap-learn"):
            ft.embed(method="umap")
    else:  # pragma: no cover - depends on optional dep
        assert ft.embed(method="umap", n_components=2).height == 40
