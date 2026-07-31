"""First-class DataFolio save/load for FeatureTable (step 6).

Persists a whole analysis under a ``<name>/…`` namespace in a DataFolio, so many
analyses live in one folio alongside arbitrary user content. Storage is
*structured*, never flattened: the cell table and each embedding are polars
items, each consensus matrix is a sparse-triplet item, and all structure (id
column, features, collections, preprocess transforms, scaler, var metadata, label
identities) lives in a JSON manifest — so masks, collections, transforms,
embeddings, cluster colors, and expensive clustering results all survive a
reload. Every item gets a content-derived
description (cell/feature counts, mask, linkage method, …) so a folio browsed via
``folio.describe()`` is self-explanatory without reloading the analysis.

    <name>/manifest              JSON: kind marker + structure
    <name>/table                 the cell table (features, metadata, masks, labels)
    <name>/embedding/<mask>__<e> embedding coordinates
    <name>/clustering/<c>        consensus matrix as (row, col, value) triplets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from datafolio import DataFolio
from scipy.sparse import coo_matrix, csr_matrix

_KIND = "cellpax_feature_table"
_VERSION = "1"


def _folio(folio: DataFolio | str | Path) -> DataFolio:
    if isinstance(folio, DataFolio):
        return folio
    return DataFolio(folio, allow_existing=True)


def _scaler_tag(factory: Any) -> str:
    from cellpax.clustering import make_clipped_scaler
    from cellpax.featuretable import _default_scaler_factory

    if factory is _default_scaler_factory:
        return "standard"
    if factory is make_clipped_scaler:
        return "clipped"
    return "custom"


def _factory_from_tag(tag: str) -> Any:
    from cellpax.clustering import make_clipped_scaler
    from cellpax.featuretable import _default_scaler_factory

    return {"standard": _default_scaler_factory, "clipped": make_clipped_scaler}.get(
        tag, _default_scaler_factory
    )


def _labels_manifest(table: Any) -> dict[str, Any]:
    """Attached label identities (color/description/mask) as JSON-safe records."""
    return {
        column: {
            "mask": record["mask"],
            "labels": [
                {
                    "id": label.id,
                    "name": label.name,
                    "color": label.color,
                    "description": label.description,
                }
                for label in record["labels"].values()
            ],
        }
        for column, record in table._label_meta.items()
    }


def _labels_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Rebuild ``FeatureTable._label_meta`` from a manifest's ``labels`` section."""
    from cellpax.labels import Label

    return {
        column: {
            "mask": record["mask"],
            "labels": {
                int(entry["id"]): Label(
                    id=int(entry["id"]),
                    name=entry["name"],
                    color=entry["color"],
                    description=entry["description"],
                )
                for entry in record["labels"]
            },
        }
        for column, record in manifest.get("labels", {}).items()
    }


def save_feature_table(
    table: Any, folio: DataFolio | str | Path, name: str, *, overwrite: bool = True
) -> None:
    """Persist a FeatureTable under the ``name`` namespace of a folio."""
    folio = _folio(folio)
    if "/" in name:
        raise ValueError("analysis name cannot contain '/'")

    n_masks = sum(1 for c in table._df.columns if c.startswith("_mask_"))
    folio.add(
        f"{name}/table",
        table._df,
        description=(
            f"{name!r} CellPax cell table: {table._df.height} cells, "
            f"{len(table._features)} features, {n_masks} masks, id column "
            f"{table._id_column!r}"
        ),
        overwrite=overwrite,
    )

    for (mask, ename), coords in table._embeddings.items():
        n_dims = len(coords.columns) - 1
        folio.add(
            f"{name}/embedding/{mask}__{ename}",
            coords,
            description=(
                f"{name!r} embedding {ename!r} on mask {mask!r}: "
                f"{coords.height} cells, {n_dims}D"
            ),
            overwrite=overwrite,
        )

    clusterings: dict[str, dict[str, Any]] = {}
    for cname, sim in table._clusterings.items():
        cx = coo_matrix(sim.similarity_matrix)
        triplets = pl.DataFrame(
            {
                "row": cx.row.astype(np.int64),
                "col": cx.col.astype(np.int64),
                "value": cx.data.astype(np.float64),
            }
        )
        folio.add(
            f"{name}/clustering/{cname}",
            triplets,
            description=(
                f"{name!r} consensus clustering {cname!r}: {sim.shape[0]}x"
                f"{sim.shape[1]} similarity matrix, {cx.nnz} nonzero entries, "
                f"{sim.method} linkage"
            ),
            overwrite=overwrite,
        )
        clusterings[cname] = {
            "shape": list(sim.shape),
            "method": sim.method,
            "max_value": float(sim.max_value),
            "normalized": float(sim.max_value) == 1.0,
        }

    manifest = {
        "kind": _KIND,
        "version": _VERSION,
        "id_column": table._id_column,
        "features": list(table._features),
        "var": table._var.to_dicts(),
        "collections": {n: list(c.columns) for n, c in table._collections.items()},
        "transforms": dict(table._transforms),
        "scaler": _scaler_tag(table._scaler_factory),
        "embeddings": [[mask, ename] for (mask, ename) in table._embeddings],
        "clusterings": clusterings,
        "labels": _labels_manifest(table),
    }
    folio.add(
        f"{name}/manifest",
        manifest,
        description=(
            f"{name!r} CellPax analysis manifest: structure needed to reload "
            f"the FeatureTable (features, collections, transforms, scaler, "
            f"embeddings, clusterings)"
        ),
        overwrite=overwrite,
    )


def load_feature_table(folio: DataFolio | str | Path, name: str) -> Any:
    """Load a FeatureTable previously saved under ``name``."""
    from cellpax.clustering import SimilarityMatrix
    from cellpax.featuretable import FeatureCollection, FeatureTable

    folio = _folio(folio)
    manifest = folio.get(f"{name}/manifest")
    if not isinstance(manifest, dict) or manifest.get("kind") != _KIND:
        raise ValueError(f"{name!r} is not a CellPax analysis in this folio")

    table = folio.get(f"{name}/table", frame="polars")
    mask_cols = [c for c in table.columns if c.startswith("_mask_")]
    base = table.drop(mask_cols)
    var = pl.DataFrame(manifest["var"]) if manifest["var"] else None

    ft = FeatureTable(
        base,
        manifest["features"],
        id_column=manifest["id_column"],
        feature_metadata=var,
        scaler_factory=_factory_from_tag(manifest["scaler"]),
    )
    ft._df = table  # restore masks and attached label columns verbatim
    ft._label_meta = _labels_from_manifest(manifest)
    ft._transforms = dict(manifest["transforms"])
    ft._collections = {
        n: FeatureCollection(n, tuple(cols))
        for n, cols in manifest["collections"].items()
    }
    for mask, ename in manifest["embeddings"]:
        ft._embeddings[(mask, ename)] = folio.get(
            f"{name}/embedding/{mask}__{ename}", frame="polars"
        )
    for cname, meta in manifest["clusterings"].items():
        triplets = folio.get(f"{name}/clustering/{cname}", frame="polars")
        matrix = csr_matrix(
            (
                triplets["value"].to_numpy(),
                (triplets["row"].to_numpy(), triplets["col"].to_numpy()),
            ),
            shape=tuple(meta["shape"]),
        )
        ft._clusterings[cname] = SimilarityMatrix(
            matrix, normalized=meta["normalized"], method=meta["method"]
        )
    return ft


def list_analyses(folio: DataFolio | str | Path) -> list[str]:
    """List CellPax analysis names stored in a folio (ignoring user content)."""
    folio = _folio(folio)
    names: list[str] = []
    for item in folio.list_contents().get("json_data", []):
        if not item.endswith("/manifest"):
            continue
        manifest = folio.get(item)
        if isinstance(manifest, dict) and manifest.get("kind") == _KIND:
            names.append(item[: -len("/manifest")])
    return sorted(names)
