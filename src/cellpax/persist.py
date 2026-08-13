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
    <name>/partitions/<c>        the individual Leiden runs behind a consensus
    <name>/settings/<c>          which setting each of those runs came from
    <name>/space/<key>           a frozen scaling + PCA, as explicit parameters
    <name>/clustering/<c>        consensus matrix as (row, col, value) triplets (v1)

Version 2 stores the *runs* rather than the consensus matrix they imply. The matrix is
derived on load, which is both far smaller and strictly more informative:
``(n_cells, n_runs)`` int32 against a triplet table whose size grows with the number of
nonzero cell pairs. At 21k cells and 100 runs that is roughly 8 MB versus 600 MB — the
latter being large enough to exceed DataFolio's eager-load limit, so an analysis could be
saved and then not load back. Keeping the runs also restores ``Clustering.restrict`` and
``merge_support`` after a reload, neither of which the matrix alone can support.

Version 1 triplet clusterings are still read, so older folios load unchanged; they come
back without partitions, exactly as they did before.
"""

from __future__ import annotations

import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from datafolio import DataFolio
from scipy.sparse import coo_matrix, csr_matrix

_KIND = "cellpax_feature_table"
_VERSION = "2"


def _folio(folio: DataFolio | str | Path) -> DataFolio:
    if isinstance(folio, DataFolio):
        return folio
    return DataFolio(folio, allow_existing=True)


def _scaler_tag(factory: Any) -> str | dict[str, Any]:
    """How to rebuild the scaler factory: a tag, or its full configuration.

    ``clipped_scaler_factory(...)`` carries its arguments, so those are recorded and the
    reload gets the same percentiles or ``n_sigma`` back. A factory that is neither
    recognised nor self-describing raises rather than being recorded as something else:
    it used to fall back to ``"custom"``, which the loader then resolved to
    ``"standard"``, so a table saved with a clipped scaler reloaded with plain
    standardisation and no warning — silently changing every scaled value and therefore
    every downstream distance. Failing the save is recoverable; that is not.
    """
    from cellpax.clustering import make_clipped_scaler
    from cellpax.featuretable import _default_scaler_factory

    if factory is _default_scaler_factory:
        return "standard"
    if factory is make_clipped_scaler:
        return "clipped"
    params = getattr(factory, "_cellpax_scaler_params", None)
    if params is not None:
        return dict(params)
    raise TypeError(
        f"cannot record scaler factory {factory!r} in the manifest: it is neither one "
        "of the known factories nor self-describing, and saving it as anything else "
        "would reload as plain standardisation, silently changing every scaled value. "
        "Use clipped_scaler_factory(...) — which carries its configuration — or attach "
        "a '_cellpax_scaler_params' dict to the factory."
    )


def _factory_from_tag(tag: str | dict[str, Any]) -> Any:
    from cellpax.clustering import clipped_scaler_factory, make_clipped_scaler
    from cellpax.featuretable import _default_scaler_factory

    if isinstance(tag, dict):
        if tag.get("kind") != "clipped":
            raise ValueError(
                f"unknown scaler configuration {tag!r} in the manifest; this analysis "
                "was saved by a version that knows a scaler this one does not"
            )
        return clipped_scaler_factory(
            lower=tag.get("lower", 0.1),
            upper=tag.get("upper", 99.9),
            mode=tag.get("mode", "percentile"),
            n_sigma=tag.get("n_sigma", 5.0),
        )
    known = {"standard": _default_scaler_factory, "clipped": make_clipped_scaler}
    if tag not in known:
        raise ValueError(
            f"unknown scaler tag {tag!r} in the manifest; this analysis was saved with "
            "a scaler this version cannot rebuild, and guessing would change every "
            "scaled value"
        )
    return known[tag]


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


def _save_partitions(
    folio: DataFolio, name: str, cname: str, partitions: Any, *, overwrite: bool
) -> None:
    """Store the raw Leiden runs plus the setting each came from.

    ``int32`` because Leiden memberships are small non-negative integers (and ``-1``),
    so half the width of the default is exact and halves the stored size.
    """
    labels = pl.DataFrame(
        {
            f"run_{run}": partitions.labels[:, run].astype(np.int32)
            for run in range(partitions.n_runs)
        }
    )
    folio.add(
        f"{name}/partitions/{cname}",
        labels,
        description=(
            f"{name!r} clustering {cname!r} runs: {partitions.n_cells} cells x "
            f"{partitions.n_runs} Leiden runs, the input the consensus matrix is "
            f"derived from"
        ),
        overwrite=overwrite,
    )
    folio.add(
        f"{name}/settings/{cname}",
        pl.DataFrame(
            {
                "run": np.arange(partitions.n_runs, dtype=np.int64),
                "graph_type": partitions.graph_type.astype(str),
                "n_neighbors": partitions.n_neighbors.astype(np.int64),
                "resolution": partitions.resolution.astype(np.float64),
            }
        ),
        description=(
            f"{name!r} clustering {cname!r} settings: the graph type, neighbourhood "
            f"size and resolution behind each of {partitions.n_runs} runs"
        ),
        overwrite=overwrite,
    )


def _load_partitions(folio: DataFolio, name: str, cname: str) -> Any:
    """Rebuild :class:`~cellpax.Partitions` from stored runs and settings."""
    from cellpax.clustering import Partitions

    labels = folio.get(f"{name}/partitions/{cname}", frame="polars")
    settings = folio.get(f"{name}/settings/{cname}", frame="polars")
    # select by run index rather than trusting stored column order, so a backend
    # that returned run_10 before run_2 could not pair runs with wrong settings
    labels = labels.select([f"run_{run}" for run in range(len(labels.columns))])
    return Partitions(
        labels=labels.to_numpy().astype(np.int64),
        n_neighbors=settings["n_neighbors"].to_numpy().astype(np.int64),
        resolution=settings["resolution"].to_numpy().astype(np.float64),
        graph_type=settings["graph_type"].to_numpy().astype("<U16"),
    )


def _space_slug(
    mask: str,
    variance: float | int,
    seed: Any,
    weight_key: str | None = None,
    columns: tuple[str, ...] = (),
) -> str:
    """A filesystem-safe key for a frozen space, unique per cache entry.

    Carries everything the cache key carries: two spaces on one mask over
    different feature subsets are different fits, so ``columns`` is digested in
    — omitting it silently overwrote one of them on save. The variance marker
    keeps the int/float distinction (``evn3`` = three components, ``ev1`` =
    100% of the variance), matching :meth:`FeatureTable.space`.
    """
    marker = (
        f"evn{int(variance)}"
        if isinstance(variance, (int, np.integer)) and not isinstance(variance, bool)
        else f"ev{float(variance):g}"
    )
    slug = f"{mask}__{marker}__seed{'none' if seed is None else seed}"
    if columns:
        slug += f"__c{zlib.crc32('|'.join(columns).encode()):08x}"
    if weight_key is not None:
        # weighted and unweighted fits of the same mask are different spaces and must
        # not collide on disk
        slug += f"__w{weight_key}"
    return slug.replace(".", "p")


def _arrays_frame(arrays: dict[str, np.ndarray]) -> pl.DataFrame:
    """Named arrays of mixed rank as one long frame: ``array, row, col, value``.

    Uniform enough to hold a 1-D mean, a 2-D rotation and a handful of scaler bounds in
    one item, and small — a 45x81 rotation is a few thousand rows.
    """
    frames = []
    for key, value in arrays.items():
        block = np.atleast_2d(np.asarray(value, dtype=float))
        rows, cols = np.indices(block.shape)
        frames.append(
            pl.DataFrame(
                {
                    "array": np.full(block.size, key),
                    "row": rows.ravel().astype(np.int64),
                    "col": cols.ravel().astype(np.int64),
                    "value": block.ravel(),
                    "ndim": np.full(block.size, np.asarray(value).ndim, dtype=np.int64),
                }
            )
        )
    return pl.concat(frames)


def _frame_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Inverse of :func:`_arrays_frame`."""
    arrays: dict[str, np.ndarray] = {}
    for (key,), part in frame.group_by(["array"], maintain_order=True):
        rows = part["row"].to_numpy()
        cols = part["col"].to_numpy()
        block = np.zeros((rows.max() + 1, cols.max() + 1), dtype=float)
        block[rows, cols] = part["value"].to_numpy()
        arrays[str(key)] = block if int(part["ndim"][0]) == 2 else block.ravel()
    return arrays


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
        partitions = getattr(sim, "partitions", None)
        meta: dict[str, Any] = {
            "shape": list(sim.shape),
            "method": sim.method,
            "max_value": float(sim.max_value),
            # the real flag, not inferred from max_value — an unnormalized
            # single-run consensus also peaks at 1.0
            "normalized": bool(getattr(sim, "normalized", float(sim.max_value) == 1.0)),
            "params": getattr(sim, "params", None),
            # provenance, so a reloaded clustering can still cut itself into labels
            "mask": getattr(sim, "mask", None),
            "columns": list(getattr(sim, "columns", ())),
            "space": getattr(sim, "space", ""),
            # only the column name: the values themselves live in the table, so
            # re-reading them on load keeps the two from drifting apart
            "order_by": getattr(sim, "order_by", None),
            "order_agg": getattr(sim, "_order_agg", "mean"),
            "order_ascending": getattr(sim, "_order_ascending", True),
        }
        folio.add(
            f"{name}/cellids/{cname}",
            pl.DataFrame({table._id_column: np.asarray(sim.cell_ids)}),
            description=(
                f"{name!r} clustering {cname!r} cell ids, in matrix-row order — "
                f"stored rather than re-derived so a mask redefined after "
                f"clustering cannot silently re-pair rows with different cells"
            ),
            overwrite=overwrite,
        )
        meta["cell_ids_stored"] = True
        if partitions is not None:
            _save_partitions(folio, name, cname, partitions, overwrite=overwrite)
            meta["storage"] = "partitions"
            meta["n_runs"] = int(partitions.n_runs)
        else:
            # No runs to store — a clustering reloaded from a v1 folio and re-saved, or
            # one built straight from a matrix. Fall back to the triplet form.
            cx = coo_matrix(sim.similarity_matrix)
            triplets = pl.DataFrame(
                {
                    "row": cx.row.astype(np.int64),
                    "col": cx.col.astype(np.int64),
                    "value": cx.data.astype(np.float32),
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
            meta["storage"] = "matrix"
        clusterings[cname] = meta

    spaces: dict[str, dict[str, Any]] = {}
    for key, space in getattr(table, "_space_cache", {}).items():
        mask, columns, variance_key, seed, weight_key = key
        variance = variance_key[1]
        slug = _space_slug(mask, variance, seed, weight_key, columns)
        space_meta, arrays = space.to_records()
        folio.add(
            f"{name}/space/{slug}",
            _arrays_frame(arrays),
            description=(
                f"{name!r} frozen space {slug!r}: {space.label}, "
                f"{space.n_components}/{space.n_total_components} components over "
                f"{len(columns)} features on mask {mask!r}"
            ),
            overwrite=overwrite,
        )
        space_meta["mask"] = mask
        space_meta["seed"] = seed
        space_meta["fit_explained_variance"] = variance
        space_meta["fit_variance_is_count"] = variance_key[0] == "n"
        space_meta["weight_key"] = weight_key
        spaces[slug] = space_meta

    manifest = {
        "kind": _KIND,
        "version": _VERSION,
        "id_column": table._id_column,
        "seed": int(getattr(table, "_seed", 0)),
        "features": list(table._features),
        "var": table._var.to_dicts(),
        "collections": {n: list(c.columns) for n, c in table._collections.items()},
        "transforms": dict(table._transforms),
        "validity": dict(getattr(table, "_validity", {})),
        "scaler": _scaler_tag(table._scaler_factory),
        "embeddings": [[mask, ename] for (mask, ename) in table._embeddings],
        # the calls behind stored coordinates — the model itself is session-only,
        # but the parameterization is what makes the coordinates re-derivable
        "embedding_params": {
            f"{mask}__{ename}": params
            for (mask, ename), params in getattr(table, "_embedding_params", {}).items()
        },
        # where externally-registered coordinates came from. Unlike a fitted model this
        # is a label the caller supplied, so losing it on reload would discard provenance
        # nothing else records.
        "external_embeddings": [
            [mask, ename, space, n_components]
            for (mask, ename), (space, n_components) in getattr(
                table, "_external_embeddings", {}
            ).items()
        ],
        "clusterings": clusterings,
        "spaces": spaces,
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
    from cellpax.clustering import Clustering
    from cellpax.featuretable import FeatureCollection, FeatureTable

    folio = _folio(folio)
    manifest = folio.get(f"{name}/manifest")
    if not isinstance(manifest, dict) or manifest.get("kind") != _KIND:
        raise ValueError(f"{name!r} is not a CellPax analysis in this folio")
    version = str(manifest.get("version", "1"))
    if version not in {"1", "2"}:
        raise ValueError(
            f"analysis {name!r} has manifest version {version}, which this version "
            f"of cellpax does not know (it reads versions 1 and 2); parsing it as "
            f"v2 could silently misread it — upgrade cellpax instead"
        )

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
        seed=int(manifest.get("seed", 0)),
    )
    ft._df = table  # restore masks and attached label columns verbatim
    ft._label_meta = _labels_from_manifest(manifest)
    ft._transforms = dict(manifest["transforms"])
    ft._validity = dict(manifest.get("validity", {}))
    ft._collections = {
        n: FeatureCollection(n, tuple(cols))
        for n, cols in manifest["collections"].items()
    }
    for mask, ename in manifest["embeddings"]:
        ft._embeddings[(mask, ename)] = folio.get(
            f"{name}/embedding/{mask}__{ename}", frame="polars"
        )
    for mask, ename, space, n_components in manifest.get("external_embeddings", []):
        ft._external_embeddings[(mask, ename)] = (space, int(n_components))
    for slug, params in manifest.get("embedding_params", {}).items():
        mask, _, ename = slug.partition("__")
        ft._embedding_params[(mask, ename)] = params
    for slug, space_meta in manifest.get("spaces", {}).items():
        from cellpax.space import FittedSpace

        arrays = _frame_arrays(folio.get(f"{name}/space/{slug}", frame="polars"))
        space = FittedSpace.from_records(space_meta, arrays)
        raw_variance = space_meta["fit_explained_variance"]
        is_count = space_meta.get(
            "fit_variance_is_count", isinstance(raw_variance, int)
        )
        variance_key = (
            ("n", int(raw_variance)) if is_count else ("var", float(raw_variance))
        )
        key = (
            space_meta["mask"],
            tuple(space_meta["columns"]),
            variance_key,
            space_meta["seed"],
            space_meta.get("weight_key"),
        )
        ft._space_cache[key] = space

    for cname, meta in manifest["clusterings"].items():
        partitions = None
        if meta.get("storage") == "partitions":
            partitions = _load_partitions(folio, name, cname)
            matrix = partitions.coclustering(normalize=meta["normalized"])
        else:
            triplets = folio.get(f"{name}/clustering/{cname}", frame="polars")
            matrix = csr_matrix(
                (
                    triplets["value"].to_numpy().astype(np.float64),
                    (triplets["row"].to_numpy(), triplets["col"].to_numpy()),
                ),
                shape=tuple(meta["shape"]),
            )
        # analyses saved before clusterings carried provenance have no mask recorded;
        # "all" reproduces what ft.label(name, distance_threshold=…) did back then
        mask = meta.get("mask") or "all"
        if meta.get("cell_ids_stored"):
            cell_ids = (
                folio.get(f"{name}/cellids/{cname}", frame="polars")
                .to_series()
                .to_numpy()
            )
        else:
            # older saves re-derive from the current mask; a same-size redefinition
            # between cluster() and save() cannot be detected here
            cell_ids = ft._cell_ids(mask)
        order_by = meta.get("order_by")
        order_values = None
        if order_by is not None and order_by in ft._df.columns:
            # matched to the stored row order by id, not by the mask's current order
            id_to_value = dict(
                zip(
                    ft._df[ft._id_column].to_list(),
                    ft._df[order_by].cast(pl.Float64).to_list(),
                )
            )
            order_values = np.asarray(
                [id_to_value.get(c, np.nan) for c in cell_ids], dtype=float
            )
        elif order_by is not None:
            warnings.warn(
                f"clustering {cname!r} was ordered by column {order_by!r}, which is "
                f"no longer in the table; its cuts will use dendrogram order",
                stacklevel=2,
            )
            order_by = None
        ft._clusterings[cname] = Clustering(
            matrix,
            cell_ids=cell_ids,
            mask=mask,
            columns=tuple(meta.get("columns", ())),
            space=meta.get("space", ""),
            normalized=meta["normalized"],
            method=meta["method"],
            order_by=order_by,
            order_values=order_values,
            order_agg=meta.get("order_agg", "mean"),
            order_ascending=meta.get("order_ascending", True),
            partitions=partitions,
            params=meta.get("params"),
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
