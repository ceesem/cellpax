"""The AnnData bridge — one adapter to the scanpy/scvi ecosystem, not a rebuild.

A :class:`~cellpax.featuretable.FeatureTable` is deliberately not an AnnData:
masks are boolean columns rather than copies, embeddings are id-keyed frames
rather than positional blocks, and features carry declared validity domains.
But the single-cell ecosystem's tooling — scanpy plotting, scvi models,
cellxgene — all speaks AnnData, and re-implementing any of it here would be a
worse use of everyone's time than a bridge. So: :func:`to_anndata` exports a
mask's view of the table, and :func:`from_anndata` builds a working table from
an AnnData, whether or not this module produced it.

The export is honest rather than lossy-by-silence. Everything AnnData has a
slot for goes in the slot (``X``, ``obs``, ``var``, ``obsm``); everything it
does not — which mask this is, whether ``X`` is scaled, the per-feature
transforms, the validity domains, the table seed — is recorded under
``uns["cellpax"]``, in the same spirit as the persistence manifest
(:mod:`cellpax.persist`). Named masks ride along as boolean ``mask_<name>``
obs columns, so subsets stay visible on the AnnData side and survive the trip
back.

``anndata`` is an optional dependency, imported lazily inside the functions —
``pip install 'cellpax[anndata]'``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from anndata import AnnData

    from cellpax.featuretable import FeatureCollection, FeatureTable

#: Prefix for the boolean obs columns that carry named masks across the bridge.
_MASK_OBS_PREFIX = "mask_"


def _require_anndata() -> Any:
    """Import ``anndata`` lazily, keeping it an optional dependency."""
    try:
        import anndata
    except ImportError as error:
        raise ImportError(
            "the AnnData bridge requires the optional 'anndata' package; "
            "pip install 'cellpax[anndata]'"
        ) from error
    return anndata


def to_anndata(
    table: "FeatureTable",
    mask: str | None = None,
    *,
    columns: "str | FeatureCollection | Sequence[str] | None" = None,
    scaled: bool = False,
) -> "AnnData":
    """Export a mask's view of the table as an :class:`~anndata.AnnData`.

    Parameters
    ----------
    table : FeatureTable
        The table to export.
    mask : str, optional
        Which cells to export. ``None`` uses every cell.
    columns : str or FeatureCollection or sequence of str, optional
        Which features become ``X``'s columns. ``None`` uses every feature.
    scaled : bool, default False
        Whether ``X`` carries scaled values (the mask's own fit, exactly what
        ``features(mask, scaled=True)`` returns) or raw ones. Recorded in
        ``uns["cellpax"]["scaled"]`` either way, so the receiving side never
        has to guess what the matrix is.

    Returns
    -------
    AnnData
        - ``X``: the ``(n_cells, n_columns)`` feature matrix, float.
        - ``obs``: every non-feature column of ``dataframe(mask)`` — metadata,
          attached labels, the id column — indexed by the id values *as
          strings* (AnnData wants string ``obs_names``). The id column itself
          stays as an obs column too, so its original dtype survives the
          string coercion.
        - ``var``: the rows of ``table.var`` for the exported features,
          indexed by ``feature_id``, in ``X``'s column order.
        - ``obsm``: each stored embedding on this mask as ``X_<name>``, rows
          re-aligned to ``X``'s row order by cell id (the stored frames are
          id-keyed, so their row order is never trusted). An embedding that no
          longer covers every cell of the mask is skipped with a warning
          rather than exported with holes.
        - ``uns["cellpax"]``: the provenance AnnData has no slot for — mask
          name, ``scaled``, id column, feature order, per-feature transforms,
          validity domains, and the table seed.
        - Named masks (except ``"all"``) as boolean ``mask_<name>`` obs
          columns, restricted to the exported cells. An obs column already
          holding one of those names is shadowed by the mask, which is
          authoritative.
    """
    anndata = _require_anndata()
    from cellpax.featuretable import _DEFAULT_MASK

    resolved_mask = mask or _DEFAULT_MASK
    cols = table._resolve_columns(columns)
    id_column = table.id_column

    X = np.asarray(table.features(mask, scaled=scaled, columns=cols), dtype=float)

    view = table.dataframe(mask)
    obs_frame = view.drop([c for c in table.feature_columns if c in view.columns])
    member = table.mask_series(mask).to_numpy()
    for name in table.masks:
        if name == _DEFAULT_MASK:
            continue
        obs_frame = obs_frame.with_columns(
            pl.Series(
                f"{_MASK_OBS_PREFIX}{name}", table.mask_series(name).to_numpy()[member]
            )
        )
    obs = obs_frame.to_pandas()
    obs.index = obs[id_column].astype(str)
    obs.index.name = None

    var = (
        pl.DataFrame({"feature_id": cols})
        .join(table.var, on="feature_id", how="left")
        .to_pandas()
        .set_index("feature_id")
    )

    # obsm rows must land in X's row order, and the stored frames are id-keyed
    # rather than order-keyed — so alignment is a join on the id column, never
    # a bet on the stored row order.
    key_frame = pl.DataFrame({id_column: table._cell_ids(mask)})
    obsm: dict[str, np.ndarray] = {}
    for embedding_mask, name in table.embeddings:
        if embedding_mask != resolved_mask:
            continue
        coords = table.embedding(embedding_mask, name=name)
        coord_columns = [c for c in coords.columns if c != id_column]
        aligned = key_frame.join(coords, on=id_column, how="left")
        n_missing = aligned.filter(
            pl.any_horizontal([pl.col(c).is_null() for c in coord_columns])
        ).height
        if n_missing:
            warnings.warn(
                f"embedding {name!r} does not cover {n_missing} of mask "
                f"{resolved_mask!r}'s {key_frame.height} cells (the mask was "
                f"redefined since the coordinates were stored); skipped rather "
                f"than exported with null rows",
                stacklevel=2,
            )
            continue
        obsm[f"X_{name}"] = aligned.select(coord_columns).to_numpy()

    provenance = {
        "mask": resolved_mask,
        "scaled": bool(scaled),
        "id_column": id_column,
        "features": list(cols),
        "transforms": {c: t for c, t in table.transforms.items() if c in cols},
        "validity": {c: m for c, m in table.validity_domains.items() if c in cols},
        "seed": int(table.seed),
        "n_cells": int(X.shape[0]),
    }
    return anndata.AnnData(
        X=X, obs=obs, var=var, obsm=obsm, uns={"cellpax": provenance}
    )


def from_anndata(
    adata: "AnnData",
    *,
    features: Sequence[str] | None = None,
    id_column: str | None = None,
    seed: int = 0,
) -> "FeatureTable":
    """Build a :class:`~cellpax.featuretable.FeatureTable` from an AnnData.

    Works on any AnnData, not only one :func:`to_anndata` produced: a foreign
    object contributes ``X`` as features, obs as metadata, var columns as
    feature metadata, and obsm entries as embeddings. One produced here also
    carries ``uns["cellpax"]``, from which the original id column (with its
    dtype), per-feature transforms, validity domains and seed are restored.

    Parameters
    ----------
    adata : AnnData
        The source object. ``X`` may be dense or sparse.
    features : sequence of str, optional
        Which var names become feature columns. ``None`` uses all of
        ``adata.var_names``.
    id_column : str, optional
        The obs column holding unique cell ids. ``None`` prefers the column
        ``uns["cellpax"]`` recorded (restoring the original dtype) and falls
        back to ``obs_names`` as string ids under ``"cell_id"``. A name that
        is not an obs column keys the table on ``obs_names`` under that name.
    seed : int, default 0
        The table seed, used only when ``uns["cellpax"]`` does not carry one.

    Returns
    -------
    FeatureTable
        With masks re-registered from boolean ``mask_<name>`` obs columns
        (consumed, not left behind as metadata), embeddings registered from
        obsm via ``add_embedding`` (an ``X_`` prefix is stripped from the
        name; ``space="anndata"``) on the ``"all"`` mask, and — when
        ``uns["cellpax"]`` is present — transforms and validity domains
        restored. A validity entry whose mask or feature did not survive the
        trip is skipped with a warning rather than invented.
    """
    _require_anndata()
    import pandas as pd

    from cellpax.featuretable import _DEFAULT_MASK, FeatureTable

    provenance = dict(adata.uns.get("cellpax", {}))

    if features is None:
        features = [str(name) for name in adata.var_names]
    else:
        features = [str(name) for name in features]
        missing = [f for f in features if f not in adata.var_names]
        if missing:
            raise ValueError(f"features not in adata.var_names: {missing}")
    if adata.X is None:
        raise ValueError("adata has no X matrix to take feature values from")

    matrix = adata[:, features].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    obs = adata.obs
    mask_columns = [
        c
        for c in obs.columns
        if c.startswith(_MASK_OBS_PREFIX) and pd.api.types.is_bool_dtype(obs[c])
    ]
    meta = obs.drop(columns=mask_columns)

    # Resolve the cell ids: an obs column when one is named (explicitly or by
    # the provenance record, which is what restores the original dtype),
    # obs_names as string ids otherwise.
    recorded = provenance.get("id_column")
    if id_column is not None:
        resolved_id, from_index = id_column, id_column not in meta.columns
    elif recorded in meta.columns:
        resolved_id, from_index = str(recorded), False
    else:
        resolved_id, from_index = "cell_id", True
        if resolved_id in meta.columns:
            raise ValueError(
                "obs already has a 'cell_id' column that is not recorded as the "
                "table's id column; pass id_column= to say which ids to key on"
            )

    frame = pl.DataFrame({feature: matrix[:, j] for j, feature in enumerate(features)})
    if meta.shape[1]:
        clash = [c for c in meta.columns if c in frame.columns]
        if clash:
            raise ValueError(
                f"obs columns collide with the requested feature names: {clash}; "
                "pass features= to exclude one side"
            )
        frame = pl.from_pandas(meta).hstack(frame)
    if from_index:
        frame = frame.with_columns(
            pl.Series(resolved_id, [str(name) for name in adata.obs_names])
        )

    feature_metadata = None
    if adata.var.shape[1]:
        feature_metadata = pl.from_pandas(
            adata.var.rename_axis("feature_id").reset_index()
        )

    ft = FeatureTable(
        frame,
        features,
        id_column=resolved_id,
        feature_metadata=feature_metadata,
        seed=int(provenance.get("seed", seed)),
    )

    # Masks first: validity domains name masks, so the order is load-bearing.
    for column in mask_columns:
        name = column[len(_MASK_OBS_PREFIX) :]
        if not name or name == _DEFAULT_MASK:
            continue
        ft.add_mask(name, obs[column].to_numpy())

    for key, coords in adata.obsm.items():
        name = key[2:] if key.startswith("X_") else key
        if not name:
            continue
        ft.add_embedding(np.asarray(coords, dtype=float), name=name, space="anndata")

    transforms = provenance.get("transforms") or {}
    ft._transforms.update(
        {
            column: transform
            for column, transform in transforms.items()
            if column in features
        }
    )

    for column, domain in (provenance.get("validity") or {}).items():
        if column in features and domain in ft.masks:
            ft.set_validity([column], where=domain)
        else:
            warnings.warn(
                f"validity domain of {column!r} names mask {domain!r}, and one of "
                f"the pair did not survive the AnnData trip; skipped rather than "
                f"guessed",
                stacklevel=2,
            )
    return ft
