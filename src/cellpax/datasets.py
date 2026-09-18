"""Joining feature tables from separate datasets into one harmonized table.

Two volumes measured with the same feature extraction still disagree about the same
cells: synapse sizes in different units, a detection threshold that enriches one tail,
a spine/shaft classifier that splits differently, sections compressed by a few percent.
Concatenating the rows and scaling once puts every one of those differences into the
geometry, and a clustering then finds the datasets before it finds the cell types.

:func:`join_datasets` removes the difference before the rows meet. Each dataset is
scaled on its own — optionally within each subclass, because a batch effect whose
direction reverses between cell types cannot be removed by any single per-feature map —
and every non-reference dataset is then mapped *through* the reference's inverse::

    x_joined = S_reference,s.inverse_transform(S_dataset,s.transform(x))

so the joined values are in the reference dataset's own raw units, differences *between*
subclasses survive intact, and only the difference between datasets within a subclass is
removed. The reference's cells are untouched.

The fits are kept on the joined table, so new cells from any dataset can be brought into
the joined units (:meth:`~cellpax.FeatureTable.harmonize`) and then projected through the
joined table's own fits. Original ids stay first-class: every row keeps its dataset and
its source id, both directions of the lookup are methods, and labels built on the joined
table can be handed back to each source keyed by its own ids.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from cellpax.featuretable import FeatureTable, FittedScaler
    from cellpax.labels import LabelSet


def join_datasets(
    tables: Mapping[str, "FeatureTable"],
    *,
    scaler_factory: Callable[[], Any],
    reference: str | None = None,
    strata: str | Mapping[str, str] | None = None,
    stratum_column: str = "stratum",
    min_cells: int = 50,
    columns: Sequence[str] | None = None,
    fit_mask: str | Mapping[str, str | None] | None = None,
    dataset_column: str = "dataset",
    id_column: str = "cell_id",
    source_id_column: str | None = None,
    dataset_masks: bool = True,
    output_scaler_factory: Callable[[], Any] | None = None,
    seed: int = 0,
) -> "FeatureTable":
    """Join tables from different datasets into one table in a common feature space.

    Every dataset gets its own scaler — one per ``(dataset, stratum)`` when ``strata``
    is given — and every non-reference dataset's cells are mapped onto the reference
    dataset's distribution for the same stratum: through their own fitted scaler, then
    back out through the reference's ``inverse_transform``. The joined feature values
    are therefore in the **reference dataset's raw units**; the reference's own cells
    are unchanged, and the joined table then scales, clusters and embeds exactly like
    any other table.

    Parameters
    ----------
    tables : mapping of str to FeatureTable
        Dataset name → table, at least two. Names become the values of
        ``dataset_column`` and, with ``dataset_masks``, mask names, so they follow mask
        naming rules. Mapping order fixes row order and id blocks.
    scaler_factory : callable
        Zero-argument factory for the per-dataset scalers. The scaler must provide
        ``inverse_transform``: :func:`~cellpax.quantile_scaler_factory` (a percentile
        clip then a rank transform, the choice that aligns nonlinear differences),
        ``StandardScaler``/``RobustScaler`` (location and scale only), or
        :func:`~cellpax.clipped_scaler_factory`.
    reference : str, optional
        Dataset whose units the joined table uses. Defaults to the first table. Every
        stratum of every other dataset must exist in it.
    strata : str or mapping of str to str, optional
        Column naming each cell's stratum — a metadata or attached label column, e.g. a
        subclass — shared by name or given per dataset. Values are compared as strings,
        so subclass names must already agree across datasets; lump ambiguous continua
        and reconcile taxonomies into a column before joining. ``None`` treats each
        dataset as a single stratum.
    stratum_column : str, default 'stratum'
        Name of the String column recording each cell's stratum in the joined table.
        Only written when ``strata`` is given.
    min_cells : int, default 50
        Fewest fit cells a mapped ``(dataset, stratum)`` fit may use, on both sides.
    columns : sequence of str, optional
        Features to join. By default every table must have the same feature set.
        Features left out are dropped rather than carried unharmonized.
    fit_mask : str or mapping of str to str or None, optional
        Mask whose cells each scaler is fit on — a quality-controlled subset, say. The
        fit is then applied to *every* cell of that dataset. One name for all tables or
        a per-dataset mapping; ``None`` fits on all cells.
    dataset_column : str, default 'dataset'
        Name of the String column recording each cell's dataset.
    id_column : str, default 'cell_id'
        Id column of the joined table, filled with freshly minted unique ``Int64`` ids.
    source_id_column : str, optional
        Column holding each cell's original id. Defaults to ``f"source_{id_column}"``.
    dataset_masks : bool, default True
        Add one mask per dataset, named after it.
    output_scaler_factory : callable, optional
        The joined table's own ``scaler_factory``, used for its lazy per-mask scaling
        after harmonization. Defaults to ``StandardScaler``.
    seed : int, default 0
        Seed of the joined table.

    Returns
    -------
    FeatureTable
        The joined table. Metadata columns are unioned (null where a dataset lacks
        one), masks are unioned (``False`` where a dataset lacks one), collections,
        validity domains and attached labels are carried; embeddings, clusterings and
        fitted spaces are not, since they live in the old per-dataset spaces.

    Raises
    ------
    ValueError
        On mismatched feature sets, reserved or conflicting names, strata that cannot
        be mapped onto the reference, too-small strata, or conflicting metadata.
    TypeError
        If the scaler cannot be inverted.

    Notes
    -----
    Ids are minted in dataset blocks sized by the next power of ten above the largest
    dataset — with every dataset under 10,000 cells, ``10000…`` for the first dataset
    and ``20000…`` for the second — so a joined id never coincides with the ``0…n`` or
    ``1…n`` row ids a source table often uses. Do not parse them; use
    :meth:`~cellpax.FeatureTable.source_ids` and
    :meth:`~cellpax.FeatureTable.cell_ids_for`.

    Aligning marginals, even within subclasses, does not align covariance: two datasets
    whose features correlate differently still separate in neighbour structure. Measure
    what is left with :meth:`~cellpax.FeatureTable.dataset_mixing`, in feature space as
    well as PCA — with few, mostly informative features PCA tends to concentrate the
    residual dataset structure rather than dilute it.
    """
    from cellpax.featuretable import (
        _DEFAULT_MASK,
        _MASK_PREFIX,
        FeatureTable,
        FittedScaler,
        _DatasetJoin,
        _validate_scaler_factory,
    )

    tables = dict(tables)
    if len(tables) < 2:
        raise ValueError(f"join_datasets needs at least two tables, got {len(tables)}")
    names = list(tables)
    for name, table in tables.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"dataset names must be non-empty strings, got {name!r}")
        if name == _DEFAULT_MASK or name.startswith(_MASK_PREFIX) or "/" in name:
            raise ValueError(
                f"invalid dataset name {name!r}: dataset names become mask names, so "
                f"they cannot be {_DEFAULT_MASK!r}, start with {_MASK_PREFIX!r}, or "
                f"contain '/'"
            )
        if not isinstance(table, FeatureTable):
            raise TypeError(
                f"dataset {name!r} is a {type(table).__name__}, not a FeatureTable"
            )
    reference = names[0] if reference is None else reference
    if reference not in tables:
        raise ValueError(f"reference {reference!r} is not one of the datasets {names}")
    if min_cells < 2:
        raise ValueError(f"min_cells must be at least 2, got {min_cells}")

    factory = _validate_scaler_factory(scaler_factory)
    if not hasattr(factory(), "inverse_transform"):
        raise TypeError(
            "the scaler from scaler_factory has no inverse_transform, so datasets "
            "cannot be mapped into the reference's units; use quantile_scaler_factory, "
            "StandardScaler, RobustScaler, or clipped_scaler_factory"
        )

    cols = _resolve_join_columns(tables, reference, columns)
    source_id_column = source_id_column or f"source_{id_column}"
    reserved = [id_column, source_id_column, dataset_column]
    if strata is not None:
        reserved.append(stratum_column)
    if len(set(reserved)) != len(reserved):
        raise ValueError(f"output column names must be distinct, got {reserved}")

    fit_masks = _per_dataset(fit_mask, names, "fit_mask")
    strata_columns = _per_dataset(strata, names, "strata")
    for name, table in tables.items():
        own = set(table.columns) - {table.id_column}
        clash = sorted(own & set(reserved))
        if clash:
            raise ValueError(
                f"dataset {name!r} already has column(s) {clash}, which the joined "
                f"table reserves; rename them or pass different dataset_column / "
                f"id_column / source_id_column / stratum_column names"
            )
        if fit_masks[name] is not None:
            table.mask_series(fit_masks[name])  # raises with the available masks
        column = strata_columns[name]
        if column is not None and column not in table.columns:
            raise ValueError(f"strata column {column!r} is not in dataset {name!r}")

    # -- per-dataset raw matrices, fit rows, and strata ----------------------------
    raw = {
        name: table._df.select(cols).to_numpy().astype(float)
        for name, table in tables.items()
    }
    fit_rows = {
        name: table.mask_series(fit_masks[name]).to_numpy()
        for name, table in tables.items()
    }
    stratum_values = {
        name: (
            np.full(table.n_cells, None, dtype=object)
            if strata_columns[name] is None
            else np.asarray(
                table._df[strata_columns[name]].cast(pl.String).to_list(),
                dtype=object,
            )
        )
        for name, table in tables.items()
    }
    _check_source_ids(tables, source_id_column)
    _check_transforms(tables, cols)

    # -- fit and map ----------------------------------------------------------------
    scalers: dict[tuple[str, str | None], FittedScaler] = {}

    def fitted(name: str, stratum: str | None) -> FittedScaler:
        key = (name, stratum)
        if key not in scalers:
            rows = fit_rows[name] & _in_stratum(stratum_values[name], stratum)
            n_fit = int(rows.sum())
            if n_fit < min_cells:
                what = (
                    f"dataset {name!r}"
                    if stratum is None
                    else f"stratum {stratum!r} of dataset {name!r}"
                )
                raise ValueError(
                    f"{what} has {n_fit} fit cells, fewer than min_cells={min_cells}; "
                    f"a distribution cannot be estimated from that few — lump it into "
                    f"a neighbouring stratum, drop it, or lower min_cells"
                )
            table = tables[name]
            transforms = [table._transforms.get(c) for c in cols]
            scaler = FittedScaler(scaler=factory(), transforms=transforms)
            scaler.fit(raw[name][rows])
            scalers[key] = scaler
        return scalers[key]

    harmonized = {reference: raw[reference]}
    for name in names:
        if name == reference:
            continue
        values = stratum_values[name]
        if strata is None:
            groups: list[str | None] = [None]
        else:
            if any(v is None for v in values):
                n_null = sum(v is None for v in values)
                raise ValueError(
                    f"{n_null} cells of dataset {name!r} have no "
                    f"{strata_columns[name]!r} value; every cell of a non-reference "
                    f"dataset needs a stratum to be mapped — label or drop them first"
                )
            groups = sorted(set(values))
            ref_groups = set(stratum_values[reference]) - {None}
            missing = [g for g in groups if g not in ref_groups]
            if missing:
                raise ValueError(
                    f"strata {missing} of dataset {name!r} do not exist in the "
                    f"reference {reference!r}, so there is no distribution to map them "
                    f"onto; reconcile the subclass names or drop those cells"
                )
        out = np.empty_like(raw[name])
        for group in groups:
            rows = _in_stratum(values, group)
            own, ref = fitted(name, group), fitted(reference, group)
            try:
                out[rows] = ref.inverse_transform(own.transform(raw[name][rows]))
            except ValueError as error:
                where = "" if group is None else f", stratum {group!r}"
                raise ValueError(
                    f"harmonizing dataset {name!r}{where} failed: {error}"
                ) from error
        harmonized[name] = out

    _warn_unrecordable(scalers)

    # -- rows -------------------------------------------------------------------------
    block = 10 ** len(str(max(t.n_cells for t in tables.values())))
    label_columns = _label_columns(tables)
    frames = []
    for position, (name, table) in enumerate(tables.items()):
        drop = [c for c in table._df.columns if c.startswith(_MASK_PREFIX)]
        drop += [
            c
            for label in label_columns
            if label in table.labels
            for c in (label, f"{label}_id", f"{label}_confidence")
            if c in table._df.columns
        ]
        drop += [c for c in table.feature_columns if c not in cols]
        frame = table._df.drop(drop).with_columns(
            [
                pl.Series(c, harmonized[name][:, j], dtype=pl.Float64, nan_to_null=True)
                for j, c in enumerate(cols)
            ]
        )
        frame = frame.rename({table.id_column: source_id_column}).with_columns(
            pl.Series(
                id_column,
                (position + 1) * block + np.arange(table.n_cells),
                dtype=pl.Int64,
            ),
            pl.lit(name, dtype=pl.String).alias(dataset_column),
        )
        if strata is not None:
            frame = frame.with_columns(
                pl.Series(
                    stratum_column, stratum_values[name].tolist(), dtype=pl.String
                )
            )
        lead = [id_column, dataset_column, source_id_column]
        if strata is not None:
            lead.append(stratum_column)
        frames.append(frame.select(lead + [c for c in frame.columns if c not in lead]))
    _check_concat_dtypes(frames, names)
    rows = pl.concat(frames, how="diagonal_relaxed")

    joined = FeatureTable(
        rows,
        cols,
        id_column=id_column,
        feature_metadata=_merge_var(tables, cols),
        scaler_factory=output_scaler_factory,
        seed=seed,
    )
    joined._transforms = {
        c: t
        for c, t in tables[reference]._transforms.items()
        if c in cols and t is not None
    }
    offsets = np.cumsum([0] + [t.n_cells for t in tables.values()])

    _carry_masks(joined, tables, offsets, dataset_masks, dataset_column)
    _carry_collections(joined, tables, cols)
    _carry_validity(joined, tables, cols)
    _carry_labels(joined, tables, label_columns, block, id_column, dataset_masks, names)

    from cellpax.persist import _scaler_tag

    try:
        factory_tag: Any = _scaler_tag(factory)
    except TypeError:
        # provenance only — the fitted scalers themselves are what get saved
        factory_tag = getattr(factory, "__name__", type(factory).__name__)
    joined._join = _DatasetJoin(
        column=dataset_column,
        source_id_column=source_id_column,
        stratum_column=stratum_column if strata is not None else None,
        reference=reference,
        features=list(cols),
        entries=[
            {
                "name": name,
                "id_column": table.id_column,
                "fit_mask": fit_masks[name],
                "strata": strata_columns[name],
                "n_cells": table.n_cells,
                "id_offset": (position + 1) * block,
                "factory": factory_tag,
            }
            for position, (name, table) in enumerate(tables.items())
        ],
        scalers=scalers,
    )
    return joined


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _per_dataset(value: Any, names: list[str], parameter: str) -> dict[str, str | None]:
    """Expand a shared value or a per-dataset mapping to one entry per dataset."""
    if value is None or isinstance(value, str):
        return {name: value for name in names}
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f"{parameter} names unknown datasets {unknown}")
        if parameter == "strata":
            missing = [name for name in names if value.get(name) is None]
            if missing:
                raise ValueError(
                    f"strata must name a column for every dataset; missing for "
                    f"{missing}"
                )
        return {name: value.get(name) for name in names}
    raise TypeError(
        f"{parameter} must be a string or a mapping of dataset name to string, got "
        f"{type(value).__name__}"
    )


def _in_stratum(values: np.ndarray, stratum: str | None) -> np.ndarray:
    """Boolean row selector for a stratum; every row when the join is unstratified."""
    if stratum is None:
        return np.ones(values.shape[0], dtype=bool)
    return values == stratum


def _resolve_join_columns(
    tables: dict[str, "FeatureTable"], reference: str, columns: Sequence[str] | None
) -> list[str]:
    """The joined feature list, in the reference's order unless given explicitly."""
    if columns is not None:
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of feature names, not a string")
        cols = list(columns)
        if not cols:
            raise ValueError("columns cannot be empty")
        for name, table in tables.items():
            missing = [c for c in cols if c not in table.feature_columns]
            if missing:
                raise ValueError(f"dataset {name!r} has no feature(s) {missing}")
        return cols
    cols = tables[reference].feature_columns
    expected = set(cols)
    for name, table in tables.items():
        present = set(table.feature_columns)
        if present != expected:
            only_here = sorted(present - expected)
            only_ref = sorted(expected - present)
            raise ValueError(
                f"dataset {name!r} has a different feature set from {reference!r} "
                f"(only in {name!r}: {only_here}; only in {reference!r}: {only_ref}); "
                f"pass columns= to join on a shared subset"
            )
    return cols


def _check_transforms(tables: dict[str, "FeatureTable"], cols: list[str]) -> None:
    """Warn when datasets were preprocessed with different per-feature transforms."""
    differing = [
        c
        for c in cols
        if len({table._transforms.get(c) for table in tables.values()}) > 1
    ]
    if differing:
        warnings.warn(
            f"datasets disagree on the preprocess transform of {differing}; each "
            f"dataset's scaler applies its own, which a rank-based scaler absorbs but "
            f"a linear one does not",
            stacklevel=3,
        )


def _warn_unrecordable(scalers: dict[tuple[str, str | None], Any]) -> None:
    """Say at join time, not after hours of clustering, that ``save`` will refuse."""
    from cellpax.space import _scaler_records

    for fitted in scalers.values():
        try:
            _scaler_records(fitted)
        except TypeError as error:
            warnings.warn(
                f"the per-dataset scalers cannot be recorded ({error}); the joined "
                f"table works in this session, but saving it will raise",
                stacklevel=3,
            )
            return


def _label_columns(tables: dict[str, "FeatureTable"]) -> list[str]:
    """Attached label columns across datasets, refusing label/plain-column clashes."""
    labels: list[str] = []
    for table in tables.values():
        labels += [c for c in table.labels if c not in labels]
    for label in labels:
        plain = [
            name
            for name, table in tables.items()
            if label not in table.labels
            and any(
                c in table.columns
                for c in (label, f"{label}_id", f"{label}_confidence")
            )
        ]
        if plain:
            raise ValueError(
                f"{label!r} is an attached label in some datasets but a plain column "
                f"in {plain}; attach it everywhere or rename it before joining"
            )
    return labels


def _check_source_ids(tables: dict[str, "FeatureTable"], column: str) -> None:
    """Refuse to widen integer ids to strings by mixing id types across datasets."""
    dtypes = {name: table._df.schema[table.id_column] for name, table in tables.items()}
    integer = {name for name, dtype in dtypes.items() if dtype.is_integer()}
    if integer and len(integer) != len(dtypes):
        raise ValueError(
            f"source ids have mixed types ({ {n: str(d) for n, d in dtypes.items()} }); "
            f"{column!r} would silently turn them all into strings — cast them to one "
            f"type first"
        )
    if not integer and len({str(d) for d in dtypes.values()}) > 1:
        raise ValueError(
            f"source ids have mixed types ({ {n: str(d) for n, d in dtypes.items()} }); "
            f"cast them to one type first"
        )


def _check_concat_dtypes(frames: list[pl.DataFrame], names: list[str]) -> None:
    """Name the metadata columns whose types cannot be unified, before polars fails."""
    seen: dict[str, dict[str, pl.DataType]] = {}
    for name, frame in zip(names, frames):
        for column, dtype in frame.schema.items():
            seen.setdefault(column, {})[name] = dtype
    bad = {
        column: {n: str(d) for n, d in by.items()}
        for column, by in seen.items()
        if len({str(d) for d in by.values()}) > 1
        and not all(d.is_numeric() for d in by.values())
    }
    if bad:
        raise ValueError(
            f"metadata columns have incompatible types across datasets: {bad}; cast "
            f"or rename them before joining"
        )


def _merge_var(
    tables: dict[str, "FeatureTable"], cols: list[str]
) -> pl.DataFrame | None:
    """One feature-metadata frame, refusing features the datasets annotate differently."""
    parts = [
        table._var.filter(pl.col("feature_id").is_in(cols)) for table in tables.values()
    ]
    try:
        var = pl.concat(parts, how="diagonal_relaxed")
    except Exception as error:  # noqa: BLE001 — polars raises several types here
        raise ValueError(
            f"feature metadata cannot be combined across datasets: {error}"
        ) from error
    attributes = [c for c in var.columns if c != "feature_id"]
    if not attributes:
        return None
    grouped = var.group_by("feature_id", maintain_order=True)
    counts = grouped.agg([pl.col(c).drop_nulls().n_unique() for c in attributes])
    conflicts = [
        f"{row['feature_id']}.{c}"
        for row in counts.iter_rows(named=True)
        for c in attributes
        if row[c] > 1
    ]
    if conflicts:
        raise ValueError(
            f"datasets annotate features differently: {conflicts[:10]}"
            f"{' …' if len(conflicts) > 10 else ''}; align the feature metadata first"
        )
    return grouped.agg([pl.col(c).drop_nulls().first() for c in attributes])


def _carry_masks(
    joined: "FeatureTable",
    tables: dict[str, "FeatureTable"],
    offsets: np.ndarray,
    dataset_masks: bool,
    dataset_column: str,
) -> None:
    from cellpax.featuretable import _DEFAULT_MASK

    masks: list[str] = []
    for table in tables.values():
        masks += [m for m in table.masks if m != _DEFAULT_MASK and m not in masks]
    if dataset_masks:
        clash = sorted(set(masks) & set(tables))
        if clash:
            raise ValueError(
                f"mask(s) {clash} share a name with a dataset, which dataset_masks "
                f"would overwrite; rename the masks or pass dataset_masks=False"
            )
    partial: list[str] = []
    for mask in masks:
        membership = np.zeros(joined.n_cells, dtype=bool)
        for position, table in enumerate(tables.values()):
            if mask in table.masks:
                start, stop = offsets[position], offsets[position + 1]
                membership[start:stop] = table.mask_series(mask).to_numpy()
            else:
                partial.append(mask)
        joined.add_mask(mask, pl.Series(mask, membership))
    if partial:
        warnings.warn(
            f"mask(s) {sorted(set(partial))} are missing from some datasets; those "
            f"datasets' cells are outside them in the joined table",
            stacklevel=3,
        )
    if dataset_masks:
        for name in tables:
            joined.add_mask(name, pl.col(dataset_column) == name)


def _carry_collections(
    joined: "FeatureTable", tables: dict[str, "FeatureTable"], cols: list[str]
) -> None:
    from cellpax.featuretable import FeatureCollection

    merged: dict[str, tuple[str, ...]] = {}
    for name, table in tables.items():
        for cname, collection in table._collections.items():
            columns = tuple(collection.columns)
            if cname in merged and merged[cname] != columns:
                raise ValueError(
                    f"collection {cname!r} has different columns in dataset {name!r}; "
                    f"rename one of them before joining"
                )
            merged[cname] = columns
    dropped = [c for c, columns in merged.items() if not set(columns) <= set(cols)]
    if dropped:
        warnings.warn(
            f"collection(s) {dropped} use features outside the joined columns and "
            f"were not carried",
            stacklevel=3,
        )
    joined._collections = {
        cname: FeatureCollection(cname, columns)
        for cname, columns in merged.items()
        if cname not in dropped
    }


def _carry_validity(
    joined: "FeatureTable", tables: dict[str, "FeatureTable"], cols: list[str]
) -> None:
    restricted = {
        feature
        for table in tables.values()
        for feature in table._validity
        if feature in cols
    }
    for feature in sorted(restricted):
        domains = {name: table._validity.get(feature) for name, table in tables.items()}
        if len(set(domains.values())) > 1:
            raise ValueError(
                f"feature {feature!r} has different validity domains across datasets "
                f"({domains}); a domain filled with False for one dataset would "
                f"misstate it — give every dataset the same domain mask "
                f"(add_mask + set_validity) before joining"
            )
        joined.set_validity([feature], where=next(iter(domains.values())))


def _carry_labels(
    joined: "FeatureTable",
    tables: dict[str, "FeatureTable"],
    label_columns: list[str],
    block: int,
    id_column: str,
    dataset_masks: bool,
    names: list[str],
) -> None:
    from cellpax.featuretable import _DEFAULT_MASK
    from cellpax.labels import LabelSet

    for label in label_columns:
        parts: list[LabelSet] = []
        confidences: list[pl.DataFrame] = []
        for position, (name, table) in enumerate(tables.items()):
            if label not in table.labels:
                continue
            source = table.labelset(label)
            new_ids = _new_ids_for(table, source.cell_ids, (position + 1) * block)
            mask = source.mask
            if mask in (None, _DEFAULT_MASK):
                mask = name if dataset_masks else None
            parts.append(
                LabelSet(new_ids, source.codes, meta=source.meta, name=label, mask=mask)
            )
            confidence = f"{label}_confidence"
            if confidence in table._df.columns:
                confidences.append(
                    pl.DataFrame(
                        {
                            id_column: _new_ids_for(
                                table,
                                table._df[table.id_column].to_numpy(),
                                (position + 1) * block,
                            ),
                            confidence: table._df[confidence].cast(pl.Float64),
                        },
                        schema={id_column: pl.Int64, confidence: pl.Float64},
                    )
                )
        combined = parts[0].combine(*parts[1:], mode="priority", name=label)
        joined.attach(combined)
        if confidences:
            joined._df = joined._df.join(
                pl.concat(confidences), on=id_column, how="left"
            )


def _new_ids_for(
    table: "FeatureTable", source_ids: np.ndarray, offset: int
) -> np.ndarray:
    """Joined ids for a source table's cells: ``offset`` plus each cell's row position."""
    position = {
        cell: row for row, cell in enumerate(table._df[table.id_column].to_list())
    }
    return np.asarray(
        [offset + position[cell] for cell in np.asarray(source_ids).tolist()],
        dtype=np.int64,
    )
