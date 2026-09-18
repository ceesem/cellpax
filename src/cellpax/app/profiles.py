"""What the clusters are made of — the views that decide whether a cut is real.

The scan says how many clusters a threshold gives and the flow says how they
relate, but neither says what any of them *is*. In a notebook that answer comes
from a handful of plots you rebuild at every level: depth by cluster, a feature
heatmap, a feature against depth. Those are the same three questions every time,
so they belong in the tool rather than in a pasted cell.

Everything here reads off the cached cut and a column of the table, so it is
free in the sense that matters: no linkage, no consensus, no refit. The
computation is a group-by.

Rendering stays in the client. These functions emit binned counts and matrices,
not pictures, which is the same contract cellpax itself keeps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

#: Columns that are ids or bookkeeping rather than measurements.
_ID_HINTS = ("_id", "id", "index", "idx")


def numeric_columns(table: Any, mask: str | None) -> list[dict[str, Any]]:
    """Numeric metadata columns worth plotting against, with their ranges.

    Feature columns are excluded: they get the heatmap, and a picker holding
    six hundred morphometrics is not a picker. What is left is the metadata a
    cluster is interpreted *against* — depth, position, completeness.
    """
    frame = table.dataframe(mask)
    features = set(table.feature_columns)
    out: list[dict[str, Any]] = []
    for name, dtype in zip(frame.columns, frame.dtypes):
        if name in features or name == table.id_column:
            continue
        if not dtype.is_numeric():
            continue
        if name.startswith("_"):
            continue
        series = frame[name].drop_nulls()
        if series.len() == 0 or series.n_unique() < 3:
            continue
        out.append(
            {
                "name": name,
                "min": float(series.min()),
                "max": float(series.max()),
                "n_unique": int(series.n_unique()),
                "looks_like_id": any(name.lower().endswith(h) for h in _ID_HINTS),
            }
        )
    return sorted(out, key=lambda c: (c["looks_like_id"], c["name"]))


def column_profile(
    table: Any,
    mask: str | None,
    cell_ids: np.ndarray,
    codes: np.ndarray,
    column: str,
    *,
    bins: int = 40,
    names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """One column's distribution per cluster, on a shared binning.

    The shared binning is the point. Per-cluster histograms on their own scales
    are unreadable side by side, and the question this view answers — do these
    clusters occupy different depths — is entirely about where they sit relative
    to each other.

    Counts come back raw *and* normalized: raw shows which clusters dominate,
    normalized shows shape, and a small cluster with a sharp laminar position is
    invisible in the first and obvious in the second.
    """
    frame = table.dataframe(mask)
    if column not in frame.columns:
        raise ValueError(f"no column {column!r} in this mask")
    values = _aligned(frame, table.id_column, cell_ids, column)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"column {column!r} has no finite values here")

    lo, hi = float(np.min(values[finite])), float(np.max(values[finite]))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    series: list[dict[str, Any]] = []
    for cluster in _cluster_ids(codes):
        member = (codes == cluster) & finite
        counts, _ = np.histogram(values[member], bins=edges)
        total = int(counts.sum())
        series.append(
            {
                "cluster": int(cluster),
                "name": (names or {}).get(int(cluster)) or _default_name(cluster),
                "n_cells": total,
                "counts": [int(c) for c in counts],
                "fraction": [float(c / total) if total else 0.0 for c in counts],
                "median": float(np.median(values[member])) if total else None,
                "q1": float(np.quantile(values[member], 0.25)) if total else None,
                "q3": float(np.quantile(values[member], 0.75)) if total else None,
            }
        )
    return {
        "column": column,
        "lo": lo,
        "hi": hi,
        "centers": [float(c) for c in centers],
        "series": series,
    }


def cluster_heatmap(
    table: Any,
    mask: str | None,
    cell_ids: np.ndarray,
    codes: np.ndarray,
    *,
    columns: str | None = None,
    top: int = 40,
    names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Z-scored feature means per cluster, most discriminative features first.

    Ordered by a one-way F statistic across clusters rather than by variance or
    alphabetically: the question is which features *separate these clusters*,
    and a feature can be highly variable while saying nothing about the split.

    Values are z-scored per feature over the masked cells, so a row is
    comparable left to right and the colour scale means the same thing
    everywhere. Reported as standard deviations from the cohort mean — which is
    what makes ``+2`` readable without a legend lookup.
    """
    scaled = table.features(mask, scaled=True, columns=columns)
    feature_names = _feature_names(table, columns)
    if scaled.shape[1] != len(feature_names):
        feature_names = [f"f{i}" for i in range(scaled.shape[1])]

    clusters = _cluster_ids(codes)
    if len(clusters) < 2:
        return {
            "features": [],
            "clusters": [],
            "matrix": [],
            "reason": "need two clusters",
        }

    # z-score per feature over the cohort, so the scale is shared
    mean = scaled.mean(axis=0)
    sd = scaled.std(axis=0)
    sd[sd == 0] = 1.0
    z = (scaled - mean) / sd

    groups = [z[codes == c] for c in clusters]
    sizes = np.array([g.shape[0] for g in groups], dtype=float)
    means = np.vstack([g.mean(axis=0) for g in groups])

    # one-way F across clusters: between-group spread over within-group spread
    grand = z.mean(axis=0)
    between = (sizes[:, None] * (means - grand) ** 2).sum(axis=0) / max(
        len(clusters) - 1, 1
    )
    within = np.vstack([((g - m) ** 2).sum(axis=0) for g, m in zip(groups, means)]).sum(
        axis=0
    ) / max(z.shape[0] - len(clusters), 1)
    within[within == 0] = np.finfo(float).eps
    f_stat = between / within

    order = np.argsort(f_stat)[::-1][: int(top)]
    return {
        "features": [feature_names[i] for i in order],
        "f_stat": [float(f_stat[i]) for i in order],
        "clusters": [
            {
                "cluster": int(c),
                "name": (names or {}).get(int(c)) or _default_name(c),
                "n_cells": int(size),
            }
            for c, size in zip(clusters, sizes)
        ],
        # rows = clusters, cols = features, in the orders above
        "matrix": [[float(means[r, i]) for i in order] for r in range(means.shape[0])],
        "vmax": float(np.abs(means[:, order]).max()) if order.size else 1.0,
    }


def feature_scatter(
    table: Any,
    mask: str | None,
    cell_ids: np.ndarray,
    codes: np.ndarray,
    *,
    feature: str,
    against: str,
    scaled: bool = False,
) -> dict[str, Any]:
    """One feature against one metadata column, per cell, coloured by cluster.

    The follow-through on the heatmap: a cluster mean says the feature differs,
    and this says whether it differs as a shift, a spread, or a gradient along
    depth — which are three different claims about what the cluster is.
    """
    frame = table.dataframe(mask, scaled=scaled)
    for column in (feature, against):
        if column not in frame.columns:
            raise ValueError(f"no column {column!r} in this mask")
    x = _aligned(frame, table.id_column, cell_ids, against)
    y = _aligned(frame, table.id_column, cell_ids, feature)
    good = np.isfinite(x) & np.isfinite(y)
    return {
        "feature": feature,
        "against": against,
        "scaled": bool(scaled),
        "x": [float(v) for v in x[good]],
        "y": [float(v) for v in y[good]],
        "cluster": [int(v) for v in codes[good]],
        "n_dropped": int((~good).sum()),
    }


# -- helpers -------------------------------------------------------------------


def _aligned(
    frame: pl.DataFrame, id_column: str, cell_ids: np.ndarray, column: str
) -> np.ndarray:
    """Column values in ``cell_ids`` order.

    A join, not a positional slice: the clustering's cell order is its own and
    need not match the frame's, and lining them up by position would silently
    scramble every plot on this page.
    """
    lookup = pl.DataFrame({id_column: np.asarray(cell_ids)}).join(
        frame.select([id_column, column]), on=id_column, how="left"
    )
    return lookup[column].cast(pl.Float64).to_numpy()


def _cluster_ids(codes: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(codes) if int(c) >= 0]


def _default_name(cluster: int) -> str:
    return f"cluster {int(cluster)}"


def _feature_names(table: Any, columns: str | None) -> list[str]:
    if columns is None:
        return list(table.feature_columns)
    try:
        return list(table.collections[columns].columns)
    except (KeyError, TypeError):
        return list(table.feature_columns)
