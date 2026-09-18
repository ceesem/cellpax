"""Compare clustering approaches — the capability dfc lacked (step 7).

Given two (or more) ``LabelSet``s over the same cells, quantify how they relate:
a contingency table, agreement metrics (ARI / NMI / FMI / Jaccard) with coverage
counts, an alluvial/sankey-ready frame, and a pairwise agreement matrix across many.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl

from cellpax.labels import LabelSet

_UNASSIGNED = -1

#: Metrics ``compare_many`` can tabulate — the symmetric keys of ``agreement()``.
_MATRIX_METRICS = ("ari", "nmi", "fmi", "jaccard", "coverage", "n")


def _display_names(labels: LabelSet) -> dict[int, str]:
    """Cluster id -> name, with the id appended where several clusters share a name."""
    holders: dict[str, list[int]] = {}
    for i, identity in labels.meta.items():
        holders.setdefault(identity.name, []).append(i)
    return {
        i: identity.name
        if len(holders[identity.name]) == 1
        else f"{identity.name} (id {i})"
        for i, identity in labels.meta.items()
    }


class Comparison:
    """A pairwise comparison of two label sets aligned on shared cells.

    Parameters
    ----------
    a, b : LabelSet
        Label sets to align by cell identifier. They need not cover identical
        populations.

    Raises
    ------
    ValueError
        If the label sets share no cells.
    """

    def __init__(self, a: LabelSet, b: LabelSet) -> None:
        self._a = a
        self._b = b
        fa = a.to_frame().rename({a.name: "a", f"{a.name}_id": "a_id"})
        fb = b.to_frame().rename({b.name: "b", f"{b.name}_id": "b_id"})
        self._aligned = fa.select("cell_id", "a", "a_id").join(
            fb.select("cell_id", "b", "b_id"), on="cell_id", how="inner"
        )
        if self._aligned.height == 0:
            raise ValueError("label sets share no cells to compare")

    def contingency(self, *, normalize: bool = False) -> pl.DataFrame:
        """Long-form cross-tab of cell counts per (a, b) cluster pair.

        Keyed on cluster *ids*, so two clusters that happen to share a name stay
        two rows — the name columns show the id alongside where that happens.
        Unassigned cells (``-1``) get their own rows with a null name, sorted
        last. ``normalize`` adds a ``fraction`` column over the shared cells.

        Parameters
        ----------
        normalize : bool, default False
            Add each pair's fraction of all shared cells.

        Returns
        -------
        polars.DataFrame
            Long-form cross-tab with cluster ids, display names, counts, and
            optional fractions.
        """
        a_names, b_names = _display_names(self._a), _display_names(self._b)
        table = (
            self._aligned.group_by("a_id", "b_id")
            .len(name="n")
            .sort(
                pl.col("a_id") == _UNASSIGNED,
                "a_id",
                pl.col("b_id") == _UNASSIGNED,
                "b_id",
            )
        )
        table = table.with_columns(
            pl.Series(
                "a", [a_names.get(int(i)) for i in table["a_id"]], dtype=pl.String
            ),
            pl.Series(
                "b", [b_names.get(int(i)) for i in table["b_id"]], dtype=pl.String
            ),
        ).select("a", "a_id", "b", "b_id", "n")
        if normalize:
            table = table.with_columns(
                (pl.col("n") / pl.col("n").sum()).alias("fraction")
            )
        return table

    def alluvial_frame(self) -> pl.DataFrame:
        """Contingency renamed to source/target/value for sankey/alluvial plots.

        Keyed on ids like ``contingency``, with names disambiguated the same way,
        so clusters sharing a name stay separate flows in the plot.

        Returns
        -------
        polars.DataFrame
            Source id/name, target id/name, and cell count per flow.
        """
        return self.contingency().rename(
            {
                "a": "source",
                "a_id": "source_id",
                "b": "target",
                "b_id": "target_id",
                "n": "value",
            }
        )

    def agreement(self) -> dict[str, float | int]:
        """ARI / NMI / FMI / Jaccard over cells assigned in both label sets.

        The metrics see only the co-assigned cells, so the coverage keys say how
        much that is: ``n`` counts the co-assigned cells the metrics were computed
        on, ``n_a_assigned`` / ``n_b_assigned`` the shared cells each side
        assigned at all, and ``coverage`` is co-assigned over the union of
        assigned — 1.0 exactly when the two sets assign the same cells, so a high
        ARI over a sliver of the cells can't pass for agreement.

        Returns
        -------
        dict
            Agreement scores and assignment-coverage counts.
        """
        from sklearn.metrics import (
            adjusted_rand_score,
            fowlkes_mallows_score,
            normalized_mutual_info_score,
        )
        from sklearn.metrics.cluster import pair_confusion_matrix

        a_assigned = self._aligned["a_id"].to_numpy() >= 0
        b_assigned = self._aligned["b_id"].to_numpy() >= 0
        both = self._aligned.filter((pl.col("a_id") >= 0) & (pl.col("b_id") >= 0))
        a = both["a_id"].to_numpy()
        b = both["b_id"].to_numpy()
        if len(a) == 0:
            raise ValueError("no cells are assigned in both label sets")
        (_, fp), (fn, tp) = pair_confusion_matrix(a, b)
        jaccard = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
        union = int((a_assigned | b_assigned).sum())
        return {
            "ari": float(adjusted_rand_score(a, b)),
            "nmi": float(normalized_mutual_info_score(a, b)),
            "fmi": float(fowlkes_mallows_score(a, b)),
            "jaccard": jaccard,
            "n": int(len(a)),
            "n_a_assigned": int(a_assigned.sum()),
            "n_b_assigned": int(b_assigned.sum()),
            "coverage": float(len(a) / union) if union else float("nan"),
        }


def compare(a: LabelSet, b: LabelSet) -> Comparison:
    """Compare two label sets over their shared cells.

    Parameters
    ----------
    a, b : LabelSet
        Label sets to align by cell identifier.

    Returns
    -------
    Comparison
        Lazy comparison exposing contingency, alluvial, and agreement views.
    """
    return Comparison(a, b)


def compare_many(labels: Iterable[LabelSet], *, metric: str = "ari") -> pl.DataFrame:
    """Pairwise agreement matrix across label sets (default ARI).

    Returns a square frame with a leading ``label`` column and one column per
    label set, indexed by ``LabelSet.name``. The diagonal holds each set's
    self-comparison value: 1.0 for the agreement metrics and ``coverage``, and
    the set's own assigned-cell count for ``metric="n"``.

    Parameters
    ----------
    labels : iterable of LabelSet
        Label sets with distinct names.
    metric : {'ari', 'nmi', 'fmi', 'jaccard', 'coverage', 'n'}, default 'ari'
        Symmetric comparison value placed in the matrix.

    Returns
    -------
    polars.DataFrame
        Square comparison matrix with a leading ``label`` column.

    Raises
    ------
    ValueError
        If ``metric`` is unknown or label-set names are not unique.
    """
    if metric not in _MATRIX_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; valid metrics: {list(_MATRIX_METRICS)}"
        )
    label_list = list(labels)
    names = [ls.name for ls in label_list]
    if len(set(names)) != len(names):
        raise ValueError("label sets must have distinct names for a matrix")
    size = len(label_list)
    matrix = np.empty((size, size), dtype=float)
    for i, ls in enumerate(label_list):
        matrix[i, i] = float(ls.assigned.sum()) if metric == "n" else 1.0
    for i in range(size):
        for j in range(i + 1, size):
            score = compare(label_list[i], label_list[j]).agreement()[metric]
            matrix[i, j] = matrix[j, i] = score
    return pl.DataFrame(
        {"label": names, **{names[j]: matrix[:, j] for j in range(size)}}
    )
