"""Compare clustering approaches — the capability dfc lacked (step 7).

Given two (or more) ``LabelSet``s over the same cells, quantify how they relate:
a contingency table, agreement metrics (ARI / NMI / FMI / Jaccard), an
alluvial/sankey-ready frame, and a pairwise agreement matrix across many.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl

from cellpax.labels import LabelSet


class Comparison:
    """A pairwise comparison of two label sets aligned on shared cells."""

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
        """Long-form cross-tab of cell counts per (a, b) label pair."""
        table = (
            self._aligned.group_by("a", "b")
            .len(name="n")
            .sort("a", "b", nulls_last=True)
        )
        if normalize:
            table = table.with_columns(
                (pl.col("n") / pl.col("n").sum()).alias("fraction")
            )
        return table

    def alluvial_frame(self) -> pl.DataFrame:
        """Contingency renamed to source/target/value for sankey/alluvial plots."""
        return self.contingency().rename({"a": "source", "b": "target", "n": "value"})

    def agreement(self) -> dict[str, float]:
        """ARI / NMI / FMI / Jaccard over cells assigned in both label sets."""
        from sklearn.metrics import (
            adjusted_rand_score,
            fowlkes_mallows_score,
            normalized_mutual_info_score,
        )
        from sklearn.metrics.cluster import pair_confusion_matrix

        both = self._aligned.filter((pl.col("a_id") >= 0) & (pl.col("b_id") >= 0))
        a = both["a_id"].to_numpy()
        b = both["b_id"].to_numpy()
        if len(a) == 0:
            raise ValueError("no cells are assigned in both label sets")
        (_, fp), (fn, tp) = pair_confusion_matrix(a, b)
        jaccard = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
        return {
            "ari": float(adjusted_rand_score(a, b)),
            "nmi": float(normalized_mutual_info_score(a, b)),
            "fmi": float(fowlkes_mallows_score(a, b)),
            "jaccard": jaccard,
            "n": int(len(a)),
        }


def compare(a: LabelSet, b: LabelSet) -> Comparison:
    """Compare two label sets over their shared cells."""
    return Comparison(a, b)


def compare_many(labels: Iterable[LabelSet], *, metric: str = "ari") -> pl.DataFrame:
    """Pairwise agreement matrix across label sets (default ARI).

    Returns a square frame with a leading ``label`` column and one column per
    label set, indexed by ``LabelSet.name``.
    """
    label_list = list(labels)
    names = [ls.name for ls in label_list]
    if len(set(names)) != len(names):
        raise ValueError("label sets must have distinct names for a matrix")
    size = len(label_list)
    matrix = np.eye(size)
    for i in range(size):
        for j in range(i + 1, size):
            score = compare(label_list[i], label_list[j]).agreement()[metric]
            matrix[i, j] = matrix[j, i] = score
    return pl.DataFrame(
        {"label": names, **{names[j]: matrix[:, j] for j in range(size)}}
    )
