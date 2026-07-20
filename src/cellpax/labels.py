"""LabelSet — first-class, well-defined cluster labels with clean relabeling.

Step 4 of the redesign (see DESIGN_PROPOSAL.md). A ``LabelSet`` gives clusters
identity — ``id -> (name, color, description)`` plus per-cell membership — and
clean relabeling verbs (``rename`` / ``merge`` / ``reorder`` / ``set_colors`` /
``combine``), replacing dfc's scattered ``add_label`` / ``add_label_names`` /
``combine_labels`` column juggling. Un-versioned and mutable-in-place (verbs
return ``self`` for chaining); no decision ledger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

_UNASSIGNED = -1


def _enum_member(name: str) -> str:
    """Coerce a label name into a valid IntEnum member identifier."""
    member = re.sub(r"\W+", "_", name).strip("_")
    if not member or member[0].isdigit():
        member = f"_{member}"
    return member


@dataclass(frozen=True)
class Label:
    """Identity and display metadata for one cluster."""

    id: int
    name: str
    color: str | None = None
    description: str | None = None


class LabelSet:
    """Per-cell cluster labels with named identity and relabeling verbs.

    Parameters
    ----------
    cell_ids:
        Cell ids this label set covers (one per label).
    labels:
        Integer cluster id per cell; ``-1`` means unassigned.
    name:
        Column name used when attaching to a table. Default ``"label"``.
    """

    def __init__(
        self,
        cell_ids: Sequence[int] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        meta: Mapping[int, Label] | None = None,
        name: str = "label",
    ) -> None:
        self._cell_ids = np.asarray(cell_ids)
        self._labels = np.asarray(labels, dtype=np.int64)
        if self._cell_ids.shape[0] != self._labels.shape[0]:
            raise ValueError("cell_ids and labels must have the same length")
        self.name = name
        self._meta: dict[int, Label] = {
            i: Label(id=i, name=str(i)) for i in self._unique_ids()
        }
        if meta:
            self._meta.update(meta)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_clustering(
        cls,
        similarity: object,
        cell_ids: Sequence[int] | np.ndarray,
        *,
        distance_threshold: float,
        min_cluster_size: int = 1,
        name: str = "label",
    ) -> "LabelSet":
        """Cut a ``SimilarityMatrix`` at a threshold into a LabelSet.

        Cluster ids are renumbered to contiguous ``0..k-1`` (in ascending order of
        the cut's raw labels); unassigned cells stay ``-1``.
        """
        raw = similarity.cluster_labels(  # type: ignore[attr-defined]
            distance_threshold, min_cluster_size=min_cluster_size
        )
        remap = {
            old: new
            for new, old in enumerate(
                sorted(int(v) for v in np.unique(raw) if v != _UNASSIGNED)
            )
        }
        labels = np.array([remap.get(int(v), _UNASSIGNED) for v in raw], dtype=np.int64)
        return cls(cell_ids, labels, name=name)

    # -- accessors -------------------------------------------------------------

    def _unique_ids(self) -> list[int]:
        return [int(i) for i in np.unique(self._labels) if int(i) != _UNASSIGNED]

    @property
    def ids(self) -> list[int]:
        """Sorted cluster ids (excluding unassigned)."""
        return self._unique_ids()

    @property
    def names(self) -> list[str]:
        """Cluster names in id order."""
        return [self._meta[i].name for i in self.ids]

    @property
    def cell_ids(self) -> np.ndarray:
        return self._cell_ids

    def label(self, key: int | str) -> Label:
        """Look up a cluster by id or name."""
        return self._meta[self._resolve(key)]

    def counts(self) -> dict[str, int]:
        """Cell count per cluster name."""
        values, counts = np.unique(self._labels, return_counts=True)
        return {
            self._meta[int(v)].name: int(c)
            for v, c in zip(values, counts)
            if int(v) != _UNASSIGNED
        }

    def to_frame(self, *, id_column: str = "cell_id") -> pl.DataFrame:
        """Return ``cell_id`` + name + id columns (unassigned → null name)."""
        names = [
            None if int(v) == _UNASSIGNED else self._meta[int(v)].name
            for v in self._labels
        ]
        return pl.DataFrame(
            {
                id_column: self._cell_ids,
                self.name: names,
                f"{self.name}_id": self._labels,
            }
        )

    # -- relabeling verbs (mutate in place, return self) -----------------------

    def _resolve(self, key: int | str) -> int:
        if isinstance(key, str):
            for i, meta in self._meta.items():
                if meta.name == key:
                    return i
            raise KeyError(f"Unknown label name {key!r}")
        if key not in self._meta:
            raise KeyError(f"Unknown label id {key!r}")
        return key

    def rename(self, mapping: Mapping[int | str, str]) -> "LabelSet":
        """Rename clusters: ``{id_or_name: new_name}``."""
        for key, new_name in mapping.items():
            i = self._resolve(key)
            self._meta[i] = replace(self._meta[i], name=new_name)
        return self

    def set_colors(self, mapping: Mapping[int | str, str]) -> "LabelSet":
        """Set cluster colors: ``{id_or_name: color}``."""
        for key, color in mapping.items():
            i = self._resolve(key)
            self._meta[i] = replace(self._meta[i], color=color)
        return self

    def merge(self, members: Iterable[int | str], *, into: str) -> "LabelSet":
        """Merge clusters into one, named ``into``."""
        ids = [self._resolve(m) for m in members]
        if len(ids) < 2:
            raise ValueError("merge requires at least two clusters")
        target = min(ids)
        for i in ids:
            if i != target:
                self._labels[self._labels == i] = target
                self._meta.pop(i, None)
        self._meta[target] = replace(self._meta[target], name=into)
        return self

    def reorder(self, order: Sequence[int | str]) -> "LabelSet":
        """Renumber cluster ids to a new order (0..k-1) given names or ids."""
        resolved = [self._resolve(k) for k in order]
        if set(resolved) != set(self.ids):
            raise ValueError("reorder must list every cluster exactly once")
        remap = {old: new for new, old in enumerate(resolved)}
        new_labels = self._labels.copy()
        for old, new in remap.items():
            new_labels[self._labels == old] = new
        self._labels = new_labels
        self._meta = {
            remap[old]: replace(meta, id=remap[old]) for old, meta in self._meta.items()
        }
        return self

    def combine(self, other: "LabelSet", *, name: str | None = None) -> "LabelSet":
        """Union two LabelSets over disjoint cells into a new one."""
        if set(self._cell_ids.tolist()) & set(other._cell_ids.tolist()):
            raise ValueError("combine requires disjoint cell sets")
        offset = (max(self.ids) + 1) if self.ids else 0
        shifted = np.where(
            other._labels == _UNASSIGNED, _UNASSIGNED, other._labels + offset
        )
        meta = dict(self._meta)
        for i, m in other._meta.items():
            meta[i + offset] = replace(m, id=i + offset)
        return LabelSet(
            np.concatenate([self._cell_ids, other._cell_ids]),
            np.concatenate([self._labels, shifted]),
            meta=meta,
            name=name or self.name,
        )

    # -- IntEnum bindings ------------------------------------------------------

    def to_enum(self, class_name: str = "Labels") -> type[IntEnum]:
        """Generate an ``IntEnum`` of this label set: ``member name -> cluster id``.

        Lets you filter and compare without remembering numbers or exact strings,
        with editor autocomplete — e.g. ``df.filter(pl.col("label_id") == L.L5IT)``
        (IntEnum members compare equal to their integer id).
        """
        members: dict[str, int] = {}
        for i in self.ids:
            member = _enum_member(self._meta[i].name)
            if member in members:
                raise ValueError(f"Label names collide as enum member {member!r}")
            members[member] = i
        return IntEnum(class_name, members)

    def apply_enum(self, enum: type[IntEnum]) -> "LabelSet":
        """Name clusters from a user-defined ``IntEnum`` (``member.value`` → id).

        Define e.g. ``class ITLabels(IntEnum): L5IT = 0; L23IT = 1`` and call
        ``labels.apply_enum(ITLabels)`` to name cluster 0 ``"L5IT"``, 1 ``"L23IT"``.
        Members without a matching cluster id are ignored.
        """
        for member in enum:
            if int(member) in self._meta:
                self._meta[int(member)] = replace(
                    self._meta[int(member)], name=member.name
                )
        return self

    def __repr__(self) -> str:
        return (
            f"LabelSet(name={self.name!r}, n_cells={len(self._cell_ids)}, "
            f"clusters={self.names})"
        )
