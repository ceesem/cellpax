"""LabelSet — first-class, well-defined cluster labels with clean relabeling.

Step 4 of the redesign (see DESIGN_PROPOSAL.md). A ``LabelSet`` gives clusters
identity — ``id -> (name, color, description)`` plus per-cell membership — and
clean relabeling verbs (``rename`` / ``merge`` / ``reorder`` / ``reorder_by`` /
``compact`` / ``set_colors`` / ``combine``), replacing dfc's scattered ``add_label`` /
``add_label_names`` / ``combine_labels`` column juggling. Un-versioned and
mutable-in-place (verbs return ``self`` for chaining); no decision ledger.

It also carries the per-cell arrays models want — ``codes`` / ``codes_for`` on the
way out, ``decode`` / ``with_codes`` on the way back — so a LabelSet round-trips
through a classifier without anyone hand-maintaining an id-to-name dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Iterable, Literal, Mapping, Sequence

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
    names:
        Cluster names, as a list in cluster-id order or a ``{id: name}`` mapping.
        Without them clusters are named after their ids (``"0"``, ``"1"``, …) and
        can be named later with ``rename``.
    meta:
        Full :class:`Label` records per cluster id, for colors and descriptions
        too; ``names`` wins over a name given here. Mostly for round-tripping
        rather than hand-written — see also ``set_colors`` / ``set_descriptions``.
    name:
        Column name used when attaching to a table. Default ``"label"``.
    mask:
        Name of the mask this label set was computed on, for provenance/display
        only (e.g. shown in ``repr``) — not used for alignment or validation.
    """

    def __init__(
        self,
        cell_ids: Sequence[int] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        names: Sequence[str] | Mapping[int, str] | None = None,
        meta: Mapping[int, Label] | None = None,
        name: str = "label",
        mask: str | None = None,
    ) -> None:
        self._cell_ids = np.asarray(cell_ids)
        self._labels = np.asarray(labels, dtype=np.int64)
        if self._cell_ids.shape[0] != self._labels.shape[0]:
            raise ValueError("cell_ids and labels must have the same length")
        if np.unique(self._cell_ids).shape[0] != self._cell_ids.shape[0]:
            raise ValueError("cell_ids must be unique")
        self.name = name
        self.mask = mask
        self._meta: dict[int, Label] = {
            i: Label(id=i, name=str(i)) for i in self._unique_ids()
        }
        if meta:
            self._meta.update(meta)
        if names is not None:
            self.rename(names)

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
        mask: str | None = None,
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
        return cls(cell_ids, labels, name=name, mask=mask)

    @classmethod
    def from_labels(
        cls,
        cell_ids: Sequence[int] | np.ndarray,
        values: Sequence[Any] | np.ndarray,
        *,
        unassigned: Any = None,
        name: str = "label",
        mask: str | None = None,
    ) -> "LabelSet":
        """Build a LabelSet directly from human-readable label values.

        ``values`` are arbitrary hashable labels (e.g. cluster type names)
        rather than pre-factorized integer ids. Unique non-unassigned values
        are sorted and assigned ids ``0..k-1``; each value becomes its
        cluster's initial name, so no separate ``.rename()`` pass is needed.
        ``None``/NaN entries (and any value equal to ``unassigned``, if given)
        map to id ``-1``.
        """

        def is_unassigned(v: Any) -> bool:
            if v is None:
                return True
            if isinstance(v, float) and np.isnan(v):
                return True
            return unassigned is not None and v == unassigned

        distinct = {v for v in values if not is_unassigned(v)}
        try:
            uniques = sorted(distinct)
        except TypeError:  # mixed types (e.g. str and int in one column)
            uniques = sorted(distinct, key=str)
        id_map = {v: i for i, v in enumerate(uniques)}
        labels = np.array(
            [_UNASSIGNED if is_unassigned(v) else id_map[v] for v in values],
            dtype=np.int64,
        )
        meta = {i: Label(id=i, name=str(v)) for v, i in id_map.items()}
        return cls(cell_ids, labels, meta=meta, name=name, mask=mask)

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
    def n_clusters(self) -> int:
        """Number of clusters (excluding unassigned)."""
        return len(self.ids)

    @property
    def meta(self) -> dict[int, Label]:
        """``{id: Label}`` for every known cluster — the identities, as a copy.

        Round-trips through the ``meta=`` constructor argument, which is how
        ``FeatureTable.attach`` keeps colors and descriptions that a plain table
        column can't hold.
        """
        return dict(self._meta)

    @property
    def cell_ids(self) -> np.ndarray:
        return self._cell_ids

    def __len__(self) -> int:
        """Number of cells covered, assigned or not."""
        return int(self._cell_ids.shape[0])

    def cluster(self, key: int | str) -> Label:
        """One cluster's :class:`Label` — its id, name, color and description.

        ``key`` is either the cluster id or its current name; this is how you
        read a color or description back out. ``catalog`` is the same thing for
        every cluster at once, as a dataframe.
        """
        return self._meta[self._resolve(key)]

    def counts(self) -> dict[str, int]:
        """Cell count per cluster name (clusters sharing a name are summed)."""
        values, counts = np.unique(self._labels, return_counts=True)
        out: dict[str, int] = {}
        for value, count in zip(values, counts):
            if int(value) == _UNASSIGNED:
                continue
            name = self._meta[int(value)].name
            out[name] = out.get(name, 0) + int(count)
        return out

    def catalog(self) -> pl.DataFrame:
        """One row per cluster: ``id`` / ``name`` / ``color`` / ``description`` / ``n_cells``.

        The per-cluster counterpart of ``to_frame``'s per-cell view — a legend
        table, and unlike ``counts`` it keeps clusters that share a name apart.
        """
        values, counts = np.unique(self._labels, return_counts=True)
        sizes = {int(v): int(c) for v, c in zip(values, counts)}
        return pl.DataFrame(
            [
                {
                    "id": i,
                    "name": self._meta[i].name,
                    "color": self._meta[i].color,
                    "description": self._meta[i].description,
                    "n_cells": sizes.get(i, 0),
                }
                for i in self.ids
            ],
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "color": pl.String,
                "description": pl.String,
                "n_cells": pl.Int64,
            },
            orient="row",
        )

    def color_map(self) -> dict[str, str]:
        """``{name: color}`` for clusters with a color, e.g. seaborn's ``palette=``."""
        return {
            self._meta[i].name: color
            for i in self.ids
            if (color := self._meta[i].color) is not None
        }

    def to_frame(self, *, id_column: str = "cell_id") -> pl.DataFrame:
        """Return ``cell_id`` + name + id columns (unassigned → null name)."""
        return pl.DataFrame(
            {
                id_column: self._cell_ids,
                self.name: self.to_names(),
                f"{self.name}_id": self._labels,
            }
        )

    # -- per-cell arrays (model fitting) ---------------------------------------

    @property
    def codes(self) -> np.ndarray:
        """Per-cell integer cluster ids in ``cell_ids`` order (``-1`` unassigned).

        The ``y`` for a classifier. A copy, so relabeling verbs can't mutate it
        underneath you; use ``codes_for`` when your rows are in another order.
        """
        return self._labels.copy()

    @property
    def assigned(self) -> np.ndarray:
        """Boolean per-cell mask of cells belonging to some cluster."""
        return self._labels != _UNASSIGNED

    @property
    def n_unassigned(self) -> int:
        """How many cells are unassigned (``-1``)."""
        return int((~self.assigned).sum())

    def to_names(self) -> list[str | None]:
        """Per-cell cluster names in ``cell_ids`` order (unassigned → ``None``)."""
        return self.decode(self._labels)

    def decode(self, codes: Sequence[int] | np.ndarray) -> list[str | None]:
        """Map integer cluster ids back to names — e.g. a classifier's output.

        ``-1`` becomes ``None``; any other id without a cluster raises.
        """
        arr = np.asarray(codes, dtype=np.int64).reshape(-1)
        self._check_codes(arr)
        return [None if int(c) == _UNASSIGNED else self._meta[int(c)].name for c in arr]

    def codes_for(
        self,
        cell_ids: Sequence[int] | np.ndarray,
        *,
        missing: int = _UNASSIGNED,
    ) -> np.ndarray:
        """Integer cluster ids aligned to an arbitrary ``cell_ids`` order.

        Use this to line a ``y`` vector up with feature rows you got elsewhere
        (e.g. ``ft.features(mask)``) instead of trusting that both are in the
        same order. Cells this label set doesn't cover get ``missing``.
        """
        by_cell = dict(
            zip((int(c) for c in self._cell_ids), (int(v) for v in self._labels))
        )
        return np.array(
            [by_cell.get(int(c), missing) for c in np.asarray(cell_ids).reshape(-1)],
            dtype=np.int64,
        )

    def _check_codes(self, codes: np.ndarray) -> None:
        unknown = sorted({int(c) for c in codes} - set(self._meta) - {_UNASSIGNED})
        if unknown:
            raise ValueError(f"no cluster for ids {unknown}; known: {self.ids}")

    # -- derived label sets (new objects, not in-place) -------------------------

    def copy(self, *, name: str | None = None) -> "LabelSet":
        """An independent copy — relabeling verbs mutate in place, so branch here."""
        return LabelSet(
            self._cell_ids.copy(),
            self._labels.copy(),
            meta=dict(self._meta),
            name=name or self.name,
            mask=self.mask,
        )

    def subset(
        self, cell_ids: Sequence[int] | np.ndarray, *, name: str | None = None
    ) -> "LabelSet":
        """A new LabelSet over a subset of these cells, in the given order.

        Cluster ids, names and colors are preserved (so codes stay comparable
        with the parent's) — handy for train/test splits.
        """
        index = {int(c): i for i, c in enumerate(self._cell_ids)}
        wanted = [int(c) for c in np.asarray(cell_ids).reshape(-1)]
        absent = [c for c in wanted if c not in index]
        if absent:
            raise KeyError(
                f"{len(absent)} cell_ids are not in this label set, e.g. {absent[:5]}"
            )
        rows = [index[c] for c in wanted]
        return LabelSet(
            self._cell_ids[rows],
            self._labels[rows],
            meta=dict(self._meta),
            name=name or self.name,
            mask=self.mask,
        )

    def drop_unassigned(self, *, name: str | None = None) -> "LabelSet":
        """A new LabelSet without the unassigned (``-1``) cells, for fitting."""
        return self.subset(self._cell_ids[self.assigned], name=name)

    def with_codes(
        self,
        cell_ids: Sequence[int] | np.ndarray,
        codes: Sequence[int] | np.ndarray,
        *,
        name: str | None = None,
        mask: str | None = None,
    ) -> "LabelSet":
        """A new LabelSet over other cells reusing this one's cluster identities.

        The inverse of ``codes_for``: hand back a classifier's predicted integer
        ids and get a LabelSet whose names and colors match the one the model was
        trained on, ready for ``ft.attach``. Ids with no cluster here raise, which
        catches an off-by-one ``num_class`` before it becomes a mislabeled column.
        """
        arr = np.asarray(codes, dtype=np.int64).reshape(-1)
        self._check_codes(arr)
        return LabelSet(
            cell_ids, arr, meta=dict(self._meta), name=name or self.name, mask=mask
        )

    # -- relabeling verbs (mutate in place, return self) -----------------------

    def _resolve(self, key: int | str) -> int:
        if isinstance(key, str):
            matches = sorted(i for i, meta in self._meta.items() if meta.name == key)
            if not matches:
                raise KeyError(f"Unknown label name {key!r}")
            if len(matches) > 1:
                raise ValueError(
                    f"Label name {key!r} is ambiguous (ids {matches}); use an id"
                )
            return matches[0]
        if key not in self._meta:
            raise KeyError(f"Unknown label id {key!r}")
        return key

    def _name_mapping(
        self, names: Mapping[int | str, str] | Sequence[str]
    ) -> dict[int, str]:
        """Coerce ``{id_or_name: new}`` or a list in id order into ``{id: new}``."""
        if isinstance(names, str):
            raise TypeError("names must be a list of names or a mapping, not a string")
        if isinstance(names, Mapping):
            return {self._resolve(key): value for key, value in names.items()}
        ordered = list(names)
        ids = self.ids
        if len(ordered) != len(ids):
            raise ValueError(
                f"expected {len(ids)} names for clusters {ids}, got {len(ordered)}"
            )
        return dict(zip(ids, ordered))

    def rename(self, names: Mapping[int | str, str] | Sequence[str]) -> "LabelSet":
        """Rename clusters, by ``{id_or_name: new_name}`` or a list in id order."""
        for i, new_name in self._name_mapping(names).items():
            self._meta[i] = replace(self._meta[i], name=new_name)
        return self

    def set_colors(self, mapping: Mapping[int | str, str]) -> "LabelSet":
        """Set cluster colors: ``{id_or_name: color}``."""
        for key, color in mapping.items():
            i = self._resolve(key)
            self._meta[i] = replace(self._meta[i], color=color)
        return self

    def set_descriptions(self, mapping: Mapping[int | str, str]) -> "LabelSet":
        """Set cluster descriptions: ``{id_or_name: description}``."""
        for key, description in mapping.items():
            i = self._resolve(key)
            self._meta[i] = replace(self._meta[i], description=description)
        return self

    def merge(self, members: Iterable[int | str], *, into: str) -> "LabelSet":
        """Merge clusters into one, named ``into``.

        The surviving id is the lowest of ``members``, so ids are left with gaps;
        call ``compact`` if you need contiguous ``0..k-1`` codes again.
        """
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

    def unassign(self, members: int | str | Iterable[int | str]) -> "LabelSet":
        """Send one or more clusters' cells back to unassigned (``-1``).

        For discarding a junk cluster — the cells stay in the label set with no
        cluster, so a following ``propagate_labels`` refills them from their
        neighbors instead of leaving a hole. Ids are left with gaps; ``compact``
        closes them.
        """
        keys = [members] if isinstance(members, (int, str)) else list(members)
        for i in [self._resolve(key) for key in keys]:
            self._labels[self._labels == i] = _UNASSIGNED
            self._meta.pop(i, None)
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
            remap[old]: replace(meta, id=remap[old])
            for old, meta in self._meta.items()
            if old in remap  # drop metadata for clusters with no cells
        }
        return self

    def compact(self) -> "LabelSet":
        """Renumber cluster ids to contiguous ``0..k-1``, keeping the current order.

        ``merge`` leaves gaps in the ids; multi-class fitting (xgboost's
        ``num_class``, sklearn's ``classes_``) wants them contiguous.
        """
        return self.reorder(self.ids)

    def reorder_by(
        self,
        values: Mapping[int, float] | Sequence[float] | np.ndarray,
        *,
        cell_ids: Sequence[int] | np.ndarray | None = None,
        agg: Literal["mean", "median"] = "mean",
        ascending: bool = True,
    ) -> "LabelSet":
        """Reorder clusters by an aggregate of arbitrary per-cell ``values``.

        ``values`` is either a ``{cell_id: value}`` mapping, or a plain array/
        sequence aligned to ``cell_ids`` (defaults to this LabelSet's own
        ``cell_ids``, i.e. the same order it was constructed with). Each cluster's
        ``values`` are aggregated with ``agg`` and clusters are renumbered so the
        lowest (or highest, ``ascending=False``) aggregate becomes cluster 0 — e.g.
        sorting clusters by mean soma depth so cluster numbers read top-to-bottom.
        """
        if isinstance(values, Mapping):
            by_cell = {int(k): float(v) for k, v in values.items()}
        else:
            ids = self._cell_ids if cell_ids is None else np.asarray(cell_ids)
            arr = np.asarray(values, dtype=float)
            if arr.shape[0] != ids.shape[0]:
                raise ValueError("values must have one entry per cell_id")
            by_cell = dict(zip((int(c) for c in ids), arr.tolist()))
        try:
            per_cell = np.array([by_cell[int(c)] for c in self._cell_ids], dtype=float)
        except KeyError as error:
            raise KeyError(f"no value for cell_id {error.args[0]}") from error
        agg_fn = np.mean if agg == "mean" else np.median
        order = sorted(self.ids, key=lambda i: agg_fn(per_cell[self._labels == i]))
        if not ascending:
            order = order[::-1]
        return self.reorder(order)

    def combine(self, other: "LabelSet", *, name: str | None = None) -> "LabelSet":
        """Union two LabelSets over disjoint cells into a new one."""
        if set(self._cell_ids.tolist()) & set(other._cell_ids.tolist()):
            raise ValueError("combine requires disjoint cell sets")
        offset = max([*self._meta, *self.ids], default=-1) + 1
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
            mask=self.mask if self.mask == other.mask else None,
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
            f"LabelSet(name={self.name!r}, mask={self.mask!r}, "
            f"n_cells={len(self._cell_ids)}, clusters={self.names})"
        )
