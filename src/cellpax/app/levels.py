"""One level of the descent, and the tree they form.

A level is the unit the notebook was missing. It is *mask -> space -> Clustering
-> threshold -> child masks*, and the descent is that same unit applied to each
child. Making it an object is the whole reason the app exists: in a notebook the
recursion has to be expressed by pasting the block again with two names changed,
because a cell is not a level and cannot be navigated, re-entered, or compared to
its sibling.

Levels hold live cellpax objects, which are not small. A ``Clustering`` at 34.6k
cells carries a sparse consensus matrix plus a ``(n_cells, n_runs)`` partition
array — a few hundred MB with a dense sweep — and ``restrict`` produces a second
one. So the tree caps how many levels keep their heavy state resident and evicts
least-recently-used, which is safe precisely because the ledger records the
parameters: an evicted level is rebuildable, not lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class CutState:
    """The cut currently being considered at a level. Not yet committed."""

    distance_threshold: float | None = None
    min_cluster_size: int = 1
    #: Names the user has typed for cluster ids, before commit.
    names: dict[int, str] = field(default_factory=dict)
    colors: dict[int, str] = field(default_factory=dict)

    def is_set(self) -> bool:
        return self.distance_threshold is not None


@dataclass
class Level:
    """A node of the descent.

    ``clustering`` is the pooled consensus over every swept run; ``restricted``
    is that pool narrowed by realised grain. They are kept apart on purpose,
    because a threshold does not mean the same thing on both: pooling runs of
    different grain lands a broad group at the *fraction of runs coarse enough to
    keep it*, so on the unrestricted pool the useful thresholds bunch up against
    the point where the tree collapses. Whichever one is active is the one the
    scan and the cut read.
    """

    name: str
    mask: str | None
    parent: str | None = None
    children: list[str] = field(default_factory=list)

    #: Feature collection this level clusters on.
    columns: str | None = None
    order_by: str | None = None

    # -- live objects, all rebuildable from the ledger --------------------------
    space: Any = None
    clustering: Any = None
    restricted: Any = None
    labels: Any = None
    #: A fitted Gradient, when this level's boundary turned out continuous.
    #: Kept beside `labels`, not instead of it: a level can hold a discrete cut
    #: for most of its clusters and a coordinate through the pair that isn't.
    gradient: Any = None
    gradient_warnings: list[str] = field(default_factory=list)

    #: Parameters actually used, mirrored from the ledger for display.
    cluster_params: dict[str, Any] = field(default_factory=dict)
    restrict_params: dict[str, Any] = field(default_factory=dict)

    cut: CutState = field(default_factory=CutState)
    committed: bool = False
    #: Bumped whenever heavy state is touched, for LRU eviction.
    touched: int = 0

    @property
    def active(self) -> Any:
        """The Clustering the scan and the cut should read.

        Restriction wins when present — choosing the grain window is a decision
        about which runs to pool, and everything downstream is meant to be read
        under it.
        """
        return self.restricted if self.restricted is not None else self.clustering

    @property
    def is_restricted(self) -> bool:
        return self.restricted is not None

    @property
    def has_clustering(self) -> bool:
        return self.clustering is not None

    @property
    def n_cells(self) -> int | None:
        active = self.active
        if active is None:
            return None
        return int(active.shape[0])

    def status(self) -> str:
        """Where this level is in the loop, for the tree rail."""
        if self.committed:
            return "committed"
        if self.cut.is_set():
            return "cut"
        if self.clustering is not None:
            return "clustered"
        return "open"

    def release(self) -> None:
        """Drop heavy state, keeping the decisions. Reversible by rebuilding."""
        self.clustering = None
        self.restricted = None
        self.space = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mask": self.mask,
            "parent": self.parent,
            "children": list(self.children),
            "columns": self.columns,
            "order_by": self.order_by,
            "status": self.status(),
            "n_cells": self.n_cells,
            "has_clustering": self.has_clustering,
            "is_restricted": self.is_restricted,
            "resident": self.clustering is not None,
            "cluster_params": self.cluster_params,
            "restrict_params": self.restrict_params,
            "committed": self.committed,
            "has_gradient": self.gradient is not None,
            "cut": {
                "distance_threshold": self.cut.distance_threshold,
                "min_cluster_size": self.cut.min_cluster_size,
                "names": {str(k): v for k, v in self.cut.names.items()},
                "colors": {str(k): v for k, v in self.cut.colors.items()},
            },
        }


class LevelTree:
    """The descent: levels keyed by name, with parent/child links.

    ``max_resident`` bounds how many levels keep their ``Clustering`` in memory
    at once. The cap exists because building a linkage is the expensive thing in
    this whole application, and holding several old ones resident is how you run
    out of room to build the next.
    """

    def __init__(self, *, max_resident: int = 3) -> None:
        self._levels: dict[str, Level] = {}
        self._order: list[str] = []
        self.max_resident = max_resident
        self._clock = 0

    def __contains__(self, name: str) -> bool:
        return name in self._levels

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[Level]:
        return (self._levels[name] for name in self._order)

    def __getitem__(self, name: str) -> Level:
        try:
            return self._levels[name]
        except KeyError:
            raise KeyError(
                f"no level named {name!r}; open levels are {sorted(self._levels)}"
            ) from None

    def get(self, name: str) -> Level | None:
        return self._levels.get(name)

    def add(
        self,
        name: str,
        *,
        mask: str | None,
        parent: str | None = None,
        columns: str | None = None,
        order_by: str | None = None,
    ) -> Level:
        if name in self._levels:
            raise ValueError(f"level {name!r} is already open")
        level = Level(
            name=name, mask=mask, parent=parent, columns=columns, order_by=order_by
        )
        self._levels[name] = level
        self._order.append(name)
        if parent is not None:
            parent_level = self._levels.get(parent)
            if parent_level is not None and name not in parent_level.children:
                parent_level.children.append(name)
        return level

    def remove(self, name: str) -> None:
        """Forget a level, and unlink it from its parent's children."""
        level = self._levels.pop(name, None)
        if level is None:
            return
        self._order = [n for n in self._order if n != name]
        parent = self._levels.get(level.parent or "")
        if parent is not None:
            parent.children = [c for c in parent.children if c != name]

    def touch(self, name: str) -> Level:
        """Mark a level as most-recently-used and evict past the cap."""
        level = self[name]
        self._clock += 1
        level.touched = self._clock
        self._evict()
        return level

    def _evict(self) -> list[str]:
        resident = [lv for lv in self._levels.values() if lv.clustering is not None]
        if len(resident) <= self.max_resident:
            return []
        resident.sort(key=lambda lv: lv.touched)
        dropped = []
        for level in resident[: len(resident) - self.max_resident]:
            level.release()
            dropped.append(level.name)
        return dropped

    def roots(self) -> list[Level]:
        return [lv for lv in self if lv.parent is None]

    def path_to(self, name: str) -> list[str]:
        """Names from the root down to ``name``, for the breadcrumb."""
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = name
        while current is not None and current not in seen:
            seen.add(current)
            path.append(current)
            level = self._levels.get(current)
            current = level.parent if level is not None else None
        return list(reversed(path))

    def to_list(self) -> list[dict[str, Any]]:
        return [level.to_dict() for level in self]
