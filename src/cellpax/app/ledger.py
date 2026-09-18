"""The append-only record of what you decided, and the replay that re-does it.

The ledger is the app's source of truth. Clicking "commit" does not primarily
mutate a ``FeatureTable`` — it appends an entry, and the FeatureTable state is
what you get by replaying entries in order. That buys three things a
write-through design does not:

* **Undo** is truncation. Drop the last *n* entries, replay, and the table is
  exactly where it was.
* **Provenance** survives the session. The entry records the threshold you chose
  *and* the restrict window and ``min_cluster_size`` it was chosen under, which
  is the context that makes a threshold mean anything.
* **The loop is re-runnable headlessly.** ``replay()`` is plain cellpax calls, so
  a finished descent runs in CI or a script with no app involved.

Determinism is the load-bearing assumption, and cellpax supports it: seeds are
derived per call from ``(op, mask, name)`` off the table's root seed, so a replay
of the same entries against the same input data produces the same clusterings.
Entries therefore record *parameters*, never fitted results.

Kept deliberately outside the library: ``DESIGN_PROPOSAL.md`` lists append-only
decision ledgers among the things the immutable-first design dropped on purpose.
That judgement is about ``cellpax``; an interactive driver is exactly where the
idea earns its keep.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Entry kinds, in the order a level moves through them.
KINDS = (
    "open_level",  # a mask becomes a level of the descent
    "cluster",  # the consensus sweep for that level
    "restrict",  # narrow the pooled runs by realised grain
    "cut",  # a threshold + min_cluster_size, producing labels
    "rename",  # names and colors for the cut's clusters
    "boundary",  # ask whether each boundary is a gap or a cut
    "parametrize",  # fit a coordinate through a continuum instead of labelling it
    "bin",  # attach that coordinate, and optionally declared cuts of it
    "commit",  # attach the labels and carve child masks
    "note",  # free text, so a judgement call can say why
)


@dataclass
class Entry:
    """One decision. ``payload`` is kind-specific and JSON-round-trippable."""

    kind: str
    level: str
    payload: dict[str, Any] = field(default_factory=dict)
    #: Wall clock, for reading the history back. Never used in replay — replay
    #: depends only on order, so a ledger stays reproducible across machines.
    at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown entry kind {self.kind!r}; expected one of {KINDS}"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Entry":
        raw = json.loads(line)
        return cls(
            kind=raw["kind"],
            level=raw["level"],
            payload=raw.get("payload", {}),
            at=raw.get("at", 0.0),
        )


class Ledger:
    """An append-only jsonl file of :class:`Entry`, with in-memory mirror.

    Appends are flushed immediately: a session that dies mid-descent should lose
    at most the decision you were in the middle of making, not the morning.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[Entry] = []
        if self.path is not None and self.path.exists():
            self._entries = list(self._read(self.path))

    @staticmethod
    def _read(path: Path) -> Iterator[Entry]:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Entry.from_json(line)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{number} is not a valid entry: {exc}"
                    ) from exc

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries)

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def append(self, kind: str, level: str, **payload: Any) -> Entry:
        entry = Entry(kind=kind, level=level, payload=payload)
        self._entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.to_json() + "\n")
                handle.flush()
        return entry

    def truncate(self, count: int) -> list[Entry]:
        """Drop the last ``count`` entries and rewrite the file. Returns them.

        The undo primitive. The caller is responsible for replaying onto a fresh
        table afterwards — this only edits the record, because rolling a
        ``FeatureTable`` backwards in place is not something cellpax offers and
        faking it would be the kind of hidden state the ledger exists to avoid.
        """
        if count <= 0:
            return []
        dropped = self._entries[-count:]
        self._entries = self._entries[:-count]
        if self.path is not None:
            with self.path.open("w", encoding="utf-8") as handle:
                for entry in self._entries:
                    handle.write(entry.to_json() + "\n")
        return dropped

    def keep(self, indices: list[int]) -> list[Entry]:
        """Keep only these entry positions, returning what was removed.

        The general form of :meth:`truncate`. Needed because "throw away my
        decisions but keep the sweep that took twenty minutes" is not a suffix
        of the ledger — it is a filter over it.
        """
        wanted = set(indices)
        removed = [e for i, e in enumerate(self._entries) if i not in wanted]
        self._entries = [e for i, e in enumerate(self._entries) if i in wanted]
        if self.path is not None:
            with self.path.open("w", encoding="utf-8") as handle:
                for entry in self._entries:
                    handle.write(entry.to_json() + "\n")
        return removed

    def for_level(self, level: str) -> list[Entry]:
        return [entry for entry in self._entries if entry.level == level]

    def latest(self, kind: str, level: str) -> Entry | None:
        for entry in reversed(self._entries):
            if entry.kind == kind and entry.level == level:
                return entry
        return None

    def to_frame(self) -> Any:
        """The whole history as a polars frame, for the history pane."""
        import polars as pl

        if not self._entries:
            return pl.DataFrame(
                schema={
                    "n": pl.Int64,
                    "kind": pl.String,
                    "level": pl.String,
                    "at": pl.Float64,
                    "detail": pl.String,
                }
            )
        return pl.DataFrame(
            {
                "n": list(range(len(self._entries))),
                "kind": [e.kind for e in self._entries],
                "level": [e.level for e in self._entries],
                "at": [e.at for e in self._entries],
                "detail": [_summarize(e) for e in self._entries],
            }
        )

    def to_script(self) -> str:
        """The descent as a plain cellpax script, no app required.

        The escape hatch: whatever the app is or isn't doing, the decisions it
        recorded should be runnable without it.
        """
        lines = [
            "# Generated from a cellpax app ledger. Runs standalone.",
            "import numpy as np",
            "import polars as pl",
            "import cellpax as cpx",
            "",
            "# ft = cpx.FeatureTable.load(folio, name)   # <- your table here",
            "",
        ]
        for entry in self._entries:
            lines.extend(_script_lines(entry))
        return "\n".join(lines) + "\n"


def _summarize(entry: Entry) -> str:
    """A one-line human reading of an entry, for the history pane."""
    p = entry.payload
    if entry.kind == "cluster":
        return (
            f"{len(p.get('resolution', []))} resolutions x "
            f"{len(p.get('graph_type', []))} graphs x "
            f"{len(p.get('n_neighbors', []))} nn x {p.get('n_times')} seeds"
        )
    if entry.kind == "restrict":
        window = {k: v for k, v in p.items() if v is not None}
        return ", ".join(f"{k}={v}" for k, v in window.items()) or "no narrowing"
    if entry.kind == "cut":
        return (
            f"threshold={p.get('distance_threshold')}, "
            f"min_cluster_size={p.get('min_cluster_size')} "
            f"-> {p.get('n_clusters')} clusters, {p.get('n_unassigned')} unassigned"
        )
    if entry.kind == "commit":
        children = p.get("children") or {}
        return f"attached {p.get('name')}; children: {', '.join(children) or 'none'}"
    if entry.kind == "rename":
        return ", ".join(f"{k}->{v}" for k, v in (p.get("names") or {}).items())
    if entry.kind == "open_level":
        return f"mask={p.get('mask')} parent={p.get('parent') or 'root'}"
    if entry.kind == "boundary":
        return f"gap-or-cut at n_neighbors={p.get('n_neighbors')}"
    if entry.kind == "parametrize":
        extra = f", nuisance={p.get('nuisance')}" if p.get("nuisance") else ""
        return (
            f"axis through {p.get('clusters')} oriented by {p.get('orient_by')}{extra}"
        )
    if entry.kind == "bin":
        return f"attached {p.get('gradient')}" + (
            f" + {p.get('bins')} declared bins"
            if p.get("bins")
            else " (coordinate only)"
        )
    if entry.kind == "note":
        return str(p.get("text", ""))
    return json.dumps(p, sort_keys=True)


def _script_lines(entry: Entry) -> list[str]:
    """Render one entry as the cellpax calls that reproduce it.

    Levels and masks are not the same name and must not be conflated: a level is
    the app's handle on a step of the descent, while the mask is what cellpax
    actually takes. The entries carry both, and only the mask reaches the script.
    """
    p = entry.payload
    level = entry.level
    var = _safe(level)
    mask = p.get("mask")
    if entry.kind == "open_level":
        parent = p.get("parent")
        return [
            "",
            f"# ---- level {level!r} on mask {p.get('mask')!r}"
            f" (parent: {parent or 'none — this is the root'}) ----",
        ]
    if entry.kind == "cluster":
        lines: list[str] = []
        weights = "None"
        if p.get("use_block_weights") and p.get("columns"):
            # The weights are part of the fit, not a preference, so a script
            # that omits them reproduces a different space and therefore a
            # different clustering. They are recomputed rather than stored
            # because they are a deterministic function of the mask's scaled
            # features — an array cannot ride in a JSON ledger anyway.
            lines += [
                f"weights_{var} = cpx.block_weights(",
                f"    ft.features({mask!r}, scaled=True, columns={p['columns']!r}),",
                f"    list(ft.collections[{p['columns']!r}].columns),",
                ")",
            ]
            weights = f"weights_{var}"
        return lines + [
            f"space_{var} = ft.space({mask!r}, columns={p.get('columns')!r},"
            f" alpha={p.get('alpha')}, feature_weights={weights})",
            f"clus_{var} = ft.cluster(",
            f"    {mask!r}, space=space_{var},",
            f"    n_neighbors={p.get('n_neighbors')!r},",
            f"    graph_type={p.get('graph_type')!r},",
            f"    resolution={_short_list(p.get('resolution', []))},",
            f"    n_times={p.get('n_times')!r}, order_by={p.get('order_by')!r},",
            f"    name={p.get('name')!r},",
            ")",
        ]
    if entry.kind == "restrict":
        args = ", ".join(
            f"{k}={v!r}" for k, v in p.items() if v is not None and k != "mask"
        )
        return [f"mid_{var} = clus_{var}.restrict({args})"]
    if entry.kind == "cut":
        source = f"mid_{var}" if p.get("restricted") else f"clus_{var}"
        lines = [
            f"lbl_{var} = {source}.label(",
            f"    distance_threshold={p.get('distance_threshold')!r},",
            f"    min_cluster_size={p.get('min_cluster_size')!r},",
            f"    name={p.get('name')!r},",
            ")",
        ]
        if p.get("names"):
            lines.append(f"lbl_{var} = lbl_{var}.rename({_int_keys(p['names'])!r})")
        if p.get("colors"):
            lines.append(
                f"lbl_{var} = lbl_{var}.set_colors({_int_keys(p['colors'])!r})"
            )
        return lines
    if entry.kind == "rename":
        # Naming happens while you are still deciding, so these entries land
        # before the cut they describe. They stay in the history as a record of
        # what you called things and when; the cut entry is what the script
        # renders, because that is the state that has to be reproduced.
        return []
    if entry.kind == "commit":
        lines = [f"ft.attach(lbl_{var}, overwrite=True)"]
        for child, members in (p.get("children") or {}).items():
            lines.append(
                f"ft.add_mask({child!r}, pl.col({p.get('name')!r})"
                f".is_in({list(members)!r}), based_on={mask!r})"
            )
        return lines
    if entry.kind == "boundary":
        # A read, not a decision — but the reason a pair was treated as a
        # continuum lives here, so it belongs in the reproduced script.
        return [
            f"boundary_{var} = ft.boundary_report(",
            f"    clus_{var}, labels=lbl_{var},",
            f"    n_neighbors={p.get('n_neighbors')!r}, space=space_{var},",
            ")",
            f"# read the 'verdict' column before parametrizing anything",
        ]
    if entry.kind == "parametrize":
        return [
            f"grad_{var} = ft.parametrize(",
            f"    {mask!r}, labels=lbl_{var}, clusters={p.get('clusters')!r},",
            f"    orient_by={p.get('orient_by')!r}, nuisance={p.get('nuisance')!r},",
            f"    name={p.get('name')!r},",
            ")",
            f"grad_{var}.loadings()",
        ]
    if entry.kind == "bin":
        lines = [f"ft.attach(grad_{var})"]
        if p.get("bins"):
            lines.append(
                f"ft.attach(grad_{var}.bin({p['bins']!r}, names={p.get('names')!r}), "
                f"overwrite=True)"
            )
        return lines
    if entry.kind == "note":
        return [f"# note: {p.get('text', '')}"]
    return []


def _short_list(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return "[]"
    return f"np.geomspace({values[0]:.4g}, {values[-1]:.4g}, {len(values)})"


def _int_keys(mapping: dict[str, Any]) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for key, value in mapping.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            out[key] = value  # type: ignore[index]
    return out


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
