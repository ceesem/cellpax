"""Skeleton loading from local files, with an optional CAVE fallback.

**Not currently wired into the app.** The loader below is kept because it is the
reusable half; the rendering half was removed after it turned out to be wrong in
three ways at once, all of which a future version has to fix before this is worth
putting back on screen:

* **Scale was per-cell, not shared.** Each thumbnail was normalised to its own
  extent, so a small interneuron and a large pyramidal cell drew the same size.
  That is precisely backwards for a view whose job is comparing morphology — the
  scale bar has to be common across the sheet.
* **Too small to read.** At 110px a cortical arbor is a smudge. A useful contact
  sheet needs either far fewer, far larger cells, or a projection chosen to
  survive the size (e.g. depth profile rather than a raw xy projection).
* **Upside down.** ``flip_y`` was applied on the theory that y needed inverting
  for screen coordinates, but EM y already increases with depth, so the flip put
  pia at the bottom. The correct rendering applies no flip at all.

Both sources stay optional and stack: local disk first because it is instant,
CAVE as the fallback, and anything fetched is written back into ``skeleton_dir``
when that directory is writable so a cell is fetched at most once.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from cellpax.app.config import AnatomyConfig

#: Vertices/edges keys tried in a loaded npz, in order. Covers the meshparty and
#: pcg_skel conventions without requiring either package to be installed.
_VERTEX_KEYS = ("vertices", "verts", "xyz", "nodes")
_EDGE_KEYS = ("edges", "links", "segments")


class SkeletonStore:
    """Local-disk skeletons with an optional CAVE fallback.

    Local is checked first because it is instant. A skeleton fetched from CAVE is
    written back into the local directory when that directory is writable, so a
    cell is fetched at most once.
    """

    def __init__(self, config: AnatomyConfig) -> None:
        self.config = config
        self._cache: dict[int, dict[str, np.ndarray] | None] = {}
        self._lock = threading.Lock()
        self._client: Any = None
        self._cave_failed = False

    @property
    def available(self) -> bool:
        return self.config.has_local or self.config.has_cave

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "local": self.config.has_local,
            "local_dir": str(self.config.skeleton_dir)
            if self.config.skeleton_dir
            else None,
            "cave": self.config.has_cave and not self._cave_failed,
            "datastack": self.config.datastack,
        }

    # -- loading ---------------------------------------------------------------

    def get(self, root_id: int) -> dict[str, np.ndarray] | None:
        root_id = int(root_id)
        with self._lock:
            if root_id in self._cache:
                return self._cache[root_id]
        skeleton = self._load_local(root_id)
        if skeleton is None:
            skeleton = self._load_cave(root_id)
        with self._lock:
            self._cache[root_id] = skeleton
        return skeleton

    def _local_path(self, root_id: int) -> Path | None:
        if not self.config.has_local:
            return None
        assert self.config.skeleton_dir is not None
        return self.config.skeleton_dir / self.config.skeleton_pattern.format(
            root_id=root_id
        )

    def _load_local(self, root_id: int) -> dict[str, np.ndarray] | None:
        path = self._local_path(root_id)
        if path is None or not path.exists():
            return None
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as handle:
                    return _from_mapping({k: handle[k] for k in handle.files})
            if path.suffix in (".swc", ".txt"):
                return _from_swc(path)
        except (OSError, ValueError, KeyError):
            return None
        return None

    def _load_cave(self, root_id: int) -> dict[str, np.ndarray] | None:
        """Fetch on demand. Degrades to None rather than propagating a failure.

        A CAVE stack that is unreachable, unauthenticated, or simply does not
        serve skeletons is a normal condition for a local tool. It disables the
        source for the session rather than erroring on every cell.
        """
        if not self.config.has_cave or self._cave_failed:
            return None
        client = self._cave_client()
        if client is None:
            return None
        try:
            raw = client.skeleton.get_skeleton(root_id, output_format="dict")
        except Exception:
            return None
        skeleton = _from_mapping(raw if isinstance(raw, dict) else {})
        if skeleton is not None:
            self._write_back(root_id, skeleton)
        return skeleton

    def _cave_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import caveclient  # noqa: PLC0415 — optional, and only on first use
        except ImportError:
            self._cave_failed = True
            return None
        try:
            kwargs: dict[str, Any] = {"datastack_name": self.config.datastack}
            if self.config.server_address:
                kwargs["server_address"] = self.config.server_address
            self._client = caveclient.CAVEclient(**kwargs)
        except Exception:
            self._cave_failed = True
            return None
        return self._client

    def _write_back(self, root_id: int, skeleton: dict[str, np.ndarray]) -> None:
        path = self._local_path(root_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, vertices=skeleton["vertices"], edges=skeleton["edges"]
            )
        except OSError:
            pass  # a read-only cache directory is not a reason to fail the request


def _from_mapping(raw: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Normalise whatever the source called its arrays into vertices/edges."""
    vertices = _first(raw, _VERTEX_KEYS)
    edges = _first(raw, _EDGE_KEYS)
    if vertices is None or edges is None:
        return None
    vertices = np.asarray(vertices, dtype=float)
    edges = np.asarray(edges)
    if vertices.ndim != 2 or vertices.shape[1] < 2 or edges.ndim != 2:
        return None
    return {"vertices": vertices, "edges": edges}


def _first(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _from_swc(path: Path) -> dict[str, np.ndarray] | None:
    """Minimal SWC reader: id, type, x, y, z, radius, parent."""
    rows: list[tuple[int, float, float, float, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(
                (
                    int(parts[0]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    int(parts[6]),
                )
            )
    if not rows:
        return None
    index = {row[0]: position for position, row in enumerate(rows)}
    vertices = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    edges = np.array(
        [[index[r[0]], index[r[4]]] for r in rows if r[4] in index], dtype=int
    )
    if edges.size == 0:
        return None
    return {"vertices": vertices, "edges": edges}
