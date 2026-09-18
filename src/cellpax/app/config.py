"""Where the app finds your data, and what it is allowed to assume about it.

Everything here is site-specific — folio location, datastack, id columns, where
skeletons live. None of it is guessed: an unset field disables the feature that
needs it rather than failing at the moment you click something.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: URI schemes a folio may live behind. Anything matching stays a string.
_REMOTE_SCHEMES = ("gs://", "s3://", "az://", "http://", "https://", "gcs://")


def is_remote_uri(value: Any) -> bool:
    """Is this a bucket/URL folio rather than a filesystem path?"""
    return isinstance(value, str) and value.startswith(_REMOTE_SCHEMES)


# The sweep defaults the UI offers for a fresh level. Deliberately denser in
# resolution and thinner in seeds than the usual notebook default: the kNN graph
# is built once per (graph_type, n_neighbors) and every resolution x seed reruns
# Leiden on that cached graph, so grid points are nearly free while each new
# n_neighbors or graph_type costs a graph build. Seeds at one setting mostly
# reproduce each other, so they add runs without adding coverage of realised
# cluster count — which is what `restrict(n_clusters_min=..., n_clusters_max=...)`
# slices on.
DEFAULT_SWEEP: dict[str, Any] = {
    "n_neighbors": [15, 30, 60],
    "graph_type": ["knn", "snn_jaccard", "umap_fuzzy"],
    "resolution_min": 0.02,
    "resolution_max": 2.0,
    "resolution_points": 30,
    "n_times": 3,
}


@dataclass
class AnatomyConfig:
    """Where cell anatomy comes from. Both sources are optional and stack.

    ``skeleton_dir`` is checked first because it is local and instant; CAVE is
    the fallback for cells that are not cached there yet. Skeletons fetched from
    CAVE are written into ``skeleton_dir`` when it is writable, so the second
    look at a cell is local.
    """

    skeleton_dir: Path | None = None
    #: ``str.format`` template resolved against ``root_id``; the default matches
    #: the meshparty/pcg_skel convention of one npz per root id.
    skeleton_pattern: str = "{root_id}.npz"
    #: CAVE datastack, e.g. "minnie65_public". Unset disables on-demand fetch.
    datastack: str | None = None
    #: Passed to caveclient; unset uses its own configured default.
    server_address: str | None = None
    #: How many cells to render in one contact sheet request.
    max_thumbnails: int = 24

    @property
    def has_local(self) -> bool:
        return self.skeleton_dir is not None and self.skeleton_dir.is_dir()

    @property
    def has_cave(self) -> bool:
        return self.datastack is not None


@dataclass
class NeuroglancerConfig:
    """How to turn a set of cells into a neuroglancer link.

    Links open in a new tab rather than an embedded frame — no handshake to keep
    alive, no CORS, and the viewer stays a viewer.
    """

    #: Viewer origin the state is appended to.
    viewer_url: str = "https://neuroglancer-demo.appspot.com"
    #: Image layer source, e.g. "precomputed://gs://...".
    image_source: str | None = None
    #: Segmentation layer source.
    segmentation_source: str | None = None
    #: nm per voxel, written into the state's dimensions.
    voxel_size: tuple[float, float, float] = (4.0, 4.0, 40.0)
    #: Column holding the segment id to load. Distinct from the FeatureTable's
    #: own id column, which is usually a stable cell_id rather than a root_id.
    root_id_column: str = "root_id"
    #: Columns used to centre the view on a selection, when present.
    position_columns: tuple[str, str, str] | None = None
    #: Cap on segments per link — a state with 30k segments will not load.
    max_segments: int = 200

    @property
    def is_configured(self) -> bool:
        return self.segmentation_source is not None


@dataclass
class AppConfig:
    """Everything the server needs to start.

    With no ``folio`` the app runs on a synthetic fixture, which is how the demo
    and the tests drive the full loop without touching real data.
    """

    #: Local directory or a remote URI (``gs://``, ``s3://``, ``http(s)://``).
    #: Remote folios stay strings — resolving one as a filesystem path would
    #: silently turn ``gs://bucket/x`` into ``<config dir>/gs:/bucket/x``.
    folio: Path | str | None = None
    table_name: str | None = None
    #: Where the ledger is appended. Local by definition — it is working state,
    #: written on every decision. Defaults beside a local folio, else cwd.
    ledger_path: Path | None = None
    #: Feature collection every level clusters on unless told otherwise.
    columns: str | None = "analysis"
    #: Column the cut orders clusters by, so ids stay comparable across cuts.
    order_by: str | None = None
    #: Root mask of the descent. None means every cell.
    root_mask: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    #: Cells above which a linkage build gets a confirmation prompt instead of
    #: just running. The scipy condensed vector is n(n-1)/2 float64 and scipy
    #: copies it, so this is roughly the 5 GB line.
    linkage_warn_cells: int = 25_000
    #: GB above which a restriction asks before building its consensus matrix.
    #: Restricting to a coarse grain window makes the matrix *denser*, not
    #: sparser — the surprising direction, and the one that eats a machine.
    consensus_warn_gb: float = 8.0
    sweep: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SWEEP))
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)
    neuroglancer: NeuroglancerConfig = field(default_factory=NeuroglancerConfig)

    @property
    def is_demo(self) -> bool:
        return self.folio is None

    @property
    def folio_is_remote(self) -> bool:
        return is_remote_uri(self.folio)

    def resolved_ledger_path(self) -> Path:
        """Where decisions get appended. Always local.

        A remote folio is read-only working data as far as this app is
        concerned; the ledger is written on every decision and belongs next to
        you, not in a bucket. With a remote folio it lands in the working
        directory under the table's name, so two analyses do not share one.
        """
        if self.ledger_path is not None:
            return self.ledger_path
        stem = self.table_name or "cellpax"
        if self.folio is not None and not self.folio_is_remote:
            return Path(self.folio).parent / f"{stem}.ledger.jsonl"
        return Path.cwd() / f"{stem}.ledger.jsonl"

    @classmethod
    def from_toml(cls, path: str | Path) -> "AppConfig":
        """Load a config file. Unknown keys raise rather than being ignored."""
        path = Path(path)
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_dict(raw, base=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base: Path | None = None) -> "AppConfig":
        raw = dict(raw)
        anatomy_raw = raw.pop("anatomy", {}) or {}
        ngl_raw = raw.pop("neuroglancer", {}) or {}
        sweep_raw = raw.pop("sweep", {}) or {}

        def _path(value: Any) -> Path | str | None:
            if value is None:
                return None
            if is_remote_uri(value):
                return str(value)  # a bucket URI is not a filesystem path
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute() and base is not None:
                candidate = (base / candidate).resolve()
            return candidate

        for key in ("folio", "ledger_path"):
            if key in raw:
                raw[key] = _path(raw[key])
        if "skeleton_dir" in anatomy_raw:
            anatomy_raw["skeleton_dir"] = _path(anatomy_raw["skeleton_dir"])
        if "voxel_size" in ngl_raw:
            ngl_raw["voxel_size"] = tuple(float(v) for v in ngl_raw["voxel_size"])
        if "position_columns" in ngl_raw and ngl_raw["position_columns"] is not None:
            ngl_raw["position_columns"] = tuple(ngl_raw["position_columns"])

        sweep = dict(DEFAULT_SWEEP)
        sweep.update(sweep_raw)

        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(
                f"unknown config keys {sorted(unknown)}; "
                f"known keys are {sorted(cls.__dataclass_fields__)}"
            )
        unknown_anatomy = set(anatomy_raw) - set(AnatomyConfig.__dataclass_fields__)
        if unknown_anatomy:
            raise ValueError(f"unknown [anatomy] keys {sorted(unknown_anatomy)}")
        unknown_ngl = set(ngl_raw) - set(NeuroglancerConfig.__dataclass_fields__)
        if unknown_ngl:
            raise ValueError(f"unknown [neuroglancer] keys {sorted(unknown_ngl)}")

        return cls(
            **raw,
            sweep=sweep,
            anatomy=AnatomyConfig(**anatomy_raw),
            neuroglancer=NeuroglancerConfig(**ngl_raw),
        )

    def resolution_grid(self) -> list[float]:
        """The geometric resolution grid the sweep defaults to.

        Geometric rather than linear because linear spacing undersamples the low
        end, where the coarse plateau lives, and a plateau sampled once reads as
        noise.
        """
        import numpy as np

        return [
            float(v)
            for v in np.geomspace(
                float(self.sweep["resolution_min"]),
                float(self.sweep["resolution_max"]),
                int(self.sweep["resolution_points"]),
            )
        ]
