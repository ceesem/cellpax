"""Neuroglancer links for a cluster's cells.

Links open in a new tab rather than an embedded frame. That is a deliberate
limit: an embedded viewer needs a message handshake to stay in sync, and a stale
one is worse than none — it shows you the previous cluster while you reason about
the current one. A link that opens fresh is always honest about what it shows.

Nothing here talks to a server. It builds a neuroglancer state, JSON-encodes it
into the viewer's URL fragment, and hands back a string.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Iterable, Sequence

from cellpax.app.config import NeuroglancerConfig

#: tab20, reordered. The hues are matplotlib's because that is what everything
#: else in this analysis is plotted in, but the *order* is not: tab20 native puts
#: green at slot 3 and orange at slot 2, and those two are ΔE 0.7 apart under
#: protanopia — indistinguishable. Since cluster ids come out ordered by
#: ``order_by`` (soma depth), adjacent ids are usually adjacent in space too, so
#: adjacent slots are exactly the pairs that must not collide. Spreading them
#: takes the worst adjacent pair from ΔE 0.7 to 17.2 and turns the validator's
#: CVD check from FAIL to PASS.
#:
#: The nine saturated hues come first so a cut with ≤9 clusters never reaches for
#: a pale variant; the pale nine follow in the same order. tab20's two greys are
#: left out on purpose — grey means unassigned everywhere in this app.
#:
#: No palette this size passes an all-pairs CVD check, and none can: beyond about
#: eight categories colour alone cannot carry identity. That is what the scatter's
#: click-to-isolate is for.
_FALLBACK = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#d62728",  # red
    "#bcbd22",  # olive
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#aec7e8",  # — pale variants, same order
    "#ffbb78",
    "#9edae5",
    "#ff9896",
    "#dbdb8d",
    "#c5b0d5",
    "#98df8a",
    "#f7b6d2",
    "#c49c94",
)

#: Cells the size floor dropped. Never a cluster colour.
UNASSIGNED = "#4a4f5e"

#: The canonical order, served to the client so the scatter, the swatches and
#: the neuroglancer links all agree on what colour a cluster is.
PALETTE = _FALLBACK


class NeuroglancerLinks:
    """Builds viewer URLs from a config plus per-cluster segment ids."""

    def __init__(self, config: NeuroglancerConfig) -> None:
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.is_configured

    def state(
        self,
        segments: Sequence[int],
        *,
        colors: dict[int, str] | None = None,
        position: tuple[float, float, float] | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """A neuroglancer viewer state. Truncated to ``max_segments``.

        The cap is not cosmetic — a state naming tens of thousands of segments
        will not load, and silently handing over one that hangs the viewer is
        worse than showing a representative subset and saying so.
        """
        if not self.config.is_configured:
            raise ValueError(
                "no segmentation_source configured; set [neuroglancer] in the app config"
            )
        shown = list(segments)[: self.config.max_segments]
        layers: list[dict[str, Any]] = []
        if self.config.image_source:
            layers.append(
                {"type": "image", "source": self.config.image_source, "name": "img"}
            )
        layer: dict[str, Any] = {
            "type": "segmentation",
            "source": self.config.segmentation_source,
            "name": title or "seg",
            "segments": [str(s) for s in shown],
        }
        if colors:
            layer["segmentColors"] = {
                str(segment): color
                for segment, color in colors.items()
                if segment in set(shown)
            }
        layers.append(layer)

        state: dict[str, Any] = {
            "dimensions": {
                axis: [size * 1e-9, "m"]
                for axis, size in zip("xyz", self.config.voxel_size)
            },
            "layers": layers,
            "layout": "xy-3d",
            "selectedLayer": {"layer": layer["name"], "visible": True},
        }
        if position is not None:
            state["position"] = [float(v) for v in position]
        return state

    def url(self, segments: Sequence[int], **kwargs: Any) -> str:
        state = self.state(segments, **kwargs)
        encoded = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
        return f"{self.config.viewer_url.rstrip('/')}/#!{encoded}"

    def link_for_cluster(
        self,
        table: Any,
        cell_ids: Iterable[int],
        *,
        color: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Resolve cells to segment ids and build the link.

        Reports what it dropped. A cluster whose cells have no root id is a real
        condition (a cohort keyed only by ``cell_id``), not an error, so it comes
        back as an unavailable link with a reason rather than an exception.
        """
        cell_ids = [int(c) for c in cell_ids]
        column = self.config.root_id_column
        if column not in table.dataframe().columns:
            return {
                "available": False,
                "reason": f"no {column!r} column to resolve segments with",
            }
        frame = table.dataframe()
        id_column = table.id_column
        matched = (
            frame.filter(frame[id_column].is_in(cell_ids))
            .select([id_column, column])
            .drop_nulls()
        )
        segments = [int(v) for v in matched[column].to_list()]
        if not segments:
            return {"available": False, "reason": "no segment ids for these cells"}
        if not self.config.is_configured:
            return {
                "available": False,
                "reason": "no segmentation_source configured",
                "n_segments": len(segments),
            }
        position = self._centroid(frame, cell_ids)
        colors = {segment: color for segment in segments} if color else None
        return {
            "available": True,
            "url": self.url(segments, colors=colors, position=position, title=title),
            "n_segments": len(segments),
            "n_shown": min(len(segments), self.config.max_segments),
            "truncated": len(segments) > self.config.max_segments,
        }

    def _centroid(
        self, frame: Any, cell_ids: Sequence[int]
    ) -> tuple[float, float, float] | None:
        columns = self.config.position_columns
        if not columns or not all(c in frame.columns for c in columns):
            return None
        import polars as pl

        subset = frame.filter(pl.col(frame.columns[0]).is_in(list(cell_ids)))
        if subset.height == 0:
            return None
        try:
            return tuple(  # type: ignore[return-value]
                float(subset[c].mean()) for c in columns
            )
        except (TypeError, ValueError):
            return None


def palette_for(
    ids: Sequence[int], existing: dict[int, str] | None = None
) -> dict[int, str]:
    """Stable colours per cluster id, honouring any already chosen."""
    colors = dict(existing or {})
    for position, cluster_id in enumerate(sorted(ids)):
        colors.setdefault(int(cluster_id), _FALLBACK[position % len(_FALLBACK)])
    return colors
