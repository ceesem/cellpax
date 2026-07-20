"""EXPERIMENTAL: a flexible, AnnData-inspired view over one Study revision.

This is a prototype facade to test a design direction — a single in-memory object
you hold, mask, switch layers on, and plot from, without register/preview/keep
ceremony. It is read-oriented: it reflects an immutable revision (plus an optional
assignment set) and never mutates the study. Transforms that should be durable
still go through the Study.

Mapping to AnnData:
    .obs   per-cell metadata + review labels        (AnnData.obs)
    .var   feature catalog for the selection        (AnnData.var)
    .X     feature matrix (raw or a chosen layer)    (AnnData.X / layers)
    .obsm  representation coordinates (embeddings)   (AnnData.obsm)
    masking -> a narrowed view (a subset of cells)   (AnnData boolean/label slicing)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import polars as pl

from cellpax.records import AssignmentSet, KeptRevision

if TYPE_CHECKING:
    import numpy as np

    from cellpax.study import Study


class CellData:
    """A maskable, layer- and embedding-aware view of one revision's cells."""

    def __init__(
        self,
        study: "Study",
        revision: KeptRevision | str,
        *,
        assignment_set: AssignmentSet | str | None = None,
        cell_ids: Sequence[int] | None = None,
    ) -> None:
        self._study = study
        self._revision = (
            study.get_revision(revision) if isinstance(revision, str) else revision
        )
        self._assignment_set = assignment_set
        self._cell_ids = None if cell_ids is None else list(cell_ids)
        self._layer_cache: dict[str, pl.DataFrame] = {}

    # -- construction / masking ------------------------------------------------

    def _with_cells(self, cell_ids: Sequence[int]) -> "CellData":
        return CellData(
            self._study,
            self._revision,
            assignment_set=self._assignment_set,
            cell_ids=list(cell_ids),
        )

    def mask(self, predicate: pl.Expr) -> "CellData":
        """Return a view of cells where a Polars predicate over ``.obs`` holds."""
        kept = self.obs.filter(predicate)["cell_id"].to_list()
        return self._with_cells(kept)

    def select_cells(self, cell_ids: Iterable[int]) -> "CellData":
        """Return a view restricted to an explicit set of cell ids."""
        return self._with_cells(list(cell_ids))

    # -- core frames -----------------------------------------------------------

    def _table(self, layer: str) -> pl.DataFrame:
        if layer not in {"raw", "normalized"}:
            raise ValueError("layer must be 'raw' or 'normalized'")
        if layer not in self._layer_cache:
            frame = self._study.feature_table(
                self._revision,
                normalized=(layer == "normalized"),
                assignment_set=self._assignment_set,
            )
            self._layer_cache[layer] = frame
        frame = self._layer_cache[layer]
        if self._cell_ids is not None:
            frame = frame.filter(pl.col("cell_id").is_in(self._cell_ids))
        return frame

    @property
    def var(self) -> pl.DataFrame:
        """Feature catalog rows for this revision's selection (AnnData.var)."""
        space = self._study.get_feature_space(self._revision.feature_space_id)
        selection = self._study.get_feature_selection(space.feature_selection_id)
        members = self._study.folio.get(selection.members_ref, frame="polars").sort(
            "position"
        )
        catalog = self._study.feature_catalog()
        return members.join(catalog, on=["feature_block_id", "feature_id"], how="left")

    @property
    def feature_names(self) -> list[str]:
        return self.var["feature_id"].to_list()

    @property
    def obs(self) -> pl.DataFrame:
        """Per-cell labels + universe metadata, no feature columns (AnnData.obs)."""
        return self._table("raw").drop(self.feature_names)

    def features(self, layer: str = "raw") -> pl.DataFrame:
        """Per-cell feature values for a layer, as cell_id + feature columns."""
        return self._table(layer).select("cell_id", *self.feature_names)

    def X(self, layer: str = "raw") -> "np.ndarray":
        """The feature matrix as a NumPy array (AnnData.X / layers)."""
        return self.features(layer).drop("cell_id").to_numpy()

    def frame(self, layer: str = "raw") -> pl.DataFrame:
        """One tidy plot-ready frame: obs + features (facet / axis / color)."""
        return self._table(layer)

    @property
    def obsm(self) -> dict[str, pl.DataFrame]:
        """Representation coordinates keyed by role (AnnData.obsm)."""
        out: dict[str, pl.DataFrame] = {}
        pairs = {
            "clustering": self._revision.clustering_representation_id,
            "visualization": self._revision.visualization_representation_id,
        }
        for role, rep_id in pairs.items():
            if rep_id is None:
                continue
            rep = self._study.get_representation(rep_id)
            coords = self._study.folio.get(rep.coords_ref, frame="polars")
            if self._cell_ids is not None:
                coords = coords.filter(pl.col("cell_id").is_in(self._cell_ids))
            out[role] = coords
        return out

    # -- niceties --------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self._table("raw").height

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def __repr__(self) -> str:
        obs_cols = [c for c in self.obs.columns if c != "cell_id"]
        return (
            f"CellData(revision={self._revision.name!r}, "
            f"n_cells={self.n_cells}, n_features={self.n_features}, "
            f"layers=['raw', 'normalized'], obsm={list(self.obsm)}, "
            f"obs={obs_cols})"
        )
