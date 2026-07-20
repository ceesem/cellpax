"""A fluent builder that threads previews into named revisions.

``RevisionBuilder`` is a thin convenience wrapper over the ``preview_*`` and
``keep`` methods of :class:`~cellpax.study.Study`. It holds the artifacts you
build so you don't re-pass them into every step, and chains kept revisions
automatically. It adds no new persistence semantics — every method delegates to
the study, and each artifact is content-addressed exactly as if you had called
the study directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import polars as pl

from cellpax.config import (
    CandidateCutConfig,
    ClusteringConfig,
    FeatureSpaceConfig,
    RepresentationConfig,
)
from cellpax.records import (
    CandidateSet,
    ClusteringRun,
    FeatureSelection,
    FeatureSpace,
    KeptRevision,
    Representation,
    Scope,
)

if TYPE_CHECKING:
    from cellpax.clustering import CandidateGenerator
    from cellpax.study import Study


class RevisionBuilder:
    """Accumulate previews and keep them as chained named revisions."""

    def __init__(
        self, study: "Study", *, parent_revision: KeptRevision | str | None = None
    ) -> None:
        self._study = study
        self._parent = parent_revision
        self._scope: Scope | None = None
        self._feature_selection: FeatureSelection | None = None
        self._feature_space: FeatureSpace | None = None
        self._clustering_representation: Representation | None = None
        self._visualization_representation: Representation | None = None
        self._clustering_run: ClusteringRun | None = None
        self._candidate_set: CandidateSet | None = None

    def scope(
        self,
        cells: pl.DataFrame | pl.Series | Iterable[int],
        *,
        derivation_text: str | None = None,
        parent: Scope | str | None = None,
    ) -> "RevisionBuilder":
        """Preview and set the working cell scope."""
        self._scope = self._study.preview_scope(
            cells, derivation_text=derivation_text, parent=parent
        )
        return self

    def use_scope(self, scope: Scope | str) -> "RevisionBuilder":
        """Adopt an existing scope instead of previewing a new one."""
        self._scope = self._study.get_scope(scope) if isinstance(scope, str) else scope
        return self

    def select(
        self,
        block_or_frame,
        feature_ids: Sequence[str] | None = None,
        *,
        derivation_text: str | None = None,
    ) -> "RevisionBuilder":
        """Preview and set the feature selection."""
        self._feature_selection = self._study.preview_feature_selection(
            block_or_frame, feature_ids, derivation_text=derivation_text
        )
        return self

    def use_selection(self, selection: FeatureSelection | str) -> "RevisionBuilder":
        """Adopt an existing feature selection."""
        self._feature_selection = (
            self._study.get_feature_selection(selection)
            if isinstance(selection, str)
            else selection
        )
        return self

    def feature_space(
        self,
        config: FeatureSpaceConfig,
        *,
        fit_scope: Scope | str | None = None,
        parent: FeatureSpace | str | None = None,
    ) -> "RevisionBuilder":
        """Preview a feature space from the current scope and selection."""
        self._require(self._scope, "a scope", "feature_space")
        self._require(self._feature_selection, "a feature selection", "feature_space")
        self._feature_space = self._study.preview_feature_space(
            scope=self._scope,
            fit_scope=fit_scope,
            feature_selection=self._feature_selection,
            config=config,
            parent=parent,
        )
        return self

    def representation(
        self,
        config: RepresentationConfig,
        *,
        fit_scope: Scope | str | None = None,
        clustering: bool = True,
        visualization: bool = True,
    ) -> "RevisionBuilder":
        """Preview a representation from the current feature space.

        By default the result is used for both clustering and visualization; set
        ``clustering=False`` or ``visualization=False`` to assign only one slot.
        """
        self._require(self._feature_space, "a feature space", "representation")
        record = self._study.preview_representation(
            scope=self._scope,
            fit_scope=fit_scope,
            feature_space=self._feature_space,
            config=config,
        )
        if clustering:
            self._clustering_representation = record
        if visualization:
            self._visualization_representation = record
        return self

    def clustering(
        self,
        config: ClusteringConfig,
        *,
        generator: "CandidateGenerator | None" = None,
    ) -> "RevisionBuilder":
        """Preview a clustering run over the current clustering representation."""
        self._require(
            self._clustering_representation,
            "a clustering representation",
            "clustering",
        )
        self._clustering_run = self._study.preview_clustering_run(
            scope=self._scope,
            representation=self._clustering_representation,
            config=config,
            generator=generator,
        )
        return self

    def candidates(
        self,
        config: CandidateCutConfig,
        *,
        generator: "CandidateGenerator | None" = None,
    ) -> "RevisionBuilder":
        """Preview a candidate set by cutting the current clustering run."""
        self._require(self._clustering_run, "a clustering run", "candidates")
        self._candidate_set = self._study.preview_candidate_set(
            clustering_run=self._clustering_run,
            config=config,
            generator=generator,
        )
        return self

    def keep(
        self,
        name: str,
        *,
        notes: str | None = None,
        created_by: str | None = None,
    ) -> KeptRevision:
        """Keep the accumulated previews as a named revision.

        The returned revision becomes the parent of the next ``keep`` on this
        builder, so successive checkpoints chain into a linear history.
        """
        if self._scope is None:
            raise ValueError("A revision requires a scope; call .scope(...) first")
        revision = self._study.keep(
            name,
            scope=self._scope,
            feature_selection=self._feature_selection,
            feature_space=self._feature_space,
            clustering_representation=self._clustering_representation,
            visualization_representation=self._visualization_representation,
            candidate_set=self._candidate_set,
            parent_revision=self._parent,
            notes=notes,
            created_by=created_by,
        )
        self._parent = revision
        return revision

    @property
    def current_scope(self) -> Scope | None:
        """The scope currently held by the builder, if any."""
        return self._scope

    @property
    def current_candidate_set(self) -> CandidateSet | None:
        """The candidate set currently held by the builder, if any."""
        return self._candidate_set

    @staticmethod
    def _require(value: object, what: str, step: str) -> None:
        if value is None:
            raise ValueError(f"{step} requires {what}; set it earlier in the builder")
