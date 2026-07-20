"""DataFolio-backed stable study substrate."""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from datafolio import DataFolio

from cellpax.artifacts import (
    commit_ref,
    component_manifest_ref,
    new_item_ref,
    resolve_checksum,
    validate_item_ref,
)
from cellpax.clustering import CandidateGenerator, builtin_generator
from cellpax.config import (
    CandidateCutConfig,
    ClusteringConfig,
    FeatureSpaceConfig,
    KeepConfig,
    PropagationConfig,
    RepresentationConfig,
)
from cellpax.contracts import (
    CONTRACTS,
    SCHEMA_VERSION,
    validate_registries,
    validate_table,
)
from cellpax.contracts.invariants import (
    assert_architecture_contracts,
    assert_registry_extension,
)
from cellpax.features import (
    FeatureDefinition,
    feature_catalog,
    feature_catalog_hash,
    prepare_feature_catalog,
    prepare_feature_selection_members,
)
from cellpax.identity import (
    canonical_hash,
    canonical_json_bytes,
    membership_hash,
    new_entity_id,
)
from cellpax.recipes import render_release_replay_script, resolved_release_recipe
from cellpax.records import (
    AnnotationRelease,
    AssignmentSet,
    CandidateSet,
    ClusteringRun,
    Decision,
    FeatureBlock,
    FeatureSelection,
    FeatureSpace,
    KeptRevision,
    PropagationRun,
    Representation,
    Scope,
    StudyCommit,
    Universe,
)
from cellpax.release import (
    RELEASE_ARTIFACT_NAMES,
    RELEASE_QUALITY,
    ReleaseBundle,
    generate_enum_binding,
    release_quality_summary,
    validate_release_manifest,
)
from cellpax.review import DecisionActionConfig, validate_decision_semantics
from cellpax.scopes import normalize_cell_ids, require_membership_subset
from cellpax.spaces import (
    assemble_selected_values,
    feature_column,
    fit_feature_space,
    fit_representation,
)
from cellpax.table_utils import table_schema_hash, validate_cell_table
from cellpax.taxonomy import taxonomy_enum, validate_taxonomy
from cellpax.universe import semantic_roles_json, validate_universe_cells

if TYPE_CHECKING:
    from cellpax.builder import RevisionBuilder
    from cellpax.celldata import CellData

_STUDY_ID = "cellpax_study_id"
_SCHEMA_VERSION = "cellpax_schema_version"
_HEAD_COMMIT = "cellpax_head_commit_id"
_CREATED_BY = "cellpax_created_by"
_PROTECTED_METADATA = frozenset({_STUDY_ID, _SCHEMA_VERSION, _HEAD_COMMIT, _CREATED_BY})
_NUMERIC_SOFTWARE = ("numpy", "scikit-learn")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _software_versions(*, include: Sequence[str] = ()) -> str:
    packages = {
        "cellpax": version("cellpax"),
        "datafolio": version("datafolio"),
        "datafolio_api": "2.0",
        "polars": pl.__version__,
        "python": platform.python_version(),
    }
    packages.update(
        {distribution: version(distribution) for distribution in sorted(set(include))}
    )
    return canonical_json_bytes(packages).decode("utf-8")


def _typed_frame(name: str, rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=CONTRACTS[name].schema)


def _normalized_refs(refs: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(dict.fromkeys(validate_item_ref(ref) for ref in refs)))
    if len(normalized) != len(refs):
        raise ValueError("source_refs cannot contain duplicates")
    return normalized


def _universe_from_row(row: Mapping[str, Any]) -> Universe:
    return Universe(
        universe_id=row["universe_id"],
        cells_ref=row["cells_ref"],
        schema_hash=row["schema_hash"],
        semantic_roles_json=row["semantic_roles_json"],
        source_refs=tuple(row["source_refs"]),
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _feature_block_from_row(row: Mapping[str, Any]) -> FeatureBlock:
    return FeatureBlock(
        feature_block_id=row["feature_block_id"],
        universe_id=row["universe_id"],
        values_ref=row["values_ref"],
        schema_hash=row["schema_hash"],
        source_refs=tuple(row["source_refs"]),
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _selection_from_row(row: Mapping[str, Any]) -> FeatureSelection:
    return FeatureSelection(
        feature_selection_id=row["feature_selection_id"],
        catalog_hash=row["catalog_hash"],
        members_ref=row["members_ref"],
        n_features=row["n_features"],
        derivation_text=row["derivation_text"],
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _scope_from_row(row: Mapping[str, Any]) -> Scope:
    return Scope(
        scope_id=row["scope_id"],
        universe_id=row["universe_id"],
        parent_scope_id=row["parent_scope_id"],
        members_ref=row["members_ref"],
        membership_hash=row["membership_hash"],
        n_cells=row["n_cells"],
        derivation_text=row["derivation_text"],
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _feature_space_from_row(row: Mapping[str, Any]) -> FeatureSpace:
    blocks = row["input_feature_block_ids"]
    return FeatureSpace(
        feature_space_id=row["feature_space_id"],
        scope_id=row["scope_id"],
        fit_scope_id=row["fit_scope_id"],
        parent_feature_space_id=row["parent_feature_space_id"],
        input_feature_block_ids=None if blocks is None else tuple(blocks),
        feature_selection_id=row["feature_selection_id"],
        transform=row["transform"],
        params_json=row["params_json"],
        fitted_state_ref=row["fitted_state_ref"],
        values_ref=row["values_ref"],
        missing_policy=row["missing_policy"],
        seed=row["seed"],
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _representation_from_row(row: Mapping[str, Any]) -> Representation:
    return Representation(
        representation_id=row["representation_id"],
        scope_id=row["scope_id"],
        fit_scope_id=row["fit_scope_id"],
        method=row["method"],
        input_feature_space_id=row["input_feature_space_id"],
        spatial_input_json=row["spatial_input_json"],
        n_components=row["n_components"],
        params_json=row["params_json"],
        fitted_state_ref=row["fitted_state_ref"],
        coords_ref=row["coords_ref"],
        seed=row["seed"],
        recompute_deterministic=row["recompute_deterministic"],
        software_versions=row["software_versions"],
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _clustering_run_from_row(row: Mapping[str, Any]) -> ClusteringRun:
    return ClusteringRun(**dict(row))


def _candidate_set_from_row(row: Mapping[str, Any]) -> CandidateSet:
    return CandidateSet(**dict(row))


def _decision_from_row(row: Mapping[str, Any]) -> Decision:
    values = dict(row)
    values["target_ids"] = (
        None if values["target_ids"] is None else tuple(values["target_ids"])
    )
    values["evidence_refs"] = tuple(values["evidence_refs"])
    return Decision(**values)


def _assignment_set_from_row(row: Mapping[str, Any]) -> AssignmentSet:
    return AssignmentSet(**dict(row))


def _annotation_release_from_row(row: Mapping[str, Any]) -> AnnotationRelease:
    return AnnotationRelease(**dict(row))


def _propagation_run_from_row(row: Mapping[str, Any]) -> PropagationRun:
    return PropagationRun(**dict(row))


def _revision_from_row(row: Mapping[str, Any]) -> KeptRevision:
    return KeptRevision(
        revision_id=row["revision_id"],
        name=row["name"],
        parent_revision_id=row["parent_revision_id"],
        scope_id=row["scope_id"],
        feature_space_id=row["feature_space_id"],
        clustering_representation_id=row["clustering_representation_id"],
        visualization_representation_id=row["visualization_representation_id"],
        candidate_set_id=row["candidate_set_id"],
        state_hash=row["state_hash"],
        config_hash=row["config_hash"],
        software_versions=row["software_versions"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        notes=row["notes"],
    )


class Study:
    """A revisioned CellPax study stored as a DataFolio 2 folio."""

    def __init__(self, folio: DataFolio) -> None:
        self._folio = folio
        self._registry_cache_head: str | None = None
        self._registry_cache: dict[str, pl.DataFrame] = {}
        self._validate_study_metadata()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        created_by: str,
        metadata: Mapping[str, object] | None = None,
    ) -> "Study":
        """Create a new study and initialize its DataFolio metadata."""
        if not created_by:
            raise ValueError("created_by is required")
        user_metadata = dict(metadata or {})
        protected = _PROTECTED_METADATA & set(user_metadata)
        if protected:
            raise ValueError(
                f"Metadata cannot override protected keys: {sorted(protected)}"
            )
        initial = {
            **user_metadata,
            _STUDY_ID: new_entity_id(),
            _SCHEMA_VERSION: SCHEMA_VERSION,
            _HEAD_COMMIT: None,
            _CREATED_BY: created_by,
        }
        return cls(DataFolio(path, metadata=initial))

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> "Study":
        """Open an existing study and verify its CellPax commit head."""
        study = cls(DataFolio(path, read_only=read_only))
        if study.head_commit_id is not None:
            study.head_commit()
        return study

    @property
    def folio(self) -> DataFolio:
        """The underlying DataFolio 2 object."""
        return self._folio

    @property
    def study_id(self) -> str:
        return str(self._folio.metadata[_STUDY_ID])

    @property
    def schema_version(self) -> str:
        return str(self._folio.metadata[_SCHEMA_VERSION])

    @property
    def head_commit_id(self) -> str | None:
        value = self._folio.metadata[_HEAD_COMMIT]
        return None if value is None else str(value)

    def _validate_study_metadata(self) -> None:
        missing = _PROTECTED_METADATA - set(self._folio.metadata)
        if missing:
            raise ValueError(f"Not a CellPax study; metadata missing {sorted(missing)}")
        if self._folio.metadata[_SCHEMA_VERSION] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CellPax schema version {self._folio.metadata[_SCHEMA_VERSION]!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def _author(self, created_by: str | None) -> str:
        return created_by or str(self._folio.metadata[_CREATED_BY])

    def _clear_registry_cache(self) -> None:
        self._registry_cache_head = None
        self._registry_cache.clear()

    def head_commit(self) -> StudyCommit | None:
        """Load and integrity-check the current CellPax commit manifest."""
        commit_id = self.head_commit_id
        if commit_id is None:
            return None
        manifest = self._folio.get(commit_ref(commit_id))
        expected_keys = {"schema_version", "parent_commit_id", "registry_refs"}
        if not isinstance(manifest, dict) or set(manifest) != expected_keys:
            raise ValueError(f"Malformed CellPax commit manifest {commit_id}")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Commit {commit_id} has an unsupported schema version")
        if canonical_hash(manifest) != commit_id:
            raise ValueError(f"CellPax commit manifest hash mismatch: {commit_id}")
        registry_refs = manifest["registry_refs"]
        if not isinstance(registry_refs, dict):
            raise ValueError(f"Commit {commit_id} registry_refs must be a mapping")
        for name, ref in registry_refs.items():
            if name not in CONTRACTS:
                raise ValueError(
                    f"Commit {commit_id} references unknown registry {name!r}"
                )
            validate_item_ref(ref)
        return StudyCommit(
            commit_id=commit_id,
            parent_commit_id=manifest["parent_commit_id"],
            registry_refs=tuple(sorted(registry_refs.items())),
        )

    def registry(self, name: str) -> pl.DataFrame:
        """Load the current immutable snapshot of a named registry."""
        table_contract = CONTRACTS.get(name)
        if table_contract is None:
            raise KeyError(f"Unknown CellPax registry {name!r}")
        commit_id = self.head_commit_id
        if self._registry_cache_head != commit_id:
            self._registry_cache_head = commit_id
            self._registry_cache.clear()
        cached = self._registry_cache.get(name)
        if cached is not None:
            return cached.clone()
        head = self.head_commit()
        if head is None or name not in head.registries:
            frame = pl.DataFrame(schema=table_contract.schema)
        else:
            frame = self._folio.get(head.registries[name], frame="polars")
            validate_table(frame, table_contract)
        self._registry_cache[name] = frame
        return frame.clone()

    def _existing_row(
        self, registry_name: str, key_column: str, key: object
    ) -> dict[str, object] | None:
        registry = self.registry(registry_name)
        matches = registry.filter(pl.col(key_column) == key)
        if matches.is_empty():
            return None
        return matches.row(0, named=True)

    def _lineage_inputs(
        self, previous_ref: str | None, rows: pl.DataFrame
    ) -> list[str]:
        candidates: list[str] = []
        if previous_ref is not None:
            candidates.append(previous_ref)
        for column in rows.columns:
            if column.endswith("_ref"):
                candidates.extend(
                    value
                    for value in rows[column].drop_nulls().to_list()
                    if isinstance(value, str)
                )
            elif column.endswith("_refs"):
                for values in rows[column].drop_nulls().to_list():
                    candidates.extend(
                        value for value in values if isinstance(value, str)
                    )
        existing: list[str] = []
        for ref in dict.fromkeys(candidates):
            try:
                self._folio.item_info(ref)
            except KeyError:
                continue
            existing.append(ref)
        return existing

    def _merge_registry(
        self,
        name: str,
        additions: pl.DataFrame,
    ) -> tuple[pl.DataFrame, bool]:
        table_contract = CONTRACTS[name]
        validate_table(additions, table_contract)
        current = self.registry(name)
        if additions.is_empty():
            return current, False

        keys = table_contract.primary_key
        current_rows = {
            tuple(row[column] for column in keys): row
            for row in current.iter_rows(named=True)
        }
        new_rows: list[dict[str, object]] = []
        for row in additions.iter_rows(named=True):
            key = tuple(row[column] for column in keys)
            existing = current_rows.get(key)
            if existing is None:
                current_rows[key] = row
                new_rows.append(row)
            elif existing != row:
                raise ValueError(
                    f"Immutable registry {name!r} already has different row for key {key!r}"
                )
        if not new_rows:
            return current, False
        appended = _typed_frame(name, new_rows)
        combined = pl.concat([current, appended], how="vertical")
        assert_registry_extension(current, combined, table_contract)
        return combined, True

    def _commit_registry_changes(
        self, changes: Mapping[str, pl.DataFrame]
    ) -> StudyCommit:
        """Publish registry snapshots and a commit inside an active folio batch."""
        parent = self.head_commit()
        registry_refs = {} if parent is None else parent.registries
        changed = False
        for name, additions in changes.items():
            if name not in CONTRACTS:
                raise KeyError(f"Unknown CellPax registry {name!r}")
            combined, did_change = self._merge_registry(name, additions)
            if not did_change:
                continue
            previous_ref = registry_refs.get(name)
            registry_ref = new_item_ref(name, collection="registries")
            self._folio.add(
                registry_ref,
                combined,
                description=f"CellPax {name} registry snapshot",
                inputs=self._lineage_inputs(previous_ref, additions),
            )
            registry_refs[name] = registry_ref
            changed = True

        if not changed:
            if parent is None:
                raise ValueError("Cannot create an empty initial CellPax commit")
            return parent

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "parent_commit_id": None if parent is None else parent.commit_id,
            "registry_refs": dict(sorted(registry_refs.items())),
        }
        commit_id = canonical_hash(manifest)
        ref = commit_ref(commit_id)
        self._folio.add(
            ref,
            manifest,
            description="Immutable CellPax registry commit",
            inputs=list(manifest["registry_refs"].values()),
        )
        self._folio.metadata[_HEAD_COMMIT] = commit_id
        self._clear_registry_cache()
        return StudyCommit(
            commit_id=commit_id,
            parent_commit_id=manifest["parent_commit_id"],
            registry_refs=tuple(manifest["registry_refs"].items()),
        )

    def universe(self) -> Universe:
        """Return the study's one authoritative universe."""
        registry = self.registry("universe")
        if registry.height != 1:
            raise ValueError(
                f"Study must contain one universe; found {registry.height}"
            )
        return _universe_from_row(registry.row(0, named=True))

    def register_universe(
        self,
        cells: pl.DataFrame | pl.Series | Iterable[int],
        *,
        semantic_roles: Mapping[str, Sequence[str]] | None = None,
        nullable_columns: Sequence[str] = (),
        source_refs: Sequence[str] = (),
        created_by: str | None = None,
    ) -> Universe:
        """Validate and atomically register the authoritative cell universe.

        ``cells`` may be a DataFrame (with a ``cell_id`` column plus any semantic
        columns) or a bare iterable / Series of integer cell ids.
        """
        if not isinstance(cells, pl.DataFrame):
            cells = normalize_cell_ids(cells)
        semantic_roles = {} if semantic_roles is None else semantic_roles
        validate_universe_cells(cells, nullable_columns=nullable_columns)
        roles_json = semantic_roles_json(cells, semantic_roles)
        schema_hash = table_schema_hash(cells, nullable_columns=nullable_columns)
        author = self._author(created_by)
        normalized_sources = _normalized_refs(source_refs)
        for ref in normalized_sources:
            self._folio.item_info(ref)

        existing = self.registry("universe")
        if not existing.is_empty():
            universe = _universe_from_row(existing.row(0, named=True))
            stored_cells = self._folio.get(universe.cells_ref, frame="polars")
            if (
                stored_cells.equals(cells)
                and universe.schema_hash == schema_hash
                and universe.semantic_roles_json == roles_json
                and universe.source_refs == normalized_sources
            ):
                return universe
            raise ValueError("A study's authoritative universe cannot be replaced")

        with self._folio.batch():
            cells_ref = new_item_ref("universe-cells")
            self._folio.add(
                cells_ref,
                cells,
                description="Authoritative CellPax cell universe",
                inputs=list(normalized_sources),
            )
            manifest = {
                "component_kind": "universe",
                "schema_version": SCHEMA_VERSION,
                "cells_checksum": resolve_checksum(self._folio, cells_ref),
                "schema_hash": schema_hash,
                "semantic_roles": json.loads(roles_json),
                "nullable_columns": sorted(nullable_columns),
                "source_refs": list(normalized_sources),
            }
            universe = Universe(
                universe_id=canonical_hash(manifest),
                cells_ref=cells_ref,
                schema_hash=schema_hash,
                semantic_roles_json=roles_json,
                source_refs=normalized_sources,
                created_at=_utcnow(),
                created_by=author,
            )
            self._folio.add(
                component_manifest_ref("universe", universe.universe_id),
                manifest,
                description="Canonical CellPax universe manifest",
                inputs=[cells_ref],
            )
            self._commit_registry_changes(
                {"universe": _typed_frame("universe", [universe.row()])}
            )
        return universe

    def register_feature_block(
        self,
        values: pl.DataFrame,
        catalog: pl.DataFrame | Iterable[FeatureDefinition],
        *,
        nullable_columns: Sequence[str] = (),
        source_refs: Sequence[str] = (),
        created_by: str | None = None,
    ) -> FeatureBlock:
        """Validate and atomically register feature values plus their catalog.

        ``catalog`` may be a strict catalog DataFrame or an iterable of
        :class:`~cellpax.features.FeatureDefinition` values, which are built into
        one for you.
        """
        if not isinstance(catalog, pl.DataFrame):
            catalog = feature_catalog(catalog)
        universe = self.universe()
        validate_cell_table(values, nullable_columns=nullable_columns)
        value_columns = [column for column in values.columns if column != "cell_id"]
        if not value_columns:
            raise ValueError("A feature block requires at least one feature column")
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        require_membership_subset(
            values.select("cell_id"), universe_cells, container_name="the universe"
        )

        schema_hash = table_schema_hash(values, nullable_columns=nullable_columns)
        author = self._author(created_by)
        normalized_sources = _normalized_refs(source_refs)
        for ref in normalized_sources:
            self._folio.item_info(ref)
        with self._folio.batch():
            values_ref = new_item_ref("feature-values")
            self._folio.add(
                values_ref,
                values,
                description="Immutable CellPax feature block values",
                inputs=list(normalized_sources),
            )
            manifest = {
                "component_kind": "feature_block",
                "schema_version": SCHEMA_VERSION,
                "universe_id": universe.universe_id,
                "values_checksum": resolve_checksum(self._folio, values_ref),
                "schema_hash": schema_hash,
                "nullable_columns": sorted(nullable_columns),
                "source_refs": list(normalized_sources),
            }
            block = FeatureBlock(
                feature_block_id=canonical_hash(manifest),
                universe_id=universe.universe_id,
                values_ref=values_ref,
                schema_hash=schema_hash,
                source_refs=normalized_sources,
                created_at=_utcnow(),
                created_by=author,
            )
            catalog_rows = prepare_feature_catalog(
                catalog,
                feature_block_id=block.feature_block_id,
                value_columns=value_columns,
            )
            existing = self._existing_row(
                "feature_block", "feature_block_id", block.feature_block_id
            )
            if existing is not None:
                existing_catalog = self.feature_catalog().filter(
                    pl.col("feature_block_id") == block.feature_block_id
                )
                if not existing_catalog.sort("feature_id").equals(
                    catalog_rows.sort("feature_id")
                ):
                    raise ValueError(
                        "Feature block content is already registered with a "
                        "different semantic catalog"
                    )
                self._folio.delete(values_ref, warn_dependents=False)
                return _feature_block_from_row(existing)

            self._folio.add(
                component_manifest_ref("feature-block", block.feature_block_id),
                manifest,
                description="Canonical CellPax feature-block manifest",
                inputs=[values_ref],
            )
            self._commit_registry_changes(
                {
                    "feature_block": _typed_frame("feature_block", [block.row()]),
                    "feature_catalog": catalog_rows,
                }
            )
        return block

    def feature_catalog(self) -> pl.DataFrame:
        """Return the current complete semantic feature catalog."""
        return self.registry("feature_catalog")

    def get_feature_block(self, feature_block_id: str) -> FeatureBlock:
        """Load a registered feature block."""
        row = self._existing_row("feature_block", "feature_block_id", feature_block_id)
        if row is None:
            raise KeyError(f"Unknown feature block {feature_block_id!r}")
        return _feature_block_from_row(row)

    def _derive_column(self, op: str, column: str) -> pl.Expr:
        expr = pl.col(column)
        if op == "identity":
            return expr
        if op == "log":
            return expr.log()
        if op == "log1p":
            return expr.log1p()
        if op == "log10":
            return expr.log(base=10)
        if op == "sqrt":
            return expr.sqrt()
        raise ValueError(f"Unsupported derivation op {op!r}")

    def derive_feature_block(
        self,
        source: FeatureBlock | str,
        ops: Mapping[str, tuple[str, str]],
        *,
        created_by: str | None = None,
    ) -> FeatureBlock:
        """Register a derived feature block by applying per-column ops to a source.

        ``ops`` maps each new feature id to a ``(source_feature_id, op)`` pair,
        where ``op`` is one of ``identity``, ``log``, ``log1p``, ``log10``,
        ``sqrt``. The derived block records the source block as lineage and marks
        every feature ``derived`` with a description naming its operation.
        """
        source_block = (
            self.get_feature_block(source) if isinstance(source, str) else source
        )
        source_values = self._folio.get(source_block.values_ref, frame="polars")
        source_catalog = self.feature_catalog().filter(
            pl.col("feature_block_id") == source_block.feature_block_id
        )
        catalog_by_feature = {
            row["feature_id"]: row for row in source_catalog.iter_rows(named=True)
        }
        if not ops:
            raise ValueError("derive_feature_block requires at least one derivation")
        exprs: list[pl.Expr] = [pl.col("cell_id")]
        definitions: list[FeatureDefinition] = []
        for new_id, spec in ops.items():
            source_id, op = spec
            catalog_row = catalog_by_feature.get(source_id)
            if catalog_row is None:
                raise ValueError(f"Unknown source feature {source_id!r}")
            exprs.append(
                self._derive_column(op, catalog_row["column_name"]).alias(new_id)
            )
            definitions.append(
                FeatureDefinition(
                    new_id,
                    modality=catalog_row["modality"],
                    family=catalog_row["family"],
                    units=catalog_row["units"],
                    description=f"{op}({source_id})",
                    raw_or_derived="derived",
                )
            )
        values = source_values.select(exprs)
        return self.register_feature_block(
            values,
            definitions,
            source_refs=(source_block.values_ref,),
            created_by=created_by,
        )

    def auto_log_features(
        self,
        source: FeatureBlock | str,
        *,
        scope: Scope | str | None = None,
        metric: str = "both",
        skew_threshold: float = 1.0,
        range_threshold: float = 10.0,
        method: str = "log1p",
        created_by: str | None = None,
    ) -> FeatureBlock:
        """Register a derived block that log-scales sufficiently wide features.

        Each source feature's distribution is measured on ``scope`` (the whole
        universe by default). A non-negative feature is log-scaled with ``method``
        when it passes the chosen ``metric``: ``skew`` (skewness > ``skew_threshold``),
        ``dynamic_range`` (p99/p1 > ``range_threshold``), or ``both`` (the default,
        requiring both). Other features pass through unchanged. The block keeps the
        same feature ids; each catalog description records the resolved decision so
        it is auditable and replay reproduces the identical set.
        """
        if metric not in {"skew", "dynamic_range", "both"}:
            raise ValueError("metric must be 'skew', 'dynamic_range', or 'both'")
        if method not in {"log", "log1p", "log10"}:
            raise ValueError("method must be 'log', 'log1p', or 'log10'")
        source_block = (
            self.get_feature_block(source) if isinstance(source, str) else source
        )
        values = self._folio.get(source_block.values_ref, frame="polars")
        if scope is not None:
            scope_record = self.get_scope(scope) if isinstance(scope, str) else scope
            members = self._folio.get(scope_record.members_ref, frame="polars")
            measured = values.join(members, on="cell_id", how="inner")
        else:
            measured = values
        source_catalog = self.feature_catalog().filter(
            pl.col("feature_block_id") == source_block.feature_block_id
        )
        exprs: list[pl.Expr] = [pl.col("cell_id")]
        definitions: list[FeatureDefinition] = []
        for row in source_catalog.sort("feature_id").iter_rows(named=True):
            column = row["column_name"]
            sample = measured[column].drop_nulls().to_numpy()
            logged, note = self._auto_log_decision(
                sample, metric, skew_threshold, range_threshold
            )
            op = method if logged else "identity"
            exprs.append(self._derive_column(op, column).alias(row["feature_id"]))
            definitions.append(
                FeatureDefinition(
                    row["feature_id"],
                    modality=row["modality"],
                    family=row["family"],
                    units=row["units"],
                    description=f"{op} (auto: {note})",
                    raw_or_derived="derived",
                )
            )
        derived = values.select(exprs)
        return self.register_feature_block(
            derived,
            definitions,
            source_refs=(source_block.values_ref,),
            created_by=created_by,
        )

    @staticmethod
    def _auto_log_decision(
        sample: Any,
        metric: str,
        skew_threshold: float,
        range_threshold: float,
    ) -> tuple[bool, str]:
        import numpy as np

        if sample.size == 0:
            return False, "no values"
        minimum = float(sample.min())
        if minimum < 0:
            return False, f"has negatives (min={minimum:.3g})"
        mean = float(sample.mean())
        std = float(sample.std())
        skewness = 0.0 if std == 0 else float((((sample - mean) / std) ** 3).mean())
        low, high = (float(value) for value in np.percentile(sample, [1, 99]))
        dynamic_range = high / low if low > 0 else float("inf")
        pass_skew = skewness > skew_threshold
        pass_range = dynamic_range > range_threshold
        if metric == "skew":
            decide = pass_skew
        elif metric == "dynamic_range":
            decide = pass_range
        else:
            decide = pass_skew and pass_range
        return decide, f"skew={skewness:.2f}, p99/p1={dynamic_range:.1f}"

    def preview_feature_selection(
        self,
        selected: pl.DataFrame | FeatureBlock | str,
        feature_ids: Sequence[str] | None = None,
        *,
        derivation_text: str | None = None,
        created_by: str | None = None,
    ) -> FeatureSelection:
        """Materialize an ordered feature selection without keeping a revision.

        ``selected`` may be a ``(feature_block_id, feature_id)`` DataFrame, or a
        feature block (or its id) plus an optional ``feature_ids`` list. Omitting
        ``feature_ids`` selects every feature in that block, in catalog order.
        """
        catalog = self.feature_catalog()
        if catalog.is_empty():
            raise ValueError(
                "Cannot select features before registering a feature block"
            )
        if isinstance(selected, (FeatureBlock, str)):
            block_id = (
                selected.feature_block_id
                if isinstance(selected, FeatureBlock)
                else selected
            )
            block_catalog = catalog.filter(pl.col("feature_block_id") == block_id)
            if block_catalog.is_empty():
                raise ValueError(f"Unknown feature block {block_id!r}")
            chosen = (
                block_catalog["feature_id"].to_list()
                if feature_ids is None
                else list(feature_ids)
            )
            selected = pl.DataFrame(
                {
                    "feature_block_id": [block_id] * len(chosen),
                    "feature_id": chosen,
                },
                schema={"feature_block_id": pl.String, "feature_id": pl.String},
            )
        elif feature_ids is not None:
            raise TypeError(
                "feature_ids is only valid with a feature block, not a DataFrame"
            )
        members = prepare_feature_selection_members(selected, catalog)
        if derivation_text is None:
            derivation_text = f"selection of {members.height} features"
        elif not derivation_text:
            raise ValueError("derivation_text must be a non-empty string when given")
        catalog_digest = feature_catalog_hash(catalog)
        semantic_members = members.select("feature_block_id", "feature_id").to_dicts()
        selection_id = canonical_hash(
            {
                "component_kind": "feature_selection",
                "schema_version": SCHEMA_VERSION,
                "catalog_hash": catalog_digest,
                "members": semantic_members,
                "n_features": members.height,
            }
        )
        existing = self._existing_row(
            "feature_selection", "feature_selection_id", selection_id
        )
        if existing is not None:
            return _selection_from_row(existing)

        members_ref = new_item_ref("feature-selection-members")
        head = self.head_commit()
        inputs = []
        if head is not None and "feature_catalog" in head.registries:
            inputs.append(head.registries["feature_catalog"])
        self._folio.add(
            members_ref,
            members,
            description="Ordered CellPax feature selection preview",
            inputs=inputs,
        )
        return FeatureSelection(
            feature_selection_id=selection_id,
            catalog_hash=catalog_digest,
            members_ref=members_ref,
            n_features=members.height,
            derivation_text=derivation_text,
            created_at=_utcnow(),
            created_by=self._author(created_by),
        )

    def select_all_features(
        self,
        block: FeatureBlock | str,
        *,
        derivation_text: str | None = None,
        created_by: str | None = None,
    ) -> FeatureSelection:
        """Select every feature in one block, in catalog order."""
        return self.preview_feature_selection(
            block, None, derivation_text=derivation_text, created_by=created_by
        )

    def get_feature_selection(self, selection_id: str) -> FeatureSelection:
        """Load a registered ordered feature selection."""
        row = self._existing_row(
            "feature_selection", "feature_selection_id", selection_id
        )
        if row is None:
            raise KeyError(f"Unknown feature selection {selection_id!r}")
        return _selection_from_row(row)

    def preview_scope(
        self,
        cells: pl.DataFrame | pl.Series | Iterable[int],
        *,
        derivation_text: str | None = None,
        parent: Scope | str | None = None,
        created_by: str | None = None,
    ) -> Scope:
        """Materialize an immutable scope preview without keeping a revision."""
        universe = self.universe()
        members = normalize_cell_ids(cells)
        if derivation_text is None:
            derivation_text = f"scope of {members.height} cells"
        elif not derivation_text:
            raise ValueError("derivation_text must be a non-empty string when given")
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        require_membership_subset(
            members, universe_cells, container_name="the universe"
        )

        parent_scope: Scope | None
        if isinstance(parent, str):
            parent_scope = self.get_scope(parent)
        else:
            parent_scope = parent
        if parent_scope is not None:
            if parent_scope.universe_id != universe.universe_id:
                raise ValueError("Parent scope belongs to a different universe")
            parent_members = self._folio.get(parent_scope.members_ref, frame="polars")
            require_membership_subset(
                members, parent_members, container_name="the parent scope"
            )

        member_digest = membership_hash(members["cell_id"].to_list())
        scope_id = canonical_hash(
            {
                "component_kind": "scope",
                "schema_version": SCHEMA_VERSION,
                "universe_id": universe.universe_id,
                "parent_scope_id": None
                if parent_scope is None
                else parent_scope.scope_id,
                "membership_hash": member_digest,
                "n_cells": members.height,
                "derivation_text": derivation_text,
            }
        )
        existing = self._existing_row("scope", "scope_id", scope_id)
        if existing is not None:
            return _scope_from_row(existing)

        members_ref = new_item_ref("scope-members")
        inputs = [universe.cells_ref]
        if parent_scope is not None:
            inputs.append(parent_scope.members_ref)
        self._folio.add(
            members_ref,
            members,
            description="Immutable CellPax scope preview",
            inputs=inputs,
        )
        return Scope(
            scope_id=scope_id,
            universe_id=universe.universe_id,
            parent_scope_id=None if parent_scope is None else parent_scope.scope_id,
            members_ref=members_ref,
            membership_hash=member_digest,
            n_cells=members.height,
            derivation_text=derivation_text,
            created_at=_utcnow(),
            created_by=self._author(created_by),
        )

    def get_scope(self, scope_id: str) -> Scope:
        """Load a registered scope."""
        row = self._existing_row("scope", "scope_id", scope_id)
        if row is None:
            raise KeyError(f"Unknown scope {scope_id!r}")
        return _scope_from_row(row)

    def get_revision(self, revision_id: str) -> KeptRevision:
        """Load a named kept revision by entity id."""
        row = self._existing_row("kept_revision", "revision_id", revision_id)
        if row is None:
            raise KeyError(f"Unknown kept revision {revision_id!r}")
        return _revision_from_row(row)

    def revision_lineage(
        self, revision: KeptRevision | str
    ) -> tuple[KeptRevision, ...]:
        """Return a revision's root-to-head ancestry."""
        cursor = self.get_revision(revision) if isinstance(revision, str) else revision
        lineage: list[KeptRevision] = []
        seen: set[str] = set()
        while True:
            if cursor.revision_id in seen:
                raise ValueError("Revision lineage contains a cycle")
            seen.add(cursor.revision_id)
            lineage.append(cursor)
            if cursor.parent_revision_id is None:
                break
            cursor = self.get_revision(cursor.parent_revision_id)
        return tuple(reversed(lineage))

    def get_feature_space(self, feature_space_id: str) -> FeatureSpace:
        """Load a registered fitted feature layer."""
        row = self._existing_row("feature_space", "feature_space_id", feature_space_id)
        if row is None:
            raise KeyError(f"Unknown feature space {feature_space_id!r}")
        return _feature_space_from_row(row)

    def get_representation(self, representation_id: str) -> Representation:
        """Load a registered coordinate representation."""
        row = self._existing_row(
            "representation", "representation_id", representation_id
        )
        if row is None:
            raise KeyError(f"Unknown representation {representation_id!r}")
        return _representation_from_row(row)

    def get_clustering_run(self, clustering_run_id: str) -> ClusteringRun:
        """Load a registered expensive clustering run."""
        row = self._existing_row(
            "clustering_run", "clustering_run_id", clustering_run_id
        )
        if row is None:
            raise KeyError(f"Unknown clustering run {clustering_run_id!r}")
        return _clustering_run_from_row(row)

    def get_candidate_set(self, candidate_set_id: str) -> CandidateSet:
        """Load a registered candidate set."""
        row = self._existing_row("candidate_set", "candidate_set_id", candidate_set_id)
        if row is None:
            raise KeyError(f"Unknown candidate set {candidate_set_id!r}")
        return _candidate_set_from_row(row)

    def preview_feature_space(
        self,
        *,
        scope: Scope | str,
        fit_scope: Scope | str | None = None,
        feature_selection: FeatureSelection | str,
        config: FeatureSpaceConfig,
        parent: FeatureSpace | str | None = None,
        created_by: str | None = None,
    ) -> FeatureSpace:
        """Fit and materialize one immutable feature-space stage.

        ``fit_scope`` defaults to ``scope`` — pass it explicitly only when the
        transform should learn from a different set of cells than it is applied to.
        """
        scope_record = self.get_scope(scope) if isinstance(scope, str) else scope
        if fit_scope is None:
            fit_scope_record = scope_record
        else:
            fit_scope_record = (
                self.get_scope(fit_scope) if isinstance(fit_scope, str) else fit_scope
            )
        selection = (
            self.get_feature_selection(feature_selection)
            if isinstance(feature_selection, str)
            else feature_selection
        )
        parent_record = (
            self.get_feature_space(parent) if isinstance(parent, str) else parent
        )
        universe = self.universe()
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        scope_members = self._validate_scope_record(
            scope_record, universe, universe_cells
        )
        fit_members = self._validate_scope_record(
            fit_scope_record, universe, universe_cells
        )
        catalog = self.feature_catalog()
        selection_members = self._validate_feature_selection_record(selection, catalog)

        source_refs: list[str]
        if parent_record is None:
            block_ids = tuple(
                sorted(set(selection_members["feature_block_id"].to_list()))
            )
            block_registry = self.registry("feature_block")
            blocks: dict[str, pl.DataFrame] = {}
            source_refs = [selection.members_ref]
            for block_id in block_ids:
                rows = block_registry.filter(pl.col("feature_block_id") == block_id)
                if rows.height != 1:
                    raise ValueError(f"Unknown selected feature block {block_id!r}")
                block = _feature_block_from_row(rows.row(0, named=True))
                blocks[block_id] = self._folio.get(block.values_ref, frame="polars")
                source_refs.append(block.values_ref)
            scope_input = assemble_selected_values(
                scope_members, selection_members, catalog, blocks
            )
            fit_input = assemble_selected_values(
                fit_members, selection_members, catalog, blocks
            )
            parent_id = None
            input_block_ids: tuple[str, ...] | None = block_ids
        else:
            if parent_record.feature_selection_id != selection.feature_selection_id:
                raise ValueError(
                    "A child feature space must retain its parent selection"
                )
            if parent_record.values_ref is None:
                raise ValueError("Parent feature space has no materialized values")
            parent_values = self._folio.get(parent_record.values_ref, frame="polars")
            require_membership_subset(
                scope_members,
                parent_values,
                container_name="the parent feature-space values",
            )
            require_membership_subset(
                fit_members,
                parent_values,
                container_name="the parent feature-space values",
            )
            scope_input = scope_members.join(parent_values, on="cell_id", how="left")
            fit_input = fit_members.join(parent_values, on="cell_id", how="left")
            source_refs = [parent_record.values_ref, selection.members_ref]
            parent_id = parent_record.feature_space_id
            input_block_ids = None

        values, estimator, report = fit_feature_space(
            scope_input, fit_input, selection_members, config
        )
        author = self._author(created_by)
        with self._folio.batch():
            report_ref = new_item_ref("feature-space-missingness")
            self._folio.add(
                report_ref,
                report,
                description="CellPax feature-space missingness report",
                inputs=source_refs,
            )
            fitted_state_ref: str | None = None
            if estimator is not None:
                fitted_state_ref = new_item_ref("feature-space-fitted-state")
                self._folio.add_model(
                    fitted_state_ref,
                    estimator,
                    description="Fitted CellPax feature-space transformer",
                    inputs=[*source_refs, report_ref],
                )
            values_ref = new_item_ref("feature-space-values")
            values_inputs = [*source_refs, report_ref]
            if fitted_state_ref is not None:
                values_inputs.append(fitted_state_ref)
            self._folio.add(
                values_ref,
                values,
                description="Materialized CellPax feature-space values",
                inputs=values_inputs,
            )
            manifest = {
                "component_kind": "feature_space",
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_record.scope_id,
                "fit_scope_id": fit_scope_record.scope_id,
                "parent_feature_space_id": parent_id,
                "input_feature_block_ids": (
                    None if input_block_ids is None else list(input_block_ids)
                ),
                "feature_selection_id": selection.feature_selection_id,
                "transform": config.transform,
                "params": config.params,
                "fitted_state_checksum": (
                    None
                    if fitted_state_ref is None
                    else resolve_checksum(self._folio, fitted_state_ref)
                ),
                "values_checksum": resolve_checksum(self._folio, values_ref),
                "missing_policy": config.missing_policy,
                "seed": config.seed,
            }
            record = FeatureSpace(
                feature_space_id=canonical_hash(manifest),
                scope_id=scope_record.scope_id,
                fit_scope_id=fit_scope_record.scope_id,
                parent_feature_space_id=parent_id,
                input_feature_block_ids=input_block_ids,
                feature_selection_id=selection.feature_selection_id,
                transform=config.transform,
                params_json=config.params_json,
                fitted_state_ref=fitted_state_ref,
                values_ref=values_ref,
                missing_policy=config.missing_policy,
                seed=config.seed,
                created_at=_utcnow(),
                created_by=author,
            )
            existing = self._existing_row(
                "feature_space", "feature_space_id", record.feature_space_id
            )
            if existing is not None:
                refs = [report_ref, values_ref]
                if fitted_state_ref is not None:
                    refs.append(fitted_state_ref)
                self._folio.delete(refs, warn_dependents=False)
                return _feature_space_from_row(existing)
            manifest_ref = component_manifest_ref(
                "feature-space", record.feature_space_id
            )
            self._folio.add(
                manifest_ref,
                manifest,
                description="Canonical CellPax feature-space manifest",
                inputs=values_inputs,
            )
        return record

    def feature_space_missingness(
        self, feature_space: FeatureSpace | str
    ) -> pl.DataFrame:
        """Load the persisted missingness report for a materialized feature space."""
        record = (
            self.get_feature_space(feature_space)
            if isinstance(feature_space, str)
            else feature_space
        )
        if record.values_ref is None:
            raise ValueError("Feature space has no materialized values")
        matches = [
            ref
            for ref in self._folio.get_inputs(record.values_ref)
            if ref.startswith("cellpax/objects/feature-space-missingness/")
        ]
        if len(matches) != 1:
            raise ValueError("Feature space missingness lineage is malformed")
        return self._folio.get(matches[0], frame="polars")

    def preview_representation(
        self,
        *,
        scope: Scope | str,
        fit_scope: Scope | str | None = None,
        feature_space: FeatureSpace | str,
        config: RepresentationConfig,
        created_by: str | None = None,
    ) -> Representation:
        """Fit and materialize versioned clustering or visualization coordinates.

        ``fit_scope`` defaults to ``scope``.
        """
        scope_record = self.get_scope(scope) if isinstance(scope, str) else scope
        if fit_scope is None:
            fit_scope_record = scope_record
        else:
            fit_scope_record = (
                self.get_scope(fit_scope) if isinstance(fit_scope, str) else fit_scope
            )
        space = (
            self.get_feature_space(feature_space)
            if isinstance(feature_space, str)
            else feature_space
        )
        if space.values_ref is None:
            raise ValueError("Input feature space has no materialized values")
        values = self._folio.get(space.values_ref, frame="polars")
        universe = self.universe()
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        scope_members = self._validate_scope_record(
            scope_record, universe, universe_cells
        )
        fit_members = self._validate_scope_record(
            fit_scope_record, universe, universe_cells
        )
        require_membership_subset(
            scope_members, values, container_name="the input feature space"
        )
        require_membership_subset(
            fit_members, values, container_name="the input feature space"
        )
        scope_values = scope_members.join(values, on="cell_id", how="left")
        fit_values = fit_members.join(values, on="cell_id", how="left")
        coords, estimator = fit_representation(scope_values, fit_values, config)
        software_versions = _software_versions(include=_NUMERIC_SOFTWARE)
        author = self._author(created_by)
        with self._folio.batch():
            fitted_state_ref: str | None = None
            if estimator is not None:
                fitted_state_ref = new_item_ref("representation-fitted-state")
                self._folio.add_model(
                    fitted_state_ref,
                    estimator,
                    description="Fitted CellPax representation transformer",
                    inputs=[space.values_ref],
                )
            coords_ref = new_item_ref("representation-coordinates")
            inputs = [space.values_ref]
            if fitted_state_ref is not None:
                inputs.append(fitted_state_ref)
            self._folio.add(
                coords_ref,
                coords,
                description="Materialized CellPax representation coordinates",
                inputs=inputs,
            )
            manifest = {
                "component_kind": "representation",
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_record.scope_id,
                "fit_scope_id": fit_scope_record.scope_id,
                "method": config.method,
                "input_feature_space_id": space.feature_space_id,
                "spatial_input": None,
                "n_components": config.n_components,
                "params": config.params,
                "fitted_state_checksum": (
                    None
                    if fitted_state_ref is None
                    else resolve_checksum(self._folio, fitted_state_ref)
                ),
                "coords_checksum": resolve_checksum(self._folio, coords_ref),
                "seed": config.seed,
                "recompute_deterministic": config.recompute_deterministic,
                "software_versions": json.loads(software_versions),
            }
            record = Representation(
                representation_id=canonical_hash(manifest),
                scope_id=scope_record.scope_id,
                fit_scope_id=fit_scope_record.scope_id,
                method=config.method,
                input_feature_space_id=space.feature_space_id,
                spatial_input_json=config.spatial_input_json,
                n_components=config.n_components,
                params_json=config.params_json,
                fitted_state_ref=fitted_state_ref,
                coords_ref=coords_ref,
                seed=config.seed,
                recompute_deterministic=config.recompute_deterministic,
                software_versions=software_versions,
                created_at=_utcnow(),
                created_by=author,
            )
            existing = self._existing_row(
                "representation", "representation_id", record.representation_id
            )
            if existing is not None:
                refs = [coords_ref]
                if fitted_state_ref is not None:
                    refs.append(fitted_state_ref)
                self._folio.delete(refs, warn_dependents=False)
                return _representation_from_row(existing)
            self._folio.add(
                component_manifest_ref("representation", record.representation_id),
                manifest,
                description="Canonical CellPax representation manifest",
                inputs=inputs,
            )
        return record

    def preview_clustering_run(
        self,
        *,
        scope: Scope | str,
        representation: Representation | str,
        config: ClusteringConfig,
        generator: CandidateGenerator | None = None,
        created_by: str | None = None,
    ) -> ClusteringRun:
        """Compute and materialize one expensive, content-addressed clustering run."""
        scope_record = self.get_scope(scope) if isinstance(scope, str) else scope
        representation_record = (
            self.get_representation(representation)
            if isinstance(representation, str)
            else representation
        )
        if representation_record.scope_id != scope_record.scope_id:
            raise ValueError("Representation and clustering scope do not match")
        adapter = builtin_generator(config.method) if generator is None else generator
        if adapter.method != config.method:
            raise ValueError("Generator method and clustering config do not match")
        coords = self._folio.get(representation_record.coords_ref, frame="polars")
        members = self._folio.get(scope_record.members_ref, frame="polars")
        if not coords.select("cell_id").equals(members):
            raise ValueError(
                "Representation does not exactly cover the clustering scope"
            )
        if coords.height < 2:
            raise ValueError("Clustering requires at least two cells")
        artifacts = adapter.compute(
            coords.drop("cell_id").to_numpy(),
            coords["cell_id"].to_numpy(),
            config,
        )
        adapter_summary = dict(artifacts.structural_summary)
        reserved_summary = {"n_cells", "has_hierarchy"} & set(adapter_summary)
        if reserved_summary:
            raise ValueError(
                f"Generator structural summary uses reserved keys: {sorted(reserved_summary)}"
            )
        structural_summary = {
            "n_cells": coords.height,
            "has_hierarchy": artifacts.hierarchy_nodes is not None,
            **adapter_summary,
        }
        _structural_summary_bytes = canonical_json_bytes(structural_summary)
        software_versions = _software_versions(include=_NUMERIC_SOFTWARE)
        author = self._author(created_by)
        with self._folio.batch():
            generator_ref = new_item_ref("clustering-generator")
            self._folio.add_model(
                generator_ref,
                artifacts.payload,
                description="Method-specific clustering generator artifact",
                inputs=[representation_record.coords_ref],
            )
            hierarchy_nodes_ref: str | None = None
            hierarchy_members_ref: str | None = None
            hierarchy_inputs = [generator_ref]
            if (artifacts.hierarchy_nodes is None) != (
                artifacts.hierarchy_members is None
            ):
                raise ValueError("A generic hierarchy requires both tables")
            staged_nodes = artifacts.hierarchy_nodes
            staged_members = artifacts.hierarchy_members
            nodes_hash = (
                None
                if staged_nodes is None
                else canonical_hash(staged_nodes.to_dicts())
            )
            members_hash = (
                None
                if staged_members is None
                else canonical_hash(staged_members.to_dicts())
            )
            manifest = {
                "component_kind": "clustering_run",
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_record.scope_id,
                "representation_id": representation_record.representation_id,
                "method": config.method,
                "compute_params": config.compute_params,
                "spatial_input": None,
                "generator_checksum": resolve_checksum(self._folio, generator_ref),
                "hierarchy_nodes_checksum": nodes_hash,
                "hierarchy_members_checksum": members_hash,
                "seed": config.seed,
                "recompute_deterministic": config.recompute_deterministic,
                "software_versions": json.loads(software_versions),
                "structural_summary": structural_summary,
            }
            run_id = canonical_hash(manifest)
            if staged_nodes is not None and staged_members is not None:
                nodes = staged_nodes.with_columns(
                    pl.lit(run_id).alias("clustering_run_id")
                ).select(*CONTRACTS["candidate_hierarchy_node"].schema.names())
                hierarchy_nodes_ref = new_item_ref("candidate-hierarchy-nodes")
                self._folio.add(
                    hierarchy_nodes_ref,
                    nodes,
                    description="Generic candidate hierarchy nodes",
                    inputs=hierarchy_inputs,
                )
                members_frame = staged_members.with_columns(
                    pl.lit(run_id).alias("clustering_run_id")
                ).select(*CONTRACTS["candidate_hierarchy_membership"].schema.names())
                hierarchy_members_ref = new_item_ref("candidate-hierarchy-members")
                self._folio.add(
                    hierarchy_members_ref,
                    members_frame,
                    description="Generic candidate hierarchy leaf membership",
                    inputs=[generator_ref, hierarchy_nodes_ref],
                )
            record = ClusteringRun(
                clustering_run_id=run_id,
                scope_id=scope_record.scope_id,
                representation_id=representation_record.representation_id,
                method=config.method,
                compute_params_json=config.compute_params_json,
                spatial_input_json=config.spatial_input_json,
                generator_ref=generator_ref,
                hierarchy_nodes_ref=hierarchy_nodes_ref,
                hierarchy_members_ref=hierarchy_members_ref,
                seed=config.seed,
                recompute_deterministic=config.recompute_deterministic,
                software_versions=software_versions,
                created_at=_utcnow(),
                created_by=author,
            )
            existing = self._existing_row(
                "clustering_run", "clustering_run_id", record.clustering_run_id
            )
            if existing is not None:
                refs = [generator_ref]
                refs.extend(
                    ref
                    for ref in (hierarchy_nodes_ref, hierarchy_members_ref)
                    if ref is not None
                )
                self._folio.delete(refs, warn_dependents=False)
                return _clustering_run_from_row(existing)
            self._folio.add(
                component_manifest_ref("clustering-run", record.clustering_run_id),
                manifest,
                description="Canonical CellPax clustering-run manifest",
                inputs=[
                    generator_ref,
                    *(
                        []
                        if hierarchy_nodes_ref is None
                        else [hierarchy_nodes_ref, hierarchy_members_ref]
                    ),
                ],
            )
        return record

    def preview_candidate_set(
        self,
        *,
        clustering_run: ClusteringRun | str,
        config: CandidateCutConfig,
        generator: CandidateGenerator | None = None,
        created_by: str | None = None,
    ) -> CandidateSet:
        """Derive a cheap candidate partition from a stored clustering run."""
        run = (
            self.get_clustering_run(clustering_run)
            if isinstance(clustering_run, str)
            else clustering_run
        )
        adapter = builtin_generator(run.method) if generator is None else generator
        if adapter.method != run.method:
            raise ValueError("Generator method and clustering run do not match")
        payload = self._folio.get_model(run.generator_ref, trusted=True)
        partition = adapter.cut(payload, config)
        scope = self.get_scope(run.scope_id)
        members = self._folio.get(scope.members_ref, frame="polars")
        labels = partition.candidate_ids
        if labels.ndim != 1 or len(labels) != members.height:
            raise ValueError(
                "Generator returned one-dimensional labels of wrong length"
            )
        strengths = partition.membership_strengths
        if strengths is not None and (
            strengths.ndim != 1 or len(strengths) != len(labels)
        ):
            raise ValueError("Generator membership strengths have wrong length")
        valid_labels = sorted({int(value) for value in labels if int(value) >= 0})
        if partition.hierarchy_node_ids is not None and len(
            partition.hierarchy_node_ids
        ) != len(valid_labels):
            raise ValueError(
                "Generator hierarchy_node_ids must align with the sorted distinct "
                "non-negative candidate labels"
            )
        remap = {value: index for index, value in enumerate(valid_labels)}
        mapped = [None if int(value) < 0 else remap[int(value)] for value in labels]
        author = self._author(created_by)
        with self._folio.batch():
            semantic_definitions = pl.DataFrame(
                [
                    {
                        "candidate_id": candidate_id,
                        "hierarchy_node_id": (
                            None
                            if partition.hierarchy_node_ids is None
                            else partition.hierarchy_node_ids[position]
                        ),
                        "n_cells": mapped.count(candidate_id),
                    }
                    for position, candidate_id in enumerate(range(len(valid_labels)))
                ],
                schema={
                    "candidate_id": pl.Int32,
                    "hierarchy_node_id": pl.String,
                    "n_cells": pl.Int64,
                },
            )
            semantic_membership = pl.DataFrame(
                {
                    "cell_id": members["cell_id"],
                    "candidate_id": pl.Series(mapped, dtype=pl.Int32),
                    "membership_strength": pl.Series(
                        [None] * len(mapped) if strengths is None else strengths,
                        dtype=pl.Float32,
                    ),
                },
                schema={
                    "cell_id": pl.Int64,
                    "candidate_id": pl.Int32,
                    "membership_strength": pl.Float32,
                },
            )
            manifest = {
                "component_kind": "candidate_set",
                "schema_version": SCHEMA_VERSION,
                "clustering_run_id": run.clustering_run_id,
                "scope_id": run.scope_id,
                "cut_method": config.cut_method,
                "cut_params": config.cut_params,
                "definitions_checksum": canonical_hash(semantic_definitions.to_dicts()),
                "membership_checksum": canonical_hash(semantic_membership.to_dicts()),
            }
            candidate_set_id = canonical_hash(manifest)
            definitions = semantic_definitions.with_columns(
                pl.lit(candidate_set_id).alias("candidate_set_id")
            ).select(*CONTRACTS["candidate_definition"].schema.names())
            membership = semantic_membership.with_columns(
                pl.lit(candidate_set_id).alias("candidate_set_id")
            ).select(*CONTRACTS["candidate_membership"].schema.names())
            definitions_ref = new_item_ref("candidate-definitions")
            membership_ref = new_item_ref("candidate-membership")
            self._folio.add(
                definitions_ref,
                definitions,
                description="Candidate definitions",
                inputs=[run.generator_ref],
            )
            self._folio.add(
                membership_ref,
                membership,
                description="Candidate membership",
                inputs=[run.generator_ref, definitions_ref],
            )
            boundary_ref: str | None = None
            if partition.boundary_evidence is not None:
                evidence = partition.boundary_evidence.with_columns(
                    pl.lit(candidate_set_id).alias("candidate_set_id")
                ).select(*CONTRACTS["candidate_boundary_evidence"].schema.names())
                boundary_ref = new_item_ref("candidate-boundary-evidence")
                self._folio.add(
                    boundary_ref,
                    evidence,
                    description="Optional candidate boundary evidence",
                    inputs=[membership_ref],
                )
            record = CandidateSet(
                candidate_set_id=candidate_set_id,
                clustering_run_id=run.clustering_run_id,
                scope_id=run.scope_id,
                cut_method=config.cut_method,
                cut_params_json=config.cut_params_json,
                definitions_ref=definitions_ref,
                membership_ref=membership_ref,
                created_at=_utcnow(),
                created_by=author,
                boundary_evidence_ref=boundary_ref,
                clustering_run_record=run,
            )
            existing = self._existing_row(
                "candidate_set", "candidate_set_id", record.candidate_set_id
            )
            if existing is not None:
                self._folio.delete(
                    [
                        definitions_ref,
                        membership_ref,
                        *([] if boundary_ref is None else [boundary_ref]),
                    ],
                    warn_dependents=False,
                )
                return _candidate_set_from_row(existing)
            self._folio.add(
                component_manifest_ref("candidate-set", record.candidate_set_id),
                manifest,
                description="Canonical CellPax candidate-set manifest",
                inputs=[definitions_ref, membership_ref],
            )
        return record

    def candidate_membership(self, candidate_set: CandidateSet | str) -> pl.DataFrame:
        """Load candidate membership without exposing method-specific artifacts."""
        record = (
            self.get_candidate_set(candidate_set)
            if isinstance(candidate_set, str)
            else candidate_set
        )
        return self._folio.get(record.membership_ref, frame="polars")

    def candidate_definitions(self, candidate_set: CandidateSet | str) -> pl.DataFrame:
        """Load candidate definitions without exposing method-specific artifacts."""
        record = (
            self.get_candidate_set(candidate_set)
            if isinstance(candidate_set, str)
            else candidate_set
        )
        return self._folio.get(record.definitions_ref, frame="polars")

    def candidate_hierarchy(
        self, clustering_run: ClusteringRun | str
    ) -> tuple[pl.DataFrame, pl.DataFrame] | None:
        """Load the optional generic hierarchy emitted by a clustering backend."""
        record = (
            self.get_clustering_run(clustering_run)
            if isinstance(clustering_run, str)
            else clustering_run
        )
        if record.hierarchy_nodes_ref is None:
            return None
        if record.hierarchy_members_ref is None:
            raise ValueError("Clustering hierarchy refs are malformed")
        return (
            self._folio.get(record.hierarchy_nodes_ref, frame="polars"),
            self._folio.get(record.hierarchy_members_ref, frame="polars"),
        )

    def clustering_run_metadata(
        self, clustering_run: ClusteringRun | str
    ) -> pl.DataFrame:
        """Load optional per-run diagnostics retained by a clustering backend."""
        record = (
            self.get_clustering_run(clustering_run)
            if isinstance(clustering_run, str)
            else clustering_run
        )
        payload = self._folio.get_model(record.generator_ref, trusted=True)
        method = getattr(payload, "run_metadata", None)
        if not callable(method):
            raise ValueError(f"Generator {record.method!r} has no run metadata")
        result = method()
        if not isinstance(result, pl.DataFrame):
            raise TypeError("Generator run_metadata() must return a Polars DataFrame")
        return result

    def clustering_run_labels(
        self, clustering_run: ClusteringRun | str, *, long: bool = False
    ) -> pl.DataFrame:
        """Load optional retained per-run labels in wide or tidy form."""
        record = (
            self.get_clustering_run(clustering_run)
            if isinstance(clustering_run, str)
            else clustering_run
        )
        payload = self._folio.get_model(record.generator_ref, trusted=True)
        method = getattr(payload, "labels", None)
        if not callable(method):
            raise ValueError(f"Generator {record.method!r} has no retained labels")
        result = method(long=long)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("Generator labels() must return a Polars DataFrame")
        return result

    def candidate_boundary_evidence(
        self, candidate_set: CandidateSet | str
    ) -> pl.DataFrame:
        """Load optional pairwise evidence for a previewed or registered set."""
        record = (
            self.get_candidate_set(candidate_set)
            if isinstance(candidate_set, str)
            else candidate_set
        )
        if record.boundary_evidence_ref is not None:
            return self._folio.get(record.boundary_evidence_ref, frame="polars")
        return self.registry("candidate_boundary_evidence").filter(
            pl.col("candidate_set_id") == record.candidate_set_id
        )

    def compare_candidate_sets(
        self, left: CandidateSet | str, right: CandidateSet | str
    ) -> pl.DataFrame:
        """Return an algorithm-agnostic cell-count contingency table."""
        left_record = self.get_candidate_set(left) if isinstance(left, str) else left
        right_record = (
            self.get_candidate_set(right) if isinstance(right, str) else right
        )
        if left_record.scope_id != right_record.scope_id:
            raise ValueError("Candidate-set comparison requires the same scope")
        return (
            self.candidate_membership(left_record)
            .select("cell_id", pl.col("candidate_id").alias("left_candidate_id"))
            .join(
                self.candidate_membership(right_record).select(
                    "cell_id", pl.col("candidate_id").alias("right_candidate_id")
                ),
                on="cell_id",
            )
            .group_by("left_candidate_id", "right_candidate_id")
            .len(name="n_cells")
            .sort("left_candidate_id", "right_candidate_id", nulls_last=True)
        )

    def compare_revisions(
        self, left: KeptRevision | str, right: KeptRevision | str
    ) -> pl.DataFrame:
        """Compare the candidate partitions selected by two kept revisions."""
        left_record = self.get_revision(left) if isinstance(left, str) else left
        right_record = self.get_revision(right) if isinstance(right, str) else right
        if (
            left_record.candidate_set_id is None
            or right_record.candidate_set_id is None
        ):
            raise ValueError("Both revisions must select a candidate set")
        return self.compare_candidate_sets(
            left_record.candidate_set_id, right_record.candidate_set_id
        )

    def get_decision(self, decision_id: str) -> Decision:
        """Load one durable review decision."""
        row = self._existing_row("decision", "decision_id", decision_id)
        if row is None:
            raise KeyError(f"Unknown decision {decision_id!r}")
        return _decision_from_row(row)

    def decision_head(self, review_branch: str) -> Decision | None:
        """Return the unique append-only head of a review branch."""
        branch = self.registry("decision").filter(
            pl.col("review_branch") == review_branch
        )
        if branch.is_empty():
            return None
        parent_ids = set(branch["parent_decision_id"].drop_nulls().to_list())
        heads = branch.filter(~pl.col("decision_id").is_in(parent_ids))
        if heads.height != 1:
            raise ValueError(f"Review branch {review_branch!r} is not linear")
        return _decision_from_row(heads.row(0, named=True))

    def append_decision(
        self,
        *,
        revision: KeptRevision | str,
        review_branch: str,
        action: str,
        target_kind: str,
        rationale: str,
        target_ids: Sequence[int] | None = None,
        target_cells: Iterable[int] | pl.DataFrame | pl.Series | None = None,
        params: Mapping[str, Any] | None = None,
        taxon_id: int | None = None,
        evidence_refs: Sequence[str] = (),
        parent_decision: Decision | str | None = None,
        author: str | None = None,
    ) -> Decision:
        """Append one validated decision to a linear review branch."""
        if not review_branch:
            raise ValueError("review_branch is required")
        if not rationale:
            raise ValueError("Every decision requires a rationale")
        revision_record = (
            self.get_revision(revision) if isinstance(revision, str) else revision
        )
        if (
            self._existing_row(
                "kept_revision", "revision_id", revision_record.revision_id
            )
            is None
        ):
            raise ValueError("Decision revision must be kept in this study")
        if (target_ids is None) == (target_cells is None):
            raise ValueError("Exactly one of target_ids and target_cells is required")
        config = DecisionActionConfig.resolve(action=action, params=params)
        normalized_ids: tuple[int, ...] | None = None
        target_ref: str | None = None
        target_members: pl.DataFrame | None = None
        if target_ids is not None:
            values = list(target_ids)
            if not values or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise TypeError("target_ids must contain non-negative integers")
            normalized_ids = tuple(sorted(set(values)))
        else:
            target_members = normalize_cell_ids(target_cells)
            universe = self.universe()
            universe_cells = self._folio.get(universe.cells_ref, frame="polars")
            require_membership_subset(
                target_members, universe_cells, container_name="the universe"
            )
        validate_decision_semantics(
            config=config,
            target_kind=target_kind,
            target_count=(len(normalized_ids) if normalized_ids is not None else None),
            taxon_id=taxon_id,
        )
        if target_kind == "candidates":
            if revision_record.candidate_set_id is None:
                raise ValueError("Candidate decisions require a revision candidate set")
            definitions = self.candidate_definitions(revision_record.candidate_set_id)
            available = set(definitions["candidate_id"].to_list())
            if normalized_ids is None or not set(normalized_ids) <= available:
                raise ValueError("Decision targets unknown candidates")
        refs = tuple(dict.fromkeys(validate_item_ref(ref) for ref in evidence_refs))
        for ref in refs:
            self._folio.item_info(ref)
        current_head = self.decision_head(review_branch)
        parent = (
            self.get_decision(parent_decision)
            if isinstance(parent_decision, str)
            else parent_decision
        )
        if parent is None:
            parent = current_head
        elif parent.review_branch != review_branch:
            raise ValueError("Parent decision must belong to the same review branch")
        elif (
            current_head is not None and parent.decision_id != current_head.decision_id
        ):
            raise ValueError("A decision must append to the current branch head")
        if (
            parent is not None
            and self._existing_row("decision", "decision_id", parent.decision_id)
            is None
        ):
            raise ValueError("Parent decision does not belong to this study")
        decision = Decision(
            decision_id=new_entity_id(),
            review_branch=review_branch,
            revision_id=revision_record.revision_id,
            parent_decision_id=None if parent is None else parent.decision_id,
            action=action,
            target_kind=target_kind,
            target_ids=normalized_ids,
            target_ref=None,
            params_json=config.params_json,
            taxon_id=taxon_id,
            rationale=rationale,
            evidence_refs=refs,
            author=author or str(self._folio.metadata[_CREATED_BY]),
            created_at=_utcnow(),
        )
        with self._folio.batch():
            if target_members is not None:
                target_ref = new_item_ref("decision-target-members")
                self._folio.add(
                    target_ref,
                    target_members,
                    description="Retained large decision cell target",
                    inputs=[],
                )
                decision = Decision(
                    **{
                        **decision.row(),
                        "target_ids": None,
                        "target_ref": target_ref,
                        "evidence_refs": decision.evidence_refs,
                    }
                )
            self._commit_registry_changes(
                {"decision": _typed_frame("decision", [decision.row()])}
            )
        return decision

    def decision_lineage(self, decision: Decision | str) -> tuple[Decision, ...]:
        """Return a decision's oldest-to-newest parent chain."""
        cursor = self.get_decision(decision) if isinstance(decision, str) else decision
        lineage: list[Decision] = []
        seen: set[str] = set()
        while True:
            if cursor.decision_id in seen:
                raise ValueError("Decision lineage contains a cycle")
            seen.add(cursor.decision_id)
            lineage.append(cursor)
            if cursor.parent_decision_id is None:
                break
            cursor = self.get_decision(cursor.parent_decision_id)
        return tuple(reversed(lineage))

    def register_taxonomy(self, taxonomy: pl.DataFrame) -> pl.DataFrame:
        """Register one immutable taxonomy vocabulary version."""
        name, taxonomy_version = validate_taxonomy(taxonomy)
        existing = self.registry("taxonomy").filter(
            (pl.col("taxonomy_name") == name)
            & (pl.col("taxonomy_version") == taxonomy_version)
        )
        if not existing.is_empty():
            if existing.equals(taxonomy):
                return existing
            raise ValueError("Taxonomy version is immutable and already registered")
        with self._folio.batch():
            self._commit_registry_changes({"taxonomy": taxonomy})
        return taxonomy

    def get_taxonomy(self, name: str, taxonomy_version: str) -> pl.DataFrame:
        """Load one registered taxonomy vocabulary version."""
        frame = self.registry("taxonomy").filter(
            (pl.col("taxonomy_name") == name)
            & (pl.col("taxonomy_version") == taxonomy_version)
        )
        if frame.is_empty():
            raise KeyError(f"Unknown taxonomy {name!r} version {taxonomy_version!r}")
        validate_taxonomy(frame)
        return frame

    def taxonomy_enum(self, name: str, taxonomy_version: str):
        """Generate a rich IntEnum binding for a registered taxonomy version."""
        return taxonomy_enum(self.get_taxonomy(name, taxonomy_version))

    def get_assignment_set(self, assignment_set_id: str) -> AssignmentSet:
        """Load one registered immutable assignment set."""
        row = self._existing_row(
            "assignment_set", "assignment_set_id", assignment_set_id
        )
        if row is None:
            raise KeyError(f"Unknown assignment set {assignment_set_id!r}")
        return _assignment_set_from_row(row)

    def assignments(self, assignment_set: AssignmentSet | str) -> pl.DataFrame:
        """Load cell-level claims for an assignment set."""
        record = (
            self.get_assignment_set(assignment_set)
            if isinstance(assignment_set, str)
            else assignment_set
        )
        return self._folio.get(record.assignments_ref, frame="polars")

    def create_assignment_set(
        self,
        assignments: pl.DataFrame,
        *,
        taxonomy_name: str,
        taxonomy_version: str,
        review_branch: str,
        decision_head: Decision | str,
        created_by: str | None = None,
        propagation_runs: Sequence[PropagationRun] = (),
    ) -> AssignmentSet:
        """Validate and persist one immutable, coverage-aware assignment snapshot."""
        taxonomy = self.get_taxonomy(taxonomy_name, taxonomy_version)
        head = (
            self.get_decision(decision_head)
            if isinstance(decision_head, str)
            else decision_head
        )
        if head.review_branch != review_branch:
            raise ValueError("Assignment set branch and decision head do not match")
        if self.decision_head(review_branch) != head:
            raise ValueError("Assignment set must reduce the current branch head")
        semantic_columns = [
            column
            for column in CONTRACTS["assignment"].schema.names()
            if column not in {"assignment_set_id", "created_at"}
        ]
        if assignments.columns != semantic_columns:
            raise ValueError(
                f"Assignment input columns mismatch; expected {semantic_columns}"
            )
        assignment_set_id = new_entity_id()
        created_at = _utcnow()
        materialized = assignments.with_columns(
            pl.lit(assignment_set_id).alias("assignment_set_id"),
            pl.lit(created_at).alias("created_at"),
        ).select(*CONTRACTS["assignment"].schema.names())
        validate_table(materialized, CONTRACTS["assignment"])
        universe = self.universe()
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        require_membership_subset(
            materialized.select("cell_id"),
            universe_cells,
            container_name="the universe",
        )
        taxa = set(taxonomy["taxon_id"].to_list())
        used_taxa = set(materialized["taxon_id"].drop_nulls().to_list())
        if not used_taxa <= taxa:
            raise ValueError("Assignments reference taxon ids outside the taxonomy")
        children = set(taxonomy["parent_id"].drop_nulls().to_list())
        parent_only = set(
            materialized.filter(pl.col("assignment_status") == "parent_only")[
                "taxon_id"
            ].to_list()
        )
        if not parent_only <= children:
            raise ValueError("parent_only assignments must reference a parent taxon")
        incomplete_coverage = materialized.filter(
            pl.col("coverage_feature_space_id").is_null()
            != pl.col("coverage").is_null()
        )
        if not incomplete_coverage.is_empty():
            raise ValueError("coverage and coverage_feature_space_id must be paired")
        for revision_id in materialized["source_revision_id"].unique():
            self.get_revision(revision_id)
        for decision_id in materialized["decision_id"].drop_nulls().unique():
            self.get_decision(decision_id)
        for feature_space_id in (
            materialized["coverage_feature_space_id"].drop_nulls().unique()
        ):
            self.get_feature_space(feature_space_id)
        run_records = {run.propagation_run_id: run for run in propagation_runs}
        for run_id in materialized["propagation_run_id"].drop_nulls().unique():
            if (
                run_id not in run_records
                and self._existing_row("propagation_run", "propagation_run_id", run_id)
                is None
            ):
                raise ValueError(f"Unknown propagation run {run_id!r}")
        semantic = materialized.drop("assignment_set_id", "created_at")
        assignments_checksum = canonical_hash(semantic.to_dicts())
        state = {
            "taxonomy_name": taxonomy_name,
            "taxonomy_version": taxonomy_version,
            "review_branch": review_branch,
            "decision_head_id": head.decision_id,
            "assignments_checksum": assignments_checksum,
        }
        record = AssignmentSet(
            assignment_set_id=assignment_set_id,
            taxonomy_name=taxonomy_name,
            taxonomy_version=taxonomy_version,
            review_branch=review_branch,
            decision_head_id=head.decision_id,
            assignments_ref=new_item_ref("assignments"),
            state_hash=canonical_hash(state),
            created_at=created_at,
            created_by=self._author(created_by),
        )
        with self._folio.batch():
            self._folio.add(
                record.assignments_ref,
                materialized,
                description="Immutable CellPax assignment set",
                inputs=[],
            )
            changes: dict[str, pl.DataFrame] = {
                "assignment_set": _typed_frame("assignment_set", [record.row()])
            }
            if run_records:
                changes["propagation_run"] = _typed_frame(
                    "propagation_run", [run.row() for run in run_records.values()]
                )
            self._commit_registry_changes(changes)
        return record

    def reduce_decisions(
        self,
        decision_head: Decision | str,
        *,
        taxonomy_name: str,
        taxonomy_version: str,
        base_assignment_set: AssignmentSet | str | None = None,
    ) -> pl.DataFrame:
        """Reduce one decision lineage into canonical assignment input rows."""
        taxonomy = self.get_taxonomy(taxonomy_name, taxonomy_version)
        valid_taxa = set(taxonomy["taxon_id"].to_list())
        base = (
            self.get_assignment_set(base_assignment_set)
            if isinstance(base_assignment_set, str)
            else base_assignment_set
        )
        state: dict[int, dict[str, object]] = {}
        if base is not None:
            if (
                base.taxonomy_name != taxonomy_name
                or base.taxonomy_version != taxonomy_version
            ):
                raise ValueError("Base assignment set uses another taxonomy version")
            for row in (
                self.assignments(base)
                .drop("assignment_set_id", "created_at")
                .iter_rows(named=True)
            ):
                state[row["cell_id"]] = row

        lineage = self.decision_lineage(decision_head)
        revisions = {
            revision_id: self.get_revision(revision_id)
            for revision_id in {decision.revision_id for decision in lineage}
        }

        def candidate_targets(
            decision: Decision,
            revision: KeptRevision,
        ) -> tuple[list[int], dict[int, float | None]]:
            if revision.candidate_set_id is None:
                raise ValueError("Candidate decision revision has no candidate set")
            membership = self.candidate_membership(revision.candidate_set_id).filter(
                pl.col("candidate_id").is_in(list(decision.target_ids or ()))
            )
            return (
                membership["cell_id"].to_list(),
                dict(membership.select("cell_id", "membership_strength").iter_rows()),
            )

        for decision in lineage:
            revision = revisions[decision.revision_id]
            if decision.target_kind == "candidates":
                cells, confidence = candidate_targets(decision, revision)
            elif decision.target_kind == "cells":
                if decision.target_ref is not None:
                    members = normalize_cell_ids(
                        self._folio.get(decision.target_ref, frame="polars")
                    )
                    cells = members["cell_id"].to_list()
                else:
                    cells = list(decision.target_ids or ())
                confidence = {cell_id: None for cell_id in cells}
            else:
                cells = [
                    cell_id
                    for cell_id, row in state.items()
                    if row["taxon_id"] in set(decision.target_ids or ())
                ]
                confidence = {cell_id: None for cell_id in cells}
            feature_space_id = revision.feature_space_id

            def assign_cells(
                selected: Iterable[int],
                *,
                taxon_id: int | None,
                status: str,
                source: str,
            ) -> None:
                if taxon_id is not None and taxon_id not in valid_taxa:
                    raise ValueError(f"Decision references unknown taxon {taxon_id}")
                for cell_id in selected:
                    state[cell_id] = {
                        "cell_id": cell_id,
                        "taxon_id": taxon_id,
                        "assignment_status": status,
                        "assignment_source": source,
                        "source_revision_id": decision.revision_id,
                        "decision_id": decision.decision_id,
                        "propagation_run_id": None,
                        "coverage_feature_space_id": (
                            feature_space_id if source == "feature_based" else None
                        ),
                        "coverage": 1.0 if source == "feature_based" else None,
                        "confidence": confidence.get(cell_id),
                        "alternatives_json": None,
                    }

            source = (
                "feature_based" if decision.target_kind == "candidates" else "manual"
            )
            if decision.action == "exclude":
                assign_cells(
                    cells,
                    taxon_id=None,
                    status="outside_taxonomy",
                    source="manual",
                )
            elif decision.action in {"assign", "merge"}:
                if decision.taxon_id is None:
                    raise ValueError(
                        f"Decision action {decision.action!r} requires taxon_id"
                    )
                assign_cells(
                    cells,
                    taxon_id=decision.taxon_id,
                    status="leaf",
                    source=source,
                )
            elif decision.action == "assign_parent_only":
                assign_cells(
                    cells,
                    taxon_id=decision.taxon_id,
                    status="parent_only",
                    source=source,
                )
            elif decision.action == "mark_ambiguous":
                for cell_id in cells:
                    existing = state.get(cell_id)
                    ambiguous_taxon = (
                        decision.taxon_id
                        if decision.taxon_id is not None
                        else None
                        if existing is None
                        else existing["taxon_id"]
                    )
                    if ambiguous_taxon is None:
                        raise ValueError(
                            "mark_ambiguous requires an existing or explicit taxon"
                        )
                    assign_cells(
                        [cell_id],
                        taxon_id=int(ambiguous_taxon),
                        status="ambiguous",
                        source="manual",
                    )
            elif decision.action == "split":
                candidate_cells = set(cells)
                for part in json.loads(decision.params_json)["parts"]:
                    members = normalize_cell_ids(
                        self._folio.get(part["target_ref"], frame="polars")
                    )
                    part_cells = members["cell_id"].to_list()
                    if not set(part_cells) <= candidate_cells:
                        raise ValueError(
                            "Split part contains cells outside its candidate"
                        )
                    assign_cells(
                        part_cells,
                        taxon_id=part["taxon_id"],
                        status="leaf",
                        source="manual",
                    )
            elif decision.action not in {"rename", "attach_local_revision"}:
                raise ValueError(f"Unsupported reducer action {decision.action!r}")
        semantic_schema = {
            column: dtype
            for column, dtype in CONTRACTS["assignment"].schema.items()
            if column not in {"assignment_set_id", "created_at"}
        }
        return pl.DataFrame(
            [state[cell_id] for cell_id in sorted(state)], schema=semantic_schema
        )

    def create_assignment_set_from_decisions(
        self,
        *,
        taxonomy_name: str,
        taxonomy_version: str,
        review_branch: str,
        decision_head: Decision | str,
        base_assignment_set: AssignmentSet | str | None = None,
        created_by: str | None = None,
    ) -> AssignmentSet:
        """Reduce a branch head and persist its immutable assignment snapshot."""
        assignments = self.reduce_decisions(
            decision_head,
            taxonomy_name=taxonomy_name,
            taxonomy_version=taxonomy_version,
            base_assignment_set=base_assignment_set,
        )
        return self.create_assignment_set(
            assignments,
            taxonomy_name=taxonomy_name,
            taxonomy_version=taxonomy_version,
            review_branch=review_branch,
            decision_head=decision_head,
            created_by=created_by,
        )

    def get_propagation_run(self, propagation_run_id: str) -> PropagationRun:
        """Load one registered label-propagation computation."""
        row = self._existing_row(
            "propagation_run", "propagation_run_id", propagation_run_id
        )
        if row is None:
            raise KeyError(f"Unknown propagation run {propagation_run_id!r}")
        return _propagation_run_from_row(row)

    def preview_propagation_run(
        self,
        *,
        fit_scope: Scope | str,
        application_scope: Scope | str,
        feature_space: FeatureSpace | str,
        source_assignment_set: AssignmentSet | str,
        config: PropagationConfig,
        coverage: pl.DataFrame | None = None,
        created_by: str | None = None,
    ) -> PropagationRun:
        """Fit KNN propagation and retain per-cell neighborhood purity output."""
        if config.method != "knn":
            raise ValueError(f"Unsupported propagation method {config.method!r}")
        from sklearn.neighbors import KNeighborsClassifier

        fit = self.get_scope(fit_scope) if isinstance(fit_scope, str) else fit_scope
        application = (
            self.get_scope(application_scope)
            if isinstance(application_scope, str)
            else application_scope
        )
        space = (
            self.get_feature_space(feature_space)
            if isinstance(feature_space, str)
            else feature_space
        )
        source = (
            self.get_assignment_set(source_assignment_set)
            if isinstance(source_assignment_set, str)
            else source_assignment_set
        )
        if space.values_ref is None:
            raise ValueError("Propagation feature space has no materialized values")
        values = self._folio.get(space.values_ref, frame="polars")
        fit_members = self._folio.get(fit.members_ref, frame="polars")
        application_members = self._folio.get(application.members_ref, frame="polars")
        require_membership_subset(
            fit_members, values, container_name="the propagation feature space"
        )
        require_membership_subset(
            application_members,
            values,
            container_name="the propagation feature space",
        )
        source_rows = self.assignments(source).filter(pl.col("taxon_id").is_not_null())
        training = fit_members.join(
            source_rows.select("cell_id", "taxon_id"), on="cell_id", how="left"
        )
        if training["taxon_id"].null_count():
            raise ValueError("Every propagation fit-scope cell requires a source taxon")
        params = config.params
        if params["n_neighbors"] > training.height:
            raise ValueError("Propagation n_neighbors exceeds labeled fit-scope size")
        selected_fit = fit_members.join(values, on="cell_id", how="left")
        selected_application = application_members.join(
            values, on="cell_id", how="left"
        )
        feature_columns = [column for column in values.columns if column != "cell_id"]
        if (
            selected_fit.select(feature_columns).null_count().sum_horizontal()[0]
            or selected_application.select(feature_columns)
            .null_count()
            .sum_horizontal()[0]
        ):
            raise ValueError("Propagation input values cannot contain nulls")
        model = KNeighborsClassifier(**params)
        model.fit(
            selected_fit.select(feature_columns).to_numpy(),
            training["taxon_id"].to_numpy(),
        )
        application_matrix = selected_application.select(feature_columns).to_numpy()
        predicted = model.predict(application_matrix).astype("int64")
        probabilities = model.predict_proba(application_matrix)
        confidence = probabilities.max(axis=1).astype("float32")
        if coverage is None:
            coverage_frame = application_members.with_columns(
                pl.lit(1.0, dtype=pl.Float32).alias("coverage")
            )
        else:
            expected_schema = pl.Schema({"cell_id": pl.Int64, "coverage": pl.Float32})
            if coverage.schema != expected_schema:
                raise TypeError(f"coverage must have schema {expected_schema}")
            coverage_frame = normalize_cell_ids(coverage.select("cell_id")).join(
                coverage, on="cell_id", how="left"
            )
            if not coverage_frame.select("cell_id").equals(application_members):
                raise ValueError("Coverage must exactly cover the application scope")
            if (
                coverage_frame["coverage"].null_count()
                or coverage_frame.filter(
                    (pl.col("coverage") < 0) | (pl.col("coverage") > 1)
                ).height
            ):
                raise ValueError("Coverage values must be non-null and within [0, 1]")
        quality = (
            pl.DataFrame(
                {
                    "cell_id": application_members["cell_id"],
                    "predicted_taxon_id": pl.Series(predicted, dtype=pl.Int64),
                    "purity": pl.Series(confidence, dtype=pl.Float32),
                    "n_neighbors": pl.Series(
                        [params["n_neighbors"]] * application_members.height,
                        dtype=pl.Int32,
                    ),
                }
            )
            .join(coverage_frame, on="cell_id", how="left")
            .select(
                "cell_id",
                "predicted_taxon_id",
                "purity",
                "n_neighbors",
                "coverage",
            )
        )
        author = self._author(created_by)
        with self._folio.batch():
            model_ref = new_item_ref("propagation-model")
            self._folio.add_model(
                model_ref,
                model,
                description="Fitted CellPax label-propagation model",
                inputs=[space.values_ref, source.assignments_ref],
            )
            quality_ref = new_item_ref("propagation-quality")
            self._folio.add(
                quality_ref,
                quality,
                description="Per-cell propagation neighborhood purity",
                inputs=[model_ref, space.values_ref, source.assignments_ref],
            )
            manifest = {
                "component_kind": "propagation_run",
                "schema_version": SCHEMA_VERSION,
                "fit_scope_id": fit.scope_id,
                "application_scope_id": application.scope_id,
                "input_feature_space_id": space.feature_space_id,
                "source_assignment_set_id": source.assignment_set_id,
                "method": config.method,
                "params": config.params,
                "model_checksum": resolve_checksum(self._folio, model_ref),
                "quality_summary_checksum": resolve_checksum(self._folio, quality_ref),
                "seed": config.seed,
                "recompute_deterministic": config.recompute_deterministic,
            }
            record = PropagationRun(
                propagation_run_id=canonical_hash(manifest),
                fit_scope_id=fit.scope_id,
                application_scope_id=application.scope_id,
                input_feature_space_id=space.feature_space_id,
                source_assignment_set_id=source.assignment_set_id,
                method=config.method,
                params_json=config.params_json,
                model_ref=model_ref,
                quality_summary_ref=quality_ref,
                seed=config.seed,
                recompute_deterministic=config.recompute_deterministic,
                created_at=_utcnow(),
                created_by=author,
            )
            existing = self._existing_row(
                "propagation_run", "propagation_run_id", record.propagation_run_id
            )
            if existing is not None:
                self._folio.delete([model_ref, quality_ref], warn_dependents=False)
                return self.get_propagation_run(record.propagation_run_id)
            self._folio.add(
                component_manifest_ref("propagation-run", record.propagation_run_id),
                manifest,
                description="Canonical CellPax propagation-run manifest",
                inputs=[model_ref, quality_ref],
            )
        return record

    def propagation_quality(
        self, propagation_run: PropagationRun | str
    ) -> pl.DataFrame:
        """Load per-cell propagation predictions, purity, and coverage."""
        record = (
            self.get_propagation_run(propagation_run)
            if isinstance(propagation_run, str)
            else propagation_run
        )
        return self._folio.get(record.quality_summary_ref, frame="polars")

    def propagated_assignment_rows(
        self,
        propagation_run: PropagationRun | str,
        *,
        source_revision: KeptRevision | str,
        decision: Decision | str | None = None,
    ) -> pl.DataFrame:
        """Project propagation output into canonical assignment input rows."""
        run = (
            self.get_propagation_run(propagation_run)
            if isinstance(propagation_run, str)
            else propagation_run
        )
        revision = (
            self.get_revision(source_revision)
            if isinstance(source_revision, str)
            else source_revision
        )
        decision_record = (
            self.get_decision(decision) if isinstance(decision, str) else decision
        )
        quality = self.propagation_quality(run)
        return quality.select(
            "cell_id",
            pl.col("predicted_taxon_id").alias("taxon_id"),
            pl.lit("leaf").alias("assignment_status"),
            pl.lit("propagated").alias("assignment_source"),
            pl.lit(revision.revision_id).alias("source_revision_id"),
            pl.lit(
                None if decision_record is None else decision_record.decision_id,
                dtype=pl.String,
            ).alias("decision_id"),
            pl.lit(run.propagation_run_id).alias("propagation_run_id"),
            pl.lit(run.input_feature_space_id).alias("coverage_feature_space_id"),
            pl.col("coverage"),
            pl.col("purity").alias("confidence"),
            pl.lit(None, dtype=pl.String).alias("alternatives_json"),
        )

    def get_annotation_release(self, identifier: str) -> AnnotationRelease:
        """Load an annotation release by immutable id or unique publication name."""
        row = self._existing_row(
            "annotation_release", "annotation_release_id", identifier
        )
        if row is None:
            matches = self.registry("annotation_release").filter(
                pl.col("name") == identifier
            )
            if matches.is_empty():
                raise KeyError(f"Unknown annotation release {identifier!r}")
            row = matches.row(0, named=True)
        return _annotation_release_from_row(row)

    def create_annotation_release(
        self,
        name: str,
        *,
        source_revision: KeptRevision | str,
        assignment_set: AssignmentSet | str,
        created_by: str | None = None,
    ) -> AnnotationRelease:
        """Publish one immutable, self-documenting taxonomy/assignment bundle."""
        if not name:
            raise ValueError("An annotation release requires a name")
        if (
            not self.registry("annotation_release")
            .filter(pl.col("name") == name)
            .is_empty()
        ):
            raise ValueError(f"Annotation release name already exists: {name!r}")
        if isinstance(source_revision, str):
            revision = self.get_revision(source_revision)
        else:
            try:
                revision = self.get_revision(source_revision.revision_id)
            except KeyError as error:
                raise ValueError(
                    "Release source revision must be kept in this study"
                ) from error
            if revision != source_revision:
                raise ValueError("Release source revision handle is stale or foreign")
        if isinstance(assignment_set, str):
            assignment = self.get_assignment_set(assignment_set)
        else:
            try:
                assignment = self.get_assignment_set(assignment_set.assignment_set_id)
            except KeyError as error:
                raise ValueError(
                    "Release assignment set must belong to this study"
                ) from error
            if assignment != assignment_set:
                raise ValueError("Release assignment-set handle is stale or foreign")
        taxonomy = self.get_taxonomy(
            assignment.taxonomy_name, assignment.taxonomy_version
        ).sort("sort_order", "taxon_id")
        assignments = self.assignments(assignment).sort("cell_id")
        scope = self.get_scope(revision.scope_id)
        scope_members = self._folio.get(scope.members_ref, frame="polars")
        require_membership_subset(
            assignments.select("cell_id"),
            scope_members,
            container_name="the release source revision scope",
        )
        ancestry_ids = {
            record.revision_id for record in self.revision_lineage(revision)
        }
        unrelated_sources = sorted(
            set(assignments["source_revision_id"].to_list()) - ancestry_ids
        )
        if unrelated_sources:
            raise ValueError(
                "Assignment source revisions are outside the release source "
                f"revision ancestry: {unrelated_sources}"
            )
        head = self.get_decision(assignment.decision_head_id)
        decisions = _typed_frame(
            "decision", [record.row() for record in self.decision_lineage(head)]
        )
        release_id = new_entity_id()
        quality = release_quality_summary(
            annotation_release_id=release_id,
            assignments=assignments,
            n_scope_cells=scope_members.height,
            n_decisions=decisions.height,
        )
        validate_table(quality, RELEASE_QUALITY)
        recipe = resolved_release_recipe(
            self, source_revision=revision, assignment_set=assignment
        )
        replay_script = render_release_replay_script(release_id)
        enum_binding = generate_enum_binding(taxonomy)
        author = self._author(created_by)
        created_at = _utcnow()

        with self._folio.batch():
            refs = {
                name: new_item_ref(f"release-{name}") for name in RELEASE_ARTIFACT_NAMES
            }
            current_head = self.head_commit()
            registry_inputs = {} if current_head is None else current_head.registries
            self._folio.add(
                refs["taxonomy"],
                taxonomy,
                description=f"Taxonomy product for annotation release {name}",
                inputs=[registry_inputs["taxonomy"]],
            )
            self._folio.add(
                refs["assignments"],
                assignments,
                description=f"Assignment product for annotation release {name}",
                inputs=[assignment.assignments_ref],
            )
            self._folio.add(
                refs["decisions"],
                decisions,
                description=f"Decision lineage for annotation release {name}",
                inputs=[registry_inputs["decision"]],
            )
            propagation_refs = [
                self.get_propagation_run(run_id).quality_summary_ref
                for run_id in assignments["propagation_run_id"]
                .drop_nulls()
                .unique()
                .to_list()
            ]
            self._folio.add(
                refs["quality_summary"],
                quality,
                description=f"Quality summary for annotation release {name}",
                inputs=[refs["assignments"], *propagation_refs],
            )
            self._folio.add(
                refs["recipe"],
                recipe,
                description=f"Resolved recipe for annotation release {name}",
                inputs=[refs["assignments"], refs["decisions"]],
            )
            self._folio.add(
                refs["replay_script"],
                replay_script,
                description=f"Replay instructions for annotation release {name}",
                inputs=[refs["recipe"]],
            )
            self._folio.add(
                refs["enum_binding"],
                enum_binding,
                description=f"Generated enum binding for annotation release {name}",
                inputs=[refs["taxonomy"]],
            )
            artifacts = {
                artifact_name: {
                    "ref": ref,
                    "checksum": resolve_checksum(self._folio, ref),
                }
                for artifact_name, ref in refs.items()
            }
            manifest = {
                "component_kind": "annotation_release",
                "schema_version": SCHEMA_VERSION,
                "annotation_release_id": release_id,
                "name": name,
                "taxonomy_name": assignment.taxonomy_name,
                "taxonomy_version": assignment.taxonomy_version,
                "assignment_set_id": assignment.assignment_set_id,
                "assignment_set_state_hash": assignment.state_hash,
                "source_revision_id": revision.revision_id,
                "source_revision_state_hash": revision.state_hash,
                "decision_head_id": assignment.decision_head_id,
                "artifacts": artifacts,
            }
            manifest_ref = new_item_ref("annotation-release-manifest")
            self._folio.add(
                manifest_ref,
                manifest,
                description=f"Self-documenting annotation release {name}",
                inputs=list(refs.values()),
            )
            record = AnnotationRelease(
                annotation_release_id=release_id,
                name=name,
                taxonomy_name=assignment.taxonomy_name,
                taxonomy_version=assignment.taxonomy_version,
                assignment_set_id=assignment.assignment_set_id,
                source_revision_id=revision.revision_id,
                manifest_ref=manifest_ref,
                created_at=created_at,
                created_by=author,
            )
            self._commit_registry_changes(
                {
                    "annotation_release": _typed_frame(
                        "annotation_release", [record.row()]
                    )
                }
            )
        return record

    def load_annotation_release(
        self, release: AnnotationRelease | str
    ) -> ReleaseBundle:
        """Load all language-neutral products named by a release manifest."""
        if isinstance(release, str):
            record = self.get_annotation_release(release)
        else:
            try:
                record = self.get_annotation_release(release.annotation_release_id)
            except KeyError as error:
                raise ValueError(
                    "Annotation release does not belong to this study"
                ) from error
            if record != release:
                raise ValueError("Annotation release handle is stale or foreign")
        manifest = validate_release_manifest(self._folio.get(record.manifest_ref))
        artifacts = manifest["artifacts"]
        return ReleaseBundle(
            release=record,
            manifest=manifest,
            taxonomy=self._folio.get(artifacts["taxonomy"]["ref"], frame="polars"),
            assignments=self._folio.get(
                artifacts["assignments"]["ref"], frame="polars"
            ),
            decisions=self._folio.get(artifacts["decisions"]["ref"], frame="polars"),
            quality_summary=self._folio.get(
                artifacts["quality_summary"]["ref"], frame="polars"
            ),
            recipe=self._folio.get(artifacts["recipe"]["ref"]),
            replay_script=self._folio.get(artifacts["replay_script"]["ref"]),
            enum_binding=self._folio.get(artifacts["enum_binding"]["ref"]),
        )

    def validate_annotation_release(self, release: AnnotationRelease | str) -> None:
        """Validate one release without loading clustering or model payloads."""
        record = (
            self.get_annotation_release(release)
            if isinstance(release, str)
            else release
        )
        if (
            self._existing_row(
                "annotation_release",
                "annotation_release_id",
                record.annotation_release_id,
            )
            is None
        ):
            raise ValueError("Annotation release does not belong to this study")
        bundle = self.load_annotation_release(record)
        manifest = bundle.manifest
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Annotation-release schema version mismatch")
        assignment = self.get_assignment_set(record.assignment_set_id)
        revision = self.get_revision(record.source_revision_id)
        expected_fields = {
            "annotation_release_id": record.annotation_release_id,
            "name": record.name,
            "taxonomy_name": record.taxonomy_name,
            "taxonomy_version": record.taxonomy_version,
            "assignment_set_id": record.assignment_set_id,
            "assignment_set_state_hash": assignment.state_hash,
            "source_revision_id": record.source_revision_id,
            "source_revision_state_hash": revision.state_hash,
            "decision_head_id": assignment.decision_head_id,
        }
        for key, value in expected_fields.items():
            if manifest[key] != value:
                raise ValueError(f"Annotation-release manifest {key} mismatch")
        artifacts = manifest["artifacts"]
        for artifact_name in RELEASE_ARTIFACT_NAMES:
            descriptor = artifacts[artifact_name]
            if descriptor["checksum"] != resolve_checksum(
                self._folio, descriptor["ref"]
            ):
                raise ValueError(
                    f"Annotation-release artifact {artifact_name!r} checksum mismatch"
                )

        validate_taxonomy(bundle.taxonomy)
        validate_table(bundle.assignments, CONTRACTS["assignment"])
        validate_table(bundle.decisions, CONTRACTS["decision"])
        validate_table(bundle.quality_summary, RELEASE_QUALITY)
        expected_taxonomy = self.get_taxonomy(
            record.taxonomy_name, record.taxonomy_version
        ).sort("sort_order", "taxon_id")
        if not bundle.taxonomy.equals(expected_taxonomy):
            raise ValueError("Annotation-release taxonomy product mismatch")
        expected_assignments = self.assignments(assignment).sort("cell_id")
        if not bundle.assignments.equals(expected_assignments):
            raise ValueError("Annotation-release assignment product mismatch")
        expected_decisions = _typed_frame(
            "decision",
            [
                decision.row()
                for decision in self.decision_lineage(assignment.decision_head_id)
            ],
        )
        if not bundle.decisions.equals(expected_decisions):
            raise ValueError("Annotation-release decision product mismatch")
        scope = self.get_scope(revision.scope_id)
        scope_members = self._folio.get(scope.members_ref, frame="polars")
        require_membership_subset(
            bundle.assignments.select("cell_id"),
            scope_members,
            container_name="the release source revision scope",
        )
        ancestry_ids = {
            ancestor.revision_id for ancestor in self.revision_lineage(revision)
        }
        unrelated_sources = sorted(
            set(bundle.assignments["source_revision_id"].to_list()) - ancestry_ids
        )
        if unrelated_sources:
            raise ValueError(
                "Assignment source revisions are outside the release source "
                f"revision ancestry: {unrelated_sources}"
            )
        expected_quality = release_quality_summary(
            annotation_release_id=record.annotation_release_id,
            assignments=expected_assignments,
            n_scope_cells=scope_members.height,
            n_decisions=expected_decisions.height,
        )
        if not bundle.quality_summary.equals(expected_quality):
            raise ValueError("Annotation-release quality summary mismatch")
        expected_recipe = resolved_release_recipe(
            self, source_revision=revision, assignment_set=assignment
        )
        if bundle.recipe != expected_recipe:
            raise ValueError("Annotation-release resolved recipe mismatch")
        if bundle.replay_script != render_release_replay_script(
            record.annotation_release_id
        ):
            raise ValueError("Annotation-release replay script mismatch")
        if bundle.enum_binding != generate_enum_binding(expected_taxonomy):
            raise ValueError("Annotation-release enum binding mismatch")

    def _validate_scope_record(
        self,
        scope: Scope,
        universe: Universe,
        universe_cells: pl.DataFrame,
    ) -> pl.DataFrame:
        try:
            members = self._folio.get(scope.members_ref, frame="polars")
        except KeyError as error:
            raise ValueError("Scope preview does not belong to this study") from error
        normalized = normalize_cell_ids(members)
        if not normalized.equals(members):
            raise ValueError(f"Scope {scope.scope_id} membership is not canonical")
        if scope.universe_id != universe.universe_id:
            raise ValueError("Scope belongs to a different universe")
        require_membership_subset(
            members, universe_cells, container_name="the universe"
        )
        if members.height != scope.n_cells:
            raise ValueError(f"Scope {scope.scope_id} n_cells mismatch")
        digest = membership_hash(members["cell_id"].to_list())
        if digest != scope.membership_hash:
            raise ValueError(f"Scope {scope.scope_id} membership hash mismatch")
        expected_scope_id = canonical_hash(
            {
                "component_kind": "scope",
                "schema_version": SCHEMA_VERSION,
                "universe_id": scope.universe_id,
                "parent_scope_id": scope.parent_scope_id,
                "membership_hash": digest,
                "n_cells": members.height,
                "derivation_text": scope.derivation_text,
            }
        )
        if expected_scope_id != scope.scope_id:
            raise ValueError(f"Scope {scope.scope_id} identity mismatch")
        return members

    def _validate_feature_selection_record(
        self, selection: FeatureSelection, catalog: pl.DataFrame
    ) -> pl.DataFrame:
        try:
            members = self._folio.get(selection.members_ref, frame="polars")
        except KeyError as error:
            raise ValueError(
                "Feature selection preview does not belong to this study"
            ) from error
        prepared = prepare_feature_selection_members(members, catalog)
        if not prepared.equals(members):
            raise ValueError(
                f"Feature selection {selection.feature_selection_id} "
                "members are not canonical"
            )
        catalog_digest = feature_catalog_hash(catalog)
        if selection.catalog_hash != catalog_digest:
            raise ValueError(
                f"Feature selection {selection.feature_selection_id} catalog hash mismatch"
            )
        if members.height != selection.n_features:
            raise ValueError(
                f"Feature selection {selection.feature_selection_id} count mismatch"
            )
        expected_selection_id = canonical_hash(
            {
                "component_kind": "feature_selection",
                "schema_version": SCHEMA_VERSION,
                "catalog_hash": catalog_digest,
                "members": members.select("feature_block_id", "feature_id").to_dicts(),
                "n_features": members.height,
            }
        )
        if expected_selection_id != selection.feature_selection_id:
            raise ValueError(
                f"Feature selection {selection.feature_selection_id} identity mismatch"
            )
        return members

    def build(
        self, *, parent_revision: KeptRevision | str | None = None
    ) -> "RevisionBuilder":
        """Return a fluent builder that threads previews into named revisions.

        The builder holds each artifact you create so you don't re-pass scope,
        feature space, representations, and candidate set into ``keep``:

        >>> b = study.build()
        >>> b.scope(cell_ids).select(block)
        >>> b.keep("inputs")
        >>> b.feature_space(FeatureSpaceConfig.standard_scaler())
        >>> b.representation(RepresentationConfig.pca(n_components=2))
        >>> revision = b.keep("scaled + pca")   # parent chains automatically
        """
        from cellpax.builder import RevisionBuilder

        return RevisionBuilder(self, parent_revision=parent_revision)

    def load(
        self,
        revision: "KeptRevision | str",
        *,
        assignment_set: "AssignmentSet | str | None" = None,
    ) -> "CellData":
        """Load a revision as a flexible, in-memory :class:`CellData` view.

        EXPERIMENTAL. Returns an AnnData-inspired object you can hold, mask, switch
        layers on (raw/normalized), and plot from — a read-oriented facade over the
        immutable revision. See :class:`cellpax.celldata.CellData`.
        """
        from cellpax.celldata import CellData

        return CellData(self, revision, assignment_set=assignment_set)

    def view(self, name: str, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Compute one named read-only view (see :mod:`cellpax.views`).

        ``study.view("cells", revision, assignment_set=...)`` is equivalent to
        ``cellpax.views.cells(study, revision, assignment_set=...)``; this method
        just makes the views discoverable from the study object.
        """
        from cellpax import views

        dispatch = {
            "cells": views.cells,
            "embedding": views.embedding,
            "feature_profiles": views.feature_profiles,
            "stability": views.stability,
            "comparison": views.comparison,
            "taxonomy": views.taxonomy,
            "history": views.history,
            "release_summary": views.release_summary,
        }
        try:
            func = dispatch[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown view {name!r}; choose one of {sorted(dispatch)}"
            ) from error
        return func(self, *args, **kwargs)

    def feature_values(
        self,
        feature_space: FeatureSpace | str,
        *,
        semantic_columns: bool = True,
    ) -> pl.DataFrame:
        """Return a feature space's per-cell values as a readable DataFrame.

        Works on both previews (pass the record) and kept spaces (pass the id).
        With ``semantic_columns`` the physical ``feature_NNNNN`` columns are
        renamed to their catalog ``feature_id``s.
        """
        record = (
            self.get_feature_space(feature_space)
            if isinstance(feature_space, str)
            else feature_space
        )
        if record.values_ref is None:
            raise ValueError("Feature space has no materialized values")
        frame = self._folio.get(record.values_ref, frame="polars")
        if not semantic_columns:
            return frame
        selection = self.get_feature_selection(record.feature_selection_id)
        members = self._folio.get(selection.members_ref, frame="polars").sort(
            "position"
        )
        rename = {
            feature_column(position): feature_id
            for position, feature_id in zip(
                members["position"].to_list(), members["feature_id"].to_list()
            )
        }
        return frame.rename(rename)

    def feature_table(
        self,
        revision: KeptRevision | str,
        *,
        normalized: bool = False,
        assignment_set: AssignmentSet | str | None = None,
        include_metadata: bool = True,
    ) -> pl.DataFrame:
        """Return one tidy per-cell frame ready for faceted plotting.

        The frame has one row per scope cell and carries, as columns you can
        facet / axis / color by: the ``cells`` view's labels (candidate, taxon,
        status, source, confidence), optional universe metadata columns, and the
        revision's feature columns (readable ``feature_id`` names).

        ``normalized`` selects the feature representation: ``True`` uses the
        revision's clustering feature space (the normalized values it was built
        on); ``False`` (the default) reconstructs raw values over the same scope
        and selection, so you can cluster on normalized features and plot the
        actual ones.
        """
        record = self.get_revision(revision) if isinstance(revision, str) else revision
        if record.feature_space_id is None:
            raise ValueError("feature_table requires a revision with a feature space")
        space = self.get_feature_space(record.feature_space_id)
        if normalized:
            features = self.feature_values(space)
        else:
            scope_input, _, selection_members = self._space_inputs(space)
            columns = [
                feature_column(position)
                for position in selection_members["position"].to_list()
            ]
            rename = {
                feature_column(position): feature_id
                for position, feature_id in zip(
                    selection_members["position"].to_list(),
                    selection_members["feature_id"].to_list(),
                )
            }
            features = scope_input.select("cell_id", *columns).rename(rename)
        frame = self.view("cells", record, assignment_set=assignment_set).drop(
            "revision_id"
        )
        if include_metadata:
            universe_cells = self._folio.get(self.universe().cells_ref, frame="polars")
            metadata_columns = [
                column
                for column in universe_cells.columns
                if column != "cell_id" and column not in frame.columns
            ]
            if metadata_columns:
                frame = frame.join(
                    universe_cells.select("cell_id", *metadata_columns),
                    on="cell_id",
                    how="left",
                )
        collisions = sorted((set(features.columns) - {"cell_id"}) & set(frame.columns))
        if collisions:
            raise ValueError(
                f"Feature ids collide with label/metadata columns: {collisions}"
            )
        return frame.join(features, on="cell_id", how="left")

    def keep(
        self,
        name: str,
        *,
        scope: Scope | str,
        feature_selection: FeatureSelection | str | None = None,
        feature_space: FeatureSpace | str | None = None,
        clustering_representation: Representation | str | None = None,
        visualization_representation: Representation | str | None = None,
        candidate_set: CandidateSet | str | None = None,
        parent_revision: KeptRevision | str | None = None,
        notes: str | None = None,
        created_by: str | None = None,
    ) -> KeptRevision:
        """Keep previews as a named immutable scope-only pointer-set revision.

        In Slice 1, a supplied feature selection is rooted in its registry and
        included in ``config_hash``. It becomes a structural revision dependency
        through ``feature_space.feature_selection_id`` in Slice 2.
        """
        if not name:
            raise ValueError("A kept revision requires a name")
        revisions = self.registry("kept_revision")
        if not revisions.filter(pl.col("name") == name).is_empty():
            raise ValueError(f"Kept revision name already exists: {name!r}")

        scope_record = self.get_scope(scope) if isinstance(scope, str) else scope
        universe = self.universe()
        universe_cells = self._folio.get(universe.cells_ref, frame="polars")
        self._validate_scope_record(scope_record, universe, universe_cells)
        registered_scope = self._existing_row(
            "scope", "scope_id", scope_record.scope_id
        )
        if registered_scope is not None and registered_scope != scope_record.row():
            raise ValueError(
                "Stale scope preview handle: this scope_id is already registered; "
                "load it with get_scope() before keeping another revision"
            )
        if (
            scope_record.parent_scope_id is not None
            and self._existing_row("scope", "scope_id", scope_record.parent_scope_id)
            is None
        ):
            raise ValueError(
                "A child scope's parent must be kept before the child scope"
            )

        if isinstance(feature_selection, str):
            selection_record = self.get_feature_selection(feature_selection)
        else:
            selection_record = feature_selection
        if selection_record is not None:
            self._validate_feature_selection_record(
                selection_record, self.feature_catalog()
            )
            registered_selection = self._existing_row(
                "feature_selection",
                "feature_selection_id",
                selection_record.feature_selection_id,
            )
            if (
                registered_selection is not None
                and registered_selection != selection_record.row()
            ):
                raise ValueError(
                    "Stale feature-selection preview handle: this selection id "
                    "is already registered; load it with get_feature_selection()"
                )

        space_record = (
            self.get_feature_space(feature_space)
            if isinstance(feature_space, str)
            else feature_space
        )
        if space_record is not None:
            if space_record.scope_id != scope_record.scope_id:
                raise ValueError("Feature space belongs to a different revision scope")
            if space_record.values_ref is None:
                raise ValueError("Feature space has no materialized values")
            self._folio.item_info(space_record.values_ref)
            if selection_record is None:
                selection_record = self.get_feature_selection(
                    space_record.feature_selection_id
                )
            elif (
                selection_record.feature_selection_id
                != space_record.feature_selection_id
            ):
                raise ValueError("Feature space and feature selection do not match")
            if (
                space_record.fit_scope_id != scope_record.scope_id
                and self._existing_row("scope", "scope_id", space_record.fit_scope_id)
                is None
            ):
                raise ValueError("Feature-space fit scope must already be kept")
            if (
                space_record.parent_feature_space_id is not None
                and self._existing_row(
                    "feature_space",
                    "feature_space_id",
                    space_record.parent_feature_space_id,
                )
                is None
            ):
                raise ValueError("Parent feature space must already be kept")
            registered_space = self._existing_row(
                "feature_space", "feature_space_id", space_record.feature_space_id
            )
            if registered_space is not None and registered_space != space_record.row():
                raise ValueError("Stale feature-space preview handle")

        def resolve_representation(
            value: Representation | str | None, *, role: str
        ) -> Representation | None:
            record = self.get_representation(value) if isinstance(value, str) else value
            if record is None:
                return None
            if space_record is None:
                raise ValueError(f"{role} representation requires a feature space")
            if record.scope_id != scope_record.scope_id:
                raise ValueError(f"{role} representation belongs to another scope")
            if record.input_feature_space_id != space_record.feature_space_id:
                raise ValueError(f"{role} representation uses another feature space")
            self._folio.item_info(record.coords_ref)
            if (
                record.fit_scope_id != scope_record.scope_id
                and self._existing_row("scope", "scope_id", record.fit_scope_id) is None
            ):
                raise ValueError(
                    f"{role} representation fit scope must already be kept"
                )
            registered = self._existing_row(
                "representation", "representation_id", record.representation_id
            )
            if registered is not None and registered != record.row():
                raise ValueError(f"Stale {role} representation preview handle")
            return record

        clustering_record = resolve_representation(
            clustering_representation, role="Clustering"
        )
        visualization_record = resolve_representation(
            visualization_representation, role="Visualization"
        )
        candidate_record = (
            self.get_candidate_set(candidate_set)
            if isinstance(candidate_set, str)
            else candidate_set
        )
        clustering_run_record: ClusteringRun | None = None
        if candidate_record is not None:
            if clustering_record is None:
                raise ValueError("A candidate set requires a clustering representation")
            if candidate_record.scope_id != scope_record.scope_id:
                raise ValueError("Candidate set belongs to another scope")
            run_row = self._existing_row(
                "clustering_run",
                "clustering_run_id",
                candidate_record.clustering_run_id,
            )
            clustering_run_record = (
                candidate_record.clustering_run_record
                if run_row is None
                else _clustering_run_from_row(run_row)
            )
            if clustering_run_record is None:
                raise ValueError("Candidate-set preview has no clustering-run handle")
            if (
                clustering_run_record.representation_id
                != clustering_record.representation_id
            ):
                raise ValueError("Candidate set uses another clustering representation")
            registered_candidate = self._existing_row(
                "candidate_set", "candidate_set_id", candidate_record.candidate_set_id
            )
            if (
                registered_candidate is not None
                and registered_candidate != candidate_record.row()
            ):
                raise ValueError("Stale candidate-set preview handle")
        if isinstance(parent_revision, str):
            parent_record = self.get_revision(parent_revision)
        else:
            parent_record = parent_revision
        if parent_record is not None:
            existing_parent = self._existing_row(
                "kept_revision", "revision_id", parent_record.revision_id
            )
            if existing_parent is None or existing_parent != parent_record.row():
                raise ValueError("Parent revision does not belong to this study")

        parent_id = None if parent_record is None else parent_record.revision_id
        resolved_config = KeepConfig.resolve(
            parent_revision_id=parent_id,
            scope_id=scope_record.scope_id,
            feature_selection_id=(
                None
                if selection_record is None
                else selection_record.feature_selection_id
            ),
            feature_space_id=(
                None if space_record is None else space_record.feature_space_id
            ),
            clustering_representation_id=(
                None
                if clustering_record is None
                else clustering_record.representation_id
            ),
            visualization_representation_id=(
                None
                if visualization_record is None
                else visualization_record.representation_id
            ),
            candidate_set_id=(
                None if candidate_record is None else candidate_record.candidate_set_id
            ),
        )
        state = {
            "parent_revision_id": parent_id,
            "scope_id": scope_record.scope_id,
            "feature_space_id": (
                None if space_record is None else space_record.feature_space_id
            ),
            "clustering_representation_id": (
                None
                if clustering_record is None
                else clustering_record.representation_id
            ),
            "visualization_representation_id": (
                None
                if visualization_record is None
                else visualization_record.representation_id
            ),
            "candidate_set_id": (
                None if candidate_record is None else candidate_record.candidate_set_id
            ),
        }
        software_versions = _software_versions(include=_NUMERIC_SOFTWARE)
        revision = KeptRevision(
            revision_id=new_entity_id(),
            name=name,
            parent_revision_id=parent_id,
            scope_id=scope_record.scope_id,
            feature_space_id=state["feature_space_id"],
            clustering_representation_id=state["clustering_representation_id"],
            visualization_representation_id=state["visualization_representation_id"],
            candidate_set_id=state["candidate_set_id"],
            state_hash=canonical_hash(state),
            config_hash=resolved_config.config_hash,
            software_versions=software_versions,
            created_by=self._author(created_by),
            created_at=_utcnow(),
            notes=notes,
        )

        changes: dict[str, pl.DataFrame] = {
            "scope": _typed_frame("scope", [scope_record.row()]),
            "kept_revision": _typed_frame("kept_revision", [revision.row()]),
        }
        if selection_record is not None:
            changes["feature_selection"] = _typed_frame(
                "feature_selection", [selection_record.row()]
            )
        if space_record is not None:
            changes["feature_space"] = _typed_frame(
                "feature_space", [space_record.row()]
            )
        representations = {
            record.representation_id: record
            for record in (clustering_record, visualization_record)
            if record is not None
        }
        if representations:
            changes["representation"] = _typed_frame(
                "representation", [record.row() for record in representations.values()]
            )
        if candidate_record is not None:
            changes["clustering_run"] = _typed_frame(
                "clustering_run", [clustering_run_record.row()]
            )
            changes["candidate_set"] = _typed_frame(
                "candidate_set", [candidate_record.row()]
            )
            if candidate_record.boundary_evidence_ref is not None:
                changes["candidate_boundary_evidence"] = self._folio.get(
                    candidate_record.boundary_evidence_ref, frame="polars"
                )
        with self._folio.batch():
            self._commit_registry_changes(changes)
        return revision

    def _component_manifest(
        self, kind: str, component_id: str, expected_keys: set[str]
    ) -> dict[str, Any]:
        manifest = self._folio.get(component_manifest_ref(kind, component_id))
        if not isinstance(manifest, dict) or set(manifest) != expected_keys:
            raise ValueError(f"Malformed {kind} manifest {component_id}")
        if manifest["component_kind"] != kind.replace("-", "_"):
            raise ValueError(f"Manifest kind mismatch for {kind} {component_id}")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version for {kind} {component_id}")
        if canonical_hash(manifest) != component_id:
            raise ValueError(f"Manifest hash mismatch for {kind} {component_id}")
        return manifest

    def _space_inputs(
        self, record: FeatureSpace
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        scope = self.get_scope(record.scope_id)
        fit_scope = self.get_scope(record.fit_scope_id)
        selection = self.get_feature_selection(record.feature_selection_id)
        scope_members = self._folio.get(scope.members_ref, frame="polars")
        fit_members = self._folio.get(fit_scope.members_ref, frame="polars")
        selection_members = self._folio.get(selection.members_ref, frame="polars")
        if record.parent_feature_space_id is None:
            catalog = self.feature_catalog()
            blocks = {
                block_id: self._folio.get(
                    _feature_block_from_row(
                        self._existing_row(
                            "feature_block", "feature_block_id", block_id
                        )
                    ).values_ref,
                    frame="polars",
                )
                for block_id in record.input_feature_block_ids or ()
            }
            scope_input = assemble_selected_values(
                scope_members, selection_members, catalog, blocks
            )
            fit_input = assemble_selected_values(
                fit_members, selection_members, catalog, blocks
            )
        else:
            parent = self.get_feature_space(record.parent_feature_space_id)
            if parent.values_ref is None:
                raise ValueError("Parent feature space has no materialized values")
            parent_values = self._folio.get(parent.values_ref, frame="polars")
            scope_input = scope_members.join(parent_values, on="cell_id", how="left")
            fit_input = fit_members.join(parent_values, on="cell_id", how="left")
        return scope_input, fit_input, selection_members

    def replay_feature_space(self, feature_space_id: str) -> pl.DataFrame:
        """Reapply stored fitted state to reconstruct materialized feature values."""
        record = self.get_feature_space(feature_space_id)
        scope_input, _, selection_members = self._space_inputs(record)
        feature_columns = [
            feature_column(position)
            for position in selection_members["position"].to_list()
        ]
        ready = scope_input
        if record.missing_policy == "drop":
            ready = ready.drop_nulls(feature_columns)
        if (
            record.missing_policy == "error"
            and ready.select(feature_columns).null_count().sum_horizontal()[0]
        ):
            raise ValueError("Stored error missing policy encounters null values")
        matrix = ready.select(feature_columns).to_numpy()
        if record.fitted_state_ref is None:
            output = matrix
        else:
            estimator = self._folio.get_model(record.fitted_state_ref, trusted=True)
            output = estimator.transform(matrix)
        return (
            pl.DataFrame(output, schema=feature_columns)
            .with_columns(pl.Series("cell_id", ready["cell_id"], dtype=pl.Int64))
            .select("cell_id", *feature_columns)
        )

    def replay_representation(self, representation_id: str) -> pl.DataFrame:
        """Reapply stored fitted state to reconstruct representation coordinates."""
        record = self.get_representation(representation_id)
        space = self.get_feature_space(record.input_feature_space_id)
        if space.values_ref is None:
            raise ValueError("Input feature space has no materialized values")
        values = self._folio.get(space.values_ref, frame="polars")
        scope = self.get_scope(record.scope_id)
        members = self._folio.get(scope.members_ref, frame="polars")
        selected = members.join(values, on="cell_id", how="left")
        feature_columns = [column for column in selected.columns if column != "cell_id"]
        matrix = selected.select(feature_columns).to_numpy()
        if record.fitted_state_ref is None:
            output = matrix
        else:
            estimator = self._folio.get_model(record.fitted_state_ref, trusted=True)
            output = estimator.transform(matrix)
        dimensions = [f"dim_{index}" for index in range(output.shape[1])]
        return (
            pl.DataFrame(output, schema=dimensions)
            .with_columns(pl.Series("cell_id", selected["cell_id"], dtype=pl.Int64))
            .select("cell_id", *dimensions)
        )

    def _validate_slice_two_artifacts(
        self,
        frames: Mapping[str, pl.DataFrame],
        *,
        trusted_models: bool,
    ) -> None:
        spaces = frames.get("feature_space")
        if spaces is not None:
            for row in spaces.iter_rows(named=True):
                record = _feature_space_from_row(row)
                manifest = self._component_manifest(
                    "feature-space",
                    record.feature_space_id,
                    {
                        "component_kind",
                        "schema_version",
                        "scope_id",
                        "fit_scope_id",
                        "parent_feature_space_id",
                        "input_feature_block_ids",
                        "feature_selection_id",
                        "transform",
                        "params",
                        "fitted_state_checksum",
                        "values_checksum",
                        "missing_policy",
                        "seed",
                    },
                )
                expected = {
                    "scope_id": record.scope_id,
                    "fit_scope_id": record.fit_scope_id,
                    "parent_feature_space_id": record.parent_feature_space_id,
                    "input_feature_block_ids": (
                        None
                        if record.input_feature_block_ids is None
                        else list(record.input_feature_block_ids)
                    ),
                    "feature_selection_id": record.feature_selection_id,
                    "transform": record.transform,
                    "params": json.loads(record.params_json),
                    "missing_policy": record.missing_policy,
                    "seed": record.seed,
                }
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(
                            f"Feature space {record.feature_space_id} {key} mismatch"
                        )
                if record.values_ref is None:
                    raise ValueError(
                        "Slice 2 feature spaces require materialized values"
                    )
                if manifest["values_checksum"] != resolve_checksum(
                    self._folio, record.values_ref
                ):
                    raise ValueError(
                        f"Feature space {record.feature_space_id} values checksum mismatch"
                    )
                state_checksum = (
                    None
                    if record.fitted_state_ref is None
                    else resolve_checksum(self._folio, record.fitted_state_ref)
                )
                if manifest["fitted_state_checksum"] != state_checksum:
                    raise ValueError(
                        f"Feature space {record.feature_space_id} fitted-state mismatch"
                    )
                stored = self._folio.get(record.values_ref, frame="polars")
                scope = self.get_scope(record.scope_id)
                members = self._folio.get(scope.members_ref, frame="polars")
                if not stored.select("cell_id").equals(members):
                    raise ValueError(
                        f"Feature space {record.feature_space_id} does not cover its scope"
                    )
                if trusted_models:
                    replayed = self.replay_feature_space(record.feature_space_id)
                    if not replayed.equals(stored):
                        raise ValueError(
                            f"Feature space {record.feature_space_id} replay mismatch"
                        )
                self.feature_space_missingness(record)

        representations = frames.get("representation")
        if representations is not None:
            for row in representations.iter_rows(named=True):
                record = _representation_from_row(row)
                manifest = self._component_manifest(
                    "representation",
                    record.representation_id,
                    {
                        "component_kind",
                        "schema_version",
                        "scope_id",
                        "fit_scope_id",
                        "method",
                        "input_feature_space_id",
                        "spatial_input",
                        "n_components",
                        "params",
                        "fitted_state_checksum",
                        "coords_checksum",
                        "seed",
                        "recompute_deterministic",
                        "software_versions",
                    },
                )
                expected = {
                    "scope_id": record.scope_id,
                    "fit_scope_id": record.fit_scope_id,
                    "method": record.method,
                    "input_feature_space_id": record.input_feature_space_id,
                    "spatial_input": (
                        None
                        if record.spatial_input_json is None
                        else json.loads(record.spatial_input_json)
                    ),
                    "n_components": record.n_components,
                    "params": json.loads(record.params_json),
                    "seed": record.seed,
                    "recompute_deterministic": record.recompute_deterministic,
                    "software_versions": json.loads(record.software_versions),
                }
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(
                            f"Representation {record.representation_id} {key} mismatch"
                        )
                if manifest["coords_checksum"] != resolve_checksum(
                    self._folio, record.coords_ref
                ):
                    raise ValueError(
                        f"Representation {record.representation_id} coords checksum mismatch"
                    )
                state_checksum = (
                    None
                    if record.fitted_state_ref is None
                    else resolve_checksum(self._folio, record.fitted_state_ref)
                )
                if manifest["fitted_state_checksum"] != state_checksum:
                    raise ValueError(
                        f"Representation {record.representation_id} fitted-state mismatch"
                    )
                stored = self._folio.get(record.coords_ref, frame="polars")
                scope = self.get_scope(record.scope_id)
                members = self._folio.get(scope.members_ref, frame="polars")
                if not stored.select("cell_id").equals(members):
                    raise ValueError(
                        f"Representation {record.representation_id} does not cover its scope"
                    )
                if trusted_models and record.recompute_deterministic:
                    replayed = self.replay_representation(record.representation_id)
                    if not replayed.equals(stored):
                        raise ValueError(
                            f"Representation {record.representation_id} replay mismatch"
                        )

    def _validate_slice_three_artifacts(
        self, frames: Mapping[str, pl.DataFrame]
    ) -> None:
        runs = frames.get("clustering_run")
        if runs is not None:
            for row in runs.iter_rows(named=True):
                record = _clustering_run_from_row(row)
                manifest = self._component_manifest(
                    "clustering-run",
                    record.clustering_run_id,
                    {
                        "component_kind",
                        "schema_version",
                        "scope_id",
                        "representation_id",
                        "method",
                        "compute_params",
                        "spatial_input",
                        "generator_checksum",
                        "hierarchy_nodes_checksum",
                        "hierarchy_members_checksum",
                        "seed",
                        "recompute_deterministic",
                        "software_versions",
                        "structural_summary",
                    },
                )
                expected = {
                    "scope_id": record.scope_id,
                    "representation_id": record.representation_id,
                    "method": record.method,
                    "compute_params": json.loads(record.compute_params_json),
                    "spatial_input": (
                        None
                        if record.spatial_input_json is None
                        else json.loads(record.spatial_input_json)
                    ),
                    "seed": record.seed,
                    "recompute_deterministic": record.recompute_deterministic,
                    "software_versions": json.loads(record.software_versions),
                }
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(
                            f"Clustering run {record.clustering_run_id} {key} mismatch"
                        )
                summary = manifest["structural_summary"]
                if not isinstance(summary, dict):
                    raise ValueError("Clustering structural summary must be a mapping")
                scope = self.get_scope(record.scope_id)
                if summary.get("n_cells") != scope.n_cells:
                    raise ValueError(
                        "Clustering structural summary scope-size mismatch"
                    )
                if summary.get("has_hierarchy") != (
                    record.hierarchy_nodes_ref is not None
                ):
                    raise ValueError("Clustering structural summary hierarchy mismatch")
                if manifest["generator_checksum"] != resolve_checksum(
                    self._folio, record.generator_ref
                ):
                    raise ValueError(
                        f"Clustering run {record.clustering_run_id} generator mismatch"
                    )
                if (record.hierarchy_nodes_ref is None) != (
                    record.hierarchy_members_ref is None
                ):
                    raise ValueError("Clustering hierarchy refs must be paired")
                if record.hierarchy_nodes_ref is None:
                    if (
                        manifest["hierarchy_nodes_checksum"] is not None
                        or manifest["hierarchy_members_checksum"] is not None
                    ):
                        raise ValueError(
                            "Absent hierarchy has non-null manifest hashes"
                        )
                    continue
                nodes = self._folio.get(record.hierarchy_nodes_ref, frame="polars")
                members = self._folio.get(record.hierarchy_members_ref, frame="polars")
                validate_table(nodes, CONTRACTS["candidate_hierarchy_node"])
                validate_table(members, CONTRACTS["candidate_hierarchy_membership"])
                if set(nodes["clustering_run_id"].to_list()) != {
                    record.clustering_run_id
                } or set(members["clustering_run_id"].to_list()) != {
                    record.clustering_run_id
                }:
                    raise ValueError("Generic hierarchy owner id mismatch")
                semantic_nodes = nodes.drop("clustering_run_id")
                semantic_members = members.drop("clustering_run_id")
                if manifest["hierarchy_nodes_checksum"] != canonical_hash(
                    semantic_nodes.to_dicts()
                ) or manifest["hierarchy_members_checksum"] != canonical_hash(
                    semantic_members.to_dicts()
                ):
                    raise ValueError("Generic hierarchy semantic hash mismatch")
                node_ids = set(nodes["node_id"].to_list())
                parent_ids = set(nodes["parent_node_id"].drop_nulls().to_list())
                if not parent_ids <= node_ids:
                    raise ValueError(
                        "Generic hierarchy references missing parent nodes"
                    )
                if not set(members["leaf_node_id"].to_list()) <= node_ids:
                    raise ValueError("Generic hierarchy references missing leaf nodes")
                scope = self.get_scope(record.scope_id)
                scope_members = self._folio.get(scope.members_ref, frame="polars")
                if not members.select("cell_id").equals(scope_members):
                    raise ValueError(
                        "Generic hierarchy does not exactly cover its scope"
                    )

        candidate_sets = frames.get("candidate_set")
        if candidate_sets is not None:
            for row in candidate_sets.iter_rows(named=True):
                record = _candidate_set_from_row(row)
                manifest = self._component_manifest(
                    "candidate-set",
                    record.candidate_set_id,
                    {
                        "component_kind",
                        "schema_version",
                        "clustering_run_id",
                        "scope_id",
                        "cut_method",
                        "cut_params",
                        "definitions_checksum",
                        "membership_checksum",
                    },
                )
                expected = {
                    "clustering_run_id": record.clustering_run_id,
                    "scope_id": record.scope_id,
                    "cut_method": record.cut_method,
                    "cut_params": json.loads(record.cut_params_json),
                }
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(
                            f"Candidate set {record.candidate_set_id} {key} mismatch"
                        )
                definitions = self.candidate_definitions(record)
                membership = self.candidate_membership(record)
                validate_table(definitions, CONTRACTS["candidate_definition"])
                validate_table(membership, CONTRACTS["candidate_membership"])
                if manifest["definitions_checksum"] != canonical_hash(
                    definitions.drop("candidate_set_id").to_dicts()
                ) or manifest["membership_checksum"] != canonical_hash(
                    membership.drop("candidate_set_id").to_dicts()
                ):
                    raise ValueError("Candidate-set semantic artifact hash mismatch")
                scope = self.get_scope(record.scope_id)
                scope_members = self._folio.get(scope.members_ref, frame="polars")
                if not membership.select("cell_id").equals(scope_members):
                    raise ValueError(
                        "Candidate membership does not exactly cover its scope"
                    )
                actual_counts = (
                    membership.drop_nulls("candidate_id")
                    .group_by("candidate_id")
                    .len(name="n_cells")
                    .sort("candidate_id")
                )
                expected_counts = definitions.select("candidate_id", "n_cells").sort(
                    "candidate_id"
                )
                if not actual_counts.equals(expected_counts):
                    raise ValueError(
                        "Candidate definition counts do not match membership"
                    )
                run_row = self._existing_row(
                    "clustering_run", "clustering_run_id", record.clustering_run_id
                )
                if run_row is not None and run_row["method"] == "fauxnograph":
                    if (
                        membership["membership_strength"].null_count()
                        != membership.height
                    ):
                        raise ValueError(
                            "Fauxnograph membership_strength must remain null"
                        )

    def _validate_slice_four_artifacts(
        self,
        frames: Mapping[str, pl.DataFrame],
        *,
        trusted_models: bool,
    ) -> None:
        taxonomy_frame = frames.get("taxonomy")
        if taxonomy_frame is not None:
            for taxonomy in taxonomy_frame.partition_by(
                "taxonomy_name", "taxonomy_version", maintain_order=True
            ):
                validate_taxonomy(taxonomy)

        decisions = frames.get("decision")
        if decisions is not None:
            decision_by_id = {
                row["decision_id"]: row for row in decisions.iter_rows(named=True)
            }
            for row in decisions.iter_rows(named=True):
                decision = _decision_from_row(row)
                if decision.parent_decision_id is not None:
                    parent = decision_by_id.get(decision.parent_decision_id)
                    if parent is None:
                        raise ValueError("Decision parent is missing from the ledger")
                    if parent["review_branch"] != decision.review_branch:
                        raise ValueError("Decision parent crosses review branches")
                config = DecisionActionConfig.resolve(
                    action=decision.action, params=json.loads(decision.params_json)
                )
                validate_decision_semantics(
                    config=config,
                    target_kind=decision.target_kind,
                    target_count=(
                        None
                        if decision.target_ids is None
                        else len(decision.target_ids)
                    ),
                    taxon_id=decision.taxon_id,
                )
                if decision.target_ref is not None:
                    members = self._folio.get(decision.target_ref, frame="polars")
                    if not normalize_cell_ids(members).equals(members):
                        raise ValueError("Decision target membership is not canonical")
                for ref in decision.evidence_refs:
                    self._folio.item_info(ref)
            for branch in decisions["review_branch"].unique().to_list():
                self.decision_head(branch)

        propagation_runs = frames.get("propagation_run")
        if propagation_runs is not None:
            for row in propagation_runs.iter_rows(named=True):
                run = _propagation_run_from_row(row)
                manifest = self._component_manifest(
                    "propagation-run",
                    run.propagation_run_id,
                    {
                        "component_kind",
                        "schema_version",
                        "fit_scope_id",
                        "application_scope_id",
                        "input_feature_space_id",
                        "source_assignment_set_id",
                        "method",
                        "params",
                        "model_checksum",
                        "quality_summary_checksum",
                        "seed",
                        "recompute_deterministic",
                    },
                )
                expected = {
                    "fit_scope_id": run.fit_scope_id,
                    "application_scope_id": run.application_scope_id,
                    "input_feature_space_id": run.input_feature_space_id,
                    "source_assignment_set_id": run.source_assignment_set_id,
                    "method": run.method,
                    "params": json.loads(run.params_json),
                    "seed": run.seed,
                    "recompute_deterministic": run.recompute_deterministic,
                }
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(
                            f"Propagation run {run.propagation_run_id} {key} mismatch"
                        )
                if manifest["model_checksum"] != resolve_checksum(
                    self._folio, run.model_ref
                ) or manifest["quality_summary_checksum"] != resolve_checksum(
                    self._folio, run.quality_summary_ref
                ):
                    raise ValueError("Propagation run payload checksum mismatch")
                quality = self._folio.get(run.quality_summary_ref, frame="polars")
                quality_schema = pl.Schema(
                    {
                        "cell_id": pl.Int64,
                        "predicted_taxon_id": pl.Int64,
                        "purity": pl.Float32,
                        "n_neighbors": pl.Int32,
                        "coverage": pl.Float32,
                    }
                )
                if quality.schema != quality_schema:
                    raise TypeError("Propagation quality summary schema mismatch")
                application = self.get_scope(run.application_scope_id)
                members = self._folio.get(application.members_ref, frame="polars")
                if not quality.select("cell_id").equals(members):
                    raise ValueError(
                        "Propagation quality does not cover application scope"
                    )
                if quality.null_count().sum_horizontal()[0]:
                    raise ValueError("Propagation quality values cannot be null")
                if quality.filter(
                    (pl.col("purity") < 0)
                    | (pl.col("purity") > 1)
                    | (pl.col("coverage") < 0)
                    | (pl.col("coverage") > 1)
                    | (pl.col("n_neighbors") < 1)
                ).height:
                    raise ValueError("Propagation quality values are out of range")
                source = self.get_assignment_set(run.source_assignment_set_id)
                taxonomy = self.get_taxonomy(
                    source.taxonomy_name, source.taxonomy_version
                )
                if not set(quality["predicted_taxon_id"].to_list()) <= set(
                    taxonomy["taxon_id"].to_list()
                ):
                    raise ValueError("Propagation predicts taxa outside its taxonomy")
                if trusted_models and run.recompute_deterministic:
                    model = self._folio.get_model(run.model_ref, trusted=True)
                    space = self.get_feature_space(run.input_feature_space_id)
                    values = self._folio.get(space.values_ref, frame="polars")
                    selected = members.join(values, on="cell_id", how="left")
                    features = [
                        column for column in selected.columns if column != "cell_id"
                    ]
                    predicted = model.predict(selected.select(features).to_numpy())
                    probabilities = model.predict_proba(
                        selected.select(features).to_numpy()
                    ).max(axis=1)
                    if predicted.tolist() != quality["predicted_taxon_id"].to_list():
                        raise ValueError("Propagation prediction replay mismatch")
                    if not pl.Series(probabilities.astype("float32")).equals(
                        quality["purity"]
                    ):
                        raise ValueError("Propagation purity replay mismatch")

        assignment_sets = frames.get("assignment_set")
        if assignment_sets is not None:
            for row in assignment_sets.iter_rows(named=True):
                record = _assignment_set_from_row(row)
                taxonomy = self.get_taxonomy(
                    record.taxonomy_name, record.taxonomy_version
                )
                head = self.get_decision(record.decision_head_id)
                if head.review_branch != record.review_branch:
                    raise ValueError("Assignment-set decision branch mismatch")
                assignments = self.assignments(record)
                validate_table(assignments, CONTRACTS["assignment"])
                if set(assignments["assignment_set_id"].to_list()) != {
                    record.assignment_set_id
                }:
                    raise ValueError("Assignment artifact owner id mismatch")
                semantic = assignments.drop("assignment_set_id", "created_at")
                state = {
                    "taxonomy_name": record.taxonomy_name,
                    "taxonomy_version": record.taxonomy_version,
                    "review_branch": record.review_branch,
                    "decision_head_id": record.decision_head_id,
                    "assignments_checksum": canonical_hash(semantic.to_dicts()),
                }
                if canonical_hash(state) != record.state_hash:
                    raise ValueError("Assignment-set state hash mismatch")
                universe = self.universe()
                universe_cells = self._folio.get(universe.cells_ref, frame="polars")
                require_membership_subset(
                    assignments.select("cell_id"),
                    universe_cells,
                    container_name="the universe",
                )
                if not set(assignments["taxon_id"].drop_nulls().to_list()) <= set(
                    taxonomy["taxon_id"].to_list()
                ):
                    raise ValueError("Assignment artifact references unknown taxa")
                incomplete_coverage = assignments.filter(
                    pl.col("coverage_feature_space_id").is_null()
                    != pl.col("coverage").is_null()
                )
                if not incomplete_coverage.is_empty():
                    raise ValueError("Assignment coverage provenance is incomplete")
                for revision_id in assignments["source_revision_id"].unique():
                    self.get_revision(revision_id)
                for decision_id in assignments["decision_id"].drop_nulls().unique():
                    self.get_decision(decision_id)
                for run_id in assignments["propagation_run_id"].drop_nulls().unique():
                    self.get_propagation_run(run_id)
                for feature_space_id in (
                    assignments["coverage_feature_space_id"].drop_nulls().unique()
                ):
                    self.get_feature_space(feature_space_id)

    def _validate_slice_six_artifacts(self, frames: Mapping[str, pl.DataFrame]) -> None:
        releases = frames.get("annotation_release")
        if releases is None:
            return
        for row in releases.iter_rows(named=True):
            self.validate_annotation_release(_annotation_release_from_row(row))

    @staticmethod
    def _manifest_nullable_columns(
        manifest: Mapping[str, Any], *, component_id: str
    ) -> tuple[str, ...]:
        values = manifest["nullable_columns"]
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"Manifest {component_id} has invalid nullable_columns")
        return tuple(values)

    def _validate_universe_artifacts(self, universe: Universe) -> pl.DataFrame:
        manifest = self._component_manifest(
            "universe",
            universe.universe_id,
            {
                "component_kind",
                "schema_version",
                "cells_checksum",
                "schema_hash",
                "semantic_roles",
                "nullable_columns",
                "source_refs",
            },
        )
        nullable = self._manifest_nullable_columns(
            manifest, component_id=universe.universe_id
        )
        cells = self._folio.get(universe.cells_ref, frame="polars")
        validate_universe_cells(cells, nullable_columns=nullable)
        actual_schema_hash = table_schema_hash(cells, nullable_columns=nullable)
        if manifest["schema_hash"] != actual_schema_hash:
            raise ValueError(f"Universe {universe.universe_id} schema hash mismatch")
        if universe.schema_hash != actual_schema_hash:
            raise ValueError(
                f"Universe registry {universe.universe_id} schema hash mismatch"
            )
        if manifest["cells_checksum"] != resolve_checksum(
            self._folio, universe.cells_ref
        ):
            raise ValueError(f"Universe {universe.universe_id} checksum mismatch")
        roles_json = semantic_roles_json(cells, manifest["semantic_roles"])
        if roles_json != universe.semantic_roles_json:
            raise ValueError(f"Universe {universe.universe_id} semantic roles mismatch")
        if tuple(manifest["source_refs"]) != universe.source_refs:
            raise ValueError(f"Universe {universe.universe_id} source refs mismatch")
        for ref in universe.source_refs:
            self._folio.item_info(ref)
        return cells

    def _validate_feature_block_artifacts(
        self,
        feature_blocks: pl.DataFrame,
        catalog: pl.DataFrame,
        universe: Universe,
        universe_cells: pl.DataFrame,
    ) -> None:
        for row in feature_blocks.iter_rows(named=True):
            block = _feature_block_from_row(row)
            manifest = self._component_manifest(
                "feature-block",
                block.feature_block_id,
                {
                    "component_kind",
                    "schema_version",
                    "universe_id",
                    "values_checksum",
                    "schema_hash",
                    "nullable_columns",
                    "source_refs",
                },
            )
            nullable = self._manifest_nullable_columns(
                manifest, component_id=block.feature_block_id
            )
            if (
                manifest["universe_id"] != universe.universe_id
                or block.universe_id != universe.universe_id
            ):
                raise ValueError(
                    f"Feature block {block.feature_block_id} universe mismatch"
                )
            values = self._folio.get(block.values_ref, frame="polars")
            validate_cell_table(values, nullable_columns=nullable)
            require_membership_subset(
                values.select("cell_id"),
                universe_cells,
                container_name="the universe",
            )
            actual_schema_hash = table_schema_hash(values, nullable_columns=nullable)
            if (
                manifest["schema_hash"] != actual_schema_hash
                or block.schema_hash != actual_schema_hash
            ):
                raise ValueError(
                    f"Feature block {block.feature_block_id} schema hash mismatch"
                )
            if manifest["values_checksum"] != resolve_checksum(
                self._folio, block.values_ref
            ):
                raise ValueError(
                    f"Feature block {block.feature_block_id} checksum mismatch"
                )
            if tuple(manifest["source_refs"]) != block.source_refs:
                raise ValueError(
                    f"Feature block {block.feature_block_id} source refs mismatch"
                )
            for ref in block.source_refs:
                self._folio.item_info(ref)

            block_catalog = catalog.filter(
                pl.col("feature_block_id") == block.feature_block_id
            )
            prepared = prepare_feature_catalog(
                block_catalog.drop("feature_block_id"),
                feature_block_id=block.feature_block_id,
                value_columns=[
                    column for column in values.columns if column != "cell_id"
                ],
            )
            if not prepared.equals(block_catalog):
                raise ValueError(
                    f"Feature block {block.feature_block_id} catalog is not canonical"
                )

    def validate(self, *, trusted_models: bool = False) -> None:
        """Validate integrity and contracts without loading pickles by default.

        Set ``trusted_models=True`` only for a self-authored or otherwise trusted
        study to additionally replay deterministic fitted-state artifacts.
        """
        self._clear_registry_cache()
        integrity = self._folio.validate()
        invalid = sorted(name for name, valid in integrity.items() if not valid)
        if invalid:
            raise ValueError(f"DataFolio integrity validation failed: {invalid}")

        head = self.head_commit()
        if head is None:
            return
        frames = {name: self.registry(name) for name in head.registries}
        complete_frames = {
            name: frames.get(name, pl.DataFrame(schema=contract.schema))
            for name, contract in CONTRACTS.items()
        }
        validate_registries(complete_frames, require_foreign_targets=True)
        assert_architecture_contracts()

        if "universe" in frames:
            universe = self.universe()
            universe_cells = self._validate_universe_artifacts(universe)
        else:
            universe = None
            universe_cells = None

        if (
            universe is not None
            and universe_cells is not None
            and "feature_block" in frames
        ):
            catalog = frames.get(
                "feature_catalog",
                pl.DataFrame(schema=CONTRACTS["feature_catalog"].schema),
            )
            self._validate_feature_block_artifacts(
                frames["feature_block"], catalog, universe, universe_cells
            )

        if universe is not None and universe_cells is not None and "scope" in frames:
            scope_members: dict[str, pl.DataFrame] = {}
            for row in frames["scope"].iter_rows(named=True):
                scope = _scope_from_row(row)
                scope_members[scope.scope_id] = self._validate_scope_record(
                    scope, universe, universe_cells
                )

            for row in frames["scope"].iter_rows(named=True):
                scope = _scope_from_row(row)
                if scope.parent_scope_id is not None:
                    require_membership_subset(
                        scope_members[scope.scope_id],
                        scope_members[scope.parent_scope_id],
                        container_name="the parent scope",
                    )

        if "feature_selection" in frames and "feature_catalog" in frames:
            catalog = frames["feature_catalog"]
            for row in frames["feature_selection"].iter_rows(named=True):
                selection = _selection_from_row(row)
                self._validate_feature_selection_record(selection, catalog)

        self._validate_slice_two_artifacts(frames, trusted_models=trusted_models)
        self._validate_slice_three_artifacts(frames)
        self._validate_slice_four_artifacts(frames, trusted_models=trusted_models)
        self._validate_slice_six_artifacts(frames)

        if "kept_revision" in frames:
            for row in frames["kept_revision"].iter_rows(named=True):
                revision = _revision_from_row(row)
                state = {
                    "parent_revision_id": revision.parent_revision_id,
                    "scope_id": revision.scope_id,
                    "feature_space_id": revision.feature_space_id,
                    "clustering_representation_id": (
                        revision.clustering_representation_id
                    ),
                    "visualization_representation_id": (
                        revision.visualization_representation_id
                    ),
                    "candidate_set_id": revision.candidate_set_id,
                }
                if canonical_hash(state) != revision.state_hash:
                    raise ValueError(
                        f"Kept revision {revision.revision_id} state hash mismatch"
                    )
