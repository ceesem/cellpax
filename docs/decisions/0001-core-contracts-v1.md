# ADR 0001: Core contracts version 1

- Status: Accepted
- Date: 2026-07-18
- Applies to: CellPax storage and table contracts, schema version `1.0`

## Context

CellPax needs immutable, replayable scientific state without making notebook state,
DataFolio item names, or serialized Parquet bytes stand in for scientific identity.
The design proposal and schema draft left several substrate decisions open: how a
study is reopened, what an artifact reference means, which bytes are hashed, how
nullability is specified, how scopes are tied to a universe, and how previews are
retained.

This record settles those questions for the first implementation. It does not
freeze the Python API. Changes to persisted contracts require a schema-version
change and a migration; implementation details behind the contracts may evolve.

## Decisions

### 1. A study is a DataFolio 2 folio with CellPax commit items

The DataFolio 2 `items.json` manifest is the study's only mutable storage locator;
CellPax does not add a parallel `study.json`. The folio metadata contains:

- `cellpax_study_id` — stable UUIDv4 entity id;
- `cellpax_schema_version` — initially `1.0`; and
- `cellpax_head_commit_id` — the current CellPax commit, or null before the first
  commit.

A CellPax commit is immutable JSON stored with
`folio.add("cellpax/commits/<commit_id>", manifest)`. The manifest contains the
schema version, parent commit id, and DataFolio item ref for every complete
registry snapshot. The `commit_id` is its canonical manifest hash; creation
metadata may travel beside the commit but is excluded from that hash. Any earlier
commit remains directly openable with `folio.get()`.

Each registry snapshot is a complete Polars table stored as one DataFolio item.
Rows already present in a parent snapshot must be semantically identical in the
child. Appending a row writes a new registry snapshot and commit, physically
enforcing immutable registries and the append-only decision ledger.

All payloads, registry snapshots, the commit item, and the metadata head update are
published in one `with folio.batch():` transaction. DataFolio 2 supplies the local
writer lock, manifest revision check, rollback of staged state, and single atomic
`items.json` publication. `ConcurrentWriteError` is propagated to the caller; a
CellPax compare-and-swap or second transaction log would duplicate that machinery.
The DataFolio manifest revision is a storage transaction number, not a kept
revision or scientific identity.

### 2. CellPax uses the DataFolio 2 public API directly

There is no shadow `ArtifactStore` abstraction in version 1. `Study` owns a
`DataFolio` instance and uses its public 2.0 API:

- `add()` for Polars tables and JSON manifests;
- `add_file()` and `add_model()` for file and fitted-state payloads;
- `get()`, `scan_table()`, `item_path()`, and `item_info()` for reads;
- `inputs=` for physical lineage;
- `batch()` for atomic CellPax commits; and
- `validate()` for payload existence and checksum verification.

One materialized component payload is one DataFolio item: universe cells,
feature-block values, feature-selection members, scope members, fitted state,
representation coordinates, candidate definitions, or memberships. Registry
snapshots and commit manifests are items too; registry rows are not individual
items. CellPax never overwrites these items. Internal names use valid DataFolio
namespaces such as `cellpax/objects/<kind>/<uuid>`,
`cellpax/registries/<registry>/<uuid>`, and
`cellpax/commits/<commit_id>`. Reconstructable component manifests use predictable
refs at `cellpax/manifests/<kind>/<component_id>`.

Slice 2 missingness reports are immutable Polars items linked as inputs to the
materialized feature-space values. They are intentionally not another registry
column or component-identity field: they are a diagnostic projection of the
already identified inputs, fit/application scopes, and missing policy. The public
API resolves the one report through `inputs=` lineage. Fitted sklearn-compatible
state is stored with `add_model()` and loaded as trusted only when replaying a
model owned by the opened study.

Feature-space values cover their declared application scope exactly. A `drop`
policy that would reduce application coverage raises with the retained membership;
the caller must create an explicit child scope and retry. Stored-state replay
reapplies the owned fitted estimator and checks its output against the retained
artifact. It is an integrity guarantee, not a claim that refitting the recipe will
reproduce estimator or coordinate bytes.

Every schema `*_ref` is a DataFolio logical item name, not a filesystem path or a
second URI scheme. `item_path()` resolves its current storage location and
`item_info()` resolves its DataFolio checksum. CellPax semantic SHA-256 ids exclude
the storage name and incorporate the resolved output checksum instead, avoiding a
hash/name cycle and allowing duplicate computations to reuse the first registered
component. DataFolio checksums cover exact stored bytes; CellPax SHA-256 hashes
cover scientific semantics and identity.

CellPax invokes `validate()` at explicit study-validation and release boundaries.
DataFolio snapshots may be used for backup or user-facing checkpoints, but they
are not the CellPax kept-revision model and do not determine component identity.

Until DataFolio 2.0 is published, CellPax development resolves it from the sibling
local checkout at `../datafolio` and targets the API documented on that checkout's
`v2-refactor` branch. No DataFolio 1.x compatibility shim or deprecated method
family is implemented. Once the local package metadata is bumped and 2.0 is
published, the source override can be removed without changing CellPax storage
semantics.

### 3. SHA-256 and canonical manifests define identity

All CellPax digests use SHA-256 and lowercase hexadecimal encoding.

Canonical JSON is UTF-8 with sorted object keys, no insignificant whitespace, and
no NaN or infinity values. Lists retain order; unordered collections are forbidden.
Datetimes included in semantic content are normalized to UTC RFC 3339. Audit
fields are excluded from semantic manifests unless a contract explicitly says
otherwise.

Content-addressed component ids are hashes of a typed canonical manifest that
includes:

- component kind and schema version;
- semantic upstream component ids;
- resolved parameters and seeds;
- fitted-state checksums where applicable; and
- output checksums resolved from DataFolio refs.

Human names, notes, authors, and creation timestamps are excluded. Config hashes
cover fully resolved computational config, including config schema version. State
hashes cover the kept revision's ordered pointer fields. Entity ids
(`revision_id`, `decision_id`, `assignment_set_id`, and `annotation_release_id`)
are lowercase canonical UUIDv4 strings and are not content hashes.

Stored-byte digests and semantic helper hashes are deliberately separate. In
particular, `membership_hash` is SHA-256 over the count followed by sorted unique
`cell_id` values encoded as signed little-endian 64-bit integers. Ordered feature
selection membership is hashed as canonical ordered records. Parquet bytes are
never assumed to be stable across library versions.

`schema_hash` covers an artifact's ordered column names, canonical dtype strings,
and nullability. `catalog_hash` covers feature-catalog semantic fields sorted by
`(feature_block_id, feature_id)`; display order and audit metadata do not affect it.

Every component manifest starts with `component_kind` and `schema_version`. Its
remaining version 1 fields are fixed as follows; JSON-valued fields are decoded and
canonicalized rather than hashed as arbitrary source strings.

| Component id | Manifest fields after kind/version |
| --- | --- |
| `universe_id` | resolved `cells_checksum`, `schema_hash`, `semantic_roles`, sorted `nullable_columns`, sorted `source_refs` |
| `feature_block_id` | `universe_id`, resolved `values_checksum`, `schema_hash`, sorted `nullable_columns`, sorted `source_refs` |
| `feature_selection_id` | `catalog_hash`, ordered semantic membership records, `n_features` |
| `scope_id` | `universe_id`, `parent_scope_id`, `membership_hash`, `n_cells`, `derivation_text` |
| `feature_space_id` | `scope_id`, `fit_scope_id`, `parent_feature_space_id`, sorted `input_feature_block_ids`, `feature_selection_id`, `transform`, resolved `params`, resolved `fitted_state_checksum`, resolved `values_checksum`, `missing_policy`, `seed` |
| `representation_id` | `scope_id`, `fit_scope_id`, `method`, `input_feature_space_id`, canonical `spatial_input`, `n_components`, resolved `params`, resolved `fitted_state_checksum`, resolved `coords_checksum`, `seed`, `recompute_deterministic`, canonical `software_versions` |
| `clustering_run_id` | `scope_id`, `representation_id`, `method`, resolved `compute_params`, canonical `spatial_input`, resolved `generator_checksum`, resolved hierarchy checksums, `seed`, `recompute_deterministic`, canonical `software_versions` |
| `candidate_set_id` | `clustering_run_id`, `scope_id`, `cut_method`, resolved `cut_params`, resolved `definitions_checksum`, resolved `membership_checksum` |
| `propagation_run_id` | `fit_scope_id`, `application_scope_id`, `input_feature_space_id`, `source_assignment_set_id`, `method`, resolved `params`, resolved `model_checksum`, resolved `quality_summary_checksum`, `seed`, `recompute_deterministic` |

Null is encoded explicitly for absent optional fields. `derivation_text` is included
for a scope because derivation lineage is part of scope identity. It is excluded
from `feature_selection_id`: an exact ordered selection is reusable regardless of
which equivalent Polars expression found it, and the first registered derivation
remains its documentation hint.

The `catalog_hash` in that identity covers the complete study catalog. An
otherwise identical selection created after unrelated catalog growth therefore
has a different id. Version 1 accepts this because foundational blocks are
registered before feature selection in the target workflow; changing to a
selected-subcatalog hash is an identity-contract change, not an implementation
detail.

Slice 1 has no `feature_selection_id` column on `kept_revision`. Passing a
selection to `keep()` roots it in the selection registry and includes its id in
the fully resolved `config_hash`, but does not create a directly queryable
revision-to-selection edge. This is intentional interim behavior: Slice 2 gives
the selection a structural home at `feature_space.feature_selection_id`, and the
revision points to that feature space.

`kept_revision.state_hash` covers, in schema order, `parent_revision_id`,
`scope_id`, `feature_space_id`, `clustering_representation_id`,
`visualization_representation_id`, and `candidate_set_id`. An
`assignment_set.state_hash` covers `taxonomy_name`, `taxonomy_version`,
`review_branch`, `decision_head_id`, and the resolved assignments checksum. Names,
entity ids, DataFolio item refs, and audit fields are excluded from both
projections.

### 4. Content-id columns are not duplicated

For a content-addressed component, its primary `*_id` is its canonical manifest
hash. Separate `content_hash` columns are removed from schema version 1 because
they duplicated that identity without adding a second invariant.

Artifact refs resolve to DataFolio entries that retain their own stored-byte
checksum. `config_hash`, `state_hash`, `schema_hash`, `catalog_hash`, and
`membership_hash` remain because they identify different semantic projections.

### 5. Schema contracts include constraints, not only Polars dtypes

Each table contract consists of:

- a `pl.Schema` for column order and physical dtypes;
- an explicit set of nullable columns;
- primary and alternate keys;
- enum and numerical-domain constraints;
- row-level rules such as exactly-one-of fields; and
- referential and cross-artifact checks evaluated in study context.

Columns are non-null unless listed as nullable. Empty collections are preferred to
null collections. Unknown columns fail validation by default. Schema evolution is
handled by an explicit migration rather than permissive reads.

### 6. Every scope is anchored directly to a universe

`scope.universe_id` is required. A child scope must have the same universe as its
parent, and every member must exist in that universe. This avoids relying on a
kept revision or feature-space traversal merely to identify the coordinate system.

Two scopes may retain the same `membership_hash` while having different
`scope_id` values because derivation lineage is part of the scope manifest.

### 7. Previews are unrooted immutable content

Materialized previews use the same content addressing and integrity rules as kept
components, but they are not reachable from a study commit unless kept. Version 1
does not delete previews automatically. A future explicit garbage collector must
default to dry-run, treat study commits and releases as roots, and respect a grace
period. Preview retention therefore cannot destroy work during the first vertical
slice.

### 8. Acceptance scenarios F-J are defined locally

The current proposal references scenarios from an earlier document that is not in
this repository. For version 1 their required behavior is:

- **F — UI selection:** a lasso selection remains transient until an explicit API
  operation creates a scope or decision; reopening the study reproduces every
  durable result without browser state.
- **G — assignment-only correction:** correcting, propagating, or marking a cell
  ambiguous creates a new assignment set without changing taxonomy vocabulary.
- **H — self-contained release:** a release can be understood from its manifest,
  taxonomy table, assignment table, decision head, quality summary, and recipe in
  a fresh process.
- **I — Trajan consumption:** a thin adapter decorates Trajan's cell universe from
  a release without importing clustering or review machinery.
- **J — reused clustering run:** two candidate cuts share one expensive clustering
  run, coexist as previews, and keeping either cut does not alter the run or the
  other cut.

## Version 1 nullability convention

The schema source of truth lists nullable columns by table. Conditional rules are
validated separately; for example, a tree root has a null parent, a propagated
assignment requires a propagation run, and a decision normally has exactly one of
`target_ids` or `target_ref`.

Canonical dtype strings and these manifest field sets will be constants in the
Phase 1 implementation and covered by golden-vector tests. They are not delegated
to `repr()` output from Polars, Python, or an artifact backend.

## Consequences

- Reopening and replay do not depend on notebook execution order.
- DataFolio physical paths can change without changing refs or scientific ids;
  logical item names remain stable storage refs within a study.
- Registry writes are heavier than in-place row updates, but the tables are small
  metadata indexes and immutable commit items make append-only behavior auditable.
- Parquet reserialization may change a DataFolio checksum and therefore a component
  id; software versions and recomputation guarantees make that visible.
- CellPax relies on DataFolio 2's public transaction and manifest guarantees rather
  than implementing a second storage transaction layer.
- Automatic preview garbage collection and cross-process cloud-writer coordination
  remain deferred until a demonstrated need.
