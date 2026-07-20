# Core Data Contracts — Version 1

**Status:** Accepted substrate contract (`schema_version="1.0"`), companion to `DESIGN_PROPOSAL.md`
**Scope:** The load-bearing tables everything else (and the future HTML tool) sits on, including their input registries and release identities.

The Phase 0 storage and identity decisions are recorded in
[`docs/decisions/0001-core-contracts-v1.md`](docs/decisions/0001-core-contracts-v1.md).
Changes to persisted fields or semantics require a schema-version change and an
explicit migration.

All tables are Polars-typed, immutable parquet items stored through the DataFolio
2.0 public API. `cell_id` is the canonical universe key (`pl.Int64`); `root_id` is
an alias column in the universe, never a join key here. Content-addressed component
ids key the cache and make replay verifiable; they are computed by the library when
a preview is materialized, not by notebook naming or workflow-sandbox state.

Three design rules are enforced as invariants, not conventions:

1. **No method-specific matrix or threshold leaks.** Required downstream operations do not reference a coclustering matrix or cut threshold. They consume candidate membership, representations, and optional generic hierarchy/boundary-evidence tables. The fauxnograph dense `SimilarityMatrix` is an internal generator artifact.
2. **Append-only.** `decision` rows are never updated or deleted. Curation is expressed as new rows, so the ledger is a complete history of *why*.
3. **Immutable content, separate naming.** Content-addressed components are never renamed when kept. A named `kept_revision` points to immutable components. DataFolio records a checksum for every owned payload and `validate()` verifies it at validation/release boundaries, independently of whether replaying a stochastic recipe is expected to reproduce identical output.

## Identity and hashing rules

The schema uses four related identities deliberately:

- `*_id` on content components is the hash of a canonical manifest containing semantic inputs, resolved parameters, and resolved fitted-state/output checksums;
- `*_ref` is a DataFolio logical item name whose manifest entry carries the exact-byte checksum;
- `config_hash` hashes a fully resolved computational configuration; and
- entity ids such as `revision_id`, `assignment_set_id`, and `annotation_release_id` identify immutable named records and need not equal a content digest.

Human names, timestamps, and authors are excluded from content/config hashes. `recompute_deterministic=False` means rerunning a recipe may produce a different valid output; it never disables integrity verification of the stored artifact.

Content-addressed component `*_id` columns are the canonical manifest hash. They
are not repeated in a separate `content_hash` column. Entity ids (`revision_id`,
`decision_id`, `assignment_set_id`, and `annotation_release_id`) are UUIDv4 strings.
Artifact refs use namespaced DataFolio item names. `item_info(ref)` resolves their
stored-byte checksum; physical file paths are never persisted in these tables.

## Study manifests and registry commits

The Polars tables below are stored as immutable DataFolio registry items. The folio
metadata stores `cellpax_study_id`, `cellpax_schema_version`, and
`cellpax_head_commit_id`; there is no separate `study.json`. Each immutable commit
is a JSON item at `cellpax/commits/<commit_id>` containing `schema_version`,
`parent_commit_id`, and the DataFolio item ref for every registry snapshot. Earlier
commits remain directly openable with `folio.get()`.

Appending registry content writes a complete new registry snapshot and commit.
Existing rows must remain semantically identical. Payloads, snapshots, the commit
item, and the metadata head are published in one `folio.batch()` transaction.
DataFolio's manifest revision and `ConcurrentWriteError` enforce the single-writer
boundary; the DataFolio revision is not a CellPax revision id.

Universe and feature-block canonical manifests are stored at predictable
`cellpax/manifests/<kind>/<component_id>` refs. Their declared
`nullable_columns` are part of content identity, so dynamic table contracts can be
reconstructed and validated after a fresh-process reopen.

## Constraint and nullability convention

`pl.Schema` specifies column order and physical dtypes, but it does not encode the
whole contract. The implementation pairs each schema with keys, enums, numerical
domains, row rules, foreign keys, and an explicit nullable-column set. Columns are
non-null unless listed below; collection columns use empty lists rather than null
unless a row rule requires otherwise. Unknown columns fail validation.

| Table | Nullable columns |
| --- | --- |
| `universe` | none |
| `feature_block` | none |
| `feature_catalog` | `units`, `description` |
| `feature_selection` | none |
| `scope` | `parent_scope_id` |
| `feature_space` | `parent_feature_space_id`, `input_feature_block_ids`, `fitted_state_ref`, `values_ref`, `seed` |
| `representation` | `spatial_input_json`, `n_components`, `fitted_state_ref`, `seed` |
| `clustering_run` | `spatial_input_json`, `hierarchy_nodes_ref`, `hierarchy_members_ref`, `seed` |
| `candidate_hierarchy_node` | `parent_node_id`, `merge_score` |
| `candidate_hierarchy_membership` | none |
| `candidate_set` | none |
| `candidate_definition` | `hierarchy_node_id` |
| `candidate_membership` | `candidate_id`, `membership_strength` |
| `candidate_boundary_evidence` | `n_permutations`, `evidence_ref` |
| `kept_revision` | `parent_revision_id`, `feature_space_id`, `clustering_representation_id`, `visualization_representation_id`, `candidate_set_id`, `notes` |
| `decision` | `parent_decision_id`, `target_ids`, `target_ref`, `taxon_id` |
| `taxonomy` | `parent_id`, `description`, `color`, `replaced_by` |
| `propagation_run` | `seed` |
| `assignment_set` | none |
| `assignment` | `taxon_id`, `decision_id`, `propagation_run_id`, `coverage_feature_space_id`, `coverage`, `confidence`, `alternatives_json` |
| `annotation_release` | none |

## Foundational inputs

### `universe`

The authoritative cell coordinate system. `cells_ref` points to the cell table containing canonical `cell_id`, aliases such as `root_id`, and stable properties. Semantic roles declare meanings such as spatial position without guessing column names.

```python
pl.Schema({
    "universe_id":         pl.Utf8,     # canonical manifest/content hash
    "cells_ref":           pl.Utf8,     # artifact: table containing unique cell_id
    "schema_hash":         pl.Utf8,
    "semantic_roles_json": pl.Utf8,     # e.g. {"position": ["soma_x", "soma_y", "soma_z"]}
    "source_refs":         pl.List(pl.Utf8),
    "created_at":          pl.Datetime("us", "UTC"),
    "created_by":          pl.Utf8,
})
# unique: universe_id
```

### `feature_block` and `feature_catalog`

A feature block is an immutable cell-keyed input, usually supplied by ossify-extraction or another external producer. Its values remain separate from corrected/scaled layers.

```python
pl.Schema({
    "feature_block_id": pl.Utf8,     # canonical manifest/content hash
    "universe_id":      pl.Utf8,
    "values_ref":       pl.Utf8,     # artifact: cell_id + physical feature columns
    "schema_hash":      pl.Utf8,
    "source_refs":      pl.List(pl.Utf8),
    "created_at":       pl.Datetime("us", "UTC"),
    "created_by":       pl.Utf8,
})
# unique: feature_block_id

pl.Schema({
    "feature_id":       pl.Utf8,     # stable semantic id, independent of column order
    "feature_block_id": pl.Utf8,
    "column_name":      pl.Utf8,
    "modality":         pl.Utf8,
    "family":           pl.Utf8,
    "units":            pl.Utf8,
    "description":      pl.Utf8,
    "raw_or_derived":   pl.Utf8,
})
# unique: (feature_block_id, feature_id); unique: (feature_block_id, column_name)
```

### `feature_selection`

Feature selection is performed with ordinary Polars expressions. Keeping it materializes the exact ordered ids; the order is part of its identity because numerical matrices depend on it.

```python
pl.Schema({
    "feature_selection_id": pl.Utf8,     # canonical manifest/content hash
    "catalog_hash":         pl.Utf8,
    "members_ref":          pl.Utf8,     # artifact: (position, feature_block_id, feature_id)
    "n_features":           pl.Int32,
    "derivation_text":      pl.Utf8,     # documentation/replay hint, not an expression DSL
    "created_at":           pl.Datetime("us", "UTC"),
    "created_by":           pl.Utf8,
})
# unique: feature_selection_id
```

The membership artifact is authoritative. `derivation_text` may contain generated Polars code, but is not treated as a stable serialization of `pl.Expr`.

`catalog_hash` covers the study's complete feature catalog, not only the selected
rows. Consequently, registering an unrelated feature block changes the identity
of an otherwise identical later selection. Version 1 intentionally assumes input
blocks are registered before selections are retained; making selection identity
stable across catalog growth would require an ADR change.

During Slice 1, `keep(..., feature_selection=selection)` roots the selection in the
`feature_selection` registry and includes its id in the revision's `config_hash`,
but `kept_revision` has no direct selection foreign key. The tables therefore do
not independently answer which selection a scope-only revision used. Slice 2
closes that temporary structural gap through
`feature_space.feature_selection_id`, with the kept revision pointing to that
feature space.

---

## Seam 1 — `representation`

The algorithm-agnostic clustering-input layer. PCA today; VAE, scaled-passthrough, or anything else later. The coordinate array is a separate artifact (`coords_ref`); this table is its identity and provenance. The `fit_scope` vs `scope` split is what makes branch-local refits (the CGE re-scale) explicit.

```python
pl.Schema({
    "representation_id":        pl.Utf8,     # content hash of (inputs + params + fitted state)
    "scope_id":                 pl.Utf8,     # cells the coords are DEFINED over (application scope)
    "fit_scope_id":             pl.Utf8,     # cells the transform was FIT on (may == scope_id)
    "method":                   pl.Utf8,     # "pca" | "vae" | "scaled_passthrough" | "banksy" | ...
    "input_feature_space_id":   pl.Utf8,     # FK -> feature_space (scaled/corrected layer that fed in)
    "spatial_input_json":       pl.Utf8,     # null unless spatially-aware: {position_role, graph:{k|radius}, weight}
    "n_components":             pl.Int32,    # output dimensionality; null if variable
    "params_json":              pl.Utf8,     # resolved hyperparameters (JSON)
    "fitted_state_ref":         pl.Utf8,     # artifact ref: sklearn pipeline / VAE weights; null if stateless
    "coords_ref":               pl.Utf8,     # artifact ref: table (cell_id, dim_0..dim_k), cell_id ⊆ scope
    "seed":                     pl.Int64,    # null if N/A
    "recompute_deterministic":  pl.Boolean,  # False means replay need not be byte-identical
    "software_versions":        pl.Utf8,     # JSON
    "created_at":               pl.Datetime("us", "UTC"),
    "created_by":               pl.Utf8,
})
# unique: representation_id
```

The upstream scaler (e.g. the clipped `(0.5, 99.5)` scaler for CGE) has its **own** identity in the `feature_space` registry below, with the same fit-scope discipline. Keeping it separate means a re-scale and a re-PCA are distinct, individually cacheable steps. Visualization embeddings use this same representation contract; a kept revision points separately to its clustering and visualization representations.

**Spatial-transcriptomics-style methods** (spatial organization *plus* feature similarity) enter here, and they take two shapes that hit different seams:

- *Augmentation-style* (Banksy, UTAG): fold each cell's spatial neighborhood into its coordinates, then cluster normally. That is a `representation` — `method="banksy"`, `spatial_input_json` set, reading cell position from the universe via a semantic `position` role. The augmented coords are then clustered by **any** downstream generator.
- *Joint-model style* (SpaGCN, BayesSpace): model features and space together and emit clusters directly. That is a `clustering_run` carrying its own `spatial_input_json` and producing a native candidate set.

Either way, position enters through a *declared* spatial input, never as ad-hoc columns — so a spatial method is as swappable as CHOIR, and everything downstream still sees only `candidate_membership`. The one universe requirement this adds: position must be a semantically-tagged role, not a guessed column name.

### Upstream — `feature_space`

One immutable feature layer with first-class identity. Feature spaces form a chain so preprocessing stages with different fit scopes are not hidden inside one pipeline. For example, a globally corrected layer can parent a branch-locally scaled layer. Exactly one of `parent_feature_space_id` or `input_feature_block_ids` is populated.

```python
pl.Schema({
    "feature_space_id":     pl.Utf8,     # content hash
    "scope_id":             pl.Utf8,     # cells the layer is defined over
    "fit_scope_id":         pl.Utf8,     # cells this stage was FIT on; may differ from application scope
    "parent_feature_space_id": pl.Utf8,  # previous corrected/scaled layer; null for a block-root layer
    "input_feature_block_ids": pl.List(pl.Utf8), # populated only when parent is null
    "feature_selection_id": pl.Utf8,     # FK -> exact ordered feature selection
    "transform":            pl.Utf8,     # one stage: "raw_join" | "spline_regress" | "robust" | ...
    "params_json":          pl.Utf8,     # resolved transform params
    "fitted_state_ref":     pl.Utf8,     # artifact ref: fitted sklearn transformer
    "values_ref":           pl.Utf8,     # optional materialized cell_id + ordered feature values
    "missing_policy":       pl.Utf8,     # "median" | "drop" | ...  (explicit missingness)
    "seed":                 pl.Int64,
    "created_at":           pl.Datetime("us", "UTC"),
    "created_by":           pl.Utf8,
})
# unique: feature_space_id
```

If `values_ref` is absent, the layer must be reconstructable from its parent/input blocks, selection, resolved transform, and fitted state. Fit scope and application scope are independent universe subsets: fitting globally and applying to a branch is valid, as is fitting and applying locally.

Materialized values cover `scope_id` exactly. Under `missing_policy="drop"`, a
preview that would remove application cells fails with the surviving membership;
the caller explicitly creates a child scope and retries against that scope. The
fit scope may drop incomplete fitting rows without changing application coverage.
This prevents a feature space from silently claiming scope `X` while only defining
values on `X′`.

---

## Seam 2 — clustering runs and candidate sets

### `clustering_run`

The expensive method-specific computation is cached independently of any cheap cut or final partition. For fauxnograph this owns the consensus/linkage basis; for an external CHOIR process it owns the returned generator artifact. Downstream does not inspect `generator_ref` directly.

```python
pl.Schema({
    "clustering_run_id":       pl.Utf8,     # canonical manifest/content hash
    "scope_id":                pl.Utf8,
    "representation_id":       pl.Utf8,     # always FK -> representation
    "method":                  pl.Utf8,     # "fauxnograph" | "choir" | "hdbscan" | ...
    "compute_params_json":     pl.Utf8,     # expensive-stage resolved params
    "spatial_input_json":      pl.Utf8,     # null unless a joint spatial generator
    "generator_ref":           pl.Utf8,     # method-specific cached artifact/process result
    "hierarchy_nodes_ref":     pl.Utf8,     # optional generic candidate_hierarchy_node table
    "hierarchy_members_ref":   pl.Utf8,     # optional generic leaf-membership table
    "seed":                    pl.Int64,
    "recompute_deterministic": pl.Boolean,
    "software_versions":       pl.Utf8,
    "created_at":              pl.Datetime("us", "UTC"),
    "created_by":              pl.Utf8,
})
# unique: clustering_run_id
```

Every generator consumes a representation. A direct method that operates on scaled features uses a `representation(method="scaled_passthrough")`; this avoids a polymorphic feature-space-or-representation foreign key.

### Optional `candidate_hierarchy`

Only methods with a meaningful nested tree populate these contracts. Non-nested Leiden resolution sweeps remain separate candidate sets and are not forced into a fake tree.

```python
pl.Schema({
    "clustering_run_id": pl.Utf8,
    "node_id":           pl.Utf8,
    "parent_node_id":    pl.Utf8,
    "level":             pl.Int32,
    "merge_score":       pl.Float64,  # method-relative; null where unavailable
    "n_cells":           pl.Int64,
    "metadata_json":     pl.Utf8,
})
# unique: (clustering_run_id, node_id)

pl.Schema({
    "clustering_run_id": pl.Utf8,
    "cell_id":           pl.Int64,
    "leaf_node_id":      pl.Utf8,
})
# unique: (clustering_run_id, cell_id)
```

### `candidate_membership`

The output of **any** clustering algorithm, in one shape. Long table, one row per (set, cell). No threshold, no matrix — just the assignment that landed.

```python
pl.Schema({
    "candidate_set_id":    pl.Utf8,     # FK -> candidate_set
    "cell_id":             pl.Int64,    # ⊆ the set's scope
    "candidate_id":        pl.Int32,    # cluster label WITHIN this set; NULL = noise/unclustered
    "membership_strength": pl.Float32,  # confidence in [0,1]; null for hard assignment
})
# unique: (candidate_set_id, cell_id)
```

- **fauxnograph:** `candidate_id` = agglomerative cut label; `membership_strength` is initially NULL because the prototype does not emit a defined per-cell strength. A later named metric may populate it.
- **HDBSCAN/DBSCAN on an embedding:** `candidate_id` = `labels_` with noise → **NULL** (not `-1`), `membership_strength` = `probabilities_`.
- **CHOIR prune:** `candidate_id` = pruned-tree leaf, `membership_strength` = leaf assignment probability.
- **Spatial (Banksy repr → any generator, or SpaGCN direct):** identical shape — spatial structure is baked into the coords or the joint model upstream, never into this table.

`candidate_id` is revision-local and disposable. It is *never* a taxon. Promotion happens only through a `decision`.

### `candidate_set`

One cheap partition derived from a clustering run. A different threshold, minimum cluster size, resolution, or selected tree cut creates a different candidate set while reusing the same run where valid.

```python
pl.Schema({
    "candidate_set_id":  pl.Utf8,     # content hash
    "clustering_run_id": pl.Utf8,     # FK -> clustering_run
    "scope_id":          pl.Utf8,
    "cut_method":        pl.Utf8,     # "distance" | "resolution" | "tree_prune" | "native"
    "cut_params_json":   pl.Utf8,     # threshold, min_cluster_size, chosen resolution, ...
    "definitions_ref":   pl.Utf8,     # artifact containing candidate_definition rows
    "membership_ref":    pl.Utf8,     # artifact containing candidate_membership rows
    "created_at":        pl.Datetime("us", "UTC"),
    "created_by":        pl.Utf8,
})
# unique: candidate_set_id
```

`candidate_definition` bridges disposable integer labels to hierarchy nodes when a cut came from a tree. It remains valid with `hierarchy_node_id=NULL` for flat methods.

```python
pl.Schema({
    "candidate_set_id":   pl.Utf8,
    "candidate_id":       pl.Int32,
    "hierarchy_node_id":  pl.Utf8,
    "n_cells":            pl.Int64,
})
# unique: (candidate_set_id, candidate_id)
```

### Optional companion — `candidate_boundary_evidence`

Where CHOIR's RF-permutation distinguishability lives, and where a hand-run ablation or silhouette score lands. Pairwise, so a reviewer (or the HTML tool) can see *how separable* two candidates are before merging them. Populated when available; empty for plain fauxnograph.

```python
pl.Schema({
    "candidate_set_id":  pl.Utf8,
    "candidate_id_a":    pl.Int32,
    "candidate_id_b":    pl.Int32,
    "metric":            pl.Utf8,     # "rf_permutation_pvalue" | "silhouette" | "ablation_survives" | ...
    "value":             pl.Float64,
    "n_permutations":    pl.Int32,    # null if N/A
    "evidence_ref":      pl.Utf8,     # optional artifact (full RF report)
})
# unique: (candidate_set_id, candidate_id_a, candidate_id_b, metric)
```

---

## `kept_revision`

A durable, named, immutable computational state that ties the component artifacts together with lineage. Branch-local children reference their parent.

```python
pl.Schema({
    "revision_id":         pl.Utf8,     # immutable entity id; not the state hash
    "name":                pl.Utf8,     # durable human name (required — that's what "kept" means)
    "parent_revision_id":  pl.Utf8,     # lineage; null for a root
    "scope_id":            pl.Utf8,     # FK -> scope
    "feature_space_id":     pl.Utf8,     # FK -> feature_space; null if scope-only
    "clustering_representation_id": pl.Utf8, # FK -> representation; null if absent
    "visualization_representation_id": pl.Utf8, # FK -> representation; null if absent
    "candidate_set_id":    pl.Utf8,     # FK -> candidate_set; null if no clustering yet
    "state_hash":           pl.Utf8,     # hash of the pointer set, excluding name/audit fields
    "config_hash":         pl.Utf8,     # hash of fully-resolved config that produced this
    "software_versions":   pl.Utf8,     # JSON
    "created_by":          pl.Utf8,
    "created_at":          pl.Datetime("us", "UTC"),
    "notes":               pl.Utf8,     # optional freeform intent ("why I kept this")
})
# unique: revision_id ; unique: name within a study
```

### Supporting registry — `scope`

Retained cell membership + how it was obtained. The membership itself is a small artifact of `cell_id`s; this row is its identity.

```python
pl.Schema({
    "scope_id":            pl.Utf8,     # canonical manifest hash including derivation lineage
    "universe_id":         pl.Utf8,     # direct FK -> universe; inherited unchanged by children
    "parent_scope_id":     pl.Utf8,     # e.g. "cge" based_on "inh_core"; null for a root
    "members_ref":         pl.Utf8,     # artifact ref: sorted cell_id column
    "membership_hash":     pl.Utf8,     # hash of sorted ids only; enables reuse across derivations
    "n_cells":             pl.Int64,
    "derivation_text":     pl.Utf8,     # generated Polars code/text OR "ui_selection"; membership is authoritative
    "created_at":          pl.Datetime("us", "UTC"),
    "created_by":          pl.Utf8,
})
# unique: scope_id
```

Two scopes may have identical `membership_hash` values but different `scope_id` values because they were derived for different reasons. Computation caches may reuse membership-dependent work through `membership_hash`; the scope id preserves scientific lineage. A child scope must name the same `universe_id` as its parent, and every member must exist in that universe.

---

## `decision` — the append-only ledger

One row per curation action over a kept revision. This is the thing that today lives in code comments and in-place `_df` edits and then evaporates.

```python
pl.Schema({
    "decision_id":         pl.Utf8,     # stable id
    "review_branch":       pl.Utf8,     # linear by default; a second value = a fork
    "revision_id":         pl.Utf8,     # FK -> kept_revision (candidate context)
    "parent_decision_id":  pl.Utf8,     # previous decision in the branch; null for first
    "action":              pl.Utf8,     # "merge"|"split"|"exclude"|"mark_ambiguous"|"assign"|"assign_parent_only"|"rename"|"attach_local_revision"
    "target_kind":         pl.Utf8,     # "candidates" | "cells" | "taxon"
    "target_ids":          pl.List(pl.Int64),   # small candidate/taxon target lists; null for large cell sets
    "target_ref":          pl.Utf8,     # retained membership artifact for large or UI-selected cell targets
    "params_json":         pl.Utf8,     # canonical output of an action-specific validated payload model
    "taxon_id":            pl.Int64,    # for assign actions; null otherwise
    "rationale":           pl.Utf8,     # free-text WHY  ← the payoff
    "evidence_refs":       pl.List(pl.Utf8),   # refs to views/plots/boundary_evidence rows inspected (discloses circularity)
    "author":              pl.Utf8,
    "created_at":          pl.Datetime("us", "UTC"),
})
# unique: decision_id ; append-only (no update/delete)
```

Storage remains one table, but `params_json` is not an untyped escape hatch. Each action has a discriminated, strictly validated payload model. Exactly one of `target_ids` and `target_ref` is normally populated; large cell selections are never embedded as giant list values.

---

## Taxonomy, propagation, and assignment releases

### `taxonomy`

A taxonomy version describes the vocabulary and its simple parent tree. It does not identify a particular set of cell assignments.

```python
pl.Schema({
    "taxonomy_name":    pl.Utf8,
    "taxonomy_version": pl.Utf8,
    "taxon_id":         pl.Int64,    # stable integer, never reused
    "key":              pl.Utf8,
    "parent_id":        pl.Int64,
    "cluster_label":    pl.Utf8,
    "short_name":       pl.Utf8,
    "long_name":        pl.Utf8,
    "description":      pl.Utf8,
    "color":            pl.Utf8,
    "sort_order":       pl.Int32,
    "status":           pl.Utf8,     # "provisional" | "active" | "deprecated"
    "introduced_in":    pl.Utf8,
    "replaced_by":      pl.Int64,
})
# unique: (taxonomy_name, taxonomy_version, taxon_id)
# unique: (taxonomy_name, taxonomy_version, key)
```

### `propagation_run`

Label propagation is a first-class fitted computation rather than provenance duplicated across individual cell rows.

```python
pl.Schema({
    "propagation_run_id":      pl.Utf8,     # canonical manifest/content hash
    "fit_scope_id":            pl.Utf8,     # labeled/core cells used to fit
    "application_scope_id":    pl.Utf8,     # cells receiving predictions
    "input_feature_space_id":  pl.Utf8,
    "source_assignment_set_id": pl.Utf8,    # labels used as training targets
    "method":                  pl.Utf8,     # "knn" | "classifier" | ...
    "params_json":             pl.Utf8,
    "model_ref":               pl.Utf8,
    "quality_summary_ref":     pl.Utf8,
    "seed":                    pl.Int64,
    "recompute_deterministic": pl.Boolean,
    "created_at":              pl.Datetime("us", "UTC"),
    "created_by":              pl.Utf8,
})
# unique: propagation_run_id
```

### `assignment_set` and `assignment`

An assignment set is one immutable collection of cell-level claims against one taxonomy version. Revising assignments creates another set without changing the taxonomy vocabulary.

```python
pl.Schema({
    "assignment_set_id": pl.Utf8,     # immutable entity id
    "taxonomy_name":     pl.Utf8,
    "taxonomy_version":  pl.Utf8,
    "review_branch":     pl.Utf8,
    "decision_head_id":  pl.Utf8,     # exact branch state reduced into this set
    "assignments_ref":   pl.Utf8,
    "state_hash":        pl.Utf8,
    "created_at":        pl.Datetime("us", "UTC"),
    "created_by":        pl.Utf8,
})
# unique: assignment_set_id

pl.Schema({
    "assignment_set_id":       pl.Utf8,
    "cell_id":                 pl.Int64,
    "taxon_id":                pl.Int64,    # null for deliberately-unassigned
    "assignment_status":       pl.Utf8,     # "leaf"|"parent_only"|"ambiguous"|"unassigned"|"outside_taxonomy"
    "assignment_source":       pl.Utf8,     # "feature_based" | "propagated" | "manual"
    "source_revision_id":      pl.Utf8,
    "decision_id":             pl.Utf8,     # null for unreviewed classifier output
    "propagation_run_id":      pl.Utf8,     # propagated only
    "coverage_feature_space_id": pl.Utf8,   # defines the denominator/context for coverage
    "coverage":                pl.Float32,
    "confidence":              pl.Float32,
    "alternatives_json":       pl.Utf8,     # deferred candidate for a long companion table
    "created_at":              pl.Datetime("us", "UTC"),
})
# unique: (assignment_set_id, cell_id)
```

### `annotation_release`

```python
pl.Schema({
    "annotation_release_id": pl.Utf8,
    "name":                  pl.Utf8,
    "taxonomy_name":         pl.Utf8,
    "taxonomy_version":      pl.Utf8,
    "assignment_set_id":     pl.Utf8,
    "source_revision_id":    pl.Utf8,
    "manifest_ref":          pl.Utf8,
    "created_at":            pl.Datetime("us", "UTC"),
    "created_by":            pl.Utf8,
})
# unique: annotation_release_id; unique: name
```

---

## Pressure test — real cases from `cluster_minnie_subclasses.ipynb`

### A. Dropping the CGE "artifact cluster" (enum value 12, and the `label_inh_core == 3` null-out)

Today: `cctx.feat_table._df.loc[... == 3] = -1` and a skipped enum value with a code comment. Both are invisible, irreproducible mutations.

Here: candidate 12 **stays** in `candidate_membership` (immutable). One `decision` row captures the intent:

```text
action="exclude", target_kind="candidates", target_ids=[12],
rationale="Artifact cluster — diffuse, no coherent morphology.",
evidence_refs=[<coclustering_heatmap view>]
```

No assignment ever references taxon-from-candidate-12. The exclusion is now a queryable, attributable fact. ✅

### B. A propagated non-core cell vs a feature-based core cell

Today: `predict_labels_from_neighborhood` writes a column indistinguishable from a real assignment except by column name; `neighborhood_purity_score` writes another loose column.

Here: two rows, same `taxon_id`, different provenance:

```text
core:     assignment_source="feature_based", coverage≈1.0,
          coverage_feature_space_id=<core feature space>, decision_id=<...>
non-core: assignment_source="propagated", coverage=0.4, confidence=<purity>,
          coverage_feature_space_id=<application feature space>, propagation_run_id=<knn run>
```

Counts and plots can now trivially separate "assigned from its own features" from "assigned by neighborhood," and from genuinely missing — Scenario E, without a fake leaf. ✅

### C. CGE branch-local re-scale

Today: `add_mask(..., scaler_function=make_clipped_scaler(0.5, 99.5))` mutates the table; the global scaling is gone.

Here: the globally corrected `feature_space` is the parent of a second row with `fit_scope_id="cge"` and `transform="clipped_scaler"`. That local layer feeds a `representation` row (`method="pca"`, `fit_scope_id="cge"`); the CGE `kept_revision` points at both and has `parent_revision_id=<inh_core>`. The **global** correction, inh scaling, and CGE scaling retain distinct ids and coexist. Both embeddings are addressable at once — Scenario C. ✅

### D. Swapping the generator: fauxnograph → HDBSCAN → CHOIR

- **fauxnograph → HDBSCAN-on-embedding:** new `clustering_run` (`method="hdbscan"`, its `representation_id` = a modest-D embedding), plus its native `candidate_set`; noise points land as `candidate_id=NULL`, `membership_strength=probabilities_`. Every `decision`/`assignment` above is untouched. ✅
- **→ CHOIR:** `clustering_run.method="choir"`, `generator_ref` points at an external R process's returned parquet, and optional hierarchy refs preserve the pruned tree; `candidate_boundary_evidence` fills with `metric="rf_permutation_pvalue"` rows, which a merge `decision` cites in `evidence_refs`. Same `candidate_membership` shape downstream. ✅

### E. A spatially-aware method (spatial organization + feature similarity)

Today: no path — position is just more columns the clusterer never sees as spatial structure.

Here, two ways, both landing in the same downstream tables:

- **Banksy as a representation:** `representation` row with `method="banksy"`, `input_feature_space_id=<scaled morphology>`, `spatial_input_json={"position_role":"soma_xyz","graph":{"k":18},"weight":0.4}`. Its coords are then clustered by *any* generator — fauxnograph, HDBSCAN, CHOIR. The spatial mixing is a property of the representation, versioned and re-runnable.
- **SpaGCN as a generator:** a `scaled_passthrough` representation points at the scaled morphology feature space; `clustering_run(method="spagcn")` consumes that representation with `spatial_input_json` set and emits a native `candidate_set` plus membership.

Downstream `decision`/`assignment` are identical in both. Position entered through a declared role, so it is reproducible and swappable — not smuggled in as ad-hoc columns. ✅

**Seam test:** replace the generator with a stub emitting random `candidate_id`s (+ NULLs), an optional fake hierarchy, and fake `boundary_evidence` p-values. `kept_revision`, `decision`, `assignment`, and required views read the generic contracts — none reach for a method-specific matrix or threshold. A hierarchy view may consume the optional generic hierarchy without knowing its generator.

### F. A browser lasso becomes durable only through the application API

A lasso initially produces a transient Polars frame containing `cell_id`. Closing
the browser loses it, as intended. Choosing **keep as scope** materializes sorted
membership and creates a `scope` through the same API used by a notebook; choosing
**review selection** creates a retained target artifact and `decision`. Reopening
the study reproduces either durable result without browser state. ✅

### G. Correcting assignments without changing vocabulary

A reviewer marks one propagated cell ambiguous and manually corrects another.
Both actions append decisions and reduce to a new `assignment_set` referencing the
same `taxonomy_name` and `taxonomy_version`. No vocabulary version is minted merely
because cell-level claims changed. ✅

### H. A self-contained annotation release

In a fresh process, loading an `annotation_release` resolves its manifest,
taxonomy, assignments, exact decision head, quality summary, and replay recipe.
The release can be understood and validated without loading generator internals or
the notebook that produced it. ✅

### I. Thin Trajan consumption

A Trajan adapter joins released assignments onto Trajan's cell universe using the
canonical cell mapping and exposes taxonomy metadata. It imports release contracts
only; clustering generators, candidate artifacts, and review implementation remain
outside Trajan's dependency graph. ✅

### J. One expensive run, multiple cheap cuts

Two candidate sets with different cut parameters reference the same
`clustering_run_id`. They coexist as previews. Keeping either creates a revision
pointer without mutating the run, the other candidate set, or any content id. ✅

---

## Validation against the real release (`minnie_v1621_fully_typed.parquet`)

The current release has 228 columns. Its major naming families map naturally onto these tables, and the current code's conventions turn out to *be* normalized artifacts stored in denormalized form. These counts should eventually be generated by a migration audit rather than maintained by hand:

| Wide-table convention | Normalizes to | Count |
| --- | --- | --- |
| `_mask_<scope>` (bool) | `scope` membership | 18 |
| `_emb_<scope>__<name>_umap{0,1}` | visualization `representation` (per scope, already name-slotted) | 28 coordinate columns |
| `label_<scope>` (int) | `candidate_membership.candidate_id` | 14 |
| `name_<scope>` (str) | `assignment`, `source="feature_based"` | 14 |
| `*_core` (combined) / `*_nn_pred` | `assignment`, `feature_based` vs **`propagated`** | 11 core + 6 prediction columns |
| morphology / spine / nucleus / soma | raw `feature_block` + `feature_catalog` | migration audit required |
| `frac_*`, `size_*`, `L2IT`…`mId2` | connectivity evidence block | migration audit required |

Three findings worth keeping:

1. **The wide table is already close to a denormalized *view* over these tables.** The current `ClusteringContext` has independently evolved toward namespaced, per-scope artifacts (`_mask_`, `_emb_…__DEFAULT`, `label_`, `name_`) — no more global `umap0`/`umap1`. Migration should be largely mechanical, but a generated audit must list mapped and deliberately ignored columns before claiming complete coverage. The conventions exist; provenance does not.

2. **The `_core` vs `_nn_pred` split is already in the data**, exactly as `assignment.assignment_source` formalizes it. The schema types a convention the notebook already relies on.

3. **The one thing categorically absent from the release is *why*.** The file records outcomes (`label_inh_core`, `name_cge`) but nothing about the threshold chosen per scope, the excluded `label_inh_core == 3`, or the skipped enum-12 artifact cluster. You cannot reconstruct the curation from this parquet. That is precise, concrete proof that the `decision` ledger is the missing layer — not a nicety. Cruft like `root_id_y` (a stray merge column) is a bonus witness for immutable-inputs discipline.

## Resolved decisions

1. **Revision granularity → pointer set.** A `kept_revision` is a named immutable entity pointing at independently content-addressed `scope`, `feature_space`, clustering/visualization `representation`, and `candidate_set` components. Its `state_hash` covers the pointer set while its entity id and name do not affect component caches.

2. **`membership_strength` is method-relative.** Store a generator-provided value and interpret it through `clustering_run.method`; **never compare strengths across methods**. Fauxnograph leaves it null until a named per-cell strength is defined rather than inventing one from the matrix.

3. **Taxonomy and assignments version independently.** Vocabulary or hierarchy changes create a taxonomy version. Cell-claim changes create an assignment set. An annotation release pairs the two; there is no per-cell in-place assignment history.

4. **UI-only scope derivation → deferred.** When a `scope` has no programmatic predicate, `derivation_text="ui_selection"` plus the sorted `members_ref` artifact is the recipe. Text is documentation, not a new expression DSL; membership is authoritative.

5. **Feature transformations → immutable layer chain.** Each corrected/scaled `feature_space` has at most one parent layer and its own fit/application scopes. `representation.input_feature_space_id` points at the terminal layer it consumed. A global correction, local re-scale, and local PCA are distinct, reusable steps.

6. **Expensive run → cheap partitions.** `clustering_run` owns the method-specific compute artifact and optional generic hierarchy. Each cut or native partition is a separate `candidate_set`, so previews can compare cuts without recomputing consensus.

7. **Representations provide one generator-input seam.** Visualization embeddings and clustering representations share one contract. Direct feature-based generators consume a `scaled_passthrough` representation rather than overloading a foreign key.

8. **Keeping never mutates content.** Component registries do not acquire names when kept. The kept revision supplies the durable name. Stored artifacts are always digest-verified, even when stochastic recomputation is not byte-identical.
