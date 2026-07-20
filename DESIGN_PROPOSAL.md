# Cell Taxonomy Workbench: Design Proposal (v2)

**Status:** Design settled enough to build a first vertical slice after the core contracts below are implemented
**Scope:** A new library, informed by `dendritic_feature_clustering` and Trajan, not a direct extension of either prototype
**Primary use case:** Interactive, reproducible, *self-documenting* construction of cell-level feature spaces, cluster structure, and curated hierarchical cell labels
**Companion:** `SCHEMA_DRAFT.md` holds the version 1 concrete table contracts and their validation against the real release. This document is the narrative, the principles, and the plan; the schema contract is the source of truth for column definitions. Phase 0 storage and identity decisions are recorded in `docs/decisions/0001-core-contracts-v1.md`.

## What changed since v1

The first proposal was a conceptual sketch. This version has been pressure-tested against the actual prototype — the code in `src/dendritic_feature_clustering/`, the current 249-cell `cluster_minnie_subclasses.ipynb` (217 code cells), and the released `minnie_v1621_fully_typed.parquet`. That grounding changed the emphasis in five ways:

1. **The primary value is self-documentation, not release engineering.** The strongest reason to build this is to make *why* a decision was made durable and replayable. Downstream consumption (Trajan, crosswalks, enum exports) is secondary and can be built when a consumer actually needs it.
2. **Clustering is a swappable seam, not a thing to rebuild.** The existing `fauxnograph` coclustering works well enough. What matters is that the *interface* admits VAE embeddings, CHOIR, HDBSCAN, and spatial-transcriptomics methods without touching anything downstream.
3. **Views are a contract, not a framework.** A "view" is a documented function returning a typed Polars frame with a declared schema — not a class hierarchy. The same frame feeds a matplotlib call and the future HTML tool.
4. **Label propagation is a first-class production step**, because feature coverage across the dataset is patchy. Assignments must record whether a cell was labeled from its own features or propagated from a neighborhood.
5. **The plan is substrate-first and gated, not seven linear phases.** Prove the self-documenting substrate on the real minnie notebook before committing to release products or an HTML application.

The second pressure test exposed five additional requirements now incorporated below: transformation stages can have different fit scopes; an expensive clustering run and a cheap cut need separate identities; optional candidate hierarchies must survive the clustering seam; keeping cannot mutate content-addressed components; and taxonomy vocabulary versions must be independent of cell-assignment releases.

The connectivity/targeting analysis in the prototype is **evidence** consumed by curation, not an input to clustering, so the boundary with Trajan is cleanly unidirectional. It is out of scope here.

## Executive summary

The library is an interactive, revisioned workbench for assembling cell-level features, exploring cluster structure, and curating biological taxonomies. It preserves the prototype's productive loop — choose cells and features; normalize and embed; generate and inspect candidate clusters; mask to an interesting scope; re-analyze locally; assign stable labels; publish — while hardening the parts that currently make it fragile.

Hand tuning is a first-class feature. Clustering metrics and CHOIR-like distinguishability tests suggest structure; they do not replace biological judgment. The reproducibility mechanism is therefore not an automated pipeline. It is an **immutable history of previews, kept revisions, and explicit curation decisions with rationale** that can be replayed as a recipe and read as a lab notebook.

The core architectural boundary:

> Use Polars to express *what* a data operation is. Use the workbench to record *why* it was done, which revision it belongs to, whether it was kept, and how it contributes to a curated taxonomy.

## Motivation: what the prototype gets right, and what it loses

The prototype's analysis pattern is scientifically productive — the recursive descent through a cell-type tree (mask from parent → recluster locally → cut → name → descend) is repeated ~15 times in the minnie notebook and produces real, published taxonomies. The design keeps that loop intact.

What it loses, verified against the code:

- **Candidate IDs masquerade as biological labels.** `assign_labels` feeds clustering output straight into a `label_<scope>` column, and a hand-written `IntEnum` then assigns meaning *by cluster integer* (`class L23Subclasses(IntEnum): L2a = 1; L2b = 0; ...`). Because `compute_coclustering` runs with `force=True` over 20–50 stochastic Leiden runs, those integers can permute on re-run and silently remap every name. This is the single most dangerous fragility.
- **The notebook is the database of record.** `FeatureTable` is a mutable pandas god-object; a kernel restart loses authoritative state; running cells out of order changes downstream products.
- **Curation is invisible and irreproducible.** Real decisions happen by reaching into private state — `feat_table._df.loc[label == 3] = -1` ("null out nonsense cluster"), a skipped enum value 12 ("artifact cluster"), hand-typed cell-ID lists. These survive only as code comments and vanish. The released parquet records outcomes (`label_*`, `name_*`) but *nothing about why* — you cannot reconstruct the curation from the file.
- **Dense co-clustering caps scaling and leaks abstraction.** `SimilarityMatrix` is a dense cell×cell matrix. It is both a scaling limit and a coupling that would block swapping in CHOIR, VAE+Leiden, or spatial methods, none of which produce such a matrix.
- **Pandas `.query()` DSL and permissive `**kwargs`** weaken stability; embeddings and scalers lack durable identity and explicit fit scope.

The good news from validation: the current `ClusteringContext` has independently evolved toward namespaced, per-scope artifacts (`_mask_<scope>`, `_emb_<scope>__DEFAULT`, `label_<scope>`, `name_<scope>`, `*_core` vs `*_nn_pred`). The release parquet is effectively a *denormalized view* over the very tables this design proposes. Migration is therefore mechanical: give those conventions durable identity, immutability, and a decision ledger.

## Design principles

1. **Keep the biologist in the loop.** Merges, splits, exclusions, re-embeddings, naming, and ambiguity calls are valid scientific actions, recorded with rationale — never hidden or prohibited. No scalar metric is treated as objective clustering truth; metrics are evidence.
2. **Make exploration safe.** Parameter changes create alternatives, not mutations. Distinguish **preview** (ephemeral, content-addressed), **keep** (immutable named revision), **review** (append-only curation over a revision), **release** (frozen product).
3. **Start from an authoritative universe.** Every analysis aligns through stable cell identifiers. Cells never silently disappear because a feature is missing; **missingness and coverage are explicit properties of a scope** — which is exactly why label propagation exists.
4. **Separate computation from interpretation.** "Cells grouped together under this feature space and algorithm" and "the researcher considers these cells one biological category" are different claims with different lifetimes.
5. **Prefer existing machinery.** Polars is the canonical table layer; NumPy/SciPy for arrays and sparse matrices; scikit-learn-compatible transformers; established PCA/UMAP/neighbors/Leiden/HDBSCAN implementations; DataFolio for artifact storage and lineage. The workbench adds domain semantics and revision control, not replacements.
6. **Produce self-documenting products.** A kept revision or a release explains itself — identifiers, names, taxonomy, assignment state, feature provenance, **decisions with rationale**, and a replayable recipe travel together.
7. **Support branch-local analysis.** Subsetting and local refitting are central, not edge cases. A child inherits selected configuration from its parent while explicitly refitting normalization, neighbors, embeddings, and clustering within its own cell scope.

## Non-goals

The first versions do not provide: a generic ontology or inference engine; a new dataframe abstraction; a workflow scheduler; a general or distributed compute platform; a universal clustering-quality score; automatic biological naming of clusters; a **view/plotting framework** (views are typed frames, not a class system); connectivity analysis (Trajan's concern); or a mandatory HTML application before the programmatic contracts are stable.

## Core conceptual model

The concrete Polars schemas for every table below are in `SCHEMA_DRAFT.md`. At a glance, the durable tables are:

| Table | Role |
| --- | --- |
| `universe` | authoritative cell coordinate system; canonical `cell_id`, aliases (e.g. `root_id`), stable properties, and **semantic roles** for important columns (including a `position` role for spatial methods) |
| `feature_block` + `feature_catalog` | immutable cell-keyed inputs and their semantic feature descriptions |
| `feature_selection` | an exact, ordered list of feature ids selected with ordinary Polars operations |
| `scope` | retained cell membership + lineage (`parent_scope_id`); the analysis-scope tree |
| `feature_space` | one immutable raw/corrected/scaled feature layer; layers form a chain so each transform has its own `fit_scope_id` |
| `representation` | clustering-input coordinates (PCA/VAE/…); **seam 1**, algorithm-agnostic, with a spatial-input hook |
| `clustering_run` | expensive generator output or basis, such as a consensus/linkage artifact, cached independently of any cut |
| `candidate_set` + `candidate_definition` + `candidate_membership` | one partition derived from a clustering run, its optional hierarchy-node mapping, and cell membership; **seam 2**, algorithm-agnostic; `candidate_id` is revision-local and disposable |
| `candidate_hierarchy` | optional generic nodes/edges/leaf membership for methods that produce a real hierarchy |
| `candidate_boundary_evidence` | optional pairwise distinguishability (CHOIR RF-permutation p-values, silhouette, ablations) |
| `kept_revision` | a durable, named **pointer set** over the components above, with `parent_revision_id` lineage |
| `decision` | append-only curation ledger: merge/split/exclude/mark-ambiguous/assign, each with **rationale** and evidence references |
| `taxonomy` | a versioned vocabulary and hierarchy of durable named biological categories |
| `assignment_set` + `assignment` | a versioned set of coverage-aware claims that cells belong to taxa |
| `annotation_release` | a published bundle pairing one taxonomy version with one assignment set |

Four concepts stay strictly distinct — conflating them is the prototype's core bug:

| Concept | Meaning | Lifetime |
| --- | --- | --- |
| Scope / mask | cells included in a computation | revision-specific |
| Candidate cluster | output of a clustering operation | ephemeral |
| Taxon | durable named biological category | stable across releases |
| Assignment | versioned claim that a cell belongs to a taxon | revisable |

Promotion from candidate to taxon is *always* an explicit `decision`. Released taxonomies keep the useful `IntEnum` interface as a typed release binding while also exporting a language-neutral table (stable integer ids never recycled; deprecations and crosswalks explicit) — as specified in `SCHEMA_DRAFT.md` and the taxon schema.

Taxonomy vocabulary and cell assignments have independent identities. Correcting a propagated cell or revising a boundary creates a new `assignment_set`, not a new taxonomy version. An `annotation_release` explicitly pairs the two. This prevents ordinary annotation updates from pretending that the biological vocabulary changed.

### Assignments are coverage-aware

Because feature coverage is patchy, an `assignment` records **how it was earned**: `assignment_source` is `feature_based`, `propagated`, or `manual`. Propagation is represented by a first-class `propagation_run` with fit scope, application scope, input feature space, model artifact, resolved parameters, and quality summary; propagated assignment rows reference that run and carry cell-level confidence. Coverage is explicitly relative to the feature space used for the assignment, rather than an uninterpretable global fraction. Parent-only, ambiguous, deliberately-unassigned, and outside-taxonomy are all valid states — no fake leaf assignment is ever required. This makes the prototype's `*_core` (feature-based) vs `*_nn_pred` (propagated) distinction first-class rather than a naming convention.

### Views are contracts, not a framework

A view is a revision-aware read model: a documented function `f(revision) -> pl.DataFrame` with a declared schema, validated by the same contract tests as every other table. matplotlib functions and the future HTML tool both consume the frame; there is no `View` class hierarchy and no new query language. Views are read-only; a browser lasso is a transient Polars frame of cell IDs until an explicit keep/review command routes it through the ledger. Most views are recomputed on demand; they are materialized only when a release or frozen figure needs exact reproducibility.

## Architecture invariants

Three rules are enforced in contract tests, not left to convention:

1. **No method-specific matrix or threshold leak.** Required downstream operations never reference a coclustering matrix or cut threshold. They depend on `candidate_membership`, representation coordinates, and optional generic hierarchy/boundary-evidence contracts. The `fauxnograph` dense `SimilarityMatrix` remains an internal generator artifact.
2. **Append-only curation.** `decision` rows are never updated or deleted. Curation history is complete and attributable.
3. **Immutable content, separate naming.** Content-addressed components are never renamed when kept. Keeping creates a named revision that points at them; it does not mutate them. Artifact digests always verify stored bytes, while a separate recomputation flag records whether executing the same recipe is expected to reproduce identical output.

**Seam test.** Replace the clustering generator with a stub emitting random `candidate_id`s (with NULLs for noise), an optional fake hierarchy, and fake boundary-evidence p-values. `kept_revision`, `decision`, `assignment`, and every view still run unchanged — because none of them reach for a method-specific matrix or threshold. A generic hierarchy-aware view may use the optional hierarchy contract without knowing which generator produced it.

## Clustering and representation as swappable seams

The library ships with the existing `fauxnograph` consensus clustering, wrapped — not rebuilt — behind the preview/keep interface. The **expensive stochastic consensus compute is a content-addressed `clustering_run`; each threshold/min-cluster-size cut is a cheap, reproducible `candidate_set` derived from that run.** Multiple cuts can coexist as previews and reuse the same expensive artifact. Keeping one creates a revision pointing at that candidate set; the cut itself is computational provenance, not a curation decision. This is the natural cache boundary the notebook already implies.

Methods that produce meaningful trees may also emit the optional generic `candidate_hierarchy` tables. Methods that only produce a flat partition leave them absent. This preserves CHOIR-like or agglomerative structure for inspection without requiring any downstream consumer to understand a coclustering matrix, R object, or method-specific tree representation.

Because the seams are algorithm-agnostic artifact contracts, known future directions slot in without downstream change (see `SCHEMA_DRAFT.md` pressure-test cases D and E):

- **VAE embeddings** — a `representation` swap; trained weights are a versioned fitted-state artifact. Stored artifacts are always integrity-checked, while `recompute_deterministic=False` means replay verifies lineage and structural contracts rather than promising byte-identical recomputation (same principle for UMAP).
- **CHOIR** (RF + permutation prune) — a `clustering_run` generator producing a candidate set, optional hierarchy, and distinguishability evidence. Since CHOIR ships as an R/Seurat package, the seam allows an **external-process generator that returns parquet**; reimplementing the method in Python (sklearn RandomForest + permutation test over a tree) is an alternative to be decided later.
- **HDBSCAN/DBSCAN** on a modest-D embedding — a clustering run with a native candidate set; noise points are `candidate_id=NULL`, `membership_strength=probabilities_`.
- **Spatial-transcriptomics methods** (organization *plus* feature similarity) — either a `representation` (augmentation-style: Banksy, UTAG) or a joint-model clustering run (SpaGCN, BayesSpace), both consuming position through a declared `spatial_input_json` and the universe's `position` role, never ad-hoc columns.

CHOIR-like pruning is an advisory strategy, kept modular so simpler clustering stays usable, and it never auto-decides a taxonomy — it pre-populates evidence a human ratifies as a decision.

## Implementation plan

Substrate-first and gated. Do not build clustering, releases, or UI before the self-documenting substrate is proven on the real minnie notebook.

**Slice 0 — contracts in the existing library scaffold.** Preserve the current `cellpax` `uv`/poe/ruff/pytest/docs/release template and use the DataFolio 2.0 public API as the native study-storage substrate, resolving the currently unpublished API from the sibling `../datafolio` checkout during development. Use direct Polars items, namespaced refs, lineage, atomic batches, and validation; do not add a DataFolio 1.x compatibility layer. Implement the accepted `pl.Schema` definitions and contract-test stubs for the substrate tables in `SCHEMA_DRAFT.md`, including the three invariants as importable checks. Add synthetic stubs for later tables only where needed to exercise the seam. *Exit:* the seam-test stub passes end to end on synthetic data.

**Slice 1 — stable study substrate (the self-documentation payoff).** `Study` over a DataFolio 2 folio, namespaced DataFolio item refs, immutable CellPax commit items, universe and feature-block registration/validation, `scope` and ordered `feature_selection` membership artifacts, strict typed config, separate entity/content/config identities, preview/keep, and reopen/replay in a fresh process. No fitted transforms or clustering yet. *Exit:* Polars cell and feature selections can be kept, reopened, validated against their universe/catalog, and traced through DataFolio `inputs` (Scenario A).

**Slice 2 — feature spaces and representations.** Chained raw/corrected/scaled feature spaces with source/value references; scikit-learn-compatible fitted transformers with an explicit fit scope per stage; missingness reports; versioned representations and visualization embeddings. *Exit:* the global preprocessing followed by the CGE branch-local re-scale replays as distinct, reusable stages with global and local fits coexisting (Scenario C).

> **Gate.** With Slices 0–2 proven on the real notebook, decide continue/stop before investing further. The remaining slices are lower personal-value or defer-until-needed.

**Slice 3 — candidate clustering behind the seam.** Wrap `fauxnograph`; implement `clustering_run` versus derived `candidate_set`; optional generic hierarchy; candidate definitions, membership, and boundary evidence; revision comparison. Keep the generator interface external-process-capable. For fauxnograph, leave `membership_strength` null until a named per-cell strength metric is explicitly defined. *Exit:* multiple cuts reuse one run, two previews coexist, one is kept, and swapping in a stub generator changes nothing downstream (Scenarios B, J).

**Slice 4 — review, taxonomy, assignments.** Append-only `decision` ledger with rationale and evidence refs; action-specific validated decision payloads; merge/split/exclude/mark-ambiguous/assign; small hierarchical taxonomy + rich `IntEnum` bindings; independent assignment sets; coverage-aware assignments; and first-class propagation runs with purity output. Large cell targets are retained membership artifacts rather than giant lists inside decision rows. *Exit:* the nulled `label_inh_core==3`, the skipped enum-12 exclusion, and a propagated non-core cell are all represented and auditable without changing the taxonomy vocabulary (Scenarios D, E, G).

**Slice 5 — view contracts and static reporting.** A small set of documented `revision -> typed frame` projections (cells, embedding, feature profiles, stability, comparison, taxonomy, history, release summary) plus thin matplotlib adapters. Accept that the long tail of bespoke paper figures stays as notebook code. *Exit:* one embedding view drives a notebook plot and a serialized report component.

**Slice 6 — release products (build when a consumer needs it).** Release validation, an `annotation_release` pairing a taxonomy version with an assignment set, self-documenting manifests, generated enum bindings, resolved recipe + replay script, quality summary, and a thin Trajan annotation adapter. *Exit:* Scenarios H and I work without importing clustering machinery into Trajan.

**Slice 7 — local interactive HTML workbench.** Linked read-only views over the view contracts; transient shared selections; preview/revision comparison; decision forms with rationale; explicit keep/review commands; history and release inspection. The server calls the same application API as notebooks. *Exit:* Scenario F works, and every durable UI action replays without the browser.

The evaluation scenarios A–J in `SCHEMA_DRAFT.md` are the acceptance tests for API
and storage proposals; a design that makes them awkward is introducing the wrong
abstraction.

## Migration from the prototype

Incremental and evidence-driven; the column mapping in `SCHEMA_DRAFT.md` shows it is mostly mechanical:

1. Register the minnie cell universe (canonical `cell_id`, `root_id` alias, `position` role) from `minnie_v1621_fully_typed.parquet`.
2. Register one ossify-derived feature block, build the feature catalog, and retain an ordered feature selection.
3. Explode `_mask_<scope>` → `scope`, `_emb_<scope>__…` → `representation`/embedding, `label_<scope>` → `candidate_membership`, `name_*`/`*_core`/`*_nn_pred` → `assignment` (with `feature_based` vs `propagated`).
4. Reproduce one global embedding + clustering as a kept revision, and one mask-and-recluster branch with explicit local fit scope.
5. Express the known manual curations (the `label==3` null-out, the enum-12 exclusion, a merge) as `decision` rows with rationale — the layer the release parquet cannot currently express.
6. Convert one `IntEnum` into a released taxonomy table + rich enum binding.
7. Produce a small release and consume its assignments in a minimal Trajan example.
8. Only then migrate plotting and build the HTML tool.

Compatibility adapters may read prototype artifacts, but new public APIs must not preserve accidental behavior (direct `_df` mutation, unnamed embeddings, permissive `**kwargs`, stray `root_id_y` merge columns).

## Testing and stability strategy

- **Contract tests** on every materialized table: required columns and dtypes; unique/non-null keys; `cell_id ⊆ declared scope`; stable ordering where semantically significant; referential integrity between assignments and taxonomies; feature-space parent-chain integrity; candidate-set derivation from a clustering run; schema-version compatibility; and the three architecture invariants as importable checks.
- **Determinism tests:** identical recipe/state hashes across processes for deterministic components; explicit seeds and recorded implementation versions for stochastic ones; integrity hashes for every stored artifact; and structural-invariant checks where recomputation cannot be bitwise identical (UMAP, VAE), with the limitation recorded rather than hidden.
- **Replay tests** on small *synthetic* studies exercising global/local normalization, alternative previews, kept revisions, merge/split/exclude, ambiguous and parent-only and propagated assignments, release generation, and loading a release in Trajan. Synthetic data only — tests must not encode conclusions from the real dataset.
- **Mutation and cache safety:** no authoritative mutable dataframes; a transformation produces a new artifact or stays an ephemeral Polars computation; caches keyed by immutable inputs + resolved params, never invalidated by convention.
- **Configuration tests:** unknown/misspelled keys fail before computation; default resolution, schema migration, environment-independent hashing, informative preflight (including estimates of obviously quadratic materializations).

## Resolved decisions

Carried from the design conversation (full context in `SCHEMA_DRAFT.md`):

1. **Revision granularity = pointer set.** A `kept_revision` gives a durable name to immutable `scope`/`feature_space`/representation/`candidate_set` pointers; components remain separately content-addressed, cacheable, and reusable.
2. **`membership_strength` is method-relative** — stored only when the generator defines it, interpreted through `clustering_run.method`, and never compared across methods. Fauxnograph initially leaves it null.
3. **Taxonomy and assignment version independently.** A revised cell claim creates a new `assignment_set`; a new taxonomy version is required only when the vocabulary or hierarchy changes. An `annotation_release` pairs them.
4. **UI-only scope derivation deferred** until the interactive layer exists; for now `derivation_text="ui_selection"` + the sorted membership artifact is the recipe. The text is documentation, not a custom expression DSL.
5. **Feature transformations form an immutable layer chain.** Each `feature_space` has at most one parent feature space, its own fit/application scopes, fitted state, and optional materialized values. This represents global correction followed by local scaling without hiding different fit scopes inside one pipeline.
6. **Consensus run and partition are separate.** `clustering_run` owns the expensive method-specific basis; a `candidate_set` owns one cheap partition/cut derived from it.
7. **Keeping never renames content.** A revision is a named immutable pointer set with its own entity id and state hash. Component content ids and stored-artifact digests do not include names or audit fields.
8. **Representations and visualization embeddings share one contract.** A kept revision points separately to clustering and visualization representation ids; a `scaled_passthrough` representation preserves a single candidate-generator input seam.
9. **Study history uses DataFolio 2 transactions.** DataFolio's atomic `items.json` is the only storage locator. Folio metadata points to an immutable CellPax commit item, which points to complete immutable registry items; one `folio.batch()` publishes payloads, registries, commit, and head together.
10. **CellPax uses the DataFolio 2 public API directly.** Schema refs are namespaced DataFolio item names; `add`/`add_file`/`add_model`, `get`/`scan_table`, `inputs`, `batch`, `item_info`, and `validate` provide storage, lineage, transactions, resolution, and exact-byte integrity. CellPax SHA-256 ids remain the separate scientific identity layer.
11. **Scopes are directly universe-anchored.** Every scope stores `universe_id`; child scopes retain the same universe and validate all members against it.
12. **Previews are retained by default in v1.** They are immutable but unrooted until kept. Automatic garbage collection is deferred; any future collector is explicit and dry-run by default.
13. **Package name is `cellpax`.** The existing library scaffold is intentional and is extended in place.

## Deferred / open questions

- Representation of probabilistic/multi-label assignments alongside a primary released taxon.
- Whether taxonomies are authored as enums→tables, tables→enums, or both under one validation step.
- Which release changes require a taxonomy major/minor/patch bump.
- The smallest generic candidate-hierarchy schema that supports agglomerative and CHOIR-like trees without pretending non-nested resolution sweeps are trees.
- Whether component aliases are useful enough to warrant a separate alias registry after v1; revisions alone provide durable names initially.

## Suggested initial package structure

A responsibility sketch, not a commitment to names:

```text
<package>/
  universe.py        universe contract, semantic roles, validation
  artifacts.py       DataFolio references, manifests, content hashing
  features.py        feature blocks, catalog, selections, chained feature spaces
  spaces.py          representations and embeddings (seam 1)
  scopes.py          retained cell membership and lineage
  candidates.py      clustering runs, partitions, hierarchy, membership, evidence (seam 2)
  generators/        fauxnograph (wrapped); adapters for external/CHOIR/spatial
  review.py          append-only decision ledger
  taxonomy.py        taxa, assignment sets, propagation runs, enums, crosswalks
  views.py           documented revision -> typed-frame projections
  plotting.py        thin matplotlib adapters over view frames
  release.py         frozen products and validation
  recipes.py         resolved recipes and replay generation
  adapters/
    dfc.py           prototype import compatibility
    trajan.py        released cell-annotation export
```

## Success criteria

The system is succeeding when: exploratory parameter changes are cheap and non-destructive; a notebook restarts without losing authoritative state; raw feature blocks, each fitted transformation stage, representations, and embeddings have unambiguous identities and explicit fit scopes; multiple cheap candidate cuts reuse expensive clustering runs; optional cluster hierarchies remain inspectable without leaking method-specific objects; branch-local re-analysis is natural and reproducible; **candidate IDs never masquerade as biological labels**; **every manual decision is easy to make and impossible to lose, with its rationale attached**; taxonomy vocabularies and assignment releases version independently; views support notebooks, plots, and an HTML tool without duplicating Polars; releases replay or are independently understood from their stored products; and Trajan consumes the result as a cell-universe decoration without coupling the packages' internals.
