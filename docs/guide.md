# User Guide

This guide is organized by task. It assumes you've skimmed the
[Tutorial](tutorial.md); here we explain the concepts behind each stage and the
options you'll reach for in real work. For exact signatures, see the
[Function Reference](reference/api.md).

## The core model

A few ideas run through everything:

- **One immutable study.** A `Study` is a single content-addressed store. Nothing
  is edited in place; new state is *appended* and the study advances to a new
  head commit. You can reopen it in a fresh process and re-validate it.
- **Content addressing.** Registering the same thing twice returns the same id —
  identity is derived from content, not from insertion order or timestamps. This
  is why re-running a notebook doesn't create duplicates.
- **Preview vs. keep.** Expensive artifacts (feature spaces, representations,
  clustering) are *previewed* — materialized and given a stable id — as often as
  you like. When you want a durable, named checkpoint you `keep` a **revision**
  that points at a chosen combination of them.
- **Revisions form a history.** Each revision names an optional `parent_revision`,
  so revisions build a tree you can walk (`history` view) and compare
  (`comparison` view).

```python
from cellpax import Study

study = Study.create("my-study", created_by="you")   # new study
study = Study.open("my-study")                        # existing, writable
study = Study.open("my-study", read_only=True)        # read-only
```

**Two ways to build revisions.** Every stage below has an explicit
`study.preview_*` method that you thread into `study.keep(...)`. The
`study.build()` **fluent builder** is sugar over exactly those calls — it holds
each artifact so you don't re-pass them, and chains kept revisions
automatically. It adds no new semantics; use whichever reads better.

```python
b = study.build()
b.scope(cell_ids).select(block)
b.keep("inputs")
b.feature_space(FeatureSpaceConfig.standard_scaler())
b.representation(RepresentationConfig.pca(n_components=2))
revision = b.keep("scaled + pca")          # parent chains automatically
```

## Registering the cell universe

The universe is the study's single source of truth for which cells exist. It is
registered once and cannot be replaced. Pass a bare list/Series of ids, or a
DataFrame when you have per-cell metadata columns to reference in
`semantic_roles`.

```python
study.register_universe(cell_ids)            # a list/Series of ids…

study.register_universe(                      # …or a DataFrame with metadata
    cells,                       # non-null unique Int64 cell_id + any columns
    semantic_roles={"position": ["soma_x", "soma_y"]},
    nullable_columns=(),         # columns allowed to contain nulls
    source_refs=(),              # provenance: refs this universe was derived from
)
study.universe()                 # the registered Universe record
```

Every scope, feature block, and assignment is checked to be a subset of the
universe, so downstream cell ids are always trustworthy.

## Feature blocks and catalogs

Feature values are registered as a **feature block** together with a **catalog**
describing each column. The catalog is required and content-checked: a block's id
is derived from its values *and* its semantic catalog. Describe columns with
`FeatureDefinition` objects and CellPax builds the catalog:

```python
from cellpax import FeatureDefinition

block = study.register_feature_block(
    values,
    [
        FeatureDefinition("morph_0", modality="morphology", family="shape"),
        FeatureDefinition("morph_1", modality="morphology", family="shape", units="um"),
    ],
)
study.feature_catalog()          # the full semantic catalog across all blocks
```

`column_name` defaults to `feature_id`; `units`/`description` are optional. You can
still pass a fully-built catalog DataFrame (columns `feature_id`, `column_name`,
`modality`, `family`, `units`, `description`, `raw_or_derived`) if you prefer, or
build one with `cellpax.feature_catalog(...)`. Register many blocks (morphology,
connectivity, transcriptomic) and select across them.

## Derived features and auto log-scaling

Nonlinear per-feature transforms (like log-scaling a skewed feature) are modeled
as **derived features**, not as a feature-space transform — the feature-space
transforms (`standard_scaler`, …) are fitted, whole-matrix estimators. A derived
block records the source block as lineage and marks every feature `derived`:

```python
# explicit: apply named ops (identity/log/log1p/log10/sqrt) per column
logged = study.derive_feature_block(
    morph, {"log_path": ("path_length", "log1p"), "sqrt_vol": ("branch_vol", "sqrt")}
)
```

`auto_log_features` log-scales the features that are *wide enough*, measured on a
scope and frozen so the choice is auditable and reproducible:

```python
pre = study.auto_log_features(
    morph,
    scope=None,              # cells whose distributions are measured (universe by default)
    metric="both",           # "skew" | "dynamic_range" | "both"
    skew_threshold=1.0,
    range_threshold=10.0,    # p99 / p1
    method="log1p",
)
```

It keeps the same feature ids (a full pass-through block: wide features logged,
others unchanged), skips features with negative values, and writes the decision
into each catalog `description` (e.g. `log1p (auto: skew=2.01, p99/p1=145.3)`), so
you can audit exactly what it did:

```python
study.feature_catalog().filter(pl.col("feature_block_id") == pre.feature_block_id)
```

Cluster on the derived (pretransformed) block; keep the original block to plot
actual values (see [feature tables](#plot-ready-tables-cluster-on-normalized-inspect-raw)).

## Selections and scopes

- A **feature selection** is an *ordered* set of `(feature_block_id, feature_id)`
  pairs — which features, in what order.
- A **scope** is an exact set of cells, optionally nested under a `parent` scope.

Both are previewed and content-addressed:

```python
# a feature block + optional feature-id list (omit the list to take all of them)
selection = study.preview_feature_selection(block, ["morph_0", "morph_1"])
selection = study.select_all_features(block)
scope = study.preview_scope(cell_ids, parent=None)
```

`derivation_text` is an optional short note on how the set was derived; when given
it participates in the id, so two selections with different rationales stay
distinct. You can still pass a `(feature_block_id, feature_id)` DataFrame to
`preview_feature_selection` when selecting across multiple blocks.

## Feature spaces

A **feature space** materializes selected values through a transform. The key idea
is the split between two scopes:

- `fit_scope` — the cells the transform *learns* from (e.g. a trusted core).
- `scope` — the cells the fitted transform is *applied* to.

```python
from cellpax import FeatureSpaceConfig

space = study.preview_feature_space(
    scope=scope,
    fit_scope=core_scope,        # defaults to scope when omitted
    feature_selection=selection,
    config=FeatureSpaceConfig.standard_scaler(),
)
```

Typed config factories cover every transform — `FeatureSpaceConfig.raw_join()`,
`.standard_scaler(with_mean=…, with_std=…)`, `.robust_scaler(quantile_range=…)`,
`.clipped_scaler(lower_percentile=…, upper_percentile=…)` — so parameters
autocomplete and typos fail immediately. (`FeatureSpaceConfig.resolve(transform=…,
params={…})` remains for dynamic use.) Feature spaces can be **chained** — pass a
`parent` feature space to re-scale a branch locally while keeping the global fit
intact.

**Missingness.** Every materialized space carries a missingness report
(`study.feature_space_missingness(space)`). Strict policies mean you can't
silently drop cells: a coverage reduction must be promoted to an explicit child
scope before the space will materialize.

## Representations

A **representation** produces coordinates for clustering or visualization from a
feature space, again with an explicit `fit_scope`.

```python
from cellpax import RepresentationConfig

rep = study.preview_representation(
    scope=scope, feature_space=space,        # fit_scope defaults to scope
    config=RepresentationConfig.pca(n_components=2),
)
```

Factories: `RepresentationConfig.pca(n_components=…)` and
`.scaled_passthrough()` (uses the feature space's own columns).

## Clustering and candidates

Clustering is deliberately two-phase so you pay for the expensive part once:

- A **clustering run** stores the expensive consensus computation.
- A **candidate set** is a cheap cut of a run into a flat, numbered partition.
  Many cuts can coexist over one run.

```python
from cellpax import ClusteringConfig, CandidateCutConfig

run = study.preview_clustering_run(
    scope=scope, representation=rep,
    config=ClusteringConfig.fauxnograph(n_neighbors=(30,)),
)
candidates = study.preview_candidate_set(
    clustering_run=run,
    config=CandidateCutConfig.distance(threshold=0.5),
)
```

Cut factories: `CandidateCutConfig.distance(threshold=…)`, `.resolution(…)`,
`.tree_prune(…)`, `.native(run_index=…)`. Inspect results without touching backend
internals:

```python
study.candidate_definitions(candidates)     # candidate_id, n_cells, ...
study.candidate_membership(candidates)       # per-cell candidate_id
study.candidate_hierarchy(run)               # optional generic hierarchy
study.compare_candidate_sets(a, b)           # contingency table
```

The built-in `fauxnograph` backend is a kNN/Leiden consensus method; it records
null `membership_strength` by design. It parallelizes over cores by default —
pass `ClusteringConfig.fauxnograph(n_jobs=1)` for a single-threaded run.

!!! tip "Bring your own clustering"
    Any object implementing the `CandidateGenerator` protocol
    (`compute` + `cut`) can be passed to `preview_clustering_run` /
    `preview_candidate_set`. The seam is designed to run out-of-process, so
    swapping backends changes nothing downstream.

## Reviewing: decisions

Review is an **append-only, linear ledger** per `review_branch`. Each decision
carries a required `rationale` and optional `evidence_refs`, and is validated
against the taxonomy and candidate set it targets.

```python
study.append_decision(
    revision=revision,
    review_branch="main",
    action="assign",
    target_kind="candidates",     # "candidates", "cells", or "taxon"
    target_ids=[4],
    taxon_id=10,
    rationale="Candidate 4 is a coherent type.",
)
```

Actions:

| Action | Meaning |
|---|---|
| `assign` | Label candidates or cells with a taxon (requires `taxon_id`). |
| `merge` | Combine two or more candidates. |
| `split` | Divide one candidate into parts (each part → a taxon). |
| `exclude` | Mark cells as outside the taxonomy (artifacts). |
| `mark_ambiguous` | Flag cells as ambiguous while retaining a taxon. |
| `assign_parent_only` | Assign only a parent taxon (coarse label). |
| `rename` | Rename a taxon target. |
| `attach_local_revision` | Attach a local revision to cell targets. |

Large cell targets and split parts are stored as canonical membership artifacts,
not inlined into decision rows. Helpers:

```python
study.decision_head("main")            # current branch head
study.decision_lineage(head)           # oldest → newest chain
study.get_decision(decision_id)
```

## Taxonomy

The taxonomy is your controlled, hierarchical vocabulary. Versions are immutable —
registering the same `(name, version)` with different rows is rejected. Build the
table from concise `TaxonDefinition` objects (input order sets display order;
omitted display/lifecycle fields get defaults):

```python
from cellpax import TaxonDefinition, taxonomy_table

study.register_taxonomy(
    taxonomy_table("cells", "1.0.0", [
        TaxonDefinition(1, "excitatory", label="Exc", color="#d62728"),
        TaxonDefinition(2, "inhibitory", label="Inh", color="#1f77b4"),
        TaxonDefinition(10, "basket", parent_id=2),   # nest with parent_id
    ])
)
study.get_taxonomy("cells", "1.0.0")
Taxon = study.taxonomy_enum("cells", "1.0.0")   # a rich IntEnum
Taxon.EXCITATORY.long_name, Taxon.EXCITATORY.color, Taxon.EXCITATORY.parent_id
```

Because assignments are versioned separately (below), you can keep correcting
labels for years without ever minting a spurious taxonomy version.

## Assignment sets

An **assignment set** reduces a decision head into one immutable row per cell,
paired with a taxonomy version. Each row records its **provenance**: the source
revision, the decision that set it, its `assignment_source`
(`feature_based` / `manual` / `propagated`), status
(`leaf` / `parent_only` / `ambiguous` / `outside_taxonomy` / `unassigned`),
coverage, and confidence.

```python
assignment_set = study.create_assignment_set_from_decisions(
    taxonomy_name="cells", taxonomy_version="1.0.0",
    review_branch="main", decision_head=head,
)
study.assignments(assignment_set)
```

To correct labels, append more decisions and reduce again against the previous
set as a base:

```python
revised = study.create_assignment_set_from_decisions(
    taxonomy_name="cells", taxonomy_version="1.0.0",
    review_branch="main", decision_head=new_head,
    base_assignment_set=assignment_set,     # carry prior state forward
)
```

The result is a *new* assignment set; the taxonomy version is untouched.

## Propagating labels to non-core cells

When you've confidently labeled a core set, you can propagate those labels to
other cells with a kNN model, keeping per-cell purity and coverage as first-class
output.

```python
from cellpax import PropagationConfig

run = study.preview_propagation_run(
    fit_scope=core_scope,
    application_scope=noncore_scope,
    feature_space=space,
    source_assignment_set=core_assignments,
    config=PropagationConfig.knn(n_neighbors=15),
    coverage=coverage_frame,        # optional per-cell coverage in [0, 1]
)
study.propagation_quality(run)      # predicted taxon, purity, coverage per cell

propagated = study.propagated_assignment_rows(run, source_revision=revision)
```

You then concatenate propagated rows with your feature-based/manual rows and call
`create_assignment_set(..., propagation_runs=[run])`. Propagated rows are tagged
`assignment_source="propagated"` and reference the run, so provenance stays
explicit.

## Views: reading a study

Call `study.view(name, ...)` for a read-only, fixed-schema Polars frame validated
against `VIEW_CONTRACTS`. (Each is also a plain function in `cellpax.views` — the
method just makes them discoverable from the study.) There is no view object or
query DSL.

| View | What it projects |
|---|---|
| `cells` | Per-cell scope + candidate + assignment. |
| `embedding` | 2-D coordinates + candidate/taxon display data. |
| `feature_profiles` | Per-group feature statistics (`all`/`candidate`/`taxon`). |
| `stability` | Candidate strength and boundary-evidence summaries. |
| `comparison` | Candidate contingency table for two revisions. |
| `taxonomy` | Taxonomy metadata with per-taxon assignment counts. |
| `history` | A revision's root-to-head ancestry. |
| `release_summary` | Release-readiness counts for a revision. |

```python
from cellpax.views import serialize_view
from cellpax.plotting import embedding_scatter

frame = study.view("embedding", revision, assignment_set=assignment_set)
embedding_scatter(frame, color_by="taxonomy")   # matplotlib; pass your own ax=
component = serialize_view("embedding", frame)   # deterministic self-describing JSON
```

`embedding_scatter` imports matplotlib lazily and accepts any matplotlib-compatible
`ax`. `serialize_view` never mutates the study.

## Plot-ready tables: cluster on normalized, inspect raw

A common pattern is to cluster on *normalized* features but plot the *actual*
values. `study.feature_table(revision)` returns one tidy per-cell frame — the
`cells`-view labels, any universe metadata columns, and the revision's feature
columns (readable `feature_id` names) — so every column can serve as a facet,
axis, or color:

```python
df = study.feature_table(revision, assignment_set=assignment_set)   # raw features
import seaborn as sns
sns.relplot(df, col="taxon_id", x="morph_0", y="morph_1", hue="candidate_id")
```

`normalized=False` (the default) gives the raw values the feature space was built
from; `normalized=True` gives the normalized values it clustered on. The label
columns are identical either way — only the feature columns change:

```python
raw  = study.feature_table(revision, assignment_set=assignment_set)
norm = study.feature_table(revision, normalized=True, assignment_set=assignment_set)
```

Pass `include_metadata=False` to drop universe columns. `feature_table` is a pure
read — it materializes nothing. For just the values of a single feature space
(previewed or kept), use `study.feature_values(space)`.

## Releases

A **release** binds a source revision and an assignment set into a checksummed,
self-documenting bundle that a consumer can use without any clustering, review, or
model-deserialization code.

```python
release = study.create_annotation_release(
    "cells-v1", source_revision=revision, assignment_set=assignment_set,
)
bundle = study.load_annotation_release("cells-v1")
```

A `ReleaseBundle` carries the resolved `taxonomy`, `assignments`, `decisions`,
`quality_summary`, `recipe`, `replay_script`, and `enum_binding`:

```python
bundle.quality_summary                 # one-row counts + confidence/coverage stats
Taxon = bundle.taxonomy_enum()         # rich enum, rebuilt from the release
exec(bundle.enum_binding, ns := {})    # or import the generated source
```

Validate a release — re-deriving every product and checking checksums — without
loading any model payloads:

```python
study.validate_annotation_release(release)
```

### Handing off to a consumer

The Trajan adapter (`cellpax.adapters.trajan`) is a thin, release-only seam:

```python
from cellpax.adapters.trajan import (
    annotation_frame, decorate_cells,
    add_to_connectivity_table, add_to_synapse_table,
)

annotation_frame(bundle)                    # canonical per-cell annotation table
decorate_cells(consumer_frame, bundle)      # left-join labels onto any Polars frame
add_to_connectivity_table(table, bundle)    # register with a Trajan table
```

## Validation and the trust model

`Study.validate()` re-checks schemas, referential integrity, content-addressed
ids, and every slice's artifacts — **without deserializing model payloads by
default**. This lets you validate a study you don't fully trust.

```python
study.validate()                    # safe, no pickle deserialization
study.validate(trusted_models=True) # additionally replays fitted state / predictions
```

Use `trusted_models=True`, and the optional generator-specific label/metadata
inspection helpers, only on a study you authored or otherwise trust — they load
pickled estimator and generator state.

## Reproducibility

- Content addressing means identical inputs produce identical ids, so pipelines
  are idempotent.
- Revisions and releases record software versions and resolved configs, and a
  release ships a resolved recipe and replay script.
- A study reopened in a fresh process re-validates and reproduces the same views
  and releases.

```python
reopened = Study.open("my-study", read_only=True)
reopened.validate()
```
