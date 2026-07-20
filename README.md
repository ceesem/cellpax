# cellpax

Cell feature universe analysis for connectomics. CellPax keeps cell universes,
feature inputs, selections, scopes, and named revisions as immutable, validated
scientific state in a DataFolio 2 folio.

## Setup

This project uses `uv` for dependency management and `poe` for task running.

```bash
# Install dependencies. During DataFolio 2 development, uv uses the editable
# sibling checkout at ../datafolio.
uv sync

# Launch Jupyter Lab
poe lab
```

## Stable study substrate

```python
import polars as pl

from cellpax import Study
from cellpax import (
    CandidateCutConfig,
    ClusteringConfig,
    FeatureSpaceConfig,
    RepresentationConfig,
)

study = Study.create("analysis.cellpax", created_by="casey")
universe = study.register_universe(
    cells,  # cell_id: Int64 plus aliases/properties
    semantic_roles={"position": ["soma_x", "soma_y", "soma_z"]},
)
block = study.register_feature_block(values, catalog)

selection = study.preview_feature_selection(
    selected_features,  # ordered feature_block_id, feature_id pairs
    derivation_text="morphology feature set",
)
scope = study.preview_scope(
    pl.DataFrame({"cell_id": retained_ids}),
    derivation_text="inhibitory core",
)
revision = study.keep(
    "inh-core", scope=scope, feature_selection=selection
)

space = study.preview_feature_space(
    scope=scope,
    fit_scope=scope,
    feature_selection=selection,
    config=FeatureSpaceConfig.resolve(
        transform="robust_scaler", missing_policy="median"
    ),
)
embedding = study.preview_representation(
    scope=scope,
    fit_scope=scope,
    feature_space=space,
    config=RepresentationConfig.resolve(method="pca", n_components=2, seed=7),
)
revision = study.keep(
    "inh-pca",
    scope=scope,
    feature_space=space,
    visualization_representation=embedding,
    parent_revision=revision,
)

run = study.preview_clustering_run(
    scope=scope,
    representation=embedding,
    config=ClusteringConfig.resolve(
        method="fauxnograph",
        compute_params={"n_neighbors": [30], "n_times": 25},
        seed=7,
    ),
)
candidates = study.preview_candidate_set(
    clustering_run=run,
    config=CandidateCutConfig.resolve(
        cut_method="distance",
        cut_params={"distance_threshold": 0.35, "min_cluster_size": 10},
    ),
)
revision = study.keep(
    "inh-candidates",
    scope=scope,
    feature_space=space,
    clustering_representation=embedding,
    candidate_set=candidates,
    parent_revision=revision,
)

reopened = Study.open("analysis.cellpax", read_only=True)
reopened.validate()
```

Previews materialize immutable membership items but do not enter the study's
registry history until `keep()` succeeds. The keep publishes registry snapshots,
an immutable content-addressed commit, and the metadata head in one DataFolio 2
batch.

In the Slice 1 schema, a kept feature selection is rooted in its registry and
represented in the revision's configuration hash. The directly queryable
revision → feature-selection relationship arrives with feature spaces in Slice 2.

Feature spaces form a one-stage-per-row immutable chain. Each stage records its
application and fit scopes independently, stores sklearn-compatible fitted state,
materializes ordered values, and retains a per-feature missingness report through
DataFolio lineage. Representations use the same fit/application split for PCA or a
scaled passthrough. `replay_feature_space()` and `replay_representation()` reapply
study-owned fitted state as an integrity check; they do not refit the estimator or
claim recipe-level fit reproducibility.

A feature space must cover its declared application scope exactly. If
`missing_policy="drop"` would remove cells, preview raises
`ScopeReductionRequiredError` with `retained_members`; create an explicit child
scope from those members and retry. Cells therefore never disappear behind an
unchanged `scope_id`.

Slice 3 keeps expensive consensus computation separate from cheap cuts. CellPax
owns its fauxnograph implementation in `cellpax.generators.fauxnograph` (kNN
graph, repeated Leiden clustering, sparse co-clustering, and hierarchical cuts),
while `cellpax.generators.base` defines the algorithm-independent seam. A
generator may execute in process or return artifacts produced by an external
process. Downstream code sees only generic candidate definitions, membership,
optional hierarchy, and boundary evidence. Fauxnograph membership strength is
left null because no named per-cell strength metric has been defined.

The fauxnograph payload retains its sparse consensus matrix, a compact
cell-by-run `Int32` label matrix, and one parameter/seed/summary row per run.
`Study.clustering_run_metadata()` and `Study.clustering_run_labels()` expose the
diagnostics without promoting method-specific state into generic registries.
These are optional, generator-specific inspection helpers—not required parts of
the downstream candidate API.
Jaccard-weighted neighbor edges and opportunity-normalized consensus are explicit
compute options. Set `build_hierarchy=False` and derive `native` cuts by
`run_index` to avoid the dense hierarchical-linkage path on larger scopes.

Hierarchy construction performs an explicit quadratic-memory preflight before
consensus computation; the backend limit can be raised deliberately when the
estimate is acceptable. Clustering structural summaries and candidate partitions
are validated without recomputing stochastic consensus. Accordingly,
`ClusteringConfig.recompute_deterministic` defaults to false and is provenance,
not an instruction to rerun clustering during validation.

`Study.validate()` never deserializes fitted-state or generator pickle payloads
by default, making integrity and structural checks safe for an untrusted shared
folio. `Study.validate(trusted_models=True)` additionally replays deterministic
Slice-2 fitted state and must only be used with a trusted study. Operations that
inherently use stored models—replay, deriving another candidate cut, and optional
generator inspection—also require trusting the study source.

Slice 4 adds an append-only review ledger and separates taxonomy vocabulary from
cell-level claims. Decisions form linear named branches, require rationale, and
validate action-specific payloads for assignment, parent-only assignment,
merge, split, exclusion, ambiguity, rename, and local-revision attachment. Small
targets may be inline IDs; large cell targets are canonical DataFolio membership
artifacts. A split names those artifacts explicitly, so its retained evidence is
auditable without embedding giant cell lists in a registry row.

Taxonomy versions are immutable hierarchical tables with generated rich
`IntEnum` bindings. Reducing a decision head produces a separate immutable
assignment set: correcting a label, excluding a cell, or marking it ambiguous
therefore changes assignment state without pretending the biological vocabulary
changed. Every assignment records its source revision and whether it was
feature-based, manual, or propagated; coverage is always relative to a named
feature space.

KNN label propagation is a first-class retained computation with distinct fit and
application scopes, a source assignment set, fitted-model provenance, and
per-cell predicted taxon, neighborhood purity, and coverage. Its model is only
deserialized by `validate(trusted_models=True)`; default validation checks hashes,
schemas, scope coverage, taxonomy membership, and assignment provenance without
executing pickle payloads.

Slice 5 exposes read-only views as documented functions returning validated,
fixed-schema Polars frames. There is no view class hierarchy or query language:

```python
from cellpax.plotting import embedding_scatter
from cellpax.views import embedding, serialize_view

frame = embedding(study, revision, assignment_set=assignments)
embedding_scatter(frame, ax=ax)  # any matplotlib Axes
component_bytes = serialize_view("embedding", frame)
```

The same pattern covers revision cells, feature profiles, generic candidate
stability, revision comparison, taxonomy counts, ancestry, and release-readiness
summary. View schemas live in `cellpax.views.VIEW_CONTRACTS` and are validated on
every projection. `serialize_view()` produces deterministic, self-describing JSON
with contract and row hashes; it does not mutate the study. The plotting adapter
accepts an existing matplotlib-compatible axes object and only imports matplotlib
lazily when asked to create one.

Slice 6 publishes a named annotation release by pairing one immutable assignment
set with its taxonomy version and source revision:

```python
release = study.create_annotation_release(
    "inh-cell-types-v1",
    source_revision=revision,
    assignment_set=assignments,
    created_by="casey",
)
study.validate_annotation_release(release)

bundle = study.load_annotation_release(release)
Taxon = bundle.taxonomy_enum()
```

The release manifest resolves exact copies of the taxonomy, assignments, decision
lineage, quality summary, resolved language-neutral recipe, replay script, and an
importable rich-enum binding. Checksums cover every product, and release
validation compares them with the immutable study records without loading fitted
estimators or clustering payloads. Every assignment's source revision must be the
published revision or one of its ancestors, so the resolved recipe cannot be
borrowed from an unrelated superset scope. `bundle.enum_binding` is generated
Python source for consumers that want a checked-in binding; execute source only
from a trusted study.

The Trajan adapter consumes only this release contract. It does not import the
study, clustering, review, or propagation layers:

```python
from cellpax.adapters.trajan import add_to_connectivity_table

add_to_connectivity_table(
    connectivity_table,
    bundle,
    name="cell_types",
    side="both",
)
```

Use `add_to_synapse_table()` for Trajan's `SynapseTable`, or
`decorate_cells()` for a plain Polars cell frame. The adapter registers one
canonical annotation row per released cell through Trajan's public annotation
methods; connectivity analysis remains Trajan's responsibility.

## Development

### Running tests

```bash
poe test
```

The suite includes fresh-process replay, lineage, transaction rollback, corruption
detection, and table-contract tests.

### Building documentation

```bash
poe doc-preview
# If port 8000 is already occupied:
poe doc-preview --port 8001
```

### Versioning

```bash
# Dry run to see what will change
poe drybump patch

# Actually bump the version
poe bump patch  # or minor, or major
```



## Profiling

```bash
# Profile with scalene (CPU + memory)
poe profile-all <your script>

# Profile with pyinstrument (CPU only, nicer output)
poe profile <your script>
```

## License

MIT License - see LICENSE file for details.
