# CellPax redesign — a FeatureTable-centered clustering toolkit

**Status:** Proposed. Supersedes the immutable-first design preserved on the
`master` baseline commit. This document is the plan for the `featuretable-redesign`
branch.

## Motivation

CellPax began as a rigorous, content-addressed rewrite of
[`dendritic_feature_clustering`](../dendritic_feature_clustering) (dfc). In
practice the rewrite inverted dfc's priorities: it made an **immutable,
content-addressed store** (`Study` over a DataFolio) the primary object and
*dissolved dfc's flexible working container into it*. The result is a `~4000`-line
`Study` God-object that fuses three concerns — a data container, a persistence
substrate, and an analysis workflow — into one, which makes the framework hard to
understand and inflexible for interactive analysis.

The clearest evidence: the recent round of ergonomic additions
(`feature_table`, feature "bundles", `auto_log_features`, cross-block combos, the
read-only `CellData` facade) were each **re-implementing a method dfc already had
on its `FeatureTable`**, one at a time. The problem was never missing features; it
was a missing *object*.

## What dfc got right (and we are keeping)

dfc had three clean layers:

1. **`FeatureTable`** — a mutable, in-memory, semantically-aware container: feature
   columns + metadata, named/hierarchical **masks** (each with its own scaler),
   **feature sets**, per-mask **embeddings**, **labels**, and a global
   `preprocess()` (skew-screened log/sqrt correction, spline regress-out). The
   flexible view is `dataframe(mask, scaled=…)`.
2. **clustering** — a functional library: `fauxnograph_coclustering → SimilarityMatrix`
   (distance / leiden / spectral cuts, cluster-count curves), scalers, and
   neighborhood prediction/purity.
3. **`ClusteringContext`** — an opinionated driver tying the two together with
   consistent defaults and plotting.

## Goals

- A **flexible, interactive** analysis experience: hold one object, mask it,
  preprocess it, cluster it, label it, plot it — no register/preview/keep ceremony.
- **First-class DataFolio save/load** of the whole working state (the pain point
  dfc never solved cleanly).
- Fix dfc's rough edges:
  - **Feature collections** — composable, first-class, not clunky ad-hoc sets.
  - **Relabeling** — clean, well-defined label operations.
  - **Comparison** — a real mechanism to compare different clustering approaches
    (dfc had none).
- Keep the **clarity of well-defined cluster labels** — named clusters with
  identity, without versioned taxonomies or a decision ledger.

## Non-goals (explicitly dropped from the immutable-first design)

- Versioned, immutable taxonomies.
- Append-only decision ledgers / breadcrumbed review history.
- Immutable content-addressed commits/registries and self-documenting releases.
- The generator-seam wrapping of clustering behind the Study.

DataFolio stays — but as **ergonomic persistence**, not a pervasive immutable
substrate.

## Architecture

Four layers, cleanly separated:

```
FeatureTable   the container: data, masks, feature collections, preprocess,
               dataframe(scaled=), pca, embeddings, labels
clustering     ported from dfc: fauxnograph consensus, SimilarityMatrix,
               scalers, neighborhood prediction
labels         LabelSet: clear named clusters + rename/merge/reorder/combine +
               IntEnum bindings (to_enum / apply_enum)
compare        NEW: contingency, agreement metrics, alluvial frames across LabelSets
persist        first-class DataFolio save/load
context        optional opinionated driver (add_subset → cluster → assign)
```

Plotting is intentionally **out of scope** — the ``dataframe`` and embedding
frames are tidy and seaborn/matplotlib-ready, and bespoke plots are easier to
write per-need in the notebook.

Data model: **polars-first** with an integer `cell_id` key (numpy at compute
boundaries; `.to_pandas()` only where seaborn needs it).

### `FeatureTable` — the container

```python
ft = FeatureTable(df, features=[...], id_column="cell_id")

ft.add_column(series, "is_proofread", fill_value=False)
ft.add_mask("exc", mask=df["is_inhib"] == False)          # named subset
ft.add_mask("l23it", mask=..., based_on="exc")            # hierarchical
ft.preprocess(skew_screen=True)                    # heavy-tail (ihs) transform

ft.dataframe(mask="l23it", scaled=True)   # the flexible frame: metadata + features
ft.features_pca("l23it", explained_variance=0.95)
ft.embed("l23it", method="umap")          # per-mask embedding
```

### Feature collections (fixes dfc's clunky feature sets)

First-class and composable, defined by family / modality / predicate / explicit
list, with set algebra:

```python
ft.define_features("axon", family="axon")
ft.define_features("dend", family="dendrite")
combo = ft.features["axon"] | ft.features["dend"]     # union; also & and -
ft.cluster(features=combo, mask="l23it")
```

### `LabelSet` — clear labels + clean relabeling (fixes dfc's relabeling)

A well-defined, mutable label object: `id → (name, color, description)` plus cell
membership. Clear identity, no versioning/ledger.

```python
sim = ft.cluster(mask="l23it", name="run")
labels = ft.label("run", mask="l23it", distance_threshold=0.6)
labels.rename({0: "L2a", 1: "L2b"})
labels.merge(["L2a", "L2b"], into="L2")
labels.reorder(["L2", "L3a", "L3b"])
ft.attach(labels, name="subclass")          # becomes a column on the table
```

`LabelSet` also binds to `IntEnum` for number/name-free work with autocomplete:

```python
L = labels.to_enum("ITLabels")              # members = names, values = ids
df.filter(pl.col("subclass_id") == L.L5IT) # IntEnum member == its int id
labels.apply_enum(ITLabels)                 # or name clusters from your own enum
```

### Comparison (new capability)

Compare different clustering approaches (params, methods, thresholds) on the same
cells:

```python
cmp = ft.compare("res0.7", "res1.2")        # two attached LabelSets
cmp.contingency()                            # cross-tab
cmp.agreement()                              # ARI / NMI / Jaccard
cmp.alluvial_frame()                         # sankey-ready
compare_many([a, b, c])                      # multi-way
```

### Clustering library (ported from dfc)

`fauxnograph_coclustering`, `SimilarityMatrix` (`cluster_labels(threshold,
min_size)`, `cluster_count_curve`, leiden/spectral, `plot_heatmap`), scaler
factories, and neighborhood prediction/purity — ported largely as-is (CellPax's
current fauxnograph is itself a port of dfc's).

### First-class DataFolio persistence

dfc's `to_dataframe_and_metadata()` **flattened everything into one dataframe**
(`_mask_*`, `_emb_*` columns) and did **not** persist fitted scalers or the
expensive `SimilarityMatrix` consensus. We store the composite object *structured*
instead:

- base cell table → a polars **item**
- feature collections, masks, preprocessor specs, label sets → a **manifest item**
- fitted scalers / UMAP models → DataFolio **`add_model`**
- expensive clustering results (consensus / `SimilarityMatrix`) → their own
  **items**, so a reload skips recomputation

One call each:

```python
ft.save(folio, name="l23it_analysis")            # frames, models, manifest, results
ft = FeatureTable.load(folio, name="l23it_analysis")
```

## Module plan

New / ported:

```
src/cellpax/featuretable.py   FeatureTable + feature collections + preprocess + embeddings
src/cellpax/labels.py         LabelSet (+ IntEnum bindings)
src/cellpax/consensus.py      ported consensus + SimilarityMatrix
src/cellpax/compare.py        comparison across LabelSets
src/cellpax/persist.py        DataFolio save/load
src/cellpax/context.py        optional opinionated driver
```

(`consensus.py` is renamed to `clustering.py` when the old modules retire in
step 9. Plotting is not provided — the user maintains their own plot adapters
against the tidy frames.)

Retired (recoverable from the `master` baseline): `study.py`, `review.py`,
`taxonomy.py` (versioning), `release.py`, `recipes.py`, `builder.py`,
`celldata.py`, `views.py`, `scopes.py`, `spaces.py`, `universe.py`,
`records.py`, `artifacts.py`, `generators/`, old `clustering.py`, `plotting.py`,
`contracts/`, `adapters/`.

Reused from current CellPax where useful: config-factory ergonomics for
clustering params, `identity` hashing helpers (for content keys where handy),
`table_utils`.

## Migration

There are no external consumers to preserve; existing dfc notebooks will move to
the new API (close to dfc's, so mostly mechanical). We keep the `cellpax` package
name and reset the version.

## Design principles (from the dfc audit)

A full read of dfc's `FeatureTable` (plus `FittedScaler`, `GlobalPreprocessor`,
the label methods) and `SimilarityMatrix` showed a strong container whose
weaknesses were all *accreted complexity*, not wrong ideas. These principles carry
the good bones forward and collapse the complexity.

**Keep (proven in dfc):**

1. **`dataframe(mask, scaled=…)` is the primary surface.** Filter to a mask,
   inject raw-or-scaled features, optionally join embeddings, hide internal
   columns. Preserve this shape (polars).
2. **`FittedScaler` = per-feature transforms + a fitted scaler, re-fit per mask,**
   so normalization is subset-specific.
3. **Masks are boolean columns with hierarchical `based_on`.** Simple and
   sufficient.
4. **`add_column(data, name, mask=, fill_value=)`** — write to a masked subset,
   fill the rest.
5. **`ihs` (inverse hyperbolic sine) is the default skew correction**, not `log` —
   it handles zeros and negatives. (Supersedes the `log1p`+skip-negatives approach
   from the immutable-first branch.)
6. **`SimilarityMatrix` is well-factored** (distance/leiden/spectral cuts,
   `cluster_count_curve`, `sparsify`, `build_umap_graph`, `plot_heatmap`) — port
   largely as-is.
7. **Method chaining** (`return self`) for interactive fluency.

**Fix (collapse the accreted complexity):**

8. **Feature collections are first-class and composable** — `FeatureCollection`
   with union/intersect/difference, defined by family/modality/predicate/explicit
   — replacing the `Dict[str, List[str]]` + magic `_DEFAULT` key with no algebra.
9. **One preprocessing concept, not two.** Drop the per-scaler `skewness_screen`
   axis; `preprocess()` decides per-feature transforms once (global, per-mask
   scaler re-fit) and `scaled=True` simply applies the fitted scaler on top. No
   `_resolve_preprocessing` / `lookup_prep` fallback logic.
10. **Scalers are lazy and single.** Fit on demand and cache one scaler per
    `(mask, collection)`. No eager, combinatorial `[mask][feature_set][bool]`
    fitting at every `add_mask`/`add_feature_set`.
11. **Embeddings are a flat keyed store** `(mask, collection, method) → coords`,
    not a 4-level nested dict with `None/True/False` keys and back-compat
    fallbacks. Round-trips to DataFolio as items, not `_emb_*` columns.
12. **Labels are a first-class `LabelSet`** (`id → name/color/description` +
    membership; `rename`/`merge`/`reorder`/`combine`, plus `to_enum`/`apply_enum`
    for `IntEnum` binding), attached as a column on demand — replacing scattered
    `add_label` (with a domain-specific `reorder_by="soma_depth"` default),
    `add_label_names` (which spawns a parallel `_named` column), and the
    overloaded `combine_labels`.
13. **Structured persistence, never flattened.** Frames as items, scalers/UMAP as
    models, consensus/`SimilarityMatrix` as items, structure in a manifest — so
    fitted state and expensive results survive a reload (dfc persisted neither).
14. **polars-first, explicit accessors.** No `__getattr__`/`__getitem__`
    passthrough to the underlying frame; no schema-version flags leaking into
    user-facing metadata.

## Build order

1. ✅ `FeatureTable` core: construction, `add_column`, masks, `dataframe(scaled=)`,
   feature columns.
2. ✅ Feature collections (composable) + unified `preprocess` (skew-screen `ihs`
   heavy-tail transform) with lazy per-mask scalers. (regress-out dropped.)
3. ✅ Port consensus (`fauxnograph` + `SimilarityMatrix`) and wire `ft.cluster(...)`.
4. ✅ `LabelSet` + `ft.label` / `ft.attach` + `IntEnum` bindings.
5. ✅ Embeddings (`ft.embed`, PCA-native / UMAP optional).
6. First-class DataFolio `save`/`load`.
7. `compare`.
8. Optional `context` driver.
9. Retire the immutable-substrate modules (rename `consensus.py` → `clustering.py`);
   docs (tutorial + guide) rewrite.

## Resolved by the audit

- **Scalers:** offer clipped / standard / robust; default to dfc's clipped scaler.
  The real fix is principle 10 — remove the `skewness_screen` second axis so
  scalers are lazy and single.
- **Preprocessing scope:** principle 9 — one unified layer (global per-feature
  transforms with per-mask scaler re-fit), not two overlapping paths.

## Open questions

- **DataFolio granularity:** one folio per analysis, or one folio holding many
  named analyses? (Lean: many named analyses per folio.)
- **`ihs` parameterization:** plain `arcsinh`, or a scaled `arcsinh(x/θ)` with a
  per-feature θ? (Lean: plain to start.)
