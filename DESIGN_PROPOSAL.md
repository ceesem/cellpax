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
labels         LabelSet: clear named clusters + rename/merge/split/reorder
compare        NEW: contingency, agreement metrics, alluvial frames across LabelSets
persist        first-class DataFolio save/load
context        optional opinionated driver (add_subset → cluster → assign → plot)
plotting       ported/trimmed + embedding_scatter
```

Data model: **polars-first** with an integer `cell_id` key (numpy at compute
boundaries; `.to_pandas()` only where seaborn needs it).

### `FeatureTable` — the container

```python
ft = FeatureTable(df, features=[...], id_column="cell_id")

ft.add_column(series, "is_proofread", fill_value=False)
ft.add_mask("exc", mask=df["is_inhib"] == False)          # named subset
ft.add_mask("l23it", mask=..., based_on="exc")            # hierarchical
ft.preprocess(skew_screen=True, regress_out="soma_depth") # global _pre_ layer

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
labels = ft.cluster(mask="l23it", ...)      # returns a LabelSet
labels.rename({0: "L2a", 1: "L2b"})
labels.merge(["L2a", "L2b"], into="L2")
labels.split("L3", submask=...)
labels.reorder(["L2", "L3a", "L3b"])
ft.attach(labels, name="subclass")          # becomes a column on the table
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
src/cellpax/featuretable.py   FeatureTable + feature collections + preprocess
src/cellpax/labels.py         LabelSet
src/cellpax/clustering.py     ported consensus + SimilarityMatrix + prediction
src/cellpax/compare.py        comparison across LabelSets
src/cellpax/persist.py        DataFolio save/load
src/cellpax/context.py        optional opinionated driver
src/cellpax/plotting.py       ported/trimmed + embedding_scatter
```

Retired (recoverable from the `master` baseline): `study.py`, `review.py`,
`taxonomy.py` (versioning), `release.py`, `recipes.py`, `builder.py`,
`celldata.py`, `views.py`, `scopes.py`, `spaces.py`, `universe.py`,
`records.py`, `artifacts.py`, `generators/`, `contracts/`, `adapters/`.

Reused from current CellPax where useful: config-factory ergonomics for
clustering params, `identity` hashing helpers (for content keys where handy),
`table_utils`.

## Migration

There are no external consumers to preserve; existing dfc notebooks will move to
the new API (close to dfc's, so mostly mechanical). We keep the `cellpax` package
name and reset the version.

## Build order

1. `FeatureTable` core: construction, `add_column`, masks, `dataframe(scaled=)`,
   feature columns.
2. Feature collections + `preprocess` (skew-screen log/sqrt, regress-out).
3. Port `clustering` (fauxnograph + `SimilarityMatrix`) and wire `ft.cluster(...)`.
4. `LabelSet` + `ft.attach`.
5. Embeddings (`ft.embed`) + `plotting`.
6. First-class DataFolio `save`/`load`.
7. `compare`.
8. Optional `context` driver.
9. Retire the immutable-substrate modules; docs (tutorial + guide) rewrite.

## Open questions

- **Scalers:** port dfc's `PercentileClipper` / clipped-scaler as the default, or
  adopt CellPax's `standard_scaler`/`robust_scaler` set? (Lean: keep both; default
  to clipped, dfc's proven choice.)
- **`preprocess` scope:** keep dfc's global `_pre_` layer with per-mask re-fit, or
  make preprocessing mask-local from the start?
- **DataFolio granularity:** one folio per analysis, or one folio holding many
  named analyses? (Lean: many named analyses per folio.)
