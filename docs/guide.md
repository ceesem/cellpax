# User Guide

Task-oriented reference for each part of CellPax. Skim the [Tutorial](tutorial.md)
first; here we go deeper. For exact signatures see the
[Function Reference](reference/api.md).

## The FeatureTable

A `FeatureTable` wraps a polars frame of cells. It needs a unique id column and
the feature columns to cluster/scale on; anything else is metadata.

```python
from cellpax import FeatureTable

ft = FeatureTable(df, features=[...], id_column="cell_id", feature_metadata=meta)
ft.n_cells, ft.n_features, ft.feature_columns, ft.columns
```

A pandas frame is accepted and converted. `feature_metadata` is an optional
per-feature frame (`feature_id` + family/modality/units/…) exposed as `ft.var`.

## Masks

A mask is a named boolean subset of cells, stored on the table. `based_on`
intersects with a parent mask so hierarchical subsets stay nested.

```python
ft.add_mask("exc", pl.col("is_inhibitory") == False)
ft.add_mask("l23", pl.col("layer") == "L2/3", based_on="exc")   # nested in "exc"
ft.masks                     # ['all', 'exc', 'l23']
ft.mask_series("l23")        # the boolean Series
```

Predicates can be a polars expression over the table or a full-length boolean
array. The implicit `"all"` mask covers every cell.

## Feature collections

Collections are named, composable subsets of features — the clean replacement for
ad-hoc column lists. Define them explicitly or from feature metadata:

```python
ft.define_features("axon", family="axon")            # by metadata
ft.define_features("core", columns=["axon_len", "dend_vol"])
ft.collections["axon"] | ft.collections["dend"]      # union; also & and -
```

Pass a collection (or its name, or a plain list) as `columns=` to `dataframe`,
`features`, `cluster`, and `embed`.

## Preprocessing and scaling

`preprocess()` resolves a per-feature transform once (a heavy-tail `ihs` screen by
default), recorded in `ft.transforms`. Scaling is lazy and per-mask: the first
time you request scaled values for a `(mask, columns)` pair, a scaler is fit and
cached.

```python
ft.preprocess(skew_screen=True, method="ihs", threshold=1.5)
ft.dataframe("l23", scaled=True)      # transformed + scaled features + metadata
ft.dataframe("l23", scaled=False)     # raw units (for plotting actual values)
ft.features("l23", scaled=True, columns="axon")   # numpy matrix
```

Cluster on the normalized features, plot the raw ones — both come from the same
table. The default scaler is `StandardScaler`; pass `scaler_factory=` (e.g.
`make_clipped_scaler`) to change it.

## Clustering

Consensus clustering (repeated kNN/Leiden, "fauxnograph") runs on scaled features
and returns a `SimilarityMatrix`:

```python
ft.cluster(
    "l23", columns="axon",
    n_neighbors=(30,),      # int or a swept list
    resolution=(1.0,),      # int/float or a swept list
    n_times=20, min_cluster_size=10,
    seed=0, n_jobs=-1, name="run",
)
sim = ft.clustering("run")
sim.cluster_count_curve()               # choose a distance threshold
sim.cluster_labels(0.6, min_cluster_size=10)   # raw integer labels
```

The `SimilarityMatrix` caches its hierarchical linkage; `cluster_labels` cuts it
at a distance threshold, and `cluster_count_curve` sweeps thresholds.

## Labels

`ft.label` cuts a stored clustering into a `LabelSet` aligned to a mask's cells,
with cluster ids renumbered 0-based. A `LabelSet` gives clusters identity and
clean relabeling:

```python
labels = ft.label("run", mask="l23", distance_threshold=0.6, name="subclass")
labels.rename({0: "L2a", 1: "L2b"})
labels.merge(["L2a", "L2b"], into="L2")
labels.reorder(["L2", "L3"])
labels.set_colors({"L2": "#1f77b4"})
labels.counts()
ft.attach(labels)                       # adds a 'subclass' column (null off-mask)
```

`combine` unions two label sets over disjoint cells (e.g. exc + inh clustered
separately). See below for `IntEnum` bindings.

### IntEnum bindings

Work by name with autocomplete instead of remembering numbers:

```python
L = labels.to_enum("SubclassLabels")            # members = names, values = ids
ft.dataframe().filter(pl.col("subclass_id") == L.L2a)

class ITLabels(IntEnum):                         # or drive naming from your enum
    L5IT = 0
    L23IT = 1
labels.apply_enum(ITLabels)
```

## Comparing approaches

```python
from cellpax import compare, compare_many

cmp = compare(labels_a, labels_b)      # aligned on shared cells
cmp.agreement()                        # ARI / NMI / FMI / Jaccard (+ n)
cmp.contingency(normalize=False)       # long-form cross-tab
cmp.alluvial_frame()                   # source / target / value for sankey plots
compare_many([a, b, c], metric="ari")  # pairwise agreement matrix
```

## Embeddings

```python
ft.embed("l23", method="pca", n_components=2, name="pca")   # sklearn PCA
ft.embed("l23", method="umap", n_components=2)              # optional umap-learn
ft.embedding("l23", name="pca")                             # stored coordinates
ft.dataframe("l23", embedding="pca")                        # joined onto the tidy view
```

## Persistence

Save the whole analysis under a name in a DataFolio — structured, not flattened.
Many analyses and arbitrary user content coexist in one folio.

```python
from cellpax import list_analyses

ft.save(folio, "l23it")                # table, masks, collections, transforms,
                                       # embeddings, and consensus matrices
ft = FeatureTable.load(folio, "l23it")
list_analyses(folio)                   # CellPax analyses in the folio

folio.add("notes/readme", {...})       # your own items live alongside, untouched
```

Scalers refit lazily on load (the resolved transforms and scaler choice are
restored, so scaled values reproduce).

## Plotting

CellPax deliberately ships no plotting layer — `dataframe(...)`, `embedding(...)`,
and `compare(...).alluvial_frame()` return tidy frames you hand straight to
seaborn/matplotlib, which is easier to tailor per figure.
