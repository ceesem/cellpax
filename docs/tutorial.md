# Tutorial: from features to labeled clusters

This walkthrough builds a complete CellPax analysis: load cells, preprocess and
cluster them, turn the result into clear labels, compare cuts, and save
everything to a folio. Every snippet is part of one runnable script.

## 0. Set up

We use a small synthetic dataset of 60 cells in two well-separated groups so the
clustering has something clean to find.

```python
import tempfile
from pathlib import Path
from enum import IntEnum

import numpy as np
import polars as pl

from cellpax import FeatureTable, compare

rng = np.random.default_rng(0)
coords = np.vstack([rng.normal(0, 0.3, (30, 3)), rng.normal(8, 0.3, (30, 3))])
df = pl.DataFrame(
    {
        "cell_id": pl.Series(range(1, 61), dtype=pl.Int64),
        "axon_len": coords[:, 0],
        "axon_tort": coords[:, 1],
        "dend_vol": coords[:, 2],
        "region": ["L"] * 30 + ["R"] * 30,
    }
)
```

## 1. Build a FeatureTable

The `FeatureTable` holds your cells: a unique `cell_id`, the feature columns to
cluster on, and any metadata. Optional `feature_metadata` describes each feature
(family, modality, units) so you can select by it later.

```python
meta = pl.DataFrame(
    {
        "feature_id": ["axon_len", "axon_tort", "dend_vol"],
        "family": ["axon", "axon", "dend"],
    }
)
ft = FeatureTable(
    df, features=["axon_len", "axon_tort", "dend_vol"], feature_metadata=meta
)
```

## 2. Masks and feature collections

A **mask** is a named subset of cells; a **feature collection** is a named,
composable subset of features. Both let you focus an analysis.

```python
ft.add_mask("left", pl.col("region") == "L")      # a named subset of cells
ft.define_features("axon", family="axon")          # a named feature collection
ft.collections["axon"] | ft.collections["axon"]    # collections compose: | & -
```

## 3. Preprocess

`preprocess()` screens each feature's skew and applies an inverse-hyperbolic-sine
(`ihs`) transform to the heavy-tailed ones, refit per mask. Scaling then happens
on demand — `dataframe(scaled=True)` and clustering use the transformed, scaled
values, while `dataframe(scaled=False)` keeps the raw units for plotting.

```python
ft.preprocess()             # ihs on wide features; threshold and method are tunable
ft.transforms              # {feature: 'ihs' | None} — the resolved decisions
```

## 4. Cluster

`ft.cluster(...)` runs repeated kNN/Leiden consensus (fauxnograph) on the scaled
features and returns a `SimilarityMatrix`. Name it to store it for later.

```python
ft.cluster(n_neighbors=15, n_times=5, seed=0, name="run")
sim = ft.clustering("run")
sim.cluster_count_curve()   # (distance thresholds, number of clusters) to pick a cut
```

!!! note "fauxnograph runs in parallel"
    Pass `n_jobs=1` for a deterministic single-threaded run or to silence joblib
    worker warnings in some environments.

## 5. Turn a cut into clear labels

Cut the consensus at a distance threshold into a `LabelSet` — clusters with
identity you can rename, merge, reorder, and color. Ids are 0-based.

```python
labels = ft.label("run", distance_threshold=0.5, name="subclass")
labels.counts()                                     # cells per cluster
labels.rename({0: "TypeA", 1: "TypeB"}).set_colors({"TypeA": "#d62728"})
ft.attach(labels)                                   # adds a 'subclass' column
```

### Number- and name-free with IntEnum

Generate an `IntEnum` so you can filter with autocomplete instead of remembering
ids or exact strings (members compare equal to their integer id):

```python
L = labels.to_enum("SubclassLabels")
ft.dataframe().filter(pl.col("subclass_id") == L.TypeA)

# ...or name clusters from your own enum:
class ITLabels(IntEnum):
    TypeA = 0
    TypeB = 1
labels.apply_enum(ITLabels)
```

## 6. Compare different approaches

Cut the same run two ways (or cluster with different parameters) and compare:

```python
tight = ft.label("run", distance_threshold=0.3, name="tight")
cmp = compare(labels, tight)
cmp.agreement()        # {'ari': ..., 'nmi': ..., 'fmi': ..., 'jaccard': ..., 'n': ...}
cmp.contingency()      # long-form cross-tab
cmp.alluvial_frame()   # source/target/value, ready for a sankey plot
```

## 7. Embed and get a plot-ready frame

```python
ft.embed(method="pca", n_components=2)               # or method="umap" (optional dep)
plot_df = ft.dataframe(embedding="pca")              # metadata + labels + pca0/pca1
# plot_df is tidy — hand it straight to seaborn: facet by region, color by subclass
```

## 8. Save the whole analysis

Persist everything (table, masks, collections, transforms, embeddings, and the
consensus matrix) under a name in a DataFolio. Many analyses — and your own
content — can share one folio.

```python
folio = Path(tempfile.mkdtemp()) / "study"
ft.save(folio, "l23it")

reloaded = FeatureTable.load(folio, "l23it")         # masks, labels, clustering restored
from cellpax import list_analyses
list_analyses(folio)                                 # -> ["l23it"]
```

## Where to go next

The **[User Guide](guide.md)** covers each piece in depth: hierarchical masks,
collection algebra, the preprocessing/scaling model, clustering parameters, the
full `LabelSet` verb set, comparison metrics, and persistence layout.
