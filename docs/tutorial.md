# Tutorial: from features to labeled clusters

This walkthrough builds a complete CellPax analysis end to end: load cells,
preprocess and cluster them, resolve clusters into clear labels (two ways),
compare approaches, and save everything to a folio. Every snippet is part of one
runnable script — paste them in order.

## 0. Set up

We use a small synthetic dataset of 90 cells in three groups, with a couple of
heavy-tailed (log-normal) features so preprocessing has something to do.

```python
import tempfile
from pathlib import Path
from enum import IntEnum

import numpy as np
import polars as pl

from cellpax import FeatureTable, compare

rng = np.random.default_rng(0)
blobs = np.vstack([rng.lognormal(m, 0.4, (30, 4)) for m in (0.0, 2.0, 4.0)])
df = pl.DataFrame(
    {
        "cell_id": pl.Series(range(1, 91), dtype=pl.Int64),
        "axon_len": blobs[:, 0],
        "axon_tort": blobs[:, 1],
        "dend_vol": blobs[:, 2],
        "dend_area": blobs[:, 3],
        "region": ["L"] * 45 + ["R"] * 45,
    }
)
```

## 1. Build a FeatureTable

The `FeatureTable` holds your cells: a unique `cell_id`, the feature columns to
cluster on, and any metadata. Optional `feature_metadata` describes each feature
(family, modality, units) so you can select features by it later.

```python
meta = pl.DataFrame(
    {
        "feature_id": ["axon_len", "axon_tort", "dend_vol", "dend_area"],
        "family": ["axon", "axon", "dend", "dend"],
    }
)
features = ["axon_len", "axon_tort", "dend_vol", "dend_area"]
ft = FeatureTable(df, features=features, feature_metadata=meta)
```

## 2. Masks and feature collections

A **mask** is a named subset of cells; a **feature collection** is a named,
composable subset of features. Both focus an analysis without copying data.

```python
ft.add_mask("left", pl.col("region") == "L")       # a named subset of cells
ft.define_features("axon", family="axon")           # a named feature collection
ft.collections["axon"] | ft.collections["axon"]     # collections compose: | & -
```

Masks can nest with `based_on` (a child is intersected with its parent), so you
can drill into a subclass and cluster it on its own.

## 3. Preprocess

`preprocess()` screens each feature's skew and applies an inverse-hyperbolic-sine
(`ihs`) transform to the heavy-tailed ones (it handles zeros and negatives),
recorded per feature. Scaling then happens on demand — `dataframe(scaled=True)`
and clustering use the transformed, per-mask–scaled values, while
`dataframe(scaled=False)` keeps raw units for plotting.

```python
ft.preprocess()
ft.transforms       # e.g. {'axon_len': None, 'dend_vol': 'ihs', 'dend_area': 'ihs', ...}
```

## 4. Cluster

`ft.cluster(...)` runs repeated kNN/Leiden consensus (fauxnograph) on the scaled
features and returns a `SimilarityMatrix`. Name it to store it.

```python
ft.cluster(columns="axon", n_neighbors=12, n_times=5, seed=0, name="run")
sim = ft.clustering("run")
sim.cluster_count_curve()   # (distance thresholds, n clusters) — helps pick a cut
```

!!! note "fauxnograph runs in parallel"
    Pass `n_jobs=1` for a deterministic single-threaded run or to silence joblib
    worker warnings in some environments.

## 5. Resolve clusters into labels — two ways

### By a distance threshold

Cut the consensus dendrogram at one threshold. Simple, but a single cut can't be
right everywhere.

```python
labels = ft.label("run", distance_threshold=0.6, name="subclass")
labels.counts()
```

### By CHOIR (no threshold)

`cluster_choir` keeps each split in the tree **only where it's statistically
justified** — a random-forest permutation test decides whether two child clusters
are distinguishable — so some branches resolve deeply while others merge, with no
threshold to tune (following [CHOIR](https://www.choirclustering.com/), Sant et
al., *Nature Genetics* 2025).

```python
labels = ft.cluster_choir("run", name="subclass", min_cluster_size=8)
len(labels.ids)             # a data-driven cluster count
```

You can also prune **any** over-clustering, not just the consensus tree — CHOIR's
design favours an intentional over-split. Feed a high-resolution Leiden partition:

```python
over = ft.overcluster(resolution=3.0)                     # single high-res Leiden
labels = ft.cluster_choir(over_clustering=over, min_cluster_size=8)
```

## 6. Name labels — and bind to an IntEnum

A `LabelSet` gives clusters identity you can rename, merge, reorder, and color.
Generate an `IntEnum` to filter by name with autocomplete instead of remembering
ids (members compare equal to their integer id):

```python
labels.rename({0: "L2a", 1: "L2b", 2: "L3"}).set_colors({"L2a": "#d62728"})
L = labels.to_enum("Subclass")
ft.attach(labels)                                          # adds a 'subclass' column
ft.dataframe().filter(pl.col("subclass_id") == L.L2a)

# ...or drive naming from your own enum:
class ITLabels(IntEnum):
    L2a = 0
    L2b = 1
    L3 = 2
labels.apply_enum(ITLabels)
```

## 7. Compare approaches

```python
threshold = ft.label("run", distance_threshold=0.6, name="threshold")
cmp = compare(labels, threshold)
cmp.agreement()        # {'ari': ..., 'nmi': ..., 'fmi': ..., 'jaccard': ..., 'n': ...}
cmp.contingency()      # long-form cross-tab
cmp.alluvial_frame()   # source / target / value, ready for a sankey plot
```

## 8. Embed and get a plot-ready frame

```python
ft.embed(method="pca", n_components=2)               # or method="umap" (optional dep)
plot_df = ft.dataframe(embedding="pca")              # metadata + labels + pca0/pca1
# tidy — hand straight to seaborn: facet by region, x/y = features or coords,
# color by subclass
```

## 9. Save the whole analysis

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
three CHOIR modes (consensus / per-node reselection / arbitrary over-clustering),
the full `LabelSet` verb set, comparison metrics, and persistence layout.
