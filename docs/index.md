# CellPax

CellPax is a flexible, **polars-native** toolkit for clustering cells by their
features and turning the result into clear, named, shareable cell-type labels.
It is built for interactive analysis: you hold one object — a `FeatureTable` —
and work with it directly.

A typical session masks a subset of cells, preprocesses heavy-tailed features,
runs consensus clustering, resolves it into well-defined labels (by a simple
threshold or by a statistical test that needs no threshold), compares approaches,
and saves the whole analysis to a [DataFolio](https://github.com/) — all through
methods on the table.

## The core idea

Everything hangs off `FeatureTable`:

- **Masks** — named, hierarchical subsets of cells (`add_mask`, `based_on`).
- **Feature collections** — named, composable subsets of features
  (`define_features`, with `|`, `&`, `-`).
- **`dataframe(mask, scaled=…)`** — the primary view: one row per cell, all
  metadata/labels intact, feature columns raw or normalized, ready for seaborn.
- **`preprocess()`** — inverse-hyperbolic-sine screening of heavy-tailed features,
  fit per mask.
- **Clustering** — `cluster` (fauxnograph kNN/Leiden consensus) and
  `overcluster` (single high-resolution Leiden), with a hierarchy toolkit
  (`merge_support`, nested labels, per-cell stability) for choosing cuts.
- **`LabelSet`** — clusters with identity (`name`, `color`), clean relabeling, and
  `IntEnum` bindings so you write `L.L5IT` with autocomplete.
- **`compare`** — contingency, ARI/NMI/FMI/Jaccard, and alluvial frames across
  labelings.
- **`save` / `load`** — first-class DataFolio persistence, many analyses per
  folio, alongside your own content.

## A 30-second look

```python
from cellpax import FeatureTable, compare

ft = FeatureTable(cells, features=[...], feature_metadata=meta)
ft.define_features("axon", family="axon")
ft.add_mask("l23", pl.col("layer") == "L2/3")
ft.preprocess()                                   # ihs on heavy-tailed features

clus = ft.cluster("l23", columns="axon", name="run")
labels = clus.label(distance_threshold=0.6, name="subclass")
labels.rename({0: "L2a", 1: "L2b"})
ft.attach(labels)

ft.dataframe("l23", embedding="pca")              # tidy frame: metadata + labels + coords
ft.save(folio, "l23it")
```

## Install

CellPax uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

During development the DataFolio 2.0 storage library is resolved from the sibling
`../datafolio` checkout.

## Where to go next

- **[Tutorial](tutorial.md)** — build a complete analysis end to end.
- **[User Guide](guide.md)** — task-oriented reference for every piece.
- **[Function Reference](reference/api.md)** — the full API.
