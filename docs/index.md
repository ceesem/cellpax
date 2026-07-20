# CellPax

CellPax is a flexible, **polars-native** toolkit for clustering cells by their
features and turning the result into clear, named, shareable cell-type labels.

You hold one object — a `FeatureTable` — and work interactively: mask subsets,
preprocess heavy-tailed features, run consensus clustering, cut it into
well-defined labels, compare different approaches, and save the whole analysis to
a [DataFolio](https://github.com/) with one call.

## What it gives you

- **One flexible container.** A `FeatureTable` wraps your cells (a polars frame)
  with named, hierarchical **masks**, composable **feature collections**, and an
  on-the-fly `dataframe(mask, scaled=…)` view — raw or normalized — ready for
  seaborn/matplotlib.
- **Heavy-tail preprocessing.** `preprocess()` screens each feature's skew and
  applies an inverse-hyperbolic-sine transform to the wide ones (handles zeros
  and negatives), fit per mask.
- **Consensus clustering.** `ft.cluster(...)` runs repeated kNN/Leiden
  (fauxnograph) into a `SimilarityMatrix` you can cut at any threshold.
- **Clear labels.** A `LabelSet` gives clusters identity (`name`, `color`) with
  clean `rename`/`merge`/`reorder`/`combine`, and binds to `IntEnum` so you can
  write `L.L5IT` with autocomplete instead of remembering numbers.
- **Compare approaches.** `compare(a, b)` gives contingency tables, ARI/NMI/FMI/
  Jaccard, and alluvial frames across labelings.
- **First-class persistence.** `ft.save(folio, name)` / `FeatureTable.load(...)`
  store many analyses per folio, alongside your own content.

## A 30-second look

```python
from cellpax import FeatureTable, compare

ft = FeatureTable(cells_df, features=[...], feature_metadata=meta)
ft.define_features("axon", family="axon")
ft.add_mask("l23", pl.col("layer") == "L2/3")
ft.preprocess()                                   # ihs on heavy-tailed features

ft.cluster("l23", columns="axon", name="run")
labels = ft.label("run", mask="l23", distance_threshold=0.6, name="subclass")
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

## Where to go next

- **[Tutorial](tutorial.md)** — build a complete analysis end to end.
- **[User Guide](guide.md)** — task-oriented reference for each piece.
- **[Function Reference](reference/api.md)** — the full API.
