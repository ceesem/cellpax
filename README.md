# cellpax

Flexible, polars-native cell feature clustering for connectomics. You hold one
`FeatureTable`, mask subsets, preprocess heavy-tailed features, run consensus
clustering, turn cuts into clear named labels, compare approaches, and save the
whole analysis to a DataFolio.

## Setup

This project uses `uv` for dependency management and `poe` for task running.

```bash
# During DataFolio development, uv uses the editable sibling checkout ../datafolio.
uv sync
poe lab        # Jupyter Lab
poe test       # pytest with coverage
```

## Quick start

```python
import polars as pl
from cellpax import FeatureTable, compare

ft = FeatureTable(cells_df, features=[...], feature_metadata=meta)
ft.define_features("axon", family="axon")
ft.add_mask("l23", pl.col("layer") == "L2/3")
ft.preprocess()                                   # ihs heavy-tail transform

ft.cluster("l23", columns="axon", n_neighbors=(30,), name="run")
labels = ft.label("run", mask="l23", distance_threshold=0.6, name="subclass")
labels.rename({0: "L2a", 1: "L2b"})
ft.attach(labels)

L = labels.to_enum("Subclass")                    # IntEnum: filter with autocomplete
ft.dataframe("l23").filter(pl.col("subclass_id") == L.L2a)

ft.save(folio, "l23it")                            # many analyses per folio
```

## What's here

- **`FeatureTable`** — polars container with hierarchical masks, composable
  feature collections, and `dataframe(mask, scaled=…)` raw/normalized views.
- **`preprocess`** — heavy-tail (`ihs`) skew screening, fit per mask.
- **clustering** — fauxnograph kNN/Leiden consensus + `SimilarityMatrix`.
- **`LabelSet`** — clear, relabelable clusters with `IntEnum` bindings.
- **`compare`** — ARI/NMI/FMI/Jaccard, contingency, alluvial across labelings.
- **persistence** — first-class DataFolio `save`/`load`, namespaced, many per folio.

See the [documentation](https://ceesem.github.io/cellpax/) (`poe doc-preview`),
`DESIGN_PROPOSAL.md` for the architecture and rationale, and the tutorial to build
an analysis end to end.
