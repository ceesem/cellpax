# Migrating an existing analysis

Notes for a project already using CellPax — specifically the MICrONS subclass clustering in
`v1dd_feature_pax`, whose notebooks drive the parts of the API that changed. Nothing here
is required to keep an old analysis *readable*: manifest version 1 folios still load, and
every new parameter defaults to the previous behaviour. What follows is what changes if you
want the new machinery, in the order it has to happen.

## 1. Re-save, because the old form may not load

This is the blocking one. `save_feature_table` used to write each consensus matrix as a COO
triplet frame, whose size grows with the number of nonzero cell *pairs*. At 21,291 cells
that is roughly 600 MB, above DataFolio's ~500 MB eager-load limit — so
`load_feature_table` raises on a folio it wrote itself, and the notebook's first cell can
no longer re-run against what its last cell saved:

```
ValueError: Table '<name>/clustering/<c>' is ~603.0 MB, above the ~500.0 MB
eager-load limit.
```

There is no in-place fix, because the runs behind those matrices were never stored. The
clusterings have to be recomputed once and re-saved:

```python
ft = cpx.load_feature_table(folio, name="reprocessed")   # fails today
```

If the folio is currently unloadable, recompute from the preprocessing notebook's output
and re-run the clustering cells. After that, clusterings persist as their runs
(`<name>/partitions/<c>`, int32) with the matrix derived on load — about an order of
magnitude smaller, scaling as `n × n_runs` rather than `n²`.

Two things that come back for free once the runs persist:

- `clus.restrict(...)` works on a reloaded clustering. It used to raise, because
  `Partitions` were session-only, which meant the expensive part of clustering was
  unrecoverable and the grain could not be revisited after a reload.
- `clus.merge_support(...)` and `axis_stability(...)` likewise.

## 2. The count curve and the cut have to agree

Every `cluster_count_curve()` call in the clustering notebook is read at the default
`min_cluster_size=1` while the corresponding `label(...)` cuts at 10, 50 or 100. At 1
nothing is dropped and every singleton counts as a cluster, so the curve promises far more
clusters than the cut delivers — usually by a wide margin, since singletons are most of the
count. The threshold chosen from that curve was chosen against a curve the cut does not
produce.

```python
clus.cluster_count_curve(min_cluster_size=50)   # the same value you will cut at
clus.threshold_scan(min_cluster_size=50)        # and how many cells survive it
```

This is a pre-existing mismatch, not something the new code introduced, but it interacts
with everything below: any threshold carried forward was picked under the wrong curve.

## 3. Eight hand-tuned thresholds become one ladder

The notebook picks a `distance_threshold` per cohort, spanning 0.55–0.80, plus three
different `min_cluster_size` values. That is the flattening step the hierarchy work
replaces:

```python
h = clus.hierarchy()          # levels at several heights, coarse to fine
h.nested_levels               # what each level is: height, frequency, n_clusters
h.merge_table()               # every merge with its stability
clus.merge_support(n_bands=4) # and which resolutions supported it
```

Then attach the level you want, or several:

```python
from cellpax import LabelSet
ft.attach(LabelSet.from_labels(
    h.nested_labels["cell_id"].to_numpy(),
    h.nested_labels["level_1"].to_numpy(),
    unassigned=-1, name="subclass", mask=clus.mask,
))
```

The recursion the notebook builds by hand — cluster a cohort, cut it, `np.isin` the labels
into a new mask, cluster again — is a different thing from this and still valid; the
hierarchy replaces the *cut*, not the recursion. But note the two now overlap: a level of
the ladder and a level of the recursion answer the same question differently, and it is
worth deciding which one owns a given split rather than having both.

## 4. New sweep parameters

```python
clus = ft.cluster(
    mask,
    columns="analysis",
    alpha=0.5,                                   # partial whitening; 0.0 is unchanged
    eigenvalue_floor=ft.space(mask).noise_floor,  # read this before using alpha=1
    graph_type=["knn", "snn_jaccard", "umap_fuzzy"],
    n_neighbors=[15, 30, 60],
    resolution=np.geomspace(0.02, 1.5, 12),
    n_times=5,
    seed=0,                                      # note: currently None everywhere
)
```

`seed=None` throughout the notebook means no run is reproducible, which matters more once
parameters are being compared: a difference between two settings is not readable if each
was measured once with an unrecorded seed.

Pool on realised grain rather than nominal resolution when `graph_type` is a list —
resolution is not comparable across weightings:

```python
clus.partitions.by_setting()                       # median_clusters is comparable
clus.restrict(n_clusters_min=8, n_clusters_max=40)
```

## 5. The clip rule, for the small cohorts

The clipped scaler is configured once in `microns_preprocessing.ipynb`:

```python
ft = cpx.FeatureTable(..., scaler_factory=cpx.make_clipped_scaler)
```

That is a percentile rule at 0.1/99.9, which is fine at 21,000 cells and not fine at the
few-hundred-cell interneuron subclasses. Check before switching:

```python
from cellpax import clip_comparison, clipped_scaler_factory
clip_comparison(ft.features(small_mask), ft.feature_columns)
```

then, if the comparison supports it:

```python
ft = cpx.FeatureTable(..., scaler_factory=clipped_scaler_factory(mode="sigma", n_sigma=4.0))
```

`n_sigma` is in IQR units, not standard deviations — see the user guide. Changing this
changes every scaled value, so it invalidates stored clusterings and embeddings; do it in
the preprocessing notebook and recompute downstream.

## 6. Scoring the changes

The notebook currently judges results by looking at UMAPs. Two problems with carrying that
forward: `ft.embed` builds its neighbour graph in the **full 81-D scaled space** while
`ft.cluster` uses `pca(0.95)`, so the figure was never a picture of the clustered space;
and at `alpha > 0` the two genuinely diverge, so labels will look *worse* on the existing
UMAP even when the clustering improved.

```python
from cellpax import subsample_stability, loo_knn_recovery, paired_recovery, label_purity
```

- **`subsample_stability`** is the primary criterion. Build the coordinates outside the
  loop, from views over one `ft.space(...)` fit, so the representation is frozen and only
  `alpha` differs.
- **`ct_combo`** is already a column on the feature table and is subclass-resolution, which
  makes it a far better recovery target than the 389 manual E/I labels. It is another
  model's output, so agreeing with it rewards reproducing its boundary including its
  errors — it ranks representations rather than certifying them.
- **`label_purity` against `is_inhib_label_nn`** is a tripwire, not a criterion. Note it
  can only fire where straddling is possible: the notebook pre-splits E and I before
  clustering, so for seven of the eight cohorts purity is 1.0 by construction. It has teeth
  on `exc_v1_core`, which is gated on `is_inhib_label` but auditable by the
  differently-provenanced `is_inhib_label_nn` — 438 + 939 population cells disagree between
  the two, so genuine straddlers exist to find.

## 6b. Optional embedding backends

`ft.embed` now takes `method="pacmap"` and `method="localmap"` as well as `"umap"`, behind
`pip install 'cellpax[embeddings]'`. Worth trying here specifically because the eight cohort
figures are read for inter-cluster structure, which is what UMAP is least able to provide.

```python
ft.embed(mask, method="pacmap", name=f"{group}_pacmap", seed=0)
ft.embed(mask, method="umap",   name=f"{group}_umap",   seed=0)
```

Three practical notes for the existing notebooks:

- **`seed=` is currently absent everywhere**, so no figure in them can be regenerated. All
  four backends accept it.
- **`n_neighbors` defaults to 10** for pacmap/localmap against UMAP's 15, so the first
  output will look more fragmented than the current figures. Not a like-for-like swap.
- **On the 81-column `analysis` set, pacmap's own `apply_pca` never fires** (it needs >100
  features), so it embeds what it is handed. It *would* fire on the full 133-column set.
  `ft.graph_provenance()` shows which happened.

Running two backends is more useful than replacing one: a cell misplaced in both is more
likely genuinely unusual, while one misplaced in only one is that algorithm's artifact.
`neighbor_label_composition` takes any coordinate array, so the same triage runs on each.

## 6c. Coordinate columns no longer need reconstructing

Every scatter call currently rebuilds coordinate column names from the embedding's name,
which means writing that name three times:

```python
sns.scatterplot(
    x=f"{embedding_name}0",
    y=f"{embedding_name}1",
    data=feat_table.dataframe(mask=mask, embedding=embedding_name),
)
```

`ft.embedding_view` returns the frame and the names together:

```python
v = feat_table.embedding_view(mask, name=embedding_name, labels=lbl)
sns.scatterplot(**v.xy, data=v.frame, hue=lbl.name, palette=lbl.color_map())
```

That collapses three things in the project:

- **The local `plot_embedding` helper** (`microns_subclass_clustering.ipynb` cell 7) becomes
  unnecessary. Worth deleting rather than porting — it accepts an `ft` argument it never
  uses and closes over the global `feat_table` instead, so it silently ignores the table
  you hand it.
- **`plotting.preview_labels`** (`plots.py:1042`) can take a view instead of an
  `embedding_name`, which removes the same triple-mention from the library side.
- **The `pl.col('umap_core0') > 10` filters** in `microns_exp_inh.ipynb` cells 27, 44 and 45
  become `v.frame.filter(pl.col(v.x) > 10)` — coordinates as predicates, which a
  plotting-only helper would not have covered.

Two related additions: `ft.embeddings` lists stored `(mask, name)` pairs, which matters for
the cells that rely on `embed`'s derived name (`microns_preprocessing.ipynb` cell 36 stores
`umap_analysis` implicitly, and later cells have to guess it); and `graph_provenance()` now
lists embeddings after a reload, which it previously did not.

If you ever pass `n_components=3`, note that the old pattern silently plotted the first two
axes. `v.xy` still gives you those two, but `v.n_components` says there are three and
`v.pair(0, 2)` makes the alternative explicit.

## 7. Before the manual labels can be used at all

The 389 manually labelled cells exist only inside CAVE state `5967058931023872`, read by a
hard-coded cell in `microns_exp_inh.ipynb`, and the validation cross-tabs computed from
them have no stored outputs. Nothing can be reproducibly scored against them until they are
snapshotted — a folio table or a checked-in parquet, with the stratum sizes and design
weights alongside. That is a prerequisite for the validation rather than part of it, and
it is worth doing even though the E/I axis is no longer the driver, because a mutable
remote state is not a held-out set.

## 8. Duplicated notebook code with library equivalents

Not required, but the drift is already causing bugs. `plotting.cluster_strip`,
`plotting.feature_v_depth_facets` and `plotting.preview_labels` all exist and are
reimplemented inline — the facet loop appears seven times, two copies of which are missing
an f-string prefix and so overwrite a single file literally named `{col}.png`, and one
plots inhibitory cells against an excitatory background. `restrict`, `by_setting`,
`threshold_scan` and `sorted_matrix` are library features that the notebook never calls and
that would replace the hand-tuning directly.

## Pinned by tests

The default path is unchanged and that is enforced, not asserted: `graph_type="knn"`
produces a byte-identical edge list to the previous implementation,
`ft.cluster(alpha=0.0)` is bitwise the plain projection, `mode="percentile"` remains the
default clip, and manifest version 1 folios load with their matrices intact. If an existing
analysis changes after upgrading without you passing a new parameter, that is a bug worth
reporting.
