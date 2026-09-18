# User Guide

Task-oriented reference for each part of CellPax. Skim the [Tutorial](tutorial.md)
first; here we go deeper. For exact signatures see the
[Function Reference](reference/api.md).

## The FeatureTable

A `FeatureTable` wraps a polars frame of cells. It needs a unique id column and
the numeric feature columns to cluster/scale on; every other column is metadata
carried alongside.

```python
from cellpax import FeatureTable

ft = FeatureTable(df, features=[...], id_column="cell_id", feature_metadata=meta)
ft.n_cells, ft.n_features, ft.feature_columns, ft.columns
```

A pandas frame is accepted and converted. `feature_metadata` is an optional
per-feature frame (`feature_id` + family/modality/units/…) exposed as `ft.var`
and used by `define_features`. Pass `scaler_factory=` to change the scaler
(default `StandardScaler`; `make_clipped_scaler` is also provided).

### Bringing in a stable cell id

Extractions (e.g. ossify) are often keyed by a segmentation `root_id` and lack a
static `cell_id`. Supply an `id_map` (a `[root_id, cell_id]`-style frame) to join
it in and key the table on `cell_id` — at construction, or after:

```python
ft = FeatureTable(ossify_df, features=[...], id_map=root_to_cell)   # at creation

ft = FeatureTable(ossify_df, features=[...], id_column="root_id")   # ...or after
ft.set_id_column("cell_id", id_map=root_to_cell)
```

The map's key column (shared with the table, e.g. `root_id`) is inferred or given
via `on=`; every cell must map and ids must be unique. The original key is kept as
a metadata column.

### Adding features from another source

Static features from a different run-once source, keyed on `cell_id`, join in with
`add_features` and become first-class features (usable in collections, scaling,
and clustering):

```python
ft.add_features(
    synapse_df,                         # features default to all columns except `on`
    on="cell_id",                       # defaults to the id column
    feature_metadata=conn_meta,         # optional family/modality for the new features
    collection="conn",                  # also define a collection of these features
)
```

Omit `features` to add every source column except the join key, or pass a list to
select a subset. Every cell must be covered by the source (pass
`allow_missing=True` to permit nulls, which then can't be scaled), and the source
must have unique keys.

## Masks

A mask is a named boolean subset of cells, stored on the table. `based_on`
intersects with a parent mask, so hierarchical subsets stay nested.

```python
ft.add_mask("exc", pl.col("is_inhibitory") == False)
ft.add_mask("l23", pl.col("layer") == "L2/3", based_on="exc")   # nested in "exc"
ft.masks                     # ['all', 'exc', 'l23']
ft.mask_series("l23")        # the boolean Series
```

A predicate is a polars expression over the table or a full-length boolean array.
The implicit `"all"` mask covers every cell.

## Feature collections

Collections are named, composable subsets of features — the clean replacement for
loose column lists. Define them explicitly or from feature metadata, and combine
with set algebra:

```python
ft.define_features("axon", family="axon")            # by metadata
ft.define_features("core", columns=["axon_len", "dend_vol"])
ft.collections["axon"] | ft.collections["dend"]      # union; also & and -

ft.collections.names                                 # what's defined
ft.collections.catalog()                             # ...with their columns
```

Pass a collection — its name, the object, or a plain list — as `columns=` anywhere
features are chosen: `dataframe`, `features`, `features_pca`, `scaler`, `space`,
`cluster`, `overcluster`, `embed`, `project`, `propagate_labels`, `project_labels`,
`assign`, `assess_labels`, `triage_labels` and `boundary_report`.

## Preprocessing and scaling

`preprocess()` resolves one per-feature transform (a heavy-tail `ihs` screen by
default) recorded in `ft.transforms`. Scaling is lazy and per-mask: the first time
you request scaled values for a `(mask, columns)` pair, a scaler is fit on that
mask and cached.

```python
ft.preprocess(skew_screen=True, method="ihs", threshold=1.5)
ft.dataframe("l23", scaled=True)      # transformed + scaled features + metadata
ft.dataframe("l23", scaled=False)     # raw units (for plotting actual values)
ft.features("l23", scaled=True, columns="axon")   # numpy matrix
```

Cluster on the normalized features, plot the raw ones — both come from the same
table. `ihs` handles zeros and negatives; `log`/`sqrt` are also available and skip
features with negative values.

## Clustering

Consensus clustering (repeated kNN/Leiden, "fauxnograph") runs on scaled features
reduced by PCA (see [below](#which-space-the-knn-graph-is-built-in)). The full path
from features to a table column is four steps:

```python
clus = ft.cluster(                      # -> Clustering
    "l23", columns="axon",
    n_neighbors=(30,),      # int or a swept list
    resolution=(1.0,),      # int/float or a swept list
    n_times=20, min_cluster_size=10,
    seed=0, n_jobs=-1, name="run",
)
clus.cluster_count_curve()              # choose a distance threshold
labels = clus.label(distance_threshold=0.6, min_cluster_size=10)   # -> LabelSet
ft.attach(labels)                       # -> table columns
```

A `Clustering` is a `SimilarityMatrix` that also remembers **which cells it covers** —
its `mask`, that mask's `cell_ids` in matrix-row order, the `columns` compared, and
the `space` they were compared in:

```python
clus.mask       # 'l23'
clus.space      # 'pca(0.95)'
clus.columns    # ('axon_length', 'axon_branches', …)
clus.cell_ids   # the mask's ids, row-aligned with the matrix
```

That provenance is what makes `clus.label(...)` safe. Rows are matched to mask
members **by position, not by id**, so cutting a clustering against the wrong mask
mislabels every cell. Because the clustering carries its own mask, there is nothing
to line up by hand — and `ft.label` takes the mask from it, raising if you pass
one that disagrees rather than silently misaligning.

Everything a `SimilarityMatrix` does still works, since `Clustering` subclasses it:
`linkage` is cached, `cluster_labels(0.6)` gives raw integer labels, and
`cluster_count_curve()` sweeps thresholds. `ft.label(sim, mask=…, …)` also remains,
for a `SimilarityMatrix` computed outside the table.

`ft.clustering("run")` retrieves a stored one, provenance intact — including after a
`save`/`load` round trip. (Analyses saved before clusterings carried provenance load
with `mask="all"`, which is what `ft.label(name, distance_threshold=…)` assumed back
then.)

### Which space the kNN graph is built in

The graph is built on the mask's scaled features **reduced by PCA to `pca=0.95`** —
the smallest number of components reaching that explained variance. This is the space
phenograph-style clustering is conventionally run in: correlated features stop each
counting separately toward the neighbor distances, and the graph is built on a
denoised, cheaper matrix. It matches `propagate_labels`' default, so clusters and the
propagation of those clusters live in one space.

```python
ft.cluster("l23", n_times=20, name="run")            # pca=0.95, the default
ft.cluster("l23", n_times=20, pca=False, name="run") # raw scaled, full dimensionality
ft.cluster("l23", n_times=20, pca=0.99, name="run")  # keep more variance
```

Two things to watch:

- **A reduced space can split more finely than the full one.** Components carrying
  little variance also carried little separation, so a group held together by their
  combined strength can come apart without them. Read `cluster_count_curve()` after
  changing `pca` rather than reusing a threshold chosen under the other setting.
- **The reduction is degenerate when features are near-perfectly correlated.** If
  every feature separates your groups the same way, 0.95 collapses to one component
  and Leiden will over-split along it. Check with
  `ft.features_pca(mask, explained_variance=0.95).shape[1]` — if that's 1 or 2 on a
  many-feature table, prefer `pca=False`.

A single high-resolution Leiden partition is available too — the quick look before
committing to a full consensus run, and the deliberate over-split that downstream
merging or manifold work starts from:

```python
ft.overcluster("l23", resolution=4.0)   # -> a LabelSet, many small clusters
```

`overcluster` runs on the **raw scaled features** rather than `pca(0.95)`. Pass
`space=ft.space(mask, columns=…)` to put it in the same representation `cluster`
uses, so the two are comparing cells in one space rather than two.

### Seeds and reproducibility

Every stochastic verb is deterministic by default. The table carries a seed
(`FeatureTable(df, features=[...], seed=0)`), and `cluster`, `embed` and
`overcluster` with `seed=None` — the default — derive a per-call seed from it and
the call's *identity*: the verb, the mask, and the run name. The same call on the
same table reproduces exactly; two differently-named runs get different streams; an
explicit `seed=` overrides the derivation.

```python
ft = FeatureTable(df, features=cols, seed=0)
ft.cluster("l23", n_times=20, name="run")     # reproducible without a seed argument
ft.cluster("l23", n_times=20, name="run2")    # a different, equally stable stream
```

The derivation deliberately ignores the *data* — feature values, column lists —
because a seed that moved with the features would confound the comparisons the
validation tools are built for: comparing two feature sets under one run name should
change the features and nothing else. Rename the run when you want a fresh stream;
hold the name when you want a controlled comparison.

What a run actually received is recorded, not remembered. `clus.params` carries the
full producing call, resolved seed included, and persists with a saved analysis —
`ft.cluster(mask, **params)` replays the ensemble bit for bit. Embedding
parameterizations persist in the manifest the same way, which matters because the
fitted models themselves do not.

One cost worth knowing: a UMAP given a `random_state` runs single-threaded, and
`embed(method="umap")` now always passes one. If you want unseeded, parallel UMAP
layout, call the backend directly and register the coordinates with
`add_embedding`.

### Whitening the PCs: `alpha`

Truncating PCA is a rotation plus a truncation, not a reweighting. A block of features
that measures one thing several ways — soma volume, soma area, soma radius; or eleven
percentiles of one depth profile — collapses into a single high-eigenvalue component that
then dominates Euclidean distance exactly as much as the raw block did. With dozens of
engineered features rather than thousands of genes, that does not average out.

`alpha` scales component *j* by `λ_j ** (-alpha/2)`: `0.0` (the default) leaves PCA's own
scaling alone, `1.0` equalises the retained components entirely.

```python
space = ft.space("l23")                     # the frozen scaling + PCA
space.condition_number                      # how ill-conditioned the inputs are
space.spectrum(ft.features("l23", scaled=True))   # scree + discarded-PC kurtosis

for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):   # one PCA fit serves all five
    ft.cluster("l23", alpha=alpha, n_times=20, seed=0, name=f"a{alpha}")
```

`ft.space(...)` caches on the *fit* only — `alpha` is applied as a view over it, so a
sweep is structurally incapable of refitting the space underneath itself. That is what
makes "only the swept axis differs" true rather than intended.

Which features are redundant is worth looking at before choosing an `alpha`:

```python
from cellpax import feature_correlation
sm = feature_correlation(ft.features("l23", columns="analysis"),
                         ft.collections["analysis"].columns)
sm.sizes            # block sizes; a block of 11 is 11 columns counting once
sm.cell_ids         # feature names in dendrogram order
```

Three things to watch:

- **Whitening amplifies the *smallest retained* component most**, so a truncation chosen
  by cumulative variance becomes a discontinuity: the last kept component gets full
  weight and the first dropped one gets none. Pass
  `eigenvalue_floor=ft.space(mask).noise_floor` — the median discarded eigenvalue, an
  estimate of the scale the truncation already judged to be noise — to bound that. At
  `alpha=1` with no floor, a near-degenerate direction can be inflated into the metric;
  the transform raises rather than doing so silently.
- **The clustering space diverges from the space `embed` builds a UMAP in.** At `alpha=0`
  the two are metrically close, since PCA at 0.95 preserves most pairwise distance. At
  `alpha > 0` they genuinely differ, so **labels will look worse on an existing UMAP even
  when the clustering improved.** Score the change on `cellpax.validate`, not on the
  figure — that divergence is the reason the figure stops being evidence.
- **A discarded component with high excess kurtosis is not noise.** `spectrum()` reports
  it per component, with `n_tail_cells`. A near-Gaussian dropped component costs nothing;
  a sharply peaked one is a small group separating along a low-variance direction, and a
  cumulative-variance cut discards it *because* few cells are involved — backwards if
  those cells are a rare type.

### Weighting feature blocks instead of components

`alpha` reweights *components*, which is a blunt way to fix a problem that lives in the
*features*: a block of eleven columns measuring one thing counts eleven times toward every
Euclidean distance. Flattening the spectrum fixes that, but it also upweights low-variance
components — whose directions a finite sample barely determines — and pays for it in
reproducibility.

Block weighting attacks the same redundancy without that cost, because it acts before the
rotation exists and so never touches the component directions:

```python
from cellpax import block_weights

names  = list(ft.collections["analysis"].columns)
scaled = ft.features(mask, scaled=True, columns="analysis")

w = block_weights(scaled, names)                   # one weight per feature
ft.cluster(mask, columns="analysis", feature_weights=w, name="run")

space = ft.space(mask, columns="analysis", feature_weights=w)   # the same fit,
space.label                                        # 'pca(0.95, weighted)'
ft.embed(mask, space=space, method="umap")         # shared by every consumer
```

`cluster` also accepts a prebuilt `space=` outright (like `embed` and
`overcluster`), which is how one weighted or whitened fit provably serves the
consensus, the embeddings, and `boundary_report` together; a passed space
*is* the representation choice, so combining it with `pca=`/`alpha=` raises
rather than silently picking one.

The default `method="mfa"` divides each correlated block by the standard deviation along
its own first principal direction — Escofier & Pagès' multiple factor analysis. That's an
*adaptive* `1/√k`: it collapses to it for a perfectly correlated block and barely touches a
loosely correlated one, so a block of eight at r≈0.3 isn't punished as hard as a block of
eight at r≈0.99. `method="sqrt_k"` is the plain size-based version.

Two things to watch:

- **Pass the weights to `ft.space(feature_weights=…)`, don't multiply by hand.** Weights
  applied outside the space are invisible to it, so `embed(space=…)` and `project` would
  silently skip them and land you in a different space than you fit — with matching shapes
  and no error. Inside the space they're frozen with the fit and persisted with it.
- **A weighted space caches and persists separately** from the unweighted fit of the same
  mask, so you can hold both and compare. `clus.space` records which is which.

`alpha` and `feature_weights` are independent — weighting is pre-PCA, whitening is
post-PCA — so they compose, and `label` reports both (`pca(0.95, weighted, alpha=0.5)`).
Whether either helps is an empirical question; see
[Validating a representation](#validating-a-representation).

### Clipping small cohorts

The default clip is a percentile rule (`0.1`/`99.9`, applied after `RobustScaler`), which
defines its bound by **rank**. That has two costs, and both bite hardest exactly where
the cohorts are smallest.

```python
from cellpax import clip_comparison, clipped_scaler_factory

clip_comparison(ft.features("l23"), ft.feature_columns)   # what each rule would take

ft = FeatureTable(df, features=cols,
                  scaler_factory=clipped_scaler_factory(mode="sigma", n_sigma=4.0))
```

- **The bound stops being robust on a few hundred cells.** The 99.9th percentile has a
  breakdown point near `0.001 × n` — below one cell when `n < 1000`. At n=500
  `np.percentile` interpolates the bound between the top two order statistics, so *the
  outlier partly sets the bound meant to clip it*: one cell at 40 robust units yields a
  bound near 15, and the same cell at 400 yields a bound near 165. The more extreme the
  cell, the weaker its own clipping. By n≈5000 the tail holds enough points that the
  bound settles.
- **It pulls rare populations toward the bulk by construction.** `f × n` cells are
  clipped however clean the data is, so a population that is both rare *and* extreme is
  partly clipped automatically. No choice of `f` removes that.

`mode="sigma"` clips at `±n_sigma` instead: a fixed value from a median and an IQR, which
a handful of extreme cells cannot move, identical at n=500 and n=21000, and clipping
*only* what is actually extreme — possibly nothing. It also has **no fitted parameters**,
so a frozen transform carries no clip bounds for a future dataset to shift.

An existing table doesn't need rebuilding to change rules:
`ft.set_scaler_factory(clipped_scaler_factory(mode="sigma", n_sigma=4.0))` swaps the
rule and lazily refits every scaler — dropping, with a warning, anything computed
under the old scaling rather than serving it against geometry that no longer exists —
and `load_feature_table(folio, name, scaler_factory=…)` is the load-time form, leaving
the saved analysis untouched until you save over it. A/B-ing two rules is swap →
measure → swap back, since refits are lazy and cheap.

Two things to watch:

- **`n_sigma` is in IQR units, not standard deviations.** `RobustScaler` divides by the
  IQR, and IQR ≈ 1.349σ for a Gaussian, so `n_sigma=5` is about ±6.7 Gaussian σ and will
  clip almost nothing on well-behaved features. A useful sweep is nearer 3–5 than 5–10;
  `clip_comparison`'s `max_abs_sigma` column says how extreme anything actually is.
- **While the percentile rule is in use, report the clip percentile against the smallest
  cluster you resolve.** At 0.1% on 21,000 cells the bound sits at ~21 cells against a
  `min_cluster_size=100` cut — clear. On a 3,000-cell cohort cut at `min_cluster_size=10`
  the bound is 3 cells, within a factor of three of the smallest cluster, and the
  pipeline cannot resolve populations of that size by construction.

### Choosing the graph, or marginalising over it

The consensus already marginalises over resolution and seed but treated graph
construction as a fixed upstream choice. `graph_type` makes it a third axis:

```python
clus = ft.cluster(
    "l23",
    graph_type=["knn", "snn_jaccard", "umap_fuzzy"],
    n_neighbors=[15, 30, 60],
    resolution=np.geomspace(0.02, 1.5, 12),
    n_times=5, seed=0, name="run",
)
```

The three weightings have known and *different* failure modes, which is why the choice is
worth marginalising over rather than defending:

- **`umap_fuzzy`** subtracts each cell's distance to its nearest neighbour before
  exponentiating, so every cell keeps at least one full-weight edge and nothing is ever
  fully disconnected. Tends to hold rare and peripheral populations together. Implemented
  in-library rather than imported, so it is a separate object from any UMAP *embedding*'s
  graph by construction.
- **`snn_jaccard`** offers no such guarantee: two cells in a sparse region can be mutual
  neighbours and share almost no neighbourhood, so `prune` can strand them. Denoises
  dense regions hardest and fragments sparse ones. Stranded cells are counted as
  `graph["n_isolated"]` and warned about.
- **`knn`** (unweighted, the default) and **`knn_distance`** as baselines.

Two things to watch:

- **Resolution is not comparable across graph types.** Fuzzy weights, Jaccard values in
  `[0, 1]`, and unit weights put Leiden's RBConfiguration null on three different scales,
  so one geomspaced grid lands at very different granularities per type — and pooling raw
  lets whichever type happened to produce mid-range grain dominate. Select on **realised
  grain** instead, which costs nothing because the runs already exist:

  ```python
  clus.partitions.by_setting()             # median_clusters is the comparable column
  mid = clus.restrict(n_clusters_min=8, n_clusters_max=40)
  ```

- **Check that no one weighting is being averaged in against the rest.** A type that
  systematically disagrees with the pool is evidence about that type, not noise to be
  diluted:

  ```python
  from cellpax import axis_stability
  axis_stability(clus.partitions, clus.cluster_labels(0.5))   # per-type ARI vs the pool
  clus.restrict(graph_type="snn_jaccard")                     # look at one alone
  ```

`ft.graph_provenance()` lists what space and graph every clustering and embedding on the
table received, and warns when a clustering and an embedding on the same mask share both
— the case where comparing their neighbourhoods becomes circular while still producing
agreeable-looking numbers.

### Reading the hierarchy instead of flattening it

The resolution sweep is geomspaced because structure exists at more than one scale.
Collapsing the consensus to one flat vector with a single `fcluster` cut throws that away
again — and in practice produces a hand-tuned `distance_threshold` per cohort.

```python
h = clus.hierarchy()             # -> ConsensusHierarchy
h.merge_table()                  # every merge, annotated with its stability
h.nested_labels                  # one row per cell, level_0 (coarsest) .. level_k
h.nested_levels                  # what each level is: height, frequency, n_clusters
h.cell_stability                 # per-cell max co-clustering frequency
```

Worth being explicit about where the merge annotation comes from: distance here *is*
`max_value - similarity`, and similarity is the fraction of runs that put a pair
together, so under average linkage `coclustering_frequency` is just
`max_value - height`. It is not a separate measurement and does not need recomputing from
the runs. Likewise, per-cell stability is the existing `consensus_strength()`.

What *is* new is telling a type from a subtype, which the pooled frequency cannot express:

```python
clus.merge_support(n_bands=4)     # per merge, support per resolution band
```

A merge holding across coarse and fine runs alike is a type; one appearing only in the
runs fine enough to create it is a subtype. `resolution_min_supporting` is the lowest
band where support passes 0.5.

Each level converts to a `LabelSet` and attaches like any other:

```python
from cellpax import LabelSet
for level in ("level_0", "level_1", "level_2"):
    ft.attach(LabelSet.from_labels(
        h.nested_labels["cell_id"].to_numpy(),
        h.nested_labels[level].to_numpy(),
        unassigned=-1, name=level, mask=clus.mask,
    ))
```

Two things to watch:

- **Nesting is checked, not assumed.** Average, complete and single linkage are monotone,
  so cuts at decreasing thresholds *are* nested and a violation means the linkage method
  is wrong — that raises. Above `min_cluster_size=1` it cannot hold strictly (a cell in a
  cluster too small to keep is `-1` at that level and assigned at others), so the check
  runs over cells assigned in both levels and any residual is warned about with a count.
- **`merge_support` is capped and says so.** Only the top `max_merges` merges are
  annotated, and each group is sampled to `max_block` cells; both are logged. Exact group
  sizes still come from `merge_table()`.

### Triaging a cell that looks misplaced

A dot inside a differently-coloured cloud has two very different explanations, and the
embedding cannot tell them apart. Its feature-space neighbourhood can:

```python
triage = ft.triage_labels(labels, mask="l23", n_neighbors=30)
triage.group_by("verdict").len()
```

| `verdict` | reading |
|---|---|
| `own` | most neighbours share its label — the embedding misplaced it, nothing to fix |
| `other` | most neighbours carry one other label — the label is wrong, or it is a real outlier |
| `mixed` | no label reaches a majority — the consensus was ambiguous; check `cell_stability` |

Which turns "some dots scattered here and there" into three countable groups.

### Scoring cells against the population

`score_cells` is the adapter between the table and the outlier-detector
ecosystem: fit any estimator with a per-sample score on a mask's scaled
features, and the scores land as an ordinary metadata column — maskable,
plottable, persisted with the table, usable as an audit label.

```python
ft.score_cells("l23", columns="analysis", name="iso")     # IsolationForest, seeded
ft.score_cells("l23", scorer=LocalOutlierFactor(n_neighbors=20), name="lof")
ft.add_mask("clean", pl.col("iso") > threshold, based_on="l23")
```

Higher scores read as more normal under every supported protocol
(`score_samples`, `decision_function`, LOF's `negative_outlier_factor_`), so
the tail is `pl.col(name) < threshold`. Scoring never filters — an extreme
cell is either a reconstruction problem or the most interesting thing in the
data, and the whole point is to look before deciding which. One comparison
worth making routinely: score under the full feature set and under a
truncation-safe collection; a cell extreme under one and ordinary under the
other is being flagged by its invalid features — truncation talking, not
biology.

### Is that boundary a gap or a cut?

Two clusters can be perfectly separable and still be two halves of one thing —
any slice through a continuum is stably distinguishable at its ends, which is
why a classifier test certifies every split it is shown (and why the CHOIR-style
resolver earlier versions shipped was removed: it stamped p-values on arbitrary
cuts). The question that matters is whether anything *happens* at the boundary,
and `boundary_report` measures that four ways per cluster pair:

```python
report = ft.boundary_report("run", labels=labels)
report.filter(pl.col("verdict") == "continuous")
```

- **`dip` / `dip_p`** — Hartigan's dip on the pair projected onto the axis
  joining the centroids (via the `diptest` package). Two real modes dip;
  a sliced Gaussian doesn't. Blind to curved boundaries.
- **`connectivity_ratio`** — observed kNN cross-edges against the
  configuration-model expectation (the statistic PAGA built cluster graphs
  on). Read it comparatively across the report's pairs, not against a
  universal constant.
- **`valley_ratio`** — saddle-to-peak density along the boundary. Near 0 is a
  gap; near 1 means the boundary runs through terrain as dense as the
  clusters themselves — the "northwest corner vs southwest corner" signature.
  `density="pak"` upgrades the estimator via the optional `dadapy` extra.
- **`cocluster_cross_mean` / `cocluster_band`** — the ensemble's own read,
  from the consensus matrix: a real gap has its cross-pair co-clustering mass
  pinned near zero, while a cut continuum shows a *band* of intermediate
  frequencies — cells the runs couldn't agree about because there is no fact
  of the matter. No other leg (and no other package) has access to this.

Each leg votes; `verdict` is the majority of the legs that expressed an
opinion, with everything else `"ambiguous"`. The verdict is a summary, not a
result — the columns are the result, the thresholds behind the votes are
parameters, and a verdict worth acting on should hold across a couple of
`n_neighbors` values. A `"continuous"` boundary is not a failure: it is the
signal to stop pretending modes and parametrize the gradient instead (coming
as `ft.parametrize`), or to merge the pair for downstream use.

The hierarchy tools above — `merge_support`, `nested_labels`, per-cell
stability — remain the complementary view: they say *where in the tree*
support lives, while the boundary report says *what kind of boundary* each
split created.

### Parametrizing a continuum

When the report says `"continuous"`, the honest object is a coordinate, not a
better cut. `parametrize` fits a principal curve (Hastie & Stuetzle 1989,
implemented in-library — the Python ecosystem has no maintained home for it)
through the selected cells and returns a `Gradient`: per-cell arc length in
`[0, 1]`, attachable like any label, and convertible back to *named interval
cuts* of a persisted coordinate:

```python
g = ft.parametrize("l23", labels="subclass", clusters=["L2a", "L2b", "L3"],
                   orient_by="soma_depth_um", nuisance=["axon_frac_inside"])
g.loadings()                     # which features vary along the axis
ft.attach(g)                     # a float column, plottable like anything

labels = g.bin(3, names=["upper", "mid", "deep"])   # honest names: declared cuts
ft.attach(labels)
```

Two guards run inside the fit rather than living in the documentation. The
**dimension gate**: the intrinsic dimension is estimated first (TwoNN, with
decimation so noise at the nearest-neighbour scale doesn't masquerade as
dimensionality), and a value that doesn't look like a curve warns before you
compress real structure into an axis. The **nuisance tripwire**: truncation
manufactures fake gradients — partial reconstruction is *graded* feature
loss, so a completeness artifact reads as a smooth biological axis, which is
more insidious than a fake cluster because a continuum is what you are now
primed to accept. Pass completeness metrics as `nuisance=` and a coordinate
that tracks them warns before it acquires a biological name.

`orient_by` fixes the arbitrary sign of the axis against a column (depth,
say), so "0 means upper" survives a refit. Branching topologies are out of
scope for the curve — that is elastic-principal-graph territory
(`elpigraph-python`), a candidate future backend.

## Labels

`ft.label` cuts a stored clustering at a threshold into a `LabelSet`, with
cluster ids 0-based. A `LabelSet` gives clusters identity and clean relabeling:

```python
labels = clus.label(distance_threshold=0.6, name="subclass")   # clus from ft.cluster
labels.rename({0: "L2a", 1: "L2b"})     # or rename(["L2a", "L2b"]) in id order
labels.merge(["L2a", "L2b"], into="L2")
labels.reorder(["L2", "L3"])
labels.set_colors({"L2": "#1f77b4"})
labels.counts()
ft.attach(labels)                       # adds a 'subclass' column (null off-mask)
ft.labels                                # ['subclass'] — names attached so far
```

Those verbs all act on whole clusters. `assign` is the per-*cell* one, for the
handful you have looked at and disagree with:

```python
labels.assign(misfiled_ids, "inh")   # six somas that came out exc and plainly aren't
labels.assign(4172, "Pvalb")         # one id; an unheld name creates the cluster
labels.assign(junk_ids, None)        # the per-cell counterpart of unassign
```

A cluster *name* nothing holds yet is created on the spot; an unknown cluster *id*
raises, since a stray integer is an off-by-one rather than a new cluster. Cells the
set doesn't cover raise too — passing root_ids to a set keyed on cell_ids would
otherwise be a silent no-op, and a silent no-op here is a mislabeled figure later.
Emptying a cluster this way drops it from `ids` and `catalog` but keeps its identity
in `meta`, so moving a cell back restores it with its color; `unassign` is how you
retire one for good.

`assign` edits a labelling in place rather than layering over it, so
[`flatten_labels`](#flattening-labels-across-masks)' `source_name=` provenance won't
show the change. Where the audit trail is the point, keep the curated calls as their
own `LabelSet` and `combine` instead.

`combine` unions label sets into one. Its default, `mode="disjoint"`, is for
labellings of separate populations (e.g. exc + inh clustered separately): the cell
sets must not overlap, and ids are offset so two clusters that happen to share a
name stay two clusters. `mode="priority"` is for layering labellings of the *same*
population — each cell takes the label of the first set that assigned it, and
clusters are matched by name, so `"L5IT"` from two sets becomes one cluster keeping
the first's color. That's what [flattening](#flattening-labels-across-masks) is
built on. A `LabelSet` itself is otherwise ephemeral — `ft.labels` only lists ones
that have been `attach`ed to the table.

### Flattening labels across masks

Labels built over a diverse collection of masks are a stack of columns, each null
where it has nothing to say. `ft.flatten_labels` collapses them into one, taking
each cell's label from the first entry in the list that assigned it — so list them
most-specific-first and the coarser labels fill the holes the finer ones left:

```python
leaf = ft.flatten_labels(["subtype_nn", "family_nn"], mask="exc", name="leaf")
leaf.color_map()          # colors merged out of both, first entry in the list wins
ft.attach(leaf)
```

Entries are attached column names or `LabelSet` objects, mixed freely. `mask` scopes
the result — the label set covers exactly that mask's cells, cells outside it are
dropped even where an entry labels them, and cells inside it that nothing labels are
unassigned, or land in the cluster named by `fill`. Pass `source_name=` to get a
second `LabelSet` back alongside, recording which entry won each cell:

```python
flat, source = ft.flatten_labels(
    ["subtype_nn", "family_nn"], mask="exc", name="leaf",
    fill="unlabelled", source_name="from",
)
source.counts()           # how many cells each label actually contributed
```

An entry that *covers* a cell but leaves it unassigned falls through to the next
one, which is what makes this the way out of [the recursive
descent](#the-recursive-descent).

`attach` writes a name column and an id column, but a column can't hold a color, a
description, or which mask the labels came from — so it records those alongside the
table and `ft.labelset(column)` restores them, before or after a `save`/`load`:

```python
labels.set_colors({"L2": "#1f77b4"}).set_descriptions({"L2": "upper layer 2"})
ft.attach(labels)
ft.save(folio, "l23it")

ft = FeatureTable.load(folio, "l23it")
ft.labelset("subclass").color_map()      # {'L2': '#1f77b4'} — still there
ft.labelset("subclass").mask             # the mask the labels were computed on
```

The table stays authoritative for names: edit the name column and `labelset` picks
that up, matching colors to it by cluster id. `detach` forgets the identities along
with the columns, and attaching over an existing label column raises rather than
quietly leaving a `subclass_right` behind — `detach` first, or pass `name=`.

Relabeling verbs mutate in place, so `labels.copy()` before trying a merge you
might want to walk back. `merge` keeps the lowest id of its members, which leaves
gaps; `compact()` renumbers back to `0..k-1`.

Building a `LabelSet` by hand, names come with it — `names=` takes a list in
cluster-id order or a `{id: name}` mapping, so there's no `Label` ceremony and no
follow-up `rename` pass:

```python
LabelSet(cell_ids, codes, names=["exc", "inh"], name="cls")
LabelSet.from_labels(cell_ids, ["exc", "inh", "exc"])   # already-named values
```

`from_labels` factorizes human-readable values (a cell type column, say) and uses
each value as its cluster's name; `names=` is for when you already have the
integer codes.

### Reading a LabelSet out

Per cluster, in id order — `ids`, `names`, `counts()`, `n_clusters`; the full
table, one row per cluster, is `catalog()` (`id` / `name` / `color` /
`description` / `n_cells`), and `color_map()` gives a `{name: color}` dict to hand
to seaborn's `palette=`. For one cluster, `labels.cluster(key)` takes an id *or* a
name and returns its `Label` — how you read a color or description back out.

Per cell, in `cell_ids` order — `codes` (the integer array, `-1` unassigned),
`to_names()` (the names, `None` unassigned), `assigned` (a boolean mask), plus
`len(labels)` and `n_unassigned`. `to_frame()` is the dataframe view of the same
thing.

```python
labels.cluster("L2").color       # '#1f77b4' — by name or by id
labels.catalog()                  # every cluster: id, name, color, description, n
labels.codes                      # array([0, 0, 1, -1, ...]) per cell
labels.to_names()                 # ['L2', 'L2', 'L3', None, ...]
```

Note `labels.cluster(...)` (a noun: one cluster's identity) is unrelated to
`ft.cluster(...)` (a verb: run a clustering) or `Clustering.label(...)` (cut one into
a `LabelSet`).

### Ordering clusters by a value, not by hand

`reorder` takes an explicit order; `reorder_by` derives one from data — renumbers
clusters so cluster 0 has the lowest (or, with `ascending=False`, highest) mean
(or `agg="median"`) of some per-cell value. Handy for e.g. sorting clusters by
mean soma depth so cluster numbers read top-to-bottom instead of arbitrarily:

```python
ft.reorder_labels(labels, "soma_depth_um")   # any column — feature or metadata
```

`reorder_labels` is a thin wrapper that pulls `column` from the table and matches
it to `labels` by cell id, then calls `LabelSet.reorder_by` — use that directly if
your values live outside the table (e.g. a `{cell_id: value}` dict or a precomputed
array). Do this **before** `ft.attach(labels)`: `reorder`/`reorder_by` renumber
cluster ids in place, so attaching first would leave the table's id column stale.

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

### Fitting a classifier on labels

A `LabelSet` is the id-to-name dictionary you'd otherwise hand-maintain around a
model. Out: `codes_for(cell_ids)` gives `y` aligned to whatever row order your
feature matrix is in (cells the label set doesn't cover come back `-1`, so
`y != -1` is the fittable subset). Back in: `with_codes` turns predicted integers
into a `LabelSet` carrying the same names and colors as the one you trained on.

```python
X = ft.features("l23", scaled=True)
cell_ids = ft.dataframe("l23")[ft.id_column].to_numpy()
y = labels.codes_for(cell_ids)                  # aligned by cell id, not row order

fit = y != -1
model = XGBClassifier(num_class=labels.n_clusters).fit(X[fit], y[fit])

predicted = labels.with_codes(cell_ids, model.predict(X), name="subclass_pred")
ft.attach(predicted)                            # names, not integers, in the table
labels.decode(model.predict(X_other))           # or just ints -> names
```

Scaled features are mask-relative by design — a scaler is fit on the mask it's
asked for, so `features("l23", scaled=True)` expresses variance *within* l23, not
against the whole population. A model fit on one mask's scaled features therefore
only means anything applied to that same mask's scaled features. Fit and predict
within one mask (as above), or `project` new cells into that mask's space (below) to
carry the model onto cells it was never fit on.

### Projecting new cells into an existing space

`project` pushes rows through a mask's *existing* fits without re-fitting anything,
which is what you want when new data arrives and has to land in the space a model or
an embedding already lives in:

```python
ft.embed("l23", method="umap", n_components=2, seed=0)

new = pl.read_parquet("later_batch.parquet")   # same feature columns, by name
X = ft.project(new, "l23")                      # transforms + l23's scaler
coords = ft.project(new, "l23", embedding="umap")   # ...and into that UMAP
model.predict(X)                                 # a model fit on l23's scaled features
```

`data` is a polars frame matched on column *name* (extra columns ignored, order
irrelevant) or a raw array in the resolved column order. Nothing is stored, so it's
safe to call repeatedly.

The two fits behind it are retrievable on their own:

```python
fitted = ft.scaler("l23", columns="stable")   # FittedScaler; .transform(matrix)
fitted.scaler.mean_, fitted.transforms        # the sklearn estimator, and ihs/log/sqrt per feature
ft.embedding_model("l23", name="umap").model  # the fitted UMAP/PCA itself
```

Two things to know. Fitted **models are session-only**: coordinates persist through
`save`/`load` but the estimator behind them does not, since a UMAP fit is neither
small nor reproducible across versions — after a `load` you have the coordinates and
must re-`embed` to project anything new. And a model is **dropped, not kept stale**,
when the mask is redefined or `preprocess`/`add_features` changes the scaled space
underneath it; `embedding_model` then raises rather than silently projecting new
cells into coordinates the stored ones no longer share.

#### Labeling cells that were never in the table

`project_labels` is `propagate_labels` for rows from elsewhere — a second dataset, a
later batch, cells you deliberately kept out. Both populations go through the mask's
*frozen* scaler and PCA, so the reference is judged in exactly the space it was
clustered in, and the table is never widened:

```python
result = ft.project_labels(new_rows, sub, n_neighbors=30)   # sub = a LabelSet on "inh"
result.frame()              # cell_id + name + id + confidence, for the new cells only
result.self_agreement()     # how well the reference recovers itself in this space
```

`new_rows` is a polars frame matched on column name (extra columns ignored) or a raw
array in the resolved column order; ids come from its id column or from `cell_ids=`,
and without either it raises rather than inventing ids that could collide with real
ones. `mask=` defaults to the reference's own. The result covers **just the new
cells** — the reference votes, it doesn't come back — and its `LabelSet` has no mask,
since these cells are in no mask of this table.

The difference from adding the rows and propagating over a widened mask is that
nothing is re-fit: no scaler moves, no clustering is invalidated, and repeated calls
on different batches all land in one comparable space. The cost is that the incoming
cells get no say in the space they're measured in — which is the right trade when the
reference is the thing you trust.

`propagate_labels`' caveats apply, sharpened by distance. `"vote"` casts exactly
`n_neighbors` votes, so *every* incoming cell gets a label however far it sits from
the reference, and `confidence` measures neighbor disagreement rather than distance.
Rows from another dataset are precisely where that bites, so use `method="spread"`
when "none of these" has to be an available answer:

```python
ft.project_labels(stray, sub, cell_ids=[9001]).labels.n_unassigned                  # 0
ft.project_labels(stray, sub, cell_ids=[9001], method="spread").labels.n_unassigned  # 1
```

Validity domains aren't checked here — they're properties of cells in the table, and
these rows aren't in it.

For rows from a dataset that was joined into this table, run them through
`ft.harmonize(rows, dataset)` first, so they arrive in the joined units (see
[Joining datasets](#joining-datasets)).

`features_pca` (the denoising step `propagate_labels` uses) deliberately doesn't
retain its PCA — it's an internal, unnamed space. Use `embed(method="pca")` when you
want a PCA basis you can project into later.

`num_class` wants contiguous classes, so call `compact()` after a `merge` and
before fitting. For held-out splits, `subset(cell_ids)` and `drop_unassigned()`
return new label sets with cluster ids and names preserved, so their codes stay
comparable with the parent's.

## Label propagation

Cluster a high-quality core, then carry those labels out to every cell that looks
like them. `propagate_labels` takes a `LabelSet` (or an attached column name) over
a subset of mask `to`, and gives each cell of `to` the majority label of its
`n_neighbors` nearest labeled cells:

```python
core = clus.label(distance_threshold=0.6, name="subclass")   # clus on mask "exc_core"
core.rename(["L2a", "L2b", "L3"])

result = ft.propagate_labels(core, to="exc", columns="stable", n_neighbors=30)
result.labels             # a LabelSet over all of "exc", named subclass_nn
result.confidence         # neighbor-vote share behind each cell's label
result.self_agreement()   # how often the vote reproduces a known core label
result.frame()            # cell_id + name + id + confidence
ft.attach(result.labels)  # nothing is attached until you say so
```

The propagated set keeps the core's cluster ids, names and colors, so the two
columns are directly comparable — `ft.compare("subclass", "subclass_nn")` means
what you'd expect.

### Vote or spread

`method="vote"` (the default) is the k-nearest-neighbor vote above: cheap, and the
standard baseline — the single-cell field's label transfer is essentially this in
PCA space. Its weakness is that it must cast exactly `k` votes for every cell, so a
cell far from anything labeled still gets a unanimous verdict. Confidence measures
*disagreement among neighbors*, never *distance to them*.

`method="spread"` diffuses labels along a mutual-nearest-neighbor graph instead:

```python
result = ft.propagate_labels(core, to="exc", method="spread", n_neighbors=30)
result.labels.n_unassigned    # cells no label could reach
```

Two things follow that a fixed `k` can't give. Evidence accumulates over however
much labeled signal is actually nearby rather than exactly `k` votes, so a cell in a
dense well-labeled region draws on all of it. And a cell with no mutual path to any
labeled cell receives nothing and stays unassigned — which is the right answer for a
cell that isn't like anything in the reference.

Mutuality is what does the rejecting: a cell that names labeled neighbors which
don't name it back gets pruned out of the graph. That adapts to local density,
unlike a global distance cutoff, which matters when the population you're labeling
is genuinely sparser than the core it came from. The cost is that mutual pruning
also discards some legitimately diffuse cells, and `n_neighbors` is the knob — with
`method="spread"` prefer a larger `k` than you would for the vote. `mutual=False`
falls back to a union graph, which rejects nothing.

`weights="distance"` weights neighbors by inverse distance (the default for
`spread`, available for `vote`), so a close labeled cell counts for more than one at
the edge of the neighborhood. Worth reaching for when clusters differ a lot in size,
since uniform votes favor the larger one near a boundary.

`self_agreement()` is leave-one-out for the vote and fold-wise recovery for spread
(a clamped cell would trivially agree with itself), counting an unreachable cell as
not recovered.

### Validity domains: declaring where features inform

A feature can be measurable for a cell and still be meaningless for it — a
truncated reconstruction yields a dendrite length, it's just not informative about
type. Those features are fine *within* a well-reconstructed core and misleading
outside it, and no importance ranking computed on the core can detect the problem,
because on the core the features behave. It has to be declared, and the
declaration is first-class:

```python
ft.add_mask("axon_complete", pl.col("axon_frac_inside") > 0.8)
ft.set_validity(columns="axon", where="axon_complete")   # or valid_where= on define_features

ft.validity_domains          # {'axon_len': 'axon_complete', ...}
ft.fully_valid(columns="axon")     # per-cell: every axon feature valid here?
ft.validity_patterns()             # the distinct patterns and their sizes
```

A domain is a named mask, so it persists with the analysis, composes with
`based_on`, and can't be dropped out from under the features that reference it.
Truncation is positional, so `validity_patterns()` usually collapses to a
handful of patterns — worth reading once before deciding how to propagate.

Two diagnostics help *build* the domains rather than guess them.
`covariate_sensitivity(features, completeness, names)` ranks features by how
strongly they track a completeness metric — a feature measuring truncation
rather than biology tops the list (near-zero is necessary, not sufficient: rank
correlation misses nonmonotone artifacts). `stratum_shift(features, dataset,
names)` does the cross-dataset version: features that measure the acquisition
rather than the cells, with `median_shift_iqr` separating "recenterable shift"
from "shape-level difference". Same abstraction, different granularity — a
per-dataset domain is just a validity mask that happens to be a dataset stratum.
When the goal is to *remove* such shifts and put the datasets in one space rather than
fence them off, see [Joining datasets](#joining-datasets).

Declared domains have teeth. `propagate_labels` warns (or raises, with
`on_invalid="raise"`) when the chosen columns don't cover the target — the
failure the audit found every safeguard silent on — and `ladder=` uses the
domains per cell:

```python
result = ft.propagate_labels(core, to="exc", ladder=["full", "stable"])
result.rungs                 # which collection labeled each cell
result.rung_recovery         # per-rung self-recovery: what each set still carries
result.frame()               # ..._rung column alongside label and confidence
```

Each cell gets the *richest* collection whose features are all valid for it, so
a global `columns=` choice no longer coarsens every cell to the worst cell's
validity: well-reconstructed cells keep the distinctions the full set supports,
truncated ones fall back to what their features can honestly say, and a cell no
rung covers stays unassigned — the honest answer for a cell whose informative
features don't exist. Each rung is its own propagation with its own scaler and
PCA over only its participants, so invalid values never contaminate a fit. Two
consequences to keep in mind: confidence is comparable within a rung, not across
rungs (each rung is its own space with its own local label competition), and
the information-theoretic limit still stands — two clusters separated *only* by
truncation-sensitive features cannot be told apart on the periphery by any
method. `rung_recovery` is the per-rung version of the old advice to compare
`self_agreement()` across column sets: if the fallback rung's recovery drops
sharply, the honest move is to merge those clusters for the peripheral
population rather than propagate a distinction its features can't support.

**The reference must live inside `to`.** One feature space is fit over all of `to`
(scaling and PCA are both mask-relative), and the reference cells are picked out of
those rows — so propagating from a core that isn't inside the target mask raises
rather than silently comparing two unrelated spaces. `pca=0.95` is the default
space; `pca=False` uses raw scaled features, and `columns=` narrows to a feature
collection.

`self_agreement()` asks how often the reference's own labels come back when they
aren't given. Near 1.0 means the labels are locally coherent in this space and
propagation is interpolating; a low value means it's guessing, which usually points
at the feature set rather than at `n_neighbors`.

### Choosing `min_confidence`

`confidence` is the winning label's share of the support that reached a cell, in
`[0, 1]`, and `min_confidence` unassigns everything below a cut. Read it as a
*plurality margin*, not a probability, and don't pick the number a priori — three
things make any given value mean something different from run to run:

- **The floor moves with local competition.** The winner among `c` labels present in
  a neighborhood can't score below `1/c`, so `0.5` gates nothing along a two-cluster
  boundary and quite a lot in a five-way muddle.
- **With `weights="uniform"` the values are quantized to `1/n_neighbors`.** At
  `n_neighbors=15` confidence only ever takes 15 values, so `0.35` and `0.45` are the
  same cut. Either place cuts on that grid or use `weights="distance"`, which spreads
  them out continuously.
- **It measures ambiguity, not novelty.** A cell far outside the reference gets
  unanimous votes at confidence `1.0`. Gating will never remove it; that is what
  `method="spread"` and `mutual` are for.

So calibrate on the reference, the only cells whose answer you know. Propagate once
with `preserve_labeled=False` — that relabels the reference from its neighborhood
too, making its confidence a winner-share comparable with the periphery's — and read
off what each cut buys:

```python
import numpy as np

probe = ft.propagate_labels(core, to="exc", preserve_labeled=False,
                            n_neighbors=15, columns="stable")
ids = probe.labels.cell_ids
truth, got, conf = core.codes_for(ids), probe.labels.codes_for(ids), probe.confidence
known = truth != -1                     # reference cells; the rest have no answer
wrong = got != truth

for cut in np.unique(conf[conf < 1.0]):  # the cuts that actually do something
    keep = known & (conf >= cut)
    if not keep.any():
        break
    print(f"{cut:.3f}  keeps {keep.sum():4d}/{known.sum()}  "
          f"error among kept {wrong[keep].mean():.0%}")
```

That prints a purity-versus-coverage curve, e.g. for three types over a diffuse
periphery at `n_neighbors=15`:

| cut | reference kept | error among kept |
|---|---|---|
| ≤0.533 | 120/120 | 38% |
| 0.600 | 91/120 | 33% |
| 0.667 | 66/120 | 23% |
| 0.733 | 48/120 | 15% |
| 0.800 | 42/120 | 5% |

Take the loosest cut whose kept-set error you can live with — here `0.8` buys a 5%
error rate for a third of the cells, and anything at or below `0.533` is a no-op.
Then apply it for real with `preserve_labeled=True`, which exempts the reference:
a curated label is not the vote's to discard.

One consequence of that exemption: under `preserve_labeled=True` a reference cell's
`confidence` is the share behind the label it was *given*, not behind the winner, so
a mislabeled curated cell sitting deep inside another cluster reports `0.0` and
keeps its label anyway. That is a useful flag for reviewing the core, but it means
the confidence column is not one population — filter to the propagated cells before
taking a quantile of it.

### Conformal assignment: sets with a certificate

`propagate_labels`' confidence is a plurality margin — useful, calibrated
against nothing. `ft.assign` answers the stronger question — *which labels can
this cell defensibly be given?* — with a finite-sample guarantee attached:

```python
assignment = ft.assign(core, to="exc", columns="stable")

assignment.prediction_set(alpha=0.1)     # labels not rejected at 90% coverage
assignment.set_sizes(alpha=0.1)          # 1 = decisive, 3 = honestly torn
ft.attach(assignment.to_labelset(alpha=0.1))   # singletons keep their label,
                                               # everything else abstains
assignment.coverage(alpha=0.1)           # the self-check, per class
```

The curated reference is split into a training part and a calibration part;
a classifier (any sklearn estimator — the guarantee doesn't depend on its
quality, only the set sizes do) is fit on the first and its surprise on the
second becomes the yardstick. Every cell then gets a **p-value per label**,
and that matrix is the stored evidence: `alpha` is a *read-time* parameter,
so one `assign` call serves every coverage level you later care about — the
same stance the boundary report takes with its thresholds.

Calibration is **Mondrian by class** by default: each label's threshold comes
from calibration cells of that label, so 90% coverage means 90% *for the rare
type too*, not on average. The price is honest arithmetic — a class needs at
least `⌈1/alpha⌉ − 1` calibration cells to back its guarantee, and
`prediction_set` names the classes that fall short rather than letting them
borrow a threshold.

**When the target drifts away from the core** — truncation being the standing
example — the plain guarantee quietly dies on the drifted cells: their scores
exceed everything in calibration and their sets come back empty (measured on
the truncation benchmark: 26% coverage at nominal 90%). Two remedies, both
kept while the toolkit is deliberately overcomplete. Passing
`shift_covariates=["completeness"]` switches to weighted conformal
(Tibshirani et al. 2019): calibration cells resembling each test cell in the
covariates count for more, restoring coverage at the honest price of
conceding both labels where the classes have genuinely merged.
`conditional_prediction_set` (optional `conditional` extra) is the
finer instrument — Gibbs–Cherian–Candès conditional coverage, holding at
*every* completeness level while keeping sets small (88% coverage at mean
set size 0.91 on the same benchmark) — at the cost of a construction-time
`alpha` and an LP per cell.

Two boundaries worth respecting. An **empty set is not novelty detection**:
the guarantee assumes exchangeability with the calibration cells, which is
precisely what a truncated or foreign cell violates — `plausibility()` is a
screen, `score_cells` and the validity machinery are the real tools, and
`assign` warns when its columns' validity domains don't cover the target for
exactly this reason. And the calibrated **probabilities that ride along**
(`assignment.probabilities`) are a different object from the sets:
probabilities rank and average nicely but can be wrong together; the sets
carry the certificate.

### Smoothing and refilling

Same verb, two variations that were workhorses of the old pipeline:

```python
# smooth a noisy clustering: relabel the core cells from their own neighborhoods
ft.propagate_labels(core, to="exc_core", preserve_labeled=False, n_neighbors=50)

# throw out a junk cluster and let its cells be reassigned
core.unassign("L2b")
ft.propagate_labels(core, to="exc_core", n_neighbors=30)
```

`unassign` sends a cluster's cells back to `-1` and drops the cluster, so the next
propagation refills them from their neighbors rather than leaving a hole. Both leave
gaps in the cluster ids; `compact()` closes them.

For the array-level primitives without a table, `propagate_knn(features, codes)` and
`propagate_spread(features, codes)` both return `(codes, confidence, recovery)` and
treat `-1` as unlabeled.

### The recursive descent

Clustering, masking and propagation compose into one loop, which is how a hierarchy
gets built: cluster a curated core, propagate to the population, carve the next
mask out of the propagated labels, and recurse.

```python
coarse = ft.cluster(mask="core", n_times=20, name="coarse")
family = coarse.label(distance_threshold=0.9, name="family")
ft.attach(ft.propagate_labels(family, to="exc", method="spread").labels)

# the propagated column defines where to look next
ft.add_mask("l23it", pl.col("family_nn") == "L23IT")
ft.add_mask("l23it_core", pl.col("is_core"), based_on="l23it")   # stays nested

fine = ft.cluster(mask="l23it_core", n_times=20, name="fine")
subtype = fine.label(distance_threshold=0.5, name="subtype")
ft.attach(ft.propagate_labels(subtype, to="l23it", method="spread").labels)
```

Each round leaves its columns in the table, so the hierarchy is legible at the end:
a cell has a `subtype_nn` only where `family_nn` put it in that branch, and null
elsewhere. Use `based_on` to keep a child mask inside its parent rather than
re-deriving the intersection by hand. When you want the leaves back as a single
column, [`ft.flatten_labels`](#flattening-labels-across-masks) collapses the stack
finest-first.

A predicate that evaluates to null counts as `False`, so masking on a label column
works directly even where propagation abstained — an unassigned cell simply isn't in
the subset. Worth knowing because scaling is per-mask: each round of the descent
rescales within its own branch, which is usually what you want (variance *within* the
family you're subdividing) and is why the reference for each propagation must live
inside that round's target mask.

## Validating a representation

Whitening strength, graph weighting, neighbourhood size and clipping rule are all
parameters with no principled default. A UMAP is the worst available way to choose between
them: it is a lossy 2-D projection, it is usually built in a different space from the one
being compared, and it rewards whichever setting produces visually tidy blobs rather than
reproducible groups. Two criteria, plus one tripwire.

### Subsample reproducibility — the primary criterion

Cluster repeated subsamples under a **frozen** representation and measure how much the
labels agree with the full-data labels. No ground truth needed, and it measures the
property actually wanted: that the structure is a property of the population rather than
of the particular cells in hand.

```python
from cellpax import subsample_stability

space = ft.space("l23")
scores = {}
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    coords = space.with_alpha(alpha).transform_scaled(ft.features("l23", scaled=True))
    scores[alpha] = subsample_stability(
        coords, distance_threshold=0.6, n_draws=10, fraction=0.8,
        n_neighbors=30, resolution=[0.3, 1.0], n_times=2, seed=0,   # held fixed
    )
{a: s.mean_ari for a, s in scores.items()}
```

Three things to watch:

- **The space must not be refit inside a draw.** `subsample_stability` takes
  *coordinates*, so there is nothing for it to refit — but that means you must build them
  outside the loop, from views over one fit as above. Refitting would measure preprocessing
  stability confounded with clustering stability, and those answer different questions.
- **The inner ensemble must be identical across representations.** Keep it small (a couple
  of resolutions and seeds); absolute ARI is then not interpretable, but the *paired*
  comparison is — the same argument that makes a stratified-sample comparison valid.
- **Cost is real.** `n_draws` times the inner ensemble, per representation. Budget it as
  comparable to the main sweep.

A representation whose groups survive dropping a fifth of the cells is describing the
population; one whose groups rearrange was describing these cells. On a genuine continuum
expect a low score — that is the property that makes this a criterion rather than a
formality.

### Recovery of an external label — the cross-check

Leave-one-out kNN recovery of labels the clustering never saw. Any population-wide
external call works.

```python
from cellpax import loo_knn_recovery, paired_recovery

truth = ft.labelset("ct_combo").codes_for(ft._cell_ids("l23"))
reps = {
    f"alpha={a}": loo_knn_recovery(
        space.with_alpha(a).transform_scaled(ft.features("l23", scaled=True)),
        truth, n_neighbors=15, name=f"alpha={a}",
    )
    for a in (0.0, 0.5, 1.0)
}
paired_recovery(reps)     # deltas, discordance counts, exact McNemar p
```

`graph_knn_recovery` does the same along a neighbour graph, which is how a `graph_type`
gets scored rather than a coordinate space. Edge weights there are *similarities* and a
shortest path needs costs, so `cost="neg_log"` (the default) or `cost="complement"` — a
choice that changes the ranking, hence a parameter.

Two things to watch:

- **Differences, not absolute numbers.** Absolute recovery is often not even well defined:
  with a stratified labelled set it depends on the design, while the paired difference
  does not. `paired_recovery`'s McNemar test is what keeps a difference of a handful of
  cells from being read as a result — with a few hundred labelled cells most differences
  between reasonable representations will not clear it, and that is the honest answer.
- **An external call that is itself a classifier's output measures agreement with that
  classifier**, decision boundary and errors alike. It ranks representations; it does not
  certify them. If it disagrees with subsample ARI about the best setting, prefer the
  reproducibility number and say so.

Design weights are supported for a stratified labelled set, but read what the weighting
does before leaning on it:

```python
score = loo_knn_recovery(coords, truth, weights=design_weights, strata=strata)
score.accuracy_ipw       # population estimate
score.by_stratum()       # what the weighting hides
```

A design that oversamples the cells where two classifiers disagree gives those cells
*small* weights, precisely because they were oversampled — so the population estimate is
dominated by the easy stratum and its standard error can swamp the effect being measured.
`by_stratum()` is the informative view when that is the case.

### Purity against an audit label — the tripwire

```python
from cellpax import label_purity
label_purity(h.nested_labels, ft.dataframe("l23")["is_inhib_label_nn"])
```

Per cluster per level, sorted worst-first: `n_audit_labels`, `dominant`, `purity`. A
cluster spanning a boundary the features had no access to is a genuine red flag rather
than something to tune away.

One thing to watch: **it only has teeth where straddling is possible.** Cluster a cohort
that was itself selected on the audit label and every purity is 1.0 by construction, which
is not evidence of anything. It bites when the audit label comes from a different source
than the one that defined the cohort — then the two genuinely disagree about some cells,
and a cluster containing both is telling you something.

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
ft.embed("l23", method="pacmap", seed=0)                    # optional pacmap
ft.embed("l23", method="localmap", seed=0)                  # LocalMAP, same package
ft.embedding("l23", name="pca")                             # stored coordinates
ft.dataframe("l23", embedding="pca")                        # joined onto the tidy view
```

Install the optional backends with `pip install 'cellpax[embeddings]'` (or `[umap]` /
`[pacmap]` individually). `pacmap` pulls `faiss-cpu` and `numba`, and numba constrains the
numpy version — worth knowing before adding it to a pinned environment.

### Choosing a backend

They differ in what they try to preserve, and the difference matters here because these
figures are read to judge clusterings.

- **`umap`** preserves local neighbourhoods. Its inter-cluster distances are *not*
  meaningful — the gap between two clouds says nothing about how different they are — so
  reading a UMAP for "how far apart are these types" is reading something that isn't there.
- **`pacmap`** adds mid-near pairs alongside neighbour and further pairs, and shifts the
  weighting across three optimisation phases, specifically so global layout carries
  information too. That is the one thing UMAP is worst at and that these figures are most
  often used for.
- **`localmap`** (Wang et al., AAAI 2025; shipped in pacmap ≥ 0.8) adjusts the graph
  locally during the final stage, aimed at clearer cluster boundaries. Its benefit is on
  *ambiguous* boundaries — on well-separated blobs it does not separate them more than
  PaCMAP does, so do not expect it to make an easy case look easier.
- **`pca`** is linear, cheap, and exactly reproducible. Worth keeping as the baseline: if
  a nonlinear embedding disagrees with it about something structural, that is worth
  understanding rather than assuming the nonlinear one is right.

Two things to watch:

- **`pacmap` and `localmap` default to `n_neighbors=10`**, against UMAP's 15 and the 30–50
  typically used for the Leiden graphs. Out of the box they give a more local, more
  fragmented picture than the existing UMAPs — not a like-for-like swap. Sweep it.
- **They reduce their own input, conditionally.** With `apply_pca=True` (their default) and
  more than 100 input features, they truncated-SVD to 100 dimensions before constructing
  pairs. Below 100 features nothing happens, so on an 81-column set the flag is not a knob
  at all. Passing `space=` sets it `False` automatically, since reducing an
  already-reduced-and-whitened space would partly undo the weighting. Whatever is in force
  shows in `graph_provenance()` as e.g. `scaled -> tsvd(100)`, so the table never reports a
  space that wasn't used.

Unseeded calls are reproducible anyway: `seed=None` derives a deterministic seed from
the table seed and the call's identity (see
[Seeds and reproducibility](#seeds-and-reproducibility)), so a figure made without
thinking about seeds can still be regenerated. Pass `seed=` explicitly when you want a
particular stream — all three optional backends accept it.

### Two embeddings are better than one

Since the backends fail differently, running two turns a single ambiguous observation into
a comparison. A cell misplaced in *both* is more likely genuinely unusual; one misplaced in
only one is that algorithm's artifact:

```python
ft.embed("l23", method="umap", name="umap", seed=0)
ft.embed("l23", method="pacmap", name="pacmap", seed=0)

from cellpax import neighbor_label_composition
for name in ("umap", "pacmap"):
    coords = ft.embedding("l23", name=name).drop(ft.id_column).to_numpy()
    print(name, neighbor_label_composition(coords, labels.codes)
          .group_by("verdict").len().sort("verdict"))
```

Compare either against `ft.triage_labels(labels)`, which runs the same check in the
*clustering* space — that three-way comparison is what separates "the label is wrong" from
"this embedding placed it badly".

One caution worth stating plainly: a better global layout looks more decisive, with cleaner
gaps and more convincing separation. That invites more confidence in the picture, not less
— and global structure being better preserved is not the same as being reliable enough to
settle whether a clustering is right. The criteria under
[Validating a representation](#validating-a-representation) do not use the embedding at
all, and that is deliberate.

### Getting at the coordinates

Coordinate columns are prefixed with the embedding's name — `umap_core0`, `umap_core1` —
because they share a flat namespace with the metadata in `dataframe()`. Rebuilding those
names by hand means writing the embedding's name three times in one call: twice as a
prefix and once as `embedding=`. `embedding_view` hands back the frame *and* the names
together, so it is written once:

```python
v = ft.embedding_view("l23", name="umap_core", labels=lbl)

sns.scatterplot(**v.xy, data=v.frame, hue=lbl.name, palette=lbl.color_map())
v.frame.filter(pl.col(v.x) > 10)          # coordinates as predicates, not just axes
```

`name` is optional when the mask holds exactly one embedding, and raises listing the
candidates when it holds several — so a defaulted name is never a silent guess.
`ft.embeddings` lists every stored `(mask, name)` pair, which matters because `embed`
derives a name when you don't pass one (`embed(method="umap", columns="analysis")` stores
`"umap_analysis"`).

### Coordinates from elsewhere

`add_embedding` registers coordinates `embed` didn't compute, after which everything else
treats them identically — `embedding_view`, `dataframe(embedding=…)`, persistence:

```python
xy = ft.dataframe(mask).select(["soma_x_um", "soma_depth_um"]).to_numpy()
ft.add_embedding(xy, mask, name="soma_xz", space="anatomical")

v = ft.embedding_view(mask, name="soma_xz")
sns.scatterplot(**v.xy, data=v.frame, hue="ct_combo")
```

The obvious use is a backend `embed` doesn't wrap. But **soma position makes a perfectly
good "embedding"**, and registering it that way lets an anatomical plot reuse the same
labels, joins and colour maps as a UMAP rather than being assembled separately.

Coordinates may be an array in mask row order, or a frame carrying the id column — in
which case row order doesn't matter and the ids are matched. Coverage is checked when you
add them, not when you read them, so a mismatch fails immediately instead of surfacing
later as silently dropped cells.

Two things to watch:

- **Set `space=` to something meaningful.** Nothing else records where the coordinates came
  from, and unlike a fitted model this label persists — `graph_provenance()` reports it
  after a reload.
- **There's no model behind them**, so `embedding_model` raises and `project` can't push new
  cells in. That's the same position a reloaded embedding is in.

On dimensionality: `coords`, `v[i]` and `n_components` describe however many components
the embedding actually has, while `xy` names its own two-dimensionality rather than
pretending the rest do not exist.

```python
v = ft.embedding_view("l23", name="three_d")
v.coords          # ('three_d0', 'three_d1', 'three_d2')
v.n_components    # 3
v.xy              # {'x': 'three_d0', 'y': 'three_d1'}   the first two
v.pair(0, 2)      # {'x': 'three_d0', 'y': 'three_d2'}   a stated choice
```

`x` and `y` are aliases for `v[0]` and `v[1]`, so on a one-component embedding `y` raises
naming the component count rather than returning something misleading.

Two things to watch:

- **The coordinate names come from the stored frame**, not from rebuilding `f"{name}{i}"`.
  That matters after a `load`: coordinates persist while fitted models do not, so a view
  built from the model would work in-session and break on reload.
- **`columns` selects what `scaled` applies to**, not what the frame carries. Every feature
  column is present either way; `columns` narrows which ones come back scaled.

To color a scatter by a `LabelSet` you haven't (or won't) `attach`, join it onto
the embedding directly — both are keyed by cell id, so no attaching or row-order
assumptions needed:

```python
plot_df = ft.embedding("l23", name="pca").join(
    labels.to_frame(id_column=ft.id_column), on=ft.id_column
)
# hand straight to seaborn: x="pca0", y="pca1", hue="subclass"
```

## The AnnData bridge

CellPax deliberately isn't AnnData — masks-as-columns and id-keyed embeddings
remove the alignment bugs a positional index invites — but the single-cell
ecosystem's tools live on AnnData, and re-implementing them here would be the
wrong use of anyone's time. The bridge makes them one call away:

```python
adata = ft.to_anndata("l23", scaled=True)     # needs pip install 'cellpax[anndata]'

import scanpy as sc
sc.tl.paga(adata, groups="subclass")           # borrow the ecosystem...

back = FeatureTable.from_anndata(adata)        # ...and come home
```

Masks travel as `mask_*` boolean obs columns, embeddings as id-aligned
`obsm["X_*"]` entries, feature metadata as `var`, and provenance — which mask,
scaled or raw, per-feature transforms, validity domains, the table seed —
under `uns["cellpax"]`, which is what lets `from_anndata` restore rather than
guess. Foreign AnnData objects import too: obs becomes metadata, obsm entries
become registered embeddings, and you name the id column if there's no
provenance record to read it from.

The reading of the round trip is one-way-ish by design: coordinates,
metadata, masks, transforms, and validity all survive; fitted models,
clusterings and the consensus matrix do not cross (they are cellpax objects
with no AnnData equivalent) — export for the ecosystem's tools, keep the
analysis of record on the folio side.

## Joining datasets

Two volumes run through the same feature extraction still disagree about the same kind
of cell. Synapse sizes come out in different units, a lower detection threshold enriches
one tail, a spine/shaft classifier splits differently, sections compress by a few
percent. Concatenate the rows and scale once, and all of that becomes geometry: the
first thing a clustering finds is the datasets.

`join_datasets` removes the difference *before* the rows meet:

```python
from cellpax import FeatureTable, join_datasets, quantile_scaler_factory

ft = join_datasets(
    {"minnie": minnie, "v1dd": v1dd},        # FeatureTables with the same features
    scaler_factory=quantile_scaler_factory(), # 1st/99th percentile clip, then rank
    strata="subclass",                        # one mapping per (dataset, subclass)
)
ft.describe()
```

Each dataset gets its own scaler, fit on its own cells. Every other dataset is then
mapped **onto the reference dataset** (the first, unless `reference=` says otherwise):
forward through its own fit, back out through the reference's inverse. The joined
values are in the reference's raw units, and the reference's cells are unchanged. After
that the joined table is an ordinary table: it scales per mask, clusters, embeds and
propagates exactly as before.

### Choosing the harmonizer

- **Normalize per dataset, never jointly.** A scaler fit on the pooled rows measures the
  datasets against a mixture of both and leaves the difference in place.
- **A rank transform aligns what linear scaling cannot.** `StandardScaler` corrects
  location and scale, and nothing else. When one dataset's feature is a nonlinear warp
  of the other's (different units compounded by a threshold that enriches one tail), only
  a quantile map removes it. On Minnie vs V1dd, clipping at the 1st/99th percentile and
  then quantile-normalizing each dataset matched 85 of 87 features' marginals (KS < 0.02),
  including features whose raw ratios were 1000×. Clip first: `quantile_scaler_factory`
  does by default.
- Anything with an `inverse_transform` works. `StandardScaler`, `RobustScaler`,
  `clipped_scaler_factory()` and `quantile_scaler_factory()` all round-trip through
  `save`/`load`. Any other scaler works in a session, but `join_datasets` warns that
  `save` will refuse it.

### Stratifying by subclass

A single per-feature map fails when the batch effect changes direction between cell
types. On Minnie vs V1dd, mid-layer IT cells have 0.55× the spine density while L6 cells
have 1.1–1.6×, so no one map fixes both. Subclasses, though, are easy to match between
datasets, and `strata=` gives every `(dataset, subclass)` its own mapping. It keeps the
differences *between* subclasses (they are in the reference's units) and removes only
the difference between datasets *within* each one. On excitatory cells that cut the
mixing gap by 78%.

Strata are compared as strings, so the names must already agree:

```python
minnie = minnie.add_column(
    minnie.dataframe().select(
        pl.col("subclass").replace({"L2IT": "IT", "L3IT": "IT", "L4IT": "IT", "L5IT": "IT"})
    ),
    "stratum_group",
)
```

- **Lump continua** whose internal boundaries the datasets drew differently: L2–5 IT as
  one stratum, rather than imposing an L2/L3 boundary that doesn't match.
- **Reconcile taxonomies** before joining, e.g. V1dd's BPC + MPC together make up Minnie's
  ITC.
- **Reference-only strata pass through.** A population only the reference has (Minnie's
  L2b) keeps its values. A stratum in another dataset that the reference lacks raises,
  because there is nothing to map it onto.
- **Stratify where it pays.** Inhibitory cells in the same comparison were nearly aligned
  by a single global quantile map. Give them one stratum value, and the excitatory cells
  their subclass groups.
- `min_cells=` (default 50) refuses a mapping estimated from too few cells, on either
  side. `fit_mask=` fits on a trusted subset (proofread cells, say) and applies the fit
  to every cell.

### Checking that it worked

`dataset_mixing` asks how much more often a cell's neighbours come from its own dataset
than its stratum's composition predicts. A gap of 0 means the datasets are
indistinguishable in that space.

```python
ft.dataset_mixing()                  # scaled features, stratified by the join's strata
ft.dataset_mixing(pca=True)          # ...the same in the 95%-variance PCA space
ft.cross_dataset_classification("subclass", train="minnie", test="v1dd").by_label()
```

**Compare representations before choosing one.** With a few dozen mostly informative
features there is no noise floor to dilute a covariance difference between the datasets,
and PCA keeps exactly the high-variance directions such a difference creates. On Minnie
vs V1dd the gap was +0.13 in the 85-D feature space and +0.42 in 30-D PCA, and running
Harmony on the PCA *after* stratified quantile normalization made mixing worse. If
`dataset_mixing(pca=True)` is clearly worse than `dataset_mixing()`, cluster with
`pca=False`.

The per-label confusion in `cross_dataset_classification` shows where the datasets still
disagree. Errors that fall on known taxonomy differences are a result about taxonomy,
not a failed alignment. `accuracy_shared` leaves out test labels the training dataset
doesn't have.

Some residual gap is expected, around +0.02 overall and more for some subclasses.
Per-feature maps align marginals, not covariance, and part of what's left is real:
regional biology, or reconstruction survival that selects different cells in each
volume.

### Getting back to the sources

Joined ids are minted in dataset blocks and should never be parsed. Every row keeps its
dataset and original id, and both directions of the lookup are methods:

```python
ft.source_ids(joined_ids)               # cell_id, dataset, source_cell_id
ft.cell_ids_for("v1dd", root_ids)       # original ids -> joined ids

parts = ft.labels_by_dataset("joint_subclass")   # one LabelSet per dataset, original ids
v1dd.attach(parts["v1dd"])                       # write the joint labels back
```

New cells from any dataset reach the joined space through the same frozen mapping,
followed by the joined table's own frozen fits:

```python
rows = ft.harmonize(new_v1dd_cells, "v1dd")      # strata read from its subclass column
ft.project_labels(rows.rename({"root_id": "cell_id"}), "joint_subclass")
```

Metadata, masks, collections, validity domains and attached labels carry over. A dataset
missing a metadata column gets nulls, and one missing a mask gets `False`, with a
warning. Same-named clusters merge. Embeddings, clusterings and spaces do not carry over,
since they were computed in the per-dataset spaces. One caveat when viewing the result:
a `root_id` column now mixes ids from different volumes, so a neuroglancer link needs the
right segmentation for each dataset.

## Persistence

Save the whole analysis under a name in a DataFolio — structured, not flattened.
Many analyses and arbitrary user content coexist in one folio.

```python
from cellpax import list_analyses

ft.save(folio, "l23it")                # table, masks, collections, transforms,
                                       # embeddings, consensus matrices, and
                                       # attached label identities (colors, …)
ft = FeatureTable.load(folio, "l23it")
list_analyses(folio)                   # CellPax analyses in the folio

folio.add("notes/readme", {...})       # your own items live alongside, untouched
```

Every item `save` writes gets a content-derived description (cell/feature counts,
mask, linkage method, …), so `folio.describe()` is legible without reloading the
analysis.

Scalers refit lazily on load (the resolved transforms and scaler choice are
restored, so scaled values reproduce).

A table built by `join_datasets` also saves its per-dataset scalers, as explicit
parameters rather than pickles, along with the dataset and source-id bookkeeping. After
a `load`, `harmonize`, `source_ids` and `labels_by_dataset` work exactly as before. A
join whose scaler cannot be recorded refuses to save before anything is written.

## Plotting

CellPax ships no plotting layer by design — `dataframe(...)`, `embedding(...)`,
and `compare(...).alluvial_frame()` return tidy frames you hand straight to
seaborn/matplotlib, which is easier to tailor per figure.

Two figures are worth the assembly, and the frames for both come out of
`ConsensusHierarchy`.

**Dendrogram with merges coloured by stability.** `dendrogram_frame()` returns the layout
as line segments rather than drawing it, because `scipy.cluster.hierarchy.dendrogram`
computes the layout and draws it in one step, which makes per-merge colouring awkward.
Leaf positions follow `leaf_order`, the same order `sorted_matrix` uses, so the dendrogram
lines up with the consensus block image.

```python
seg = h.dendrogram_frame()
norm = plt.Normalize(0, 1)
for row in seg.iter_rows(named=True):
    ax.plot([row["x0"], row["x1"]], [row["y0"], row["y1"]],
            color=plt.cm.viridis(norm(row["coclustering_frequency"])), lw=0.8)
```

Merges that survive across many resolutions are types; merges appearing only at high
resolution are subtypes. Colour by `coclustering_frequency` for overall stability, or join
`merge_support()` on `merge` and colour by `resolution_min_supporting` to separate the two
directly.

**UMAP coloured by per-cell stability**, as the companion to one coloured by hard labels:

```python
v = ft.embedding_view("l23", name="umap")
frame = v.frame.join(h.cell_stability_frame(), on="cell_id")
sns.scatterplot(**v.xy, data=frame.to_pandas(),
                hue="stability", palette="magma", s=1, ax=ax)
```

The expectation to check: low-stability cells should concentrate along interdigitated
cluster boundaries and along continuous streaks between clusters. Scattered uniformly
instead, the instability is not about boundaries and the cut is not the thing to adjust —
look at `ft.graph_provenance()` and `axis_stability` instead.

The ordered correlation matrix uses the existing `SortedMatrix` shape, so the recipe in
its docstring applies unchanged:

```python
sm = feature_correlation(ft.features("l23", columns="analysis"),
                         ft.collections["analysis"].columns)
ax.imshow(sm.matrix, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(sm.n_cells), sm.cell_ids, rotation=90, fontsize=4)
ax.hlines(sm.boundaries[1:-1] - 0.5, *ax.get_xlim(), lw=0.5)
```
