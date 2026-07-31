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
```

Pass a collection (its name, the object, or a plain list) as `columns=` to
`dataframe`, `features`, `cluster`, `cluster_choir`, and `embed`.

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
at a distance threshold and `cluster_count_curve` sweeps thresholds.

A single high-resolution Leiden partition (for CHOIR, below) is available too:

```python
ft.overcluster("l23", resolution=4.0)   # -> a LabelSet, many small clusters
```

## CHOIR: statistically-validated clusters (no single threshold)

A single agglomeration cut can't be right everywhere — some branches should merge
while others at the same level shouldn't. `ft.cluster_choir` resolves a hierarchy
without a global threshold: following [CHOIR](https://www.choirclustering.com/)
(Sant et al., *Nature Genetics* 2025), it keeps each split only where the two
child clusters are random-forest–distinguishable beyond a permutation null, with a
variance condition so the separation must be *stably* high, and merges the rest.

```python
labels = ft.cluster_choir(
    "run", mask="l23", name="subclass",
    alpha=0.05, min_cluster_size=20,
    n_iterations=100, use_variance=True,   # use_variance=False is less conservative
)                                          # -> a LabelSet, data-driven cluster count
```

Three ways to use it:

- **Prune the consensus tree** — pass the `SimilarityMatrix` (or stored name).
- **Prune any over-clustering** — pass `over_clustering=` a per-cell partition
  (array or `LabelSet`), e.g. from `ft.overcluster` (high-res Leiden) or KMeans; a
  hierarchy is built over the cluster centroids and pruned. CHOIR's design favours
  an intentional over-split.
- **Per-node feature reselection** — with `reselect=True`, the `n_features` most
  variable features *within each subtree* are used for that node's test
  (optionally projected to `n_pcs` PCs), since the features distinguishing coarse
  types differ from those distinguishing fine ones. Selection is unsupervised (it
  uses the node's cells, not the child labels under test), so it doesn't bias the
  test.

```python
over = ft.overcluster("l23", resolution=4.0)
ft.cluster_choir(over_clustering=over, mask="l23", reselect=True, n_features=30)
```

The random-forest test always runs on the feature matrix; the over-clustering and
the hierarchy over it are what you choose (features via Leiden is the standard,
CHOIR-like source; the consensus is a denoised alternative). `use_variance` is the
key anti-over-clustering guard; raise `n_iterations` for more stable decisions.

## Labels

`ft.label` cuts a stored clustering at a threshold into a `LabelSet`;
`ft.cluster_choir` returns one directly. Cluster ids are 0-based. A `LabelSet`
gives clusters identity and clean relabeling:

```python
labels = ft.label("run", mask="l23", distance_threshold=0.6, name="subclass")
labels.rename({0: "L2a", 1: "L2b"})     # or rename(["L2a", "L2b"]) in id order
labels.merge(["L2a", "L2b"], into="L2")
labels.reorder(["L2", "L3"])
labels.set_colors({"L2": "#1f77b4"})
labels.counts()
ft.attach(labels)                       # adds a 'subclass' column (null off-mask)
ft.labels                                # ['subclass'] — names attached so far
```

`combine` unions two label sets over disjoint cells (e.g. exc + inh clustered
separately). A `LabelSet` itself is otherwise ephemeral — `ft.labels` only lists
ones that have been `attach`ed to the table.

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
`ft.cluster(...)` and `ft.label(...)` (verbs: run a clustering, cut it into a
`LabelSet`).

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
within one mask (as above), and use `features(..., scaled=False)` if you genuinely
need a population-level space to carry a model across masks.

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
core = ft.label("run", mask="exc_core", distance_threshold=0.6, name="subclass")
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

### Propagating on features that are valid everywhere

A feature can be measurable for a cell and still be meaningless for it — a
truncated reconstruction yields a dendrite length, it's just not informative about
type. Those features are fine *within* a well-reconstructed core and misleading
outside it, so no importance ranking computed on the core can detect the problem;
it has to be declared:

```python
ft.define_features("stable", columns=[...])   # valid for every cell, not just the core
ft.propagate_labels(core, to="exc", columns="stable")
```

This is a different kind of collection from the ones above — defined by *validity
domain* rather than by family or modality — but it's the same mechanism.

The trade-off is real: fewer features means coarser distinctions can be
transferred, and two clusters separated *only* by truncation-sensitive features
cannot be told apart on the periphery by any method, because the information isn't
there. Comparing `self_agreement()` between the full and stable column sets on the
core is the cheap way to find out before propagating — if it drops sharply, the
honest move is to merge those clusters for the peripheral population rather than to
propagate a distinction the features can't support.

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
ft.cluster(mask="core", n_times=20, name="coarse")
family = ft.label("coarse", mask="core", distance_threshold=0.9, name="family")
ft.attach(ft.propagate_labels(family, to="exc", method="spread").labels)

# the propagated column defines where to look next
ft.add_mask("l23it", pl.col("family_nn") == "L23IT")
ft.add_mask("l23it_core", pl.col("is_core"), based_on="l23it")   # stays nested

ft.cluster(mask="l23it_core", n_times=20, name="fine")
subtype = ft.label("fine", mask="l23it_core", distance_threshold=0.5, name="subtype")
ft.attach(ft.propagate_labels(subtype, to="l23it", method="spread").labels)
```

Each round leaves its columns in the table, so the hierarchy is legible at the end:
a cell has a `subtype_nn` only where `family_nn` put it in that branch, and null
elsewhere. Use `based_on` to keep a child mask inside its parent rather than
re-deriving the intersection by hand.

A predicate that evaluates to null counts as `False`, so masking on a label column
works directly even where propagation abstained — an unassigned cell simply isn't in
the subset. Worth knowing because scaling is per-mask: each round of the descent
rescales within its own branch, which is usually what you want (variance *within* the
family you're subdividing) and is why the reference for each propagation must live
inside that round's target mask.

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

To color a scatter by a `LabelSet` you haven't (or won't) `attach`, join it onto
the embedding directly — both are keyed by cell id, so no attaching or row-order
assumptions needed:

```python
plot_df = ft.embedding("l23", name="pca").join(
    labels.to_frame(id_column=ft.id_column), on=ft.id_column
)
# hand straight to seaborn: x="pca0", y="pca1", hue="subclass"
```

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

## Plotting

CellPax ships no plotting layer by design — `dataframe(...)`, `embedding(...)`,
and `compare(...).alluvial_frame()` return tidy frames you hand straight to
seaborn/matplotlib, which is easier to tailor per figure.
