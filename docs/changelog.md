# Changelog

## Unreleased — conformal assignment

- **`join_datasets(tables, scaler_factory=..., strata=...)`** (also
  `FeatureTable.join_datasets`) puts tables from separate datasets into one space.
  - Each dataset, or each `(dataset, stratum)`, gets its own fit, and every other
    dataset is mapped onto the reference's distribution through the reference's
    `inverse_transform`. Joined values are in the reference's raw units, differences
    between subclasses survive, and the reference's cells are unchanged.
  - Metadata, masks, collections, validity domains and attached labels carry over.
    Ids are freshly minted, and the original ids stay first-class: `source_ids`,
    `cell_ids_for`, and `labels_by_dataset` for writing joint labels back to each source.
  - `harmonize(rows, dataset)` brings new cells into the joined units, ready for
    `project`/`project_labels`.
  - The per-dataset scalers are saved and reloaded.
- **`quantile_scaler_factory(clip=(1, 99))`**: a percentile clip followed by a
  `QuantileTransformer`, the per-dataset harmonizer that aligned 85 of 87 features across
  Minnie and V1dd. Quantile scalers, with or without the clip, now also freeze into
  `FittedSpace` and persist.
- **`dataset_mixing` / `ft.dataset_mixing`**: how much more often a cell's neighbours
  share its dataset than its stratum's composition predicts, overall and per
  `(dataset, stratum)`, in scaled features, PCA or an embedding.
- **`cross_dataset_classification` / `ft.cross_dataset_classification`**: train on one
  dataset's labels and score another's. It returns a `TransferScore` with
  `accuracy_shared` (only labels both datasets have), `confusion()` and `by_label()`.
- **`FittedScaler.inverse_transform`**, and identity inverses on `PercentileClipper` /
  `SigmaClipper`, so clipped pipelines can be inverted. `PercentileClipper` now fits
  NaN-aware bounds: a feature with missing values used to get NaN bounds, which turned
  every clipped value into NaN.

- **`ft.collections.names`** lists the defined feature collections, with
  `len()`, a `catalog()` frame (name / n_features / columns) and a `__repr__`
  that shows the names. The accessor was already iterable, but nothing said so —
  printing it gave `<_CollectionAccessor object at 0x…>`.
- **`ft.project_labels(data, labels)`** — `propagate_labels` for cells that
  aren't in the table. Reference and incoming rows both go through the mask's
  frozen scaler and PCA (`ft.space`), so the reference is judged in the space it
  was clustered in, nothing is re-fit, no mask is widened and no clustering is
  invalidated. Ids come from the frame's id column or `cell_ids=`; the result is
  a `Propagation` over just the new cells, with the reference's cluster ids,
  names and colors. `method="spread"` still abstains on rows near nothing.
- **`LabelSet.assign(cell_ids, to)`** — move individual cells between clusters,
  the per-cell counterpart to the cluster-level relabeling verbs. `to` is a
  cluster id or name, or `None` to unassign; an unheld *name* creates the
  cluster, an unknown *id* raises. Cells the set doesn't cover raise rather than
  being skipped, which is what catches root_ids handed to a cell_id-keyed set —
  previously this took a `codes` / `np.isin` / `with_codes` round trip whose
  failure mode was a silent no-op.
- **`add_column` and `add_mask` squeeze a one-column DataFrame.**
  `ft.add_column(ft.dataframe().select(expr), ...)` used to land as an
  `Array(Boolean, shape=(1,))` column — one length-1 row per cell — and only
  showed up much later, wherever the values were finally read (a `tag_bool_cols`
  segment property, say). A width-1 frame (polars, pandas) or an `(n, 1)` array
  now squeezes to its column; anything wider raises.
- **Save/load/list now run under `folio.pinned()`** (when datafolio provides
  it; a no-op otherwise): one staleness check per batch instead of two cloud
  round trips per item. Together with lazy consensus derivation this is the
  remote-folio load fix — the profiled 66s cloud load spent ~30s rechecking an
  unchanged manifest ~60 times.
- **Load fetches every frame item as one concurrent batch** via datafolio v2's
  `get_many` (sequential fallback when absent, tested identical): every item
  path is knowable from the manifest, so cellpax hands over the list and
  datafolio owns the concurrency. Real `gs://` folio (62k cells, ~20
  clusterings): ~150s originally → 66s (lazy consensus) → 33.8s (pinned) →
  **5.3s** (batched).
- **Fixed: a cross-mask `space=` silently mixed two scalings.** Every
  `space=` consumer (`cluster`, `overcluster`, `embed`, `triage_labels`,
  `boundary_report`, and `project` through a space-built embedding) fed the
  space `transform_scaled(features(mask, scaled=True))` — the *target mask's*
  scaler followed by the *space's* rotation. Correct when the space was fit on
  that same mask (the default), a silent chimera when a parent's space was
  passed for a child mask. All paths now call `space.transform(raw)`, so the
  space applies its own frozen scaler — bit-identical in the same-mask case,
  coherent in the cross-mask case, with regression tests projecting a child
  mask through a parent space. In the same pass, `columns=` became officially
  redundant alongside `space=` everywhere (`embed`/`overcluster`/
  `triage_labels` now match `cluster`): the space carries its columns, and a
  *conflicting* `columns=` raises instead of being silently discarded.
- **Fixed: loading a saved analysis re-derived every consensus matrix eagerly.**
  Manifest v2 stores the runs and derives the n×n matrix on load — right at 21k
  cells, minutes-per-clustering at 100k (measured: 188s for one 100k-cell
  clustering, of which IO was under a second; a notebook storing twenty of them
  loaded in the better part of an hour). The matrix now derives **lazily on
  first use**: a load stores the runs and returns immediately (measured: 0.2s
  for the same clustering), `describe()`/`shape`/provenance never materialize,
  and the first `label()`/`linkage`/`soft_labels` call pays the derivation it
  actually needs. Also fixed the normalization correction inside
  `coclustering_matrix`, which materialized an ``(nnz, n_runs)`` sparse
  intermediate — gigabytes at consensus scale; it now restricts to
  both-ever-dropped pairs and reads a precomputed M·Mᵀ by key search.
- **Added `ft.describe()`** — the session state at a glance, as formatted text:
  masks with sizes and validity-domain markers, collections, validity domains,
  stored clusterings (space, run counts, seeds), embeddings (with whether a
  fitted model is still live or only coordinates survive), attached labels,
  scaler rule, transforms, and the table seed. The in-memory companion to
  `folio.describe()`.
- **Added `ft.set_scaler_factory` and a `scaler_factory=` override on
  `load_feature_table`** — the second shakedown gap: reusing an existing table
  under a different clip rule (percentile → sigma, say) meant rebuilding it and
  losing masks, collections, and label metadata. The swap refits scalers lazily
  and drops everything computed under the old scaling with a warning naming what
  went; the load-time override leaves the folio untouched until saved over.
  Deliberately no per-call factory choice — that would key every cache on the
  factory and resurrect the combinatorial fitting the redesign removed.
- **`ft.cluster` now accepts `feature_weights=` and `space=`** — the gap the first
  workflow shakedown found: block weights reached `embed`/`overcluster`/`project`
  but not the headline consumer. `feature_weights` folds
  :func:`block_weights` multipliers into the space the graph is built in;
  `space=` supplies a prebuilt `FittedSpace`, so one weighted/whitened fit
  provably serves clustering, embeddings, and the boundary report. A passed
  space is the representation choice — combining it with `pca=`/`alpha=`/
  `feature_weights=` raises. `params` records the weight digest (arrays don't
  ride in a JSON manifest; the weighted space itself persists), and
  `boundary_report` resolves weighted runs through the space cache by that
  digest — refusing, rather than silently examining unweighted geometry, when
  the space is unavailable (pass `space=` explicitly then).

- **Added `ft.assign` → `Assignment`**: split-conformal label assignment over a
  curated reference, Mondrian by class so the coverage guarantee holds for rare
  types and not just on average (engine: the `crepes` package, new core
  dependency — numpy/pandas/scipy only). The stored object is the *evidence* — a
  per-cell, per-label p-value matrix — and `alpha` is a read-time parameter:
  `prediction_set(alpha)`, `set_sizes`, `to_labelset` (singletons keep their
  label, ambiguity and no-fit both abstain), `coverage` (the per-class
  self-check), and `frame()` all derive from it. Classes with too few
  calibration cells to back a requested `alpha` are named in a warning instead
  of silently borrowing a threshold; sigmoid-calibrated probabilities ride along
  as a separate, certificate-free object. Deterministic under the derived table
  seed, params recorded. The docstrings are explicit that an empty set is not
  novelty detection and that the guarantee assumes exchangeability — which is
  why `assign` warns when its columns' validity domains don't cover the target.
- **Added shift-aware coverage, two ways, benchmarked.** `ft.assign(...,
  shift_covariates=["completeness"])` computes weighted conformal p-values
  (Tibshirani et al. 2019; in-library, keeps the read-time-alpha evidence
  matrix), and `conditional_prediction_set` adapts MAPIE's
  Gibbs–Cherian–Candès conditional conformal (new `conditional` extra:
  `mapie` + `cvxpy` — deliberately *not* MAPIE's own extra, which drags torch).
  On the truncation-shift benchmark (features drift with completeness, core
  curated from complete cells; nominal 90%): vanilla conformal covers the
  truncated tier at **26%** — undercoverage by silent empty sets; weighted
  restores **100%** by conceding both labels there (mean set size 2.0);
  conditional restores **88%** at mean set size **0.91** — near-nominal
  coverage while staying discriminative, at ~1s per thousand cells and a
  construction-time alpha. Both live behind adapters pending the
  overcomplete-then-prune review.

## Unreleased — the AnnData bridge, the boundary report, and gradients

The verification pass before this phase set the policy: own the bookkeeping and the
consensus-derived statistics, import every algorithm with a maintained home, and
bridge to foreign data models rather than depending on them.

- **Added `cellpax.interop`**: `ft.to_anndata()` / `FeatureTable.from_anndata()`
  bridge the scanpy/scvi ecosystem instead of re-implementing it. Masks travel as
  `mask_*` obs columns, embeddings as id-aligned `obsm["X_*"]`, feature metadata as
  `var`, and provenance (mask, scaled, transforms, validity domains, seed) under
  `uns["cellpax"]`, so a cellpax export round-trips and a foreign AnnData imports.
  `anndata` is a new optional extra, imported lazily.
- **Added `boundary_report`** (module function and `ft.boundary_report`): per
  cluster pair, four independent reads on whether the boundary is a *gap* or a
  *cut through one thing* — Hartigan's dip on the centroid axis (via the new
  `diptest` dependency), kNN cross-edge connectivity against the configuration
  null (the PAGA statistic, implemented in-library), saddle-to-peak density along
  the boundary (in-library kNN proxy, or dadapy's PAk estimator via
  `density="pak"`), and the consensus co-clustering profile (cross mean + the
  intermediate-frequency band — the ensemble's own read, available nowhere else).
  Per-leg votes and a majority `verdict` (`discrete` / `continuous` /
  `ambiguous`); the columns are the result, the verdict is a summary. The table
  wrapper rebuilds the space the clustering was computed in from its recorded
  parameters. This is the replacement for what CHOIR pretended to answer:
  separability certifies every cut, so the report measures the boundary instead.
- **Added `ft.parametrize` → `Gradient`** — the follow-through on a `continuous`
  verdict: a Hastie–Stuetzle principal curve (in-library; the Python ecosystem has
  no maintained implementation) yields per-cell arc length in `[0, 1]` with
  feature loadings, `ft.attach` support, and `bin()` back to named labels as
  *declared interval cuts* of a persisted coordinate rather than pretended modes.
  Two guards run inside the fit: an intrinsic-dimension gate (decimated TwoNN —
  a structure that isn't a curve warns before being compressed into one) and a
  nuisance tripwire (a coordinate that tracks a completeness metric warns before
  the artifact acquires a biological name).
- **Added `information_imbalance` and `feature_relevance`** (Glielmo et al., PNAS
  Nexus 2022; plain rank-based numpy, no jax — dadapy's DiffImbalance is the
  learned-weights upgrade): does one coordinate space predict another's
  neighbourhoods, and which features carry the full space's structure, in
  single-feature and drop-one modes. Notes document the sharp edge the tests
  pinned: after z-scoring, a binary cluster-separating feature carries *less*
  neighbourhood information than a continuous coordinate — "separates my
  clusters" and "carries neighbourhood information" are different claims.
- **Added the `dadapy` optional extra**, pinned to a commit: the 0.3.4 PyPI wheel
  still declares `numpy<2.0` and an exact jax pin, while the repo has since
  unpinned both — the extra moves to `dadapy>=0.3.5` when a fixed release exists.
- `diptest` becomes a core dependency (small, C-backed, maintained).

## Unreleased — validity domains

A feature can be measurable for a cell and still be uninformative about it — a
truncated reconstruction yields an axon length that says nothing about type. That
distinction cannot be recovered from the values, so the library now lets you declare
it, and the declaration has teeth.

- **Added `ft.set_validity(columns=…, where=<mask>)`** (and `valid_where=` on
  `define_features`): a feature's *validity domain* is a named mask — the cells it is
  informative for. Domains persist with the analysis, compose with `based_on`, and a
  mask backing a domain can't be dropped while it does. Read side:
  `ft.validity_domains`, `ft.validity()` (the per-cell-per-feature boolean matrix),
  `ft.fully_valid()` (a collection's per-cell coverage), and `ft.validity_patterns()`
  (the distinct patterns and their sizes — truncation is positional, so expect a
  handful).
- **Added `ladder=` to `propagate_labels`**: an ordered list of collections, richest
  first; each cell is labeled with the first one whose columns are all valid *for it*.
  Each rung runs as its own propagation (own scaler, own PCA, only participating
  cells), so invalid values never contaminate a fit; a cell no rung covers stays
  unassigned. The result records which rung labeled each cell (`result.rungs`, a
  `{name}_rung` frame column) and per-rung recovery (`result.rung_recovery`) — the
  per-rung answer to "what can this feature set still carry?". Confidence is
  comparable within a rung, not across rungs.
- **Propagating on columns whose domain doesn't cover the target now warns** — the
  gap the audit found every safeguard silent on. `on_invalid="raise"` escalates,
  `"ignore"` accepts.
- **Added `covariate_sensitivity`** (per-feature Spearman against a completeness
  metric — the features measuring truncation rather than biology) and
  **`stratum_shift`** (per-feature KS + `median_shift_iqr` across dataset strata —
  the features measuring the acquisition rather than the cells). These are the
  diagnostics that *build* validity domains instead of guessing them; the same
  abstraction covers per-cell truncation and per-dataset portability.
- **Added `ft.score_cells`**: fit any per-sample-scoring estimator (sklearn's
  detectors directly; `IsolationForest` seeded from the table by default) on a mask's
  scaled features and store the scores as an ordinary metadata column. Scores, never
  filters. Full-vs-safe-collection score comparison doubles as a truncation
  diagnostic.

## Unreleased — deterministic by default, honest bookkeeping, CHOIR removed

An audit pass: reproducibility becomes the default rather than an intention, a set of
silently-wrong-number bugs are fixed, and CHOIR is deleted.

**Removed: CHOIR** (`cellpax.choir`, `ft.cluster_choir`, `choir_labels`). Its
random-forest test certifies *separability*, and in morphological feature data
separability is everywhere — a continuum sliced anywhere is stably distinguishable at
its ends, so the test kept essentially every split and lent statistical authority to
arbitrary cuts. A tool whose failure mode is false confidence is worth deleting rather
than fixing. `ft.overcluster` stays (it never depended on CHOIR); the hierarchy
machinery (`merge_support`, `nested_labels`) remains the way to reason about splits
until the planned boundary report lands.

**Added: deterministic-by-default seeds.** `FeatureTable(seed=)` (default 0) is the
root every stochastic verb derives from: `cluster`/`embed`/`overcluster` with
`seed=None` now derive a stable per-call seed from the table seed and the call's
identity (verb, mask, name) — the same call reproduces exactly, different names get
different streams, an explicit `seed=` overrides. Derivation deliberately ignores the
feature list, so comparing two feature sets under one run name holds the seed fixed.
One consequence: `embed(method="umap")` now always passes a `random_state`, which
single-threads UMAP — call the backend directly and `add_embedding` the coordinates if
you want unseeded parallel layout.

**Added: recorded provenance.** `Clustering.params` records the full producing call
(seed included) — replaying it reproduces the ensemble bit for bit — and persists in
the manifest, as do embedding parameterizations and the table seed. `restrict(...)`
records its filters on top of the parent's params. A reproducibility test now pins
same-seed ⇒ same-consensus across `n_jobs`.

**Fixed (silently wrong numbers):**

- The per-feature `log` shift was computed from whatever batch was being transformed,
  not from the fit — the same raw value landed at different transformed values per
  batch, breaking `project`'s core promise. Shifts are now fitted parameters
  (persisted with frozen spaces); out-of-domain values raise instead of going NaN.
- `add_column` could silently overwrite a feature or the id column and invalidated
  nothing, so cached scalers served numbers from data that no longer existed. It now
  refuses (features need `overwrite=True`, which drops the affected fits; the id
  column always refuses in favor of `set_id_column`).
- Stored embeddings and clusterings were never invalidated: after `preprocess()` or a
  mask redefinition they were served as current despite being computed in a scaled
  space that no longer existed. Both are now dropped with a warning naming what was
  dropped (external embeddings, which never depended on scaling, stay).
- Consensus normalization divided by `min(|A_i|, |A_j|)` instead of the documented
  `|A_i ∩ A_j|`, biasing similarity low for pairs of cells dropped in *different* runs.
- `subsample_stability` scored a draw that discarded an entire cluster as ARI 1.0 /
  retention 1.0 — an optimistic bias, since subsampling systematically pushes small
  clusters below `min_cluster_size`. Draw-unassigned cells now count against retention
  and `Stability` reports per-draw assigned fractions.
- Saving two spaces on one mask over different feature subsets silently kept only one
  (`_space_slug` omitted the columns); slugs now digest the column set, and the
  ``1``-component vs ``1.0``-variance distinction is kept in both the cache key and
  the manifest.
- Clustering `cell_ids` were re-derived from the *current* mask on load, so a
  same-size mask redefinition between `cluster()` and `save()` silently re-paired
  matrix rows with different cells. The ids are now stored with the clustering.
- Sentinel codes below ``-1`` wrapped around one-hot indexing in propagation,
  corrupting votes and confidence; `LabelSet` and `propagate_*` now reject them.
- `propagate_spread` pruned zero-distance edges, so exact-duplicate cells were
  disconnected and abstained; duplicates now inherit their twin's label.
- `cluster_labels` (and therefore `nested_labels`) returned scipy's 1-based codes
  while `label()` returned 0-based — the same cut now yields ``0..k-1`` everywhere.
- `restrict()` silently forced `normalize=True`; it now preserves the parent's
  normalization, so thresholds chosen on the parent still mean the same thing.
- `label_purity` reported the unassigned pile as the worst "cluster", burying the real
  red flags; it is excluded and counted in an `n_unassigned` column instead.
- `neighborhood_self_predictions` (and everything routed through it: purity, triage)
  used the self-exclusion strategy the same module documents as unsafe with duplicate
  rows; it now shares the safe positional exclusion.
- `graph_knn_recovery` counted unreachable vertices as *wrong* rather than abstained.
- `embed(method="pca", **kwargs)` silently dropped the kwargs; `space(1)` and
  `space(1.0)` returned the same cached object; `preprocess(skew_screen=False,
  method=…)` discarded the method and wiped every transform (it now applies the method
  to all selected features, and `method=None` is the explicit way to clear);
  `agreement_folds=0` was unreachable through `propagate_labels`; NaNs silently
  zeroed `tie_report` extremes and blanked `clip_comparison` rows;
  `reorder_by(ascending=False)` put value-less clusters first; `compact()` after
  `subset()` discarded empty clusters' colors; `merge(into=)` could create the
  duplicate name `_resolve` later raises on; `Comparison.contingency()` merged
  same-named clusters (now keyed on ids); `compare_many` raised a bare `KeyError` on
  unknown metrics; non-integer cell ids broke `neighborhood_purity`/`triage_labels`;
  `n_components_for(1.0)` reported one more component than exists; a `similarity=False`
  sparse matrix raised `NotImplementedError`; the manifest version was written but
  never checked on load (an unknown version now refuses instead of misreading);
  linkage method typos are rejected at construction instead of surfacing as a
  nesting-violation raise; mask names may no longer contain ``/`` (it corrupted saved
  item paths).

**Added (smaller):**

- `Clustering.soft_labels(labels_or_threshold)` — each cell's mean co-clustering
  frequency with every cluster's members, the ensemble-derived soft assignment behind
  a hard cut. Free, since the consensus matrix already holds it.
- `ft.attach(propagation)` — attaching a `Propagation` directly writes
  `{name}_confidence` alongside the label columns instead of dropping the confidence
  on the floor; `detach` removes it with the pair.
- `confidence_curve(truth, predicted, confidence)` — the kept-fraction vs error-rate
  curve the guide's `min_confidence` calibration recipe computed by hand.
- `ft.cluster(metric=…)` — non-Euclidean neighbour metrics were already supported by
  the graph layer but unreachable from the table API.
- `agreement()` now reports coverage (`n_a_assigned`, `n_b_assigned`, `coverage`), so
  a labelling of 10% of cells no longer scores identically to one of 100%.
- `set_id_column` re-keys stored embeddings and clusterings instead of leaving them
  on the old ids; string cell ids work throughout the label-alignment paths.
- `linkage` warns with a size estimate before allocating the dense condensed matrix
  (~1.6 GB at 20k cells); `hierarchy()` no longer computes its cuts three times.
- Docstrings claiming partitions are session-only (they persist since manifest v2) and
  the phantom `clus.axis_stability` method reference are fixed.

## Unreleased — swept representation, graph, and hierarchy

The three upstream choices the pipeline used to make silently — how PCs are weighted, how
the neighbour graph is built, and where the dendrogram is cut — become swept parameters
with diagnostics, each defaulting to the previous behaviour so the change is inspectable.
Plus two fixes that had made a saved analysis unloadable.

**Fixed:** `embed(method="pacmap"/"localmap")` and `project(embedding=…)` could hang
forever, at 0% CPU and immune to `KeyboardInterrupt`. Both reach faiss, and faiss's wheel
vendors a private `libomp`; sklearn's vendors a second; xgboost's macOS wheel links
`@rpath` against Homebrew's. A process that has imported all three holds three OpenMP
runtimes, and a parallel region opened in one can block on a barrier owned by another —
master and worker pool never find each other. Observed in both directions: faiss's
`IndexHNSW::add` from `embed`, and xgboost's `QuantileDMatrix` construction from unrelated
SHAP code in the same kernel. `Ctrl-C` cannot clear it, because the main thread is parked
in native code and never returns to the interpreter to notice the signal. Both call sites
now pin faiss to one OpenMP thread and restore the previous count afterwards, which
removes the barrier entirely; the cost is unmeasurable at these sizes (30k × 80 embeds in
~5s either way). Restoring matters because pacmap sets the same knob itself whenever it is
given a `random_state` and never puts it back, silently single-threading faiss for the
rest of the session. This guard covers cellpax's own faiss use only — a session that also
wants sklearn and xgboost protected should set `OMP_NUM_THREADS=1` before importing
anything.

**Fixed:** a saved analysis could fail to load. `save_feature_table` wrote each consensus
matrix as a COO triplet frame, whose size grows with the number of nonzero *cell pairs* —
about 600 MB at 21k cells, above DataFolio's eager-load limit, so `load_feature_table`
raised on a folio it had just written. Clusterings now persist as the runs behind them
(`<name>/partitions/<c>` int32 plus `<name>/settings/<c>`), with the matrix derived on
load: strictly more informative and roughly an order of magnitude smaller, growing as
`n × n_runs` rather than `n²`. Manifest version 2; version 1 triplet clusterings still
load, and come back without partitions exactly as before.

**Fixed:** `Clustering.restrict` was dead after a reload, because `Partitions` were
session-only — so the expensive part of clustering was unrecoverable and the grain could
not be revisited. The runs now persist, which also brings back `merge_support` and
`axis_stability` on a reloaded clustering.

**Fixed:** a custom `scaler_factory` round-tripped as `"standard"`. `_scaler_tag` recorded
anything it did not recognise as `"custom"`, which the loader then resolved to plain
standardisation — silently changing every scaled value and therefore every downstream
distance. `clipped_scaler_factory(...)` now carries its configuration so the actual
percentiles or `n_sigma` persist, and a genuinely unrecognisable factory raises at save
time instead. Failing the save is recoverable; silently changing the preprocessing is not.

- Added `ft.flatten_labels`, which collapses labels built over a diverse collection of
  masks into one. Each cell takes its label from the first entry in the list that
  assigned it, so listing them most-specific-first lets coarse labels fill the holes
  the fine ones left — the way out of the recursive descent, where every round leaves
  its own column and nothing put the leaves back together. Cluster identities merge by
  name, so colors and descriptions come along rather than being rebuilt by hand. `mask`
  scopes the result, `fill` names the leftovers, and `source_name` returns a second
  `LabelSet` recording which entry won each cell.
- **Changed:** `LabelSet.combine` is now n-ary and takes `mode`. The default,
  `mode="disjoint"`, is what it always did — cell sets must not overlap, ids are offset,
  same-named clusters stay distinct — so existing calls are unaffected. `mode="priority"`
  is the new contract `flatten_labels` is built on: overlaps are expected, the earlier
  set wins, a set that covers a cell without assigning it falls through, and clusters
  are matched by name so `"L5IT"` from two sets becomes one cluster keeping the first's
  color. Rather than a second way to union labels, since one overloaded `combine_labels`
  is what `LabelSet` was built to replace.
- Added `LabelSet.reindex`, the lenient counterpart of `subset`: a new label set over
  exactly the cells you ask for, whether this one covers them or not. `subset` raises on
  an unknown cell because silently dropping a train/test split is worse than stopping;
  here the missing cells are the point, coming back unassigned or in the cluster named
  `fill`. This is how a label set computed on one population is widened to a whole mask.
- Added `LabelSet.to_series`, the one-column form of `to_frame` for when the rows are
  already lined up and only the names are wanted.
- Added `FittedSpace` (`cellpax.space`), the missing third frozen fit alongside
  `FittedScaler` and `FittedEmbedding`. `features_pca` used to refit PCA on every call and
  throw it away, so the space a clustering was computed in could neither be reapplied to
  new cells nor persisted, and a sweep over a downstream parameter refit the space
  underneath itself. `ft.space(...)` returns it, `ft.features_pca` projects through it, and
  it persists as explicit parameter arrays rather than a pickled estimator — the point of
  freezing a space being to reapply it *unchanged* to a future dataset, which a pickle ties
  to a library version.
- Added `alpha` to `ft.cluster` / `ft.features_pca`: partial whitening, scaling component
  *j* by `λ_j ** (-alpha/2)`. `0.0` is the default and reproduces the previous behaviour;
  `1.0` equalises the retained components. Truncating PCA is a rotation plus a truncation
  and not a reweighting, so a block of features measuring one thing several ways still
  dominates Euclidean distance exactly as much as the raw block did — which with dozens of
  engineered features does not average out. **`alpha` is deliberately not part of the space
  cache key**: it is applied as a view over one fit, so a sweep across whitening strengths
  is structurally incapable of refitting the space, rather than merely intended not to.
- Added `eigenvalue_floor`, and read it before using `alpha=1`. Whitening amplifies the
  *smallest retained* component most, so a truncation chosen by cumulative variance becomes
  a discontinuity — the last kept component gets full weight and the first dropped one gets
  none. `ft.space(mask).noise_floor` (the median discarded eigenvalue) bounds that. A
  near-degenerate retained direction now raises rather than being inflated into the metric
  by a factor of round-off.
- Added `FittedSpace.spectrum()`: the full eigenvalue spectrum with, per component, excess
  kurtosis and a tail-cell count. Fitting *all* components and treating the truncation as a
  view is what makes this free. A discarded component with high excess kurtosis is not
  noise — it is a small group separating along a low-variance direction, which a
  cumulative-variance cut discards *because* few cells are involved.
- Added `SigmaClipper` and `mode="sigma"` on `make_clipped_scaler` /
  `clipped_scaler_factory`, clipping at `±n_sigma` in `RobustScaler` (IQR) units.
  `mode="percentile"` remains the default. The percentile rule defines its bound by rank,
  which costs twice: at a few hundred cells the 99.9th percentile has a breakdown point
  below one cell, so `np.percentile` interpolates between the top two observations and *the
  outlier partly sets the bound meant to clip it* — one cell at 40 robust units yields a
  bound near 15, the same cell at 400 yields one near 165, so the more extreme the cell the
  weaker its own clipping. And `f × n` cells are clipped however clean the data is, which
  pulls a rare-and-extreme population toward the bulk by construction. The sigma rule has
  neither property, and has no fitted parameters at all, so a frozen transform carries no
  clip bounds for a future dataset to shift. Note `n_sigma` is in **IQR units, not standard
  deviations** — IQR ≈ 1.349σ, so `5.0` is about ±6.7 Gaussian σ.
- Added `graph_type` as a third consensus axis on `kneighbor_graph`,
  `fauxnograph_coclustering` and `ft.cluster`: `"knn"` (the default, unchanged and pinned
  byte-for-byte), `"knn_distance"`, `"snn_jaccard"`, and `"umap_fuzzy"`. The three
  weightings have known and different failure modes — the fuzzy set subtracts each cell's
  distance to its nearest neighbour so nothing is ever fully disconnected, while Jaccard
  offers no such guarantee and can strand cells in sparse regions — so the choice is worth
  marginalising over rather than defending. `umap_fuzzy` is implemented in-library rather
  than imported from `umap-learn`, which keeps the dependency optional and makes the Leiden
  graph a separate object from any UMAP *embedding*'s graph by construction, so comparing
  feature-space with embedding-space neighbourhoods cannot become circular.
- **Changed:** `cluster_leiden` now passes edge weights when the graph carries them.
  Ignoring them would quietly turn every weighting back into an unweighted graph.
  `graph_type="knn"` carries none, so its behaviour is unchanged.
- **Changed:** `Partitions` gained `graph_type` (defaulting to `"knn"`, so older runs and
  reloads still describe themselves) and its `filter` gained `graph_type=`,
  `n_clusters_min=` and `n_clusters_max=`; `Clustering.restrict` forwards all three. The
  cluster-count filters exist because **resolution is not comparable across graph types** —
  fuzzy weights, Jaccard values in `[0, 1]` and unit weights put RBConfiguration's null on
  three different scales, so one geomspaced grid samples very different granularities per
  type and pooling raw lets whichever type landed mid-range dominate the consensus.
  Selecting on realised grain costs nothing, since the runs already exist.
- Added `axis_stability`, reporting per-graph-type (or per-`n_neighbors`, or
  per-resolution) ARI against the pooled consensus. Marginalising over a choice is only
  honest if the contributions are comparable; a weighting that systematically disagrees is
  evidence about that weighting, not noise to dilute.
- Added `ft.graph_provenance()`, one row per clustering and embedding with its mask, space,
  graph type and neighbourhood size — and a warning when a clustering and an embedding on
  the same mask share both. Checking embedding-space neighbourhoods against feature-space
  neighbourhoods is a real diagnostic only while the two are built separately; build them
  from one graph and the comparison becomes a tautology that still produces agreeable
  numbers.
- Added `Clustering.hierarchy()` → `ConsensusHierarchy`, with `merge_table()`,
  `nested_labels` / `nested_levels`, `cell_stability`, and `dendrogram_frame()`. The
  resolution sweep is geomspaced because structure exists at more than one scale, and a
  single `fcluster` cut discards that — in practice producing a hand-tuned
  `distance_threshold` per cohort. Worth knowing where the merge annotation comes from:
  distance *is* `max_value - similarity`, so under average linkage
  `coclustering_frequency` is exactly `max_value - height`, not a separate measurement.
  Likewise per-cell stability is the existing `consensus_strength()`, now also reachable as
  `cell_stability()` and `cell_stability_frame()`.
- Added `Clustering.merge_support()`, which is the genuinely new part: per merge, the
  fraction of runs *per resolution band* that kept the two groups together. A merge holding
  across coarse and fine runs alike is a type; one appearing only in the runs fine enough
  to create it is a subtype. Both of its caps (`max_merges`, `max_block`) are logged rather
  than silent, since a bounded diagnostic that does not say what it bounded reads as
  complete.
- **Changed:** `nested_labels` asserts nesting rather than assuming it. Average, complete
  and single linkage are monotone, so cuts at decreasing thresholds *are* nested and a
  violation means the linkage method is wrong — that raises. Above `min_cluster_size=1` it
  cannot hold strictly, since a cell in a cluster too small to keep is `-1` at that level
  and assigned at others, so the check runs over cells assigned in both and any residual
  warns with a count. A cut where `min_cluster_size` dropped *every* cluster is skipped
  instead of emitted, since an all-unassigned column reads as a granularity.
- Added `neighbor_label_composition` and `ft.triage_labels`, classifying each cell by what
  its feature-space neighbours are labelled: `own` (the embedding misplaced it), `other`
  (the label is wrong or it is a real outlier), or `mixed` (the consensus was ambiguous —
  cross-check `cell_stability`). Turns "some dots scattered here and there" into three
  countable groups.
- Added `cellpax.diagnostics`: `feature_correlation` (returning a `SortedMatrix`, so the
  existing imshow recipe applies and redundant blocks are contiguous), `tie_report`,
  `duplicate_rows`, and `clip_comparison`. Expect `duplicate_rows` to find nothing across a
  few dozen continuous morphometrics — that answer is worth pinning rather than assuming,
  and it changes on a small subset of discrete features. `clip_comparison` is the one to
  read: it says how many cells each clipping rule takes at *this* cohort's size.
- Added `cellpax.validate`. `subsample_stability` is the primary criterion — cluster
  repeated subsamples under a frozen representation and measure agreement with the
  full-data labels — because it needs no ground truth and measures the property actually
  wanted. It takes coordinates rather than a table precisely so the representation cannot
  be refit inside a draw, which would confound preprocessing stability with clustering
  stability. `loo_knn_recovery` (built on the existing `propagate_knn`, so the
  leave-one-out vote is not reimplemented), `graph_knn_recovery`, and `paired_recovery`
  cover external-label recovery, reported as paired differences with an exact McNemar test
  — absolute recovery is often not well defined, and the paired test is what keeps a
  difference of a handful of cells from reading as a result. `label_purity` is the tripwire.
  Design weights are supported and documented with their catch: a design that oversamples
  the cells where two classifiers disagree gives those cells *small* weights, so the
  population estimate is dominated by the easy stratum; `RecoveryScore.by_stratum()` is the
  informative view when that happens.
- **Changed:** `ft.overcluster` and `ft.cluster_choir` take an explicit `space=`. They run
  on raw scaled features while `ft.cluster` reduces to `pca(0.95)`, which used to be an
  undocumented asymmetry; passing `ft.space(mask)` puts them in the clustering space
  instead. Named `space=` rather than `alpha=` because `cluster_choir.alpha` is already
  CHOIR's significance level. `ft.embed` takes `space=` for the same reason, so a whitened
  UMAP is available under its own name without displacing the separately-built default.
- Added `feature_weights` to `FittedSpace` and `ft.space(...)`, plus
  `block_weights` — per-feature multipliers applied before the PCA, so a block of eleven
  columns measuring one thing counts once rather than eleven times. This is the
  better-targeted instrument for the redundancy that motivated `alpha`: whitening flattens
  the whole spectrum and pays for it by amplifying low-variance components, whose
  *directions* a finite sample barely determines, whereas weighting acts on the features
  and never touches those directions. Default `method="mfa"` divides each block by the
  scale of its own leading direction (Escofier & Pagès), an adaptive `1/√k` that leaves a
  loosely correlated block nearly alone. The weights live *inside* the space — applied by
  `transform`, carried through `to_records`, reflected in `label` — because weights applied
  outside it are invisible to `embed(space=…)` and `project`, which would then silently
  build a different space with matching shapes and no error.
- Added `ft.add_embedding(coords, mask, name=…, space=…)`, registering coordinates
  computed elsewhere so `embedding_view`, `dataframe(embedding=…)` and persistence treat
  them like any other embedding. Beyond backends `embed` doesn't wrap, **soma position
  makes a perfectly good embedding** — registering it that way lets an anatomical plot
  reuse the same labels, joins and colour maps as a UMAP. Accepts an array in mask row
  order or a frame matched on the id column, and validates coverage at add time rather than
  letting a mismatch surface later as silently dropped cells. The `space` label persists,
  unlike a fitted model, since it is provenance the caller supplied and nothing else
  records it.
- **Fixed:** the diagnostics functions silently accepted a collection *name*. They take a
  feature matrix rather than a `FeatureTable`, so they cannot resolve `"analysis"` — and
  `list("analysis")` becomes eight one-character "names", which the width check catches
  only when the widths happen to differ. Now a `TypeError` naming the mistake.
- **Fixed:** the space cache key used `hash()` on the weight bytes, which is salted per
  process, so a reloaded weighted space would never match a fresh request for the same
  weights. Uses a stable digest instead.
- Added `ft.embedding_view(...)` returning an `EmbeddingView` — the tidy frame *and* the
  coordinate column names inside it. Coordinate columns are prefixed with the embedding's
  name because they share a flat namespace with the metadata in `dataframe()`, so
  reconstructing them by hand meant writing that name three times in one call: twice as a
  string prefix and once as `embedding=`. Now once. `xy` splats into any call taking
  `x=`/`y=`, and `v.frame.filter(pl.col(v.x) > 10)` covers the sites that use coordinates
  as predicates rather than as axes. `name` is optional when a mask holds exactly one
  embedding and raises listing the candidates otherwise, so a defaulted name is never a
  silent guess.

  Dimensionality is described rather than assumed: `coords`, `v[i]` and `n_components`
  cover however many components exist, `xy` names its own two-dimensionality, and
  `pair(i, j)` makes any other projection a stated choice — `embed(n_components=3)` already
  worked and was silently plotted as its first two axes. `x`/`y` are aliases for `v[0]`/
  `v[1]`, so a one-component embedding raises on `y` naming the count rather than
  returning something misleading. The names come from the stored coordinate frame rather
  than being rebuilt as `f"{name}{i}"`, which is what makes the view survive a reload:
  coordinates persist, fitted models do not.
- Added the `ft.embeddings` property, listing stored `(mask, name)` pairs. `masks`,
  `collections` and `transforms` were already properties, and its absence mattered because
  `embed` derives a name when one is not given — so that name previously had to be guessed
  from how `embed` built it before anything could read the coordinates back.
- **Fixed:** `graph_provenance()` reported no embeddings at all on a reloaded table. It
  iterated the fitted models, which are session-only, rather than the coordinate frames,
  which persist — the same root cause as the ergonomics problem above. Embeddings now
  always appear; `space`, `graph_type` and `n_columns` are null after a reload, since the
  fit that knew them is gone and null is the honest answer.
- Added `method="pacmap"` and `method="localmap"` to `ft.embed`, alongside the existing
  `"pca"` and `"umap"`. PaCMAP (Wang et al., JMLR 2021) weights mid-near pairs against
  neighbour and further pairs across three optimisation phases so that *global* layout
  carries information — which is the one thing UMAP is worst at and, in practice, exactly
  what these figures get read for. LocalMAP (AAAI 2025, shipped in pacmap ≥ 0.8) adjusts
  the graph locally in the final stage for clearer boundaries; measured on well-separated
  blobs it does *not* separate them more than PaCMAP, so its benefit is on ambiguous
  boundaries rather than easy ones. Both accept `seed=` for exact reproducibility, which
  matters more than it sounds: an embedding nobody can regenerate cannot be compared
  against anything.
- Added `[project.optional-dependencies]`, which did not exist — `umap-learn` was a lazy
  import declared nowhere at all. Now `cellpax[umap]`, `cellpax[pacmap]` and
  `cellpax[embeddings]`, with the ImportErrors naming the extra to install. Note `pacmap`
  pulls `faiss-cpu` and `numba`, and numba constrains the numpy version, so it is a real
  cost to a pinned environment rather than a free addition.
- **Changed:** `FittedEmbedding` gained `internal_reduction`, and `graph_provenance()`
  reports it. This exists because pacmap and localmap **reduce their own input**: with
  `apply_pca=True` (their default) and more than 100 features they truncated-SVD to 100
  before constructing pairs, so the space they built pairs in is not the space they were
  handed. Reporting only the latter would have made the provenance table state something
  untrue. Below 100 features the flag does nothing at all, so on an 81-column set it is not
  a knob. Passing `space=` sets it `False` automatically, since reducing an
  already-reduced-and-whitened space partly undoes the weighting — an explicit
  `apply_pca=True` still wins.
- pacmap and localmap are constructed with `save_tree=True` so `ft.project(...,
  embedding=name)` works. Their `transform` otherwise raises, requiring the original
  training matrix to be passed back as `basis=`, which `FittedEmbedding.transform` has no
  way to supply. The cost is keeping the neighbour index in memory.
- **Documented:** `umap-learn` does **not** reduce its input before the neighbour search —
  the graph is built on whatever matrix it is handed, and PCA enters only as an optional
  layout initialisation (`init="pca"`; the default is `"spectral"`). The impression that it
  does comes from scanpy, where `sc.pp.neighbors` computes the graph in PCA space first. At
  `alpha=0` the clustering and embedding spaces are metrically close anyway; at `alpha > 0`
  they diverge, so **labels will look worse on an existing UMAP even when the clustering
  improved.** That divergence is exactly why the parameter choice has to be scored on
  `cellpax.validate` rather than on the figure.
- **Fixed:** `__version__` in `__init__.py` disagreed with `pyproject.toml`, which would
  have made `poe bump` fail its version search under `ignore_missing_version = false`.

## 0.1.0 — FeatureTable redesign

CellPax was rebuilt around a flexible, polars-native `FeatureTable` container as
the successor to `dendritic_feature_clustering` (dfc), replacing the earlier
immutable content-addressed `Study` design (preserved on the `master` baseline).

- Added the `FeatureTable` container: polars-native construction, named
  hierarchical masks (`add_mask`, `based_on`), `add_column`, and a
  `dataframe(mask, scaled=…)` view backed by lazy, single, per-mask scalers.
- Added composable `FeatureCollection`s (set algebra; `define_features` by
  columns / family / modality / predicate over feature metadata).
- Added a unified `preprocess()` layer: heavy-tail (`ihs`) skew screening applied
  before per-mask scaling, replacing dfc's dual skew paths.
- Ported the fauxnograph kNN/Leiden consensus library and `SimilarityMatrix`
  (hierarchical linkage, cluster labels at a distance threshold, count curve);
  wired `ft.cluster(...)`.
- Added `LabelSet` — clear named clusters with `rename` / `merge` / `reorder` /
  `set_colors` / `combine`, `ft.label` / `ft.attach`, and `IntEnum` bindings
  (`to_enum` / `apply_enum`) for number- and name-free filtering with autocomplete.
- Added `ft.embed` (PCA-native, UMAP optional) and `dataframe(embedding=…)`.
- **Changed:** `ft.cluster` now returns a `Clustering` — a `SimilarityMatrix` that
  also carries the `mask` it covers, that mask's `cell_ids` in row order, the
  `columns` compared and the `space` (`'pca(0.95)'` / `'scaled'`). It cuts itself
  into labels with `clus.label(distance_threshold=…)`, completing the chain
  `ft.cluster` → `Clustering` → `LabelSet` → `ft.attach`. Because it subclasses
  `SimilarityMatrix`, existing use (`linkage`, `cluster_labels`,
  `cluster_count_curve`, `cluster_choir`, persistence) is unaffected, and
  `ft.label(sim, mask=…)` still works for matrices built outside the table.
  Clustering rows are matched to mask members positionally, so `ft.label` and
  `ft.cluster_choir` now take the mask from a `Clustering` and raise on a
  conflicting `mask=` instead of silently mislabeling. Provenance survives
  `save`/`load`; analyses saved before it existed load as `mask="all"`, matching
  what they previously assumed.
- **Changed:** `ft.cluster` now reduces with PCA before building the kNN/Leiden
  graph, `pca=0.95` by default (matching `propagate_labels`, so clusters and their
  propagation share one space). This is the space phenograph-style clustering is
  conventionally run in. Pass `pca=False` for the previous full-dimensional
  behavior. Existing clusterings will shift and a pinned distance threshold may cut
  differently — re-read `cluster_count_curve()`. Note the reduction is degenerate
  when features are near-perfectly correlated (all separating groups the same way):
  0.95 collapses to one component and Leiden over-splits along it, so check
  `features_pca(...).shape[1]` and prefer `pca=False` there.
- Added projection of new cells into an existing space: `ft.project(data, mask,
  embedding=…)` runs a mask's frozen transforms/scaler (and optionally its fitted
  embedding) over rows that were never in the fit, plus the accessors behind it —
  `ft.scaler(mask, columns=…)` → `FittedScaler` and `ft.embedding_model(mask,
  name=…)` → `FittedEmbedding`. Embedding coordinates persist but their estimators
  are session-only, and are dropped rather than left stale when a mask or the
  preprocessing changes.
- Added cross-approach comparison (`compare` / `compare_many`): contingency,
  ARI / NMI / FMI / Jaccard, and alluvial frames.
- Added CHOIR-style statistically-validated cluster resolution (`ft.cluster_choir`,
  `cellpax.choir`): keep each split only where the children pass a random-forest
  permutation test (variance-adjusted); optional per-node feature reselection; and
  `ft.overcluster` (high-res Leiden) so any over-clustering can be pruned, not just
  the consensus tree.
- Added first-class DataFolio persistence: `ft.save` / `FeatureTable.load` store
  many analyses per folio under a `<name>/…` namespace, structured (not
  flattened) and coexisting with arbitrary user content; `list_analyses`.
- Added `ft.cluster(..., order_by="soma_depth_um")`: the column is carried onto the
  `Clustering`, so every `clus.label(distance_threshold=…)` cut comes back with
  cluster ids already ordered by that column's per-cluster aggregate (`order_agg`,
  `order_ascending`; `order=False` at cut time for raw dendrogram order). Ordering
  at cut time rather than after the fact is what keeps two thresholds of one
  clustering numbered consistently. `ft.label` routes through it, and the ordering
  survives `save`/`load` (the column name is stored and re-read, so values can't
  drift). `reorder_by` now ignores nulls instead of poisoning the aggregate with
  `NaN`, and sorts a value-less cluster to the end.
- Added `clus.sorted_matrix(...)` → `SortedMatrix`: the co-clustering matrix
  permuted into cluster blocks, which is the picture the consensus matrix is for.
  Blocks come from a `LabelSet`, a plain code array, or a fresh `distance_threshold`
  cut; rows keep dendrogram leaf order within a block so substructure stays visible;
  unassigned cells form a trailing block. Carries `order`, `codes`, `cell_ids`,
  `names`, `boundaries`/`centers` for tick and divider placement, and
  `block_means()` — the `(k, k)` cohesion/confusion summary that names which two
  clusters a higher threshold would merge first. Dense, so `max_cells` refuses to
  silently allocate a huge array and `subsample` samples proportionally across
  blocks.
- Added `clus.threshold_scan(...)`, a per-threshold table of `n_clusters`,
  `n_assigned` / `n_unassigned` and the largest/median cluster size. Choosing a cut
  on cluster count alone hides how many cells the cut keeps.
  **Read it at the `min_cluster_size` you intend to cut at**: at the default of 1
  nothing is dropped and every singleton counts as a cluster, so a curve read at 1
  and a cut made at 10 routinely disagree threefold. `cluster_count_curve` takes the
  same argument and always did.
- Added `clus.consensus_strength()`: each cell's strongest co-clustering with any
  other cell. Cells scoring ~0 never grouped with anything in any kNN/Leiden run, so
  they sit at maximum distance from everything and can only merge in the dendrogram's
  final collapse — no `distance_threshold` folds them into a neighbour, and they are
  what `min_cluster_size` drops.
- Added `Partitions` — the individual kNN/Leiden runs behind a consensus, kept on
  `clus.partitions` (session-only; recomputing them *is* the cost of clustering, so
  they aren't persisted and a reloaded `Clustering` has `None`). Restores what dfc
  exposed as `return_dataframe=True` and the port had dropped, plus `summary()` per
  run and `by_setting()` per `(n_neighbors, resolution)`. The consensus can never be
  coarser than the runs it averages, so when a matrix comes out sparser or more
  finely split than expected this is where the answer is — and it is not visible in
  the consensus itself.
- Added `clus.restrict(resolution_max=…, resolution_min=…, n_neighbors=…)`, which
  re-consenses from a subset of the existing runs and returns a new `Clustering`
  (mask, cell ids, columns, space and `order_by` carried over). Sweeping a wide
  resolution range does **not** give you both grains: cells of one broad group
  co-cluster only in the runs coarse enough to hold them together, so the broad
  blocks arrive at that fraction rather than at 1.0, the threshold needed to recover
  them sits against the point where the dendrogram collapses, and low thresholds
  strand cells in small clusters. Narrowing to the coarse runs puts the structure
  back at full strength for free. Also `Partitions.filter(...)` /
  `Partitions.coclustering(...)` for the same thing at the array level, and
  `fauxnograph_coclustering(..., return_partitions=True)`.
- Added `p_adjust` to `ft.cluster_choir` / `choir_labels`, defaulting to
  `"bonferroni"` to match upstream CHOIR. The port previously tested every split at
  an uncorrected `alpha`, where upstream re-decides all of them against
  `alpha / n_comparisons` once the tree walk has fixed the denominator
  (`combineTrees.R`). Implemented as upstream does it, in two passes: walk at the
  unadjusted alpha to enumerate the comparisons, then re-decide from cached
  statistics, so no forests are refit. `p_adjust="none"` restores the old behavior.
  Expect fewer clusters than before by default.
- **Documented** a property of CHOIR worth knowing before trusting a fine split: the
  null permutes cluster *labels*, so it sits at chance for any spatially contiguous
  pair, and the test measures whether two groups are *distinguishable* rather than
  whether a boundary separates them. Two halves of a gradient — or of an isotropic
  Gaussian — are distinguishable and are kept. Upstream's remedy, `countsplit`, is
  off by default there and relies on Poisson thinning of counts, so it has no
  analogue for continuous features. Pinned in `test_choir.py` so a future change to
  the null shows up rather than passing silently.
- Added `ft.attach(labels, overwrite=True)`, which swaps an attached label pair in
  place. `LabelSet`'s default name is `"label"`, so re-cutting a clustering collides
  with the previous cut immediately, and a `detach` every time round the loop is
  friction with no safety value once you've said you mean it. Without `overwrite` it
  still refuses.
- **Fixed:** `attach`'s clash error advised `detach(name)` even when `detach` would
  refuse — it required *both* `{name}` and `{name}_id`, so a half-written pair (or a
  column of your own that happened to collide) left no way forward. `detach` now
  drops whichever of the pair exists, and the error distinguishes an attached label
  from a foreign column instead of calling both the same thing.
- **Fixed:** `dataframe(labels=…)` silently lost to an attached column of the same
  name. Polars suffixes a colliding join column `_right`, so the frame came back with
  the *attached* column under `labels.name` and the `LabelSet` you passed hidden in
  `{name}_right`. Every plotting helper reads `labels.name`, so previewing a new cut
  under a name already attached — cut, look, re-cut, the normal loop — drew the old
  labelling and silently dropped every cell it didn't cover, while the `LabelSet`
  itself checked out perfectly. What you pass now wins its own name in the view
  (the table is untouched; `attach` still refuses to overwrite). Same for
  `embedding=`.
- **Fixed:** `dataframe(embedding=…)` left-joined stored coordinates onto the mask,
  so an embedding computed before its mask was redefined gave the uncovered cells
  null coordinates. Nothing complained — but every plotting path drops null
  coordinates silently, producing a figure missing most of its cells while the
  `LabelSet` behind it looked perfectly healthy. Redefining a mask already dropped
  the embedding *model* as stale; the coordinates were left behind. It now raises,
  naming both counts and the `ft.embed(...)` call that fixes it.
- **Fixed:** the condensed-distance index in `SimilarityMatrix.linkage` was computed
  in scipy's int32 index dtype, which overflows past ~46k cells and would scatter
  distances onto the wrong pairs rather than raising. Now computed in int64.
- Dropped, relative to dfc/the old design: versioned taxonomies, decision
  ledgers, immutable releases, regress-out, and the bundled plotting layer.
