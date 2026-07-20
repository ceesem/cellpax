# Changelog

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
- Added cross-approach comparison (`compare` / `compare_many`): contingency,
  ARI / NMI / FMI / Jaccard, and alluvial frames.
- Added first-class DataFolio persistence: `ft.save` / `FeatureTable.load` store
  many analyses per folio under a `<name>/…` namespace, structured (not
  flattened) and coexisting with arbitrary user content; `list_analyses`.
- Dropped, relative to dfc/the old design: versioned taxonomies, decision
  ledgers, immutable releases, regress-out, and the bundled plotting layer.
