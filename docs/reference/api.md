# Function Reference

## FeatureTable

The container and its methods (construction, masks, collections, preprocessing,
clustering, labeling, embedding, comparison, and persistence).

::: cellpax.featuretable.FeatureTable

## Feature collections & scaling

::: cellpax.featuretable.FeatureCollection

::: cellpax.featuretable.FittedScaler

::: cellpax.featuretable.FittedEmbedding

::: cellpax.featuretable.EmbeddingView

## The clustering space

::: cellpax.space.FittedSpace

::: cellpax.clustering.PercentileClipper

::: cellpax.clustering.SigmaClipper

::: cellpax.clustering.make_clipped_scaler

::: cellpax.clustering.clipped_scaler_factory

::: cellpax.clustering.quantile_scaler_factory

## Joining datasets

Tables from separate datasets, harmonized into one space (see the guide's
[Joining datasets](../guide.md#joining-datasets)). The per-dataset lookups —
`harmonize`, `source_ids`, `cell_ids_for`, `labels_by_dataset`, `dataset_mixing`,
`cross_dataset_classification` — are methods on `FeatureTable`.

::: cellpax.datasets.join_datasets

::: cellpax.diagnostics.dataset_mixing

::: cellpax.validate.cross_dataset_classification

::: cellpax.validate.TransferScore

## Labels

::: cellpax.labels.LabelSet

::: cellpax.labels.Label

## Clustering

::: cellpax.clustering.Clustering

::: cellpax.clustering.SimilarityMatrix

::: cellpax.clustering.Partitions

::: cellpax.clustering.ConsensusHierarchy

::: cellpax.clustering.SortedMatrix

::: cellpax.clustering.CutSuggestion

::: cellpax.clustering.fauxnograph_coclustering

::: cellpax.clustering.kneighbor_graph

::: cellpax.clustering.axis_stability

::: cellpax.clustering.consensus_density

::: cellpax.clustering.neighborhood_purity

::: cellpax.clustering.neighborhood_self_predictions

::: cellpax.clustering.neighbor_label_composition

## Diagnostics

::: cellpax.diagnostics.feature_correlation

::: cellpax.diagnostics.block_weights

::: cellpax.diagnostics.tie_report

::: cellpax.diagnostics.duplicate_rows

::: cellpax.diagnostics.clip_comparison

::: cellpax.diagnostics.covariate_sensitivity

::: cellpax.diagnostics.stratum_shift

::: cellpax.diagnostics.information_imbalance

::: cellpax.diagnostics.feature_relevance

::: cellpax.diagnostics.discriminative_features

## Validation

::: cellpax.validate.subsample_stability

::: cellpax.validate.clustering_stability

::: cellpax.validate.Stability

::: cellpax.validate.loo_knn_recovery

::: cellpax.validate.graph_knn_recovery

::: cellpax.validate.paired_recovery

::: cellpax.validate.RecoveryScore

::: cellpax.validate.label_purity

## Boundary report

::: cellpax.boundary.boundary_report

## Gradients

::: cellpax.gradient.Gradient

::: cellpax.gradient.fit_principal_curve

::: cellpax.gradient.twonn_dimension

::: cellpax.gradient.twonn_profile

## AnnData bridge

::: cellpax.interop.to_anndata

::: cellpax.interop.from_anndata

## Conformal assignment

::: cellpax.assign.Assignment

::: cellpax.assign.weighted_p_values

::: cellpax.assign.conditional_prediction_set

## Label propagation

::: cellpax.propagate.confidence_curve

::: cellpax.propagate.propagate_knn

::: cellpax.propagate.propagate_spread

::: cellpax.propagate.Propagation

::: cellpax.propagate.Recovery

## Comparison

::: cellpax.compare.compare

::: cellpax.compare.compare_many

::: cellpax.compare.Comparison

## Persistence

::: cellpax.persist.save_feature_table

::: cellpax.persist.load_feature_table

::: cellpax.persist.list_analyses
