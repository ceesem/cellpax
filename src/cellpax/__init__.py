"""CellPax — flexible, polars-native cell feature clustering for connectomics.

A ``FeatureTable`` holds cell features with named masks, composable feature
collections, and a heavy-tail preprocessing layer; consensus clustering, clear
relabelable ``LabelSet``s (with ``IntEnum`` bindings), embeddings, kNN label
propagation from a curated core, cross-approach comparison, and first-class
DataFolio save/load build on it.
"""

from cellpax.clustering import (
    Clustering,
    ConsensusHierarchy,
    Partitions,
    PercentileClipper,
    SigmaClipper,
    SimilarityMatrix,
    SortedMatrix,
    axis_stability,
    clipped_scaler_factory,
    fauxnograph_coclustering,
    kneighbor_graph,
    make_clipped_scaler,
    neighbor_label_composition,
    neighborhood_purity,
    neighborhood_self_predictions,
)
from cellpax.compare import Comparison, compare, compare_many
from cellpax.diagnostics import (
    block_weights,
    clip_comparison,
    covariate_sensitivity,
    duplicate_rows,
    feature_correlation,
    stratum_shift,
    tie_report,
)
from cellpax.featuretable import (
    EmbeddingView,
    FeatureCollection,
    FeatureTable,
    FittedEmbedding,
    FittedScaler,
)
from cellpax.labels import Label, LabelSet
from cellpax.persist import list_analyses, load_feature_table, save_feature_table
from cellpax.propagate import (
    Propagation,
    Recovery,
    confidence_curve,
    propagate_knn,
    propagate_spread,
)
from cellpax.space import FittedSpace
from cellpax.validate import (
    RecoveryScore,
    Stability,
    clustering_stability,
    graph_knn_recovery,
    label_purity,
    loo_knn_recovery,
    paired_recovery,
    subsample_stability,
)

__version__ = "0.0.1"

__all__ = [
    "Clustering",
    "Comparison",
    "ConsensusHierarchy",
    "EmbeddingView",
    "FeatureCollection",
    "FeatureTable",
    "FittedEmbedding",
    "FittedScaler",
    "FittedSpace",
    "Label",
    "LabelSet",
    "Partitions",
    "PercentileClipper",
    "Propagation",
    "Recovery",
    "RecoveryScore",
    "SigmaClipper",
    "SimilarityMatrix",
    "SortedMatrix",
    "Stability",
    "__version__",
    "axis_stability",
    "block_weights",
    "clip_comparison",
    "clipped_scaler_factory",
    "clustering_stability",
    "compare",
    "compare_many",
    "confidence_curve",
    "covariate_sensitivity",
    "duplicate_rows",
    "fauxnograph_coclustering",
    "feature_correlation",
    "graph_knn_recovery",
    "kneighbor_graph",
    "label_purity",
    "list_analyses",
    "load_feature_table",
    "loo_knn_recovery",
    "make_clipped_scaler",
    "neighbor_label_composition",
    "neighborhood_purity",
    "neighborhood_self_predictions",
    "paired_recovery",
    "propagate_knn",
    "propagate_spread",
    "save_feature_table",
    "stratum_shift",
    "subsample_stability",
    "tie_report",
]
