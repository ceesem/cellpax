"""CellPax — flexible, polars-native cell feature clustering for connectomics.

A ``FeatureTable`` holds cell features with named masks, composable feature
collections, and a heavy-tail preprocessing layer; consensus clustering, clear
relabelable ``LabelSet``s (with ``IntEnum`` bindings), embeddings, kNN label
propagation from a curated core, cross-approach comparison, and first-class
DataFolio save/load build on it.
"""

from cellpax_loky_compat import apply as _apply_loky_tracker_compat

# Before anything can start a joblib worker pool: loky's resource-tracker server
# cannot parse CPython 3.13.10's tracker messages, and floods stderr with one
# traceback per shared resource. Self-disabling once joblib catches up; see
# cellpax_loky_compat.
_apply_loky_tracker_compat()

from cellpax.assign import (
    Assignment,
    conditional_prediction_set,
    weighted_p_values,
)
from cellpax.boundary import boundary_report
from cellpax.clustering import (
    Clustering,
    ConsensusHierarchy,
    CutSuggestion,
    Partitions,
    PercentileClipper,
    SigmaClipper,
    SimilarityMatrix,
    SortedMatrix,
    axis_stability,
    clipped_scaler_factory,
    consensus_density,
    fauxnograph_coclustering,
    kneighbor_graph,
    make_clipped_scaler,
    neighbor_label_composition,
    neighborhood_purity,
    neighborhood_self_predictions,
    quantile_scaler_factory,
)
from cellpax.compare import Comparison, compare, compare_many
from cellpax.datasets import join_datasets
from cellpax.diagnostics import (
    block_weights,
    clip_comparison,
    covariate_sensitivity,
    dataset_mixing,
    discriminative_features,
    duplicate_rows,
    feature_correlation,
    feature_relevance,
    information_imbalance,
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
from cellpax.gradient import (
    Gradient,
    fit_principal_curve,
    twonn_dimension,
    twonn_profile,
)
from cellpax.interop import from_anndata, to_anndata
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
    TransferScore,
    clustering_stability,
    cross_dataset_classification,
    graph_knn_recovery,
    label_purity,
    loo_knn_recovery,
    paired_recovery,
    subsample_stability,
)

__version__ = "0.0.1"

__all__ = [
    "Assignment",
    "Clustering",
    "CutSuggestion",
    "Comparison",
    "ConsensusHierarchy",
    "EmbeddingView",
    "FeatureCollection",
    "FeatureTable",
    "FittedEmbedding",
    "FittedScaler",
    "FittedSpace",
    "Gradient",
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
    "TransferScore",
    "__version__",
    "axis_stability",
    "block_weights",
    "boundary_report",
    "clip_comparison",
    "clipped_scaler_factory",
    "consensus_density",
    "clustering_stability",
    "compare",
    "compare_many",
    "conditional_prediction_set",
    "confidence_curve",
    "covariate_sensitivity",
    "cross_dataset_classification",
    "dataset_mixing",
    "discriminative_features",
    "duplicate_rows",
    "fauxnograph_coclustering",
    "feature_correlation",
    "feature_relevance",
    "fit_principal_curve",
    "from_anndata",
    "graph_knn_recovery",
    "information_imbalance",
    "join_datasets",
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
    "quantile_scaler_factory",
    "save_feature_table",
    "stratum_shift",
    "subsample_stability",
    "tie_report",
    "to_anndata",
    "twonn_dimension",
    "twonn_profile",
    "weighted_p_values",
]
