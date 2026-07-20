"""CellPax — flexible, polars-native cell feature clustering for connectomics.

A ``FeatureTable`` holds cell features with named masks, composable feature
collections, and a heavy-tail preprocessing layer; consensus clustering, clear
relabelable ``LabelSet``s (with ``IntEnum`` bindings), embeddings, cross-approach
comparison, and first-class DataFolio save/load build on it.
"""

from cellpax.clustering import (
    SimilarityMatrix,
    clipped_scaler_factory,
    fauxnograph_coclustering,
    make_clipped_scaler,
)
from cellpax.compare import Comparison, compare, compare_many
from cellpax.featuretable import FeatureCollection, FeatureTable, FittedScaler
from cellpax.labels import Label, LabelSet
from cellpax.persist import list_analyses, load_feature_table, save_feature_table

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "FeatureCollection",
    "FeatureTable",
    "FittedScaler",
    "Label",
    "LabelSet",
    "SimilarityMatrix",
    "__version__",
    "clipped_scaler_factory",
    "compare",
    "compare_many",
    "fauxnograph_coclustering",
    "list_analyses",
    "load_feature_table",
    "make_clipped_scaler",
    "save_feature_table",
]
