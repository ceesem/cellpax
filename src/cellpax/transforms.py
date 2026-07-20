"""Small sklearn-compatible transformers supported by Slice 2."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClippedScaler(TransformerMixin, BaseEstimator):
    """Clip each feature to fitted percentiles and scale it to ``[0, 1]``."""

    def __init__(self, lower_percentile: float = 0.5, upper_percentile: float = 99.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, values: np.ndarray, y: object = None) -> "ClippedScaler":
        del y
        self.lower_ = np.percentile(values, self.lower_percentile, axis=0)
        self.upper_ = np.percentile(values, self.upper_percentile, axis=0)
        self.scale_ = self.upper_ - self.lower_
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lower_, self.upper_)
        return (clipped - self.lower_) / self.scale_
