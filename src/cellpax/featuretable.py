"""The FeatureTable container — a flexible, polars-native, maskable cell table.

Step 1 of the FeatureTable-centered redesign (see DESIGN_PROPOSAL.md): the core
container — construction, ``add_column``, masks (with hierarchical ``based_on``),
feature columns, and ``dataframe(mask, scaled=…)`` backed by lazy, single,
per-mask scalers. Feature collections, the unified ``preprocess`` layer (ihs skew
correction, regress-out), embeddings, labels, clustering, comparison, and
DataFolio persistence arrive in later steps.

Design principles honored here (from the dfc audit):
- ``dataframe(mask, scaled=…)`` is the primary surface (principle 1).
- ``FittedScaler`` = per-feature transforms + a fitted scaler, re-fit per mask
  (principle 2); step 1 uses identity transforms until ``preprocess`` lands.
- masks are boolean columns with hierarchical ``based_on`` (principle 3).
- ``add_column(data, name, mask=, fill_value=)`` (principle 4).
- scalers are lazy and single — one per mask, fit on demand (principle 10).
- polars-first, explicit accessors, no ``__getattr__`` passthrough (principle 14).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

_DEFAULT_MASK = "all"
_MASK_PREFIX = "_mask_"


@dataclass
class FittedScaler:
    """A fitted scaler paired with the per-feature transforms decided at fit time.

    In step 1 ``transforms`` is all ``None`` (identity); the unified ``preprocess``
    layer will populate it with ``"ihs"`` / ``"log"`` / ``"sqrt"`` per feature.
    """

    scaler: Any
    transforms: list[str | None]

    def apply_transforms(self, features: np.ndarray) -> np.ndarray:
        if all(transform is None for transform in self.transforms):
            return features
        features = features.astype(float, copy=True)
        for index, transform in enumerate(self.transforms):
            if transform is None:
                continue
            column = features[:, index]
            if transform == "ihs":
                features[:, index] = np.arcsinh(column)
            elif transform == "sqrt":
                features[:, index] = np.sqrt(column)
            elif transform == "log":
                shift = 0.0 if column.min() > 0 else (1e-9 - column.min())
                features[:, index] = np.log(column + shift)
            else:
                raise ValueError(f"Unknown per-feature transform {transform!r}")
        return features

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.apply_transforms(features))


def _default_scaler_factory():
    from sklearn.preprocessing import StandardScaler

    return StandardScaler()


class FeatureTable:
    """A polars-native cell table with named masks and on-the-fly scaling.

    Parameters
    ----------
    df:
        The per-cell data (polars ``DataFrame``; a pandas frame is accepted and
        converted). Holds the id column, feature columns, and any metadata.
    features:
        The feature column names to cluster/scale on.
    id_column:
        The unique per-cell key. Default ``"cell_id"``.
    scaler_factory:
        Zero-argument callable returning a fresh unfitted scaler. Default
        ``StandardScaler``.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        features: Sequence[str],
        *,
        id_column: str = "cell_id",
        scaler_factory: Any = None,
    ) -> None:
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        frame = df.clone()

        if id_column not in frame.columns:
            raise ValueError(f"id_column {id_column!r} is not a column")
        if frame[id_column].null_count() or frame[id_column].n_unique() != frame.height:
            raise ValueError(f"{id_column!r} must be non-null and unique")

        features = list(features)
        if not features:
            raise ValueError("A FeatureTable requires at least one feature column")
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        non_numeric = [
            column for column in features if not frame.schema[column].is_numeric()
        ]
        if non_numeric:
            raise TypeError(f"Feature columns must be numeric: {non_numeric}")

        reserved = [
            column for column in frame.columns if column.startswith(_MASK_PREFIX)
        ]
        if reserved:
            raise ValueError(
                f"Column names starting with {_MASK_PREFIX!r} are reserved: {reserved}"
            )

        self._df = frame.with_columns(pl.lit(True).alias(_mask_column(_DEFAULT_MASK)))
        self._id_column = id_column
        self._features = features
        self._scaler_factory = scaler_factory or _default_scaler_factory
        self._scaler_cache: dict[str, FittedScaler] = {}

    # -- accessors -------------------------------------------------------------

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def feature_columns(self) -> list[str]:
        """The feature column names."""
        return list(self._features)

    @property
    def n_cells(self) -> int:
        return self._df.height

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def masks(self) -> list[str]:
        """Names of all defined masks (including the implicit ``'all'``)."""
        return [
            column[len(_MASK_PREFIX) :]
            for column in self._df.columns
            if column.startswith(_MASK_PREFIX)
        ]

    @property
    def columns(self) -> list[str]:
        """Non-internal columns (everything except the ``_mask_*`` columns)."""
        return [c for c in self._df.columns if not c.startswith(_MASK_PREFIX)]

    def mask_series(self, mask: str | None = None) -> pl.Series:
        """The boolean membership Series for a mask."""
        name = mask or _DEFAULT_MASK
        column = _mask_column(name)
        if column not in self._df.columns:
            raise KeyError(f"Unknown mask {name!r}; available: {self.masks}")
        return self._df[column]

    # -- construction / mutation ----------------------------------------------

    def add_column(
        self,
        data: Sequence[Any] | np.ndarray | pl.Series,
        name: str,
        *,
        mask: str | None = None,
        fill_value: Any = None,
    ) -> "FeatureTable":
        """Add or replace a column, writing ``data`` into a masked subset.

        ``data`` must have one value per True entry of ``mask``; the remaining
        rows are filled with ``fill_value``. Dtype is inferred by polars.
        """
        if name.startswith(_MASK_PREFIX):
            raise ValueError(f"Column names cannot start with {_MASK_PREFIX!r}")
        mask_np = self.mask_series(mask).to_numpy()
        values = list(
            data.to_list() if isinstance(data, pl.Series) else np.asarray(data)
        )
        positions = np.flatnonzero(mask_np)
        if len(values) != len(positions):
            raise ValueError(
                f"data has {len(values)} values but mask selects {len(positions)} cells"
            )
        column = [fill_value] * self._df.height
        for position, value in zip(positions, values):
            column[position] = value
        self._df = self._df.with_columns(pl.Series(name, column))
        return self

    def add_mask(
        self,
        name: str,
        predicate: pl.Expr | pl.Series | np.ndarray | Sequence[bool],
        *,
        based_on: str | None = None,
    ) -> "FeatureTable":
        """Define a named boolean mask.

        ``predicate`` is a polars expression evaluated over the full table, or a
        full-length boolean array/Series. ``based_on`` intersects the result with
        an existing (parent) mask, so hierarchical subsets stay nested.
        """
        if not name or name.startswith(_MASK_PREFIX) or name == _DEFAULT_MASK:
            raise ValueError(f"Invalid mask name {name!r}")
        if isinstance(predicate, pl.Expr):
            series = self._df.select(predicate.alias("m")).to_series()
        elif isinstance(predicate, pl.Series):
            series = predicate.rename("m")
        else:
            series = pl.Series("m", list(predicate))
        if series.dtype != pl.Boolean:
            raise TypeError("mask predicate must be boolean")
        if series.len() != self._df.height:
            raise ValueError("mask length must match the number of cells")
        if series.null_count():
            raise ValueError("mask cannot contain nulls")
        if based_on is not None:
            series = series & self.mask_series(based_on)
        self._df = self._df.with_columns(series.alias(_mask_column(name)))
        self._scaler_cache.pop(name, None)
        return self

    # -- scaling / views -------------------------------------------------------

    def _scaler(self, mask: str) -> FittedScaler:
        if mask not in self._scaler_cache:
            matrix = (
                self._df.filter(self.mask_series(mask))
                .select(self._features)
                .to_numpy()
            )
            scaler = self._scaler_factory()
            scaler.fit(matrix)
            self._scaler_cache[mask] = FittedScaler(
                scaler=scaler, transforms=[None] * len(self._features)
            )
        return self._scaler_cache[mask]

    def features(self, mask: str | None = None, *, scaled: bool = False) -> np.ndarray:
        """The feature matrix for a mask, raw or scaled, as a NumPy array."""
        name = mask or _DEFAULT_MASK
        matrix = (
            self._df.filter(self.mask_series(name)).select(self._features).to_numpy()
        )
        if scaled:
            matrix = self._scaler(name).transform(matrix)
        return matrix

    def dataframe(
        self, mask: str | None = None, *, scaled: bool = False
    ) -> pl.DataFrame:
        """Return the masked table with feature columns raw or scaled.

        The primary interactive surface: one row per masked cell, all
        metadata/label columns intact, feature columns raw (``scaled=False``) or
        normalized (``scaled=True``). Internal ``_mask_*`` columns are dropped.
        """
        name = mask or _DEFAULT_MASK
        frame = self._df.filter(self.mask_series(name))
        if scaled:
            matrix = self._scaler(name).transform(
                frame.select(self._features).to_numpy()
            )
            frame = frame.with_columns(
                [
                    pl.Series(column, matrix[:, index])
                    for index, column in enumerate(self._features)
                ]
            )
        drop = [c for c in frame.columns if c.startswith(_MASK_PREFIX)]
        return frame.drop(drop)

    def __repr__(self) -> str:
        return (
            f"FeatureTable(n_cells={self.n_cells}, n_features={self.n_features}, "
            f"masks={self.masks})"
        )


def _mask_column(name: str) -> str:
    return f"{_MASK_PREFIX}{name}"
