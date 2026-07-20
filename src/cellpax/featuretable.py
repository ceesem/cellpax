"""The FeatureTable container — a flexible, polars-native, maskable cell table.

Steps 1–2 of the FeatureTable-centered redesign (see DESIGN_PROPOSAL.md): the
core container — construction, ``add_column``, masks (with hierarchical
``based_on``), feature columns, and ``dataframe(mask, scaled=…)`` backed by lazy,
single, per-mask scalers — plus composable feature collections and the unified
``preprocess`` layer (ihs skew correction). Regress-out, embeddings, labels,
clustering, comparison, and DataFolio persistence arrive in later steps.

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


def _dedup(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))


def _skew(values: np.ndarray) -> float:
    """Fisher-Pearson skewness (matches scipy.stats.skew, bias=True)."""
    values = values.astype(float)
    std = values.std()
    if std == 0:
        return 0.0
    return float((((values - values.mean()) / std) ** 3).mean())


@dataclass(frozen=True)
class FeatureCollection:
    """A named, ordered, composable set of feature columns.

    Supports set algebra: ``a | b`` (union), ``a & b`` (intersection), ``a - b``
    (difference), each returning a new ``FeatureCollection``.
    """

    name: str
    columns: tuple[str, ...]

    def __or__(self, other: "FeatureCollection") -> "FeatureCollection":
        return FeatureCollection(
            f"{self.name}|{other.name}", _dedup(self.columns + other.columns)
        )

    def __and__(self, other: "FeatureCollection") -> "FeatureCollection":
        rhs = set(other.columns)
        return FeatureCollection(
            f"{self.name}&{other.name}",
            tuple(c for c in self.columns if c in rhs),
        )

    def __sub__(self, other: "FeatureCollection") -> "FeatureCollection":
        rhs = set(other.columns)
        return FeatureCollection(
            f"{self.name}-{other.name}",
            tuple(c for c in self.columns if c not in rhs),
        )

    def __iter__(self):
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


class _CollectionAccessor:
    """``ft.collections["axon"]`` access to defined feature collections."""

    def __init__(self, table: "FeatureTable") -> None:
        self._table = table

    def __getitem__(self, name: str) -> FeatureCollection:
        return self._table._collection(name)

    def __iter__(self):
        return iter(self._table._collection_names)

    def __contains__(self, name: str) -> bool:
        return name in self._table._collection_names


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
        feature_metadata: pl.DataFrame | None = None,
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

        if feature_metadata is not None:
            if "feature_id" not in feature_metadata.columns:
                raise ValueError("feature_metadata requires a 'feature_id' column")
            self._var = pl.DataFrame({"feature_id": features}).join(
                feature_metadata.filter(pl.col("feature_id").is_in(features)),
                on="feature_id",
                how="left",
            )
        else:
            self._var = pl.DataFrame({"feature_id": features})

        self._df = frame.with_columns(pl.lit(True).alias(_mask_column(_DEFAULT_MASK)))
        self._id_column = id_column
        self._features = features
        self._scaler_factory = scaler_factory or _default_scaler_factory
        self._scaler_cache: dict[tuple[str, tuple[str, ...]], FittedScaler] = {}
        self._collections: dict[str, FeatureCollection] = {}
        self._transforms: dict[str, str | None] = {}
        self._clusterings: dict[str, Any] = {}

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

    # -- feature collections ---------------------------------------------------

    @property
    def var(self) -> pl.DataFrame:
        """Per-feature metadata frame (``feature_id`` + any supplied columns)."""
        return self._var

    @property
    def collections(self) -> _CollectionAccessor:
        """Accessor for defined feature collections: ``ft.collections['axon']``."""
        return _CollectionAccessor(self)

    @property
    def transforms(self) -> dict[str, str | None]:
        """Per-feature preprocessing transforms resolved by ``preprocess``."""
        return dict(self._transforms)

    @property
    def _collection_names(self) -> list[str]:
        return list(self._collections)

    def _collection(self, name: str) -> FeatureCollection:
        if name not in self._collections:
            raise KeyError(
                f"Unknown collection {name!r}; defined: {self._collection_names}"
            )
        return self._collections[name]

    def define_features(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        family: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        predicate: pl.Expr | None = None,
    ) -> "FeatureTable":
        """Define a named, composable feature collection.

        Provide exactly one selector: an explicit ``columns`` list, a ``family`` or
        ``modality`` value(s) (requires that column in ``feature_metadata``), or a
        polars ``predicate`` over the feature metadata.
        """
        selectors = [
            columns is not None,
            family is not None,
            modality is not None,
            predicate is not None,
        ]
        if sum(selectors) != 1:
            raise ValueError(
                "Provide exactly one of columns / family / modality / predicate"
            )
        if columns is not None:
            chosen = list(columns)
        elif predicate is not None:
            chosen = self._var.filter(predicate)["feature_id"].to_list()
        else:
            key = "family" if family is not None else "modality"
            want = family if family is not None else modality
            if key not in self._var.columns:
                raise ValueError(f"No {key!r} column in feature_metadata")
            want = [want] if isinstance(want, str) else list(want)
            chosen = self._var.filter(pl.col(key).is_in(want))["feature_id"].to_list()
        unknown = [c for c in chosen if c not in self._features]
        if unknown:
            raise ValueError(f"Collection references non-feature columns: {unknown}")
        if not chosen:
            raise ValueError(f"Collection {name!r} selects no features")
        self._collections[name] = FeatureCollection(name, _dedup(chosen))
        return self

    def _resolve_columns(
        self, columns: str | FeatureCollection | Sequence[str] | None
    ) -> list[str]:
        if columns is None:
            return list(self._features)
        if isinstance(columns, FeatureCollection):
            chosen = list(columns.columns)
        elif isinstance(columns, str):
            chosen = list(self._collection(columns).columns)
        else:
            chosen = list(columns)
        unknown = [c for c in chosen if c not in self._features]
        if unknown:
            raise ValueError(f"Not declared as features: {unknown}")
        return chosen

    # -- preprocessing ---------------------------------------------------------

    def preprocess(
        self,
        *,
        skew_screen: bool = True,
        method: str = "ihs",
        threshold: float = 1.5,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> "FeatureTable":
        """Resolve per-feature transforms applied before scaling (unified layer).

        With ``skew_screen`` (default), each selected feature whose right-skewness
        exceeds ``threshold`` is transformed with ``method`` (``"ihs"`` by default —
        it handles zeros and negatives; ``"log"``/``"sqrt"`` are skipped for
        features with negative values). Transforms are recorded per feature and
        applied whenever features are scaled; they invalidate cached scalers.
        """
        if method not in {"ihs", "log", "sqrt"}:
            raise ValueError("method must be 'ihs', 'log', or 'sqrt'")
        for column in self._resolve_columns(columns):
            transform: str | None = None
            if skew_screen:
                sample = self._df[column].drop_nulls().to_numpy()
                if sample.size and _skew(sample) > threshold:
                    if method in {"log", "sqrt"} and float(sample.min()) < 0:
                        transform = None
                    else:
                        transform = method
            self._transforms[column] = transform
        self._scaler_cache.clear()
        return self

    # -- clustering ------------------------------------------------------------

    def cluster(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        n_neighbors: int | Sequence[int] = 30,
        resolution: float | Sequence[float] = 1.0,
        n_times: int = 1,
        min_cluster_size: int = 1,
        mutual_only: bool = False,
        normalize: bool = True,
        method: str = "average",
        seed: int | None = None,
        n_jobs: int = -1,
        name: str | None = None,
    ) -> Any:
        """Consensus-cluster a mask's scaled features into a ``SimilarityMatrix``.

        Clustering runs on the *scaled* features (so ``preprocess`` transforms and
        the per-mask scaler apply). Returns the consensus ``SimilarityMatrix``; if
        ``name`` is given it is also stored for later labeling and comparison.
        """
        from cellpax.consensus import SimilarityMatrix, fauxnograph_coclustering

        data = self.features(mask, scaled=True, columns=columns)
        matrix = fauxnograph_coclustering(
            data,
            n_neighbors=list(n_neighbors)
            if isinstance(n_neighbors, (list, tuple))
            else n_neighbors,
            resolution_parameter=list(resolution)
            if isinstance(resolution, (list, tuple))
            else resolution,
            n_times=n_times,
            min_cluster_size=min_cluster_size,
            mutual_only=mutual_only,
            normalize=normalize,
            seed=seed,
            n_jobs=n_jobs,
        )
        result = SimilarityMatrix(matrix, normalized=normalize, method=method)
        if name is not None:
            self._clusterings[name] = result
        return result

    def clustering(self, name: str) -> Any:
        """Return a stored ``SimilarityMatrix`` by name."""
        if name not in self._clusterings:
            raise KeyError(
                f"Unknown clustering {name!r}; stored: {list(self._clusterings)}"
            )
        return self._clusterings[name]

    def label(
        self,
        similarity: Any,
        *,
        mask: str | None = None,
        distance_threshold: float,
        min_cluster_size: int = 1,
        name: str = "label",
    ) -> Any:
        """Cut a clustering into a :class:`~cellpax.labels.LabelSet` for a mask.

        ``similarity`` is a ``SimilarityMatrix`` or the name of a stored one; its
        rows must align with the mask's cells (as produced by ``cluster``).
        """
        from cellpax.labels import LabelSet

        if isinstance(similarity, str):
            similarity = self.clustering(similarity)
        cell_ids = self._df.filter(self.mask_series(mask))[self._id_column].to_numpy()
        return LabelSet.from_clustering(
            similarity,
            cell_ids,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            name=name,
        )

    def attach(self, labels: Any, *, name: str | None = None) -> "FeatureTable":
        """Attach a ``LabelSet`` as a column, joined on the id column.

        Cells outside the label set get a null label.
        """
        column = name or labels.name
        frame = labels.to_frame(id_column=self._id_column).rename({labels.name: column})
        keep = [self._id_column, column]
        self._df = self._df.join(frame.select(keep), on=self._id_column, how="left")
        return self

    # -- scaling / views -------------------------------------------------------

    def _scaler(self, mask: str, columns: Sequence[str]) -> FittedScaler:
        key = (mask, tuple(columns))
        if key not in self._scaler_cache:
            matrix = self._df.filter(self.mask_series(mask)).select(columns).to_numpy()
            transforms = [self._transforms.get(c) for c in columns]
            fitted = FittedScaler(scaler=self._scaler_factory(), transforms=transforms)
            fitted.scaler.fit(fitted.apply_transforms(matrix))
            self._scaler_cache[key] = fitted
        return self._scaler_cache[key]

    def features(
        self,
        mask: str | None = None,
        *,
        scaled: bool = False,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> np.ndarray:
        """The feature matrix for a mask, raw or scaled, as a NumPy array."""
        name = mask or _DEFAULT_MASK
        cols = self._resolve_columns(columns)
        matrix = self._df.filter(self.mask_series(name)).select(cols).to_numpy()
        if scaled:
            matrix = self._scaler(name, cols).transform(matrix)
        return matrix

    def dataframe(
        self,
        mask: str | None = None,
        *,
        scaled: bool = False,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Return the masked table with feature columns raw or scaled.

        The primary interactive surface: one row per masked cell, all
        metadata/label columns intact, feature columns raw (``scaled=False``) or
        normalized (``scaled=True``). With ``columns`` only that collection's
        features are scaled. Internal ``_mask_*`` columns are dropped.
        """
        name = mask or _DEFAULT_MASK
        frame = self._df.filter(self.mask_series(name))
        if scaled:
            cols = self._resolve_columns(columns)
            matrix = self._scaler(name, cols).transform(frame.select(cols).to_numpy())
            frame = frame.with_columns(
                [
                    pl.Series(column, matrix[:, index])
                    for index, column in enumerate(cols)
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
