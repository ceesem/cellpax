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


def _apply_id_map(
    frame: pl.DataFrame,
    id_map: Any,
    id_column: str,
    *,
    on: str | None = None,
) -> pl.DataFrame:
    """Left-join ``id_map`` to bring an ``id_column`` into ``frame``.

    ``id_map`` carries ``id_column`` plus a key column already in ``frame`` (e.g.
    ``root_id``). The key is ``on`` if given, else inferred as the single shared
    column. Every row of ``frame`` must map, and ids must be unique.
    """
    if not isinstance(id_map, pl.DataFrame):
        id_map = pl.from_pandas(id_map)
    if id_column not in id_map.columns:
        raise ValueError(f"id_map must contain the id column {id_column!r}")
    if on is None:
        shared = [c for c in id_map.columns if c != id_column and c in frame.columns]
        if len(shared) != 1:
            raise ValueError(
                f"cannot infer the id_map join key; shared columns={shared}. "
                "Pass the key explicitly."
            )
        on = shared[0]
    elif on not in frame.columns or on not in id_map.columns:
        raise ValueError(f"id_map join key {on!r} must be in both the table and map")
    if id_map[on].n_unique() != id_map.height:
        raise ValueError(f"id_map has duplicate {on!r} keys")
    joined = frame.join(id_map.select(on, id_column), on=on, how="left")
    unmapped = joined.filter(pl.col(id_column).is_null())
    if unmapped.height:
        sample = unmapped[on].head(5).to_list()
        raise ValueError(
            f"{unmapped.height} rows have no {id_column!r} in id_map; "
            f"sample {on}={sample}"
        )
    return joined


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
    id_map:
        Optional ``[<key>, id_column]`` frame joined in when ``id_column`` is not
        already present (e.g. mapping ``root_id`` → ``cell_id``). See
        ``set_id_column`` to do this after construction.
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
        id_map: pl.DataFrame | None = None,
        feature_metadata: pl.DataFrame | None = None,
        scaler_factory: Any = None,
    ) -> None:
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        frame = df.clone()

        if id_column not in frame.columns and id_map is not None:
            frame = _apply_id_map(frame, id_map, id_column)
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
        self._embeddings: dict[tuple[str, str], pl.DataFrame] = {}

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

    def set_id_column(
        self,
        name: str,
        *,
        id_map: Any = None,
        on: str | None = None,
    ) -> "FeatureTable":
        """Set (or bring in) the unique cell-id column after construction.

        If ``name`` is already a column it just becomes the id. Otherwise pass an
        ``id_map`` (a ``[<key>, name]`` frame, e.g. ``['root_id', 'cell_id']``); it
        is left-joined on the shared key (or ``on``) to add ``name``. Every cell
        must map and ids must be unique.
        """
        if name not in self._df.columns:
            if id_map is None:
                raise ValueError(f"{name!r} is not a column; pass an id_map to add it")
            self._df = _apply_id_map(self._df, id_map, name, on=on)
        column = self._df[name]
        if column.null_count() or column.n_unique() != self._df.height:
            raise ValueError(f"{name!r} must be non-null and unique")
        self._id_column = name
        return self

    def add_features(
        self,
        source: pl.DataFrame,
        features: Sequence[str],
        *,
        on: str | None = None,
        feature_metadata: pl.DataFrame | None = None,
        allow_missing: bool = False,
    ) -> "FeatureTable":
        """Join additional feature columns from another source and register them.

        ``source`` is keyed on ``on`` (default: the id column) and must carry the
        named ``features``. They are left-joined onto the table and added to the
        feature set; ``feature_metadata`` (a ``feature_id`` + attribute frame)
        extends ``var`` for them. By default every cell must be covered
        (``allow_missing=True`` permits nulls, which then can't be scaled).
        """
        if not isinstance(source, pl.DataFrame):
            source = pl.from_pandas(source)
        on = on or self._id_column
        features = list(features)
        if on not in source.columns or on not in self._df.columns:
            raise ValueError(f"join key {on!r} must be in both the table and source")
        if source[on].n_unique() != source.height:
            raise ValueError(f"source has duplicate {on!r} keys")
        missing = [f for f in features if f not in source.columns]
        if missing:
            raise ValueError(f"features not in source: {missing}")
        clash = [f for f in features if f in self._df.columns]
        if clash:
            raise ValueError(f"features already present in the table: {clash}")
        non_numeric = [f for f in features if not source.schema[f].is_numeric()]
        if non_numeric:
            raise TypeError(f"features must be numeric: {non_numeric}")

        joined = self._df.join(source.select(on, *features), on=on, how="left")
        if not allow_missing:
            uncovered = [f for f in features if joined[f].null_count()]
            if uncovered:
                raise ValueError(
                    f"source does not cover every cell for: {uncovered} "
                    "(pass allow_missing=True to permit nulls)"
                )
        self._df = joined
        self._features = self._features + features
        new_var = pl.DataFrame({"feature_id": features})
        if feature_metadata is not None:
            new_var = new_var.join(
                feature_metadata.filter(pl.col("feature_id").is_in(features)),
                on="feature_id",
                how="left",
            )
        self._var = pl.concat([self._var, new_var], how="diagonal")
        self._scaler_cache.clear()
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
        from cellpax.clustering import SimilarityMatrix, fauxnograph_coclustering

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

    def overcluster(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        resolution: float = 2.0,
        n_neighbors: int = 30,
        mutual_only: bool = False,
        min_cluster_size: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        """A single high-resolution Leiden partition of a mask's scaled features.

        Returns a per-cell integer over-clustering (many small clusters) aligned
        to ``features(mask)`` rows — a good CHOIR starting point to prune via
        ``cluster_choir(over_clustering=...)``.
        """
        from cellpax.clustering import cluster_leiden, kneighbor_graph

        data = self.features(mask, scaled=True, columns=columns)
        graph = kneighbor_graph(data, n_neighbors=n_neighbors, mutual_only=mutual_only)
        return cluster_leiden(
            graph,
            resolution_parameter=resolution,
            seed=seed,
            min_cluster_size=min_cluster_size,
        )

    def cluster_choir(
        self,
        clustering: Any = None,
        *,
        over_clustering: Any = None,
        linkage_method: str = "ward",
        mask: str | None = None,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        name: str = "label",
        alpha: float = 0.05,
        min_cluster_size: int = 20,
        min_accuracy: float = 0.5,
        n_iterations: int = 100,
        n_estimators: int = 100,
        sample_max: int = 1000,
        use_variance: bool = True,
        reselect: bool = False,
        n_features: int | None = None,
        n_pcs: int | None = None,
        seed: int | None = None,
        n_jobs: int = -1,
    ) -> Any:
        """Resolve a hierarchy into a ``LabelSet`` via CHOIR-style split testing.

        Instead of cutting a dendrogram at one distance threshold, keep each split
        only where the two child clusters pass CHOIR's random-forest permutation
        test (see :mod:`cellpax.choir`). Provide exactly one starting hierarchy:

        - ``clustering``: a consensus ``SimilarityMatrix`` (or stored name) — its
          full per-cell tree is pruned.
        - ``over_clustering``: a per-cell over-clustering (integer array aligned to
          ``features(mask)`` rows, or a ``LabelSet``), e.g. from ``overcluster``
          (high-res Leiden) or KMeans. A hierarchy is built over its cluster
          centroids (``linkage_method``) and pruned.

        The RF test always runs on the feature matrix. With ``reselect``, features
        are re-chosen per node. Returns a mask-aligned ``LabelSet``.
        """
        from cellpax.choir import choir_labels, overcluster_linkage
        from cellpax.labels import LabelSet

        if (clustering is None) == (over_clustering is None):
            raise ValueError("provide exactly one of clustering or over_clustering")
        features = self.features(mask, scaled=True, columns=columns)
        cell_ids = self._df.filter(self.mask_series(mask))[self._id_column].to_numpy()

        if over_clustering is not None:
            over = self._align_over_clustering(over_clustering, cell_ids)
            hierarchy, leaf_members = overcluster_linkage(
                features, over, method=linkage_method
            )
        else:
            if isinstance(clustering, str):
                clustering = self.clustering(clustering)
            hierarchy, leaf_members = clustering.linkage, None

        labels = choir_labels(
            hierarchy,
            features,
            leaf_members=leaf_members,
            alpha=alpha,
            min_cluster_size=min_cluster_size,
            min_accuracy=min_accuracy,
            n_iterations=n_iterations,
            n_estimators=n_estimators,
            sample_max=sample_max,
            use_variance=use_variance,
            reselect=reselect,
            n_features=n_features,
            n_pcs=n_pcs,
            seed=seed,
            n_jobs=n_jobs,
        )
        return LabelSet(cell_ids, labels, name=name)

    def _align_over_clustering(
        self, over_clustering: Any, cell_ids: np.ndarray
    ) -> np.ndarray:
        """Coerce an over-clustering (array or LabelSet) to feature-row order."""
        if hasattr(over_clustering, "to_frame"):  # a LabelSet
            frame = over_clustering.to_frame(id_column=self._id_column)
            id_col, value_col = frame.columns[0], frame.columns[2]
            mapping = dict(zip(frame[id_col].to_list(), frame[value_col].to_list()))
            return np.array([mapping.get(int(c), -1) for c in cell_ids], dtype=np.int64)
        values = np.asarray(over_clustering)
        if values.shape[0] != len(cell_ids):
            raise ValueError("over_clustering length must match the mask's cells")
        return values

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

    # -- embeddings ------------------------------------------------------------

    def embed(
        self,
        mask: str | None = None,
        *,
        method: str = "pca",
        columns: str | FeatureCollection | Sequence[str] | None = None,
        n_components: int = 2,
        name: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Compute a low-dimensional embedding of a mask's scaled features.

        ``method="pca"`` (default) uses scikit-learn; ``method="umap"`` requires
        ``umap-learn`` (imported lazily). Coordinates are stored under
        ``(mask, name)`` and returned as ``cell_id`` + ``{name}0..{name}{k-1}``.
        """
        label = name or method
        data = self.features(mask, scaled=True, columns=columns)
        if method == "pca":
            from sklearn.decomposition import PCA

            coords = PCA(n_components=n_components, random_state=seed).fit_transform(
                data
            )
        elif method == "umap":
            try:
                import umap
            except ImportError as error:
                raise ImportError(
                    "method='umap' requires the optional 'umap-learn' package"
                ) from error
            coords = umap.UMAP(
                n_components=n_components, random_state=seed, **kwargs
            ).fit_transform(data)
        else:
            raise ValueError(f"Unknown embedding method {method!r}")
        cell_ids = self._df.filter(self.mask_series(mask))[self._id_column].to_numpy()
        frame = pl.DataFrame(
            {
                self._id_column: cell_ids,
                **{f"{label}{i}": coords[:, i] for i in range(n_components)},
            }
        )
        self._embeddings[(mask or _DEFAULT_MASK, label)] = frame
        return frame

    def embedding(self, mask: str | None = None, *, name: str = "pca") -> pl.DataFrame:
        """Return a stored embedding's coordinates."""
        key = (mask or _DEFAULT_MASK, name)
        if key not in self._embeddings:
            raise KeyError(
                f"No embedding {name!r} for mask {mask or _DEFAULT_MASK!r}; "
                f"stored: {list(self._embeddings)}"
            )
        return self._embeddings[key]

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
        embedding: str | None = None,
    ) -> pl.DataFrame:
        """Return the masked table with feature columns raw or scaled.

        The primary interactive surface: one row per masked cell, all
        metadata/label columns intact, feature columns raw (``scaled=False``) or
        normalized (``scaled=True``). With ``columns`` only that collection's
        features are scaled; with ``embedding`` a stored embedding's coordinates
        are joined on. Internal ``_mask_*`` columns are dropped.
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
        frame = frame.drop(drop)
        if embedding is not None:
            frame = frame.join(
                self.embedding(name, name=embedding), on=self._id_column, how="left"
            )
        return frame

    # -- comparison ------------------------------------------------------------

    def compare(self, a: Any, b: Any) -> Any:
        """Compare two ``LabelSet``s over shared cells (see :mod:`cellpax.compare`)."""
        from cellpax.compare import compare

        return compare(a, b)

    # -- persistence -----------------------------------------------------------

    def save(self, folio: Any, name: str, *, overwrite: bool = True) -> "FeatureTable":
        """Persist this analysis under ``name`` in a DataFolio (see persist)."""
        from cellpax.persist import save_feature_table

        save_feature_table(self, folio, name, overwrite=overwrite)
        return self

    @classmethod
    def load(cls, folio: Any, name: str) -> "FeatureTable":
        """Load a FeatureTable saved under ``name`` in a DataFolio."""
        from cellpax.persist import load_feature_table

        return load_feature_table(folio, name)

    def __repr__(self) -> str:
        return (
            f"FeatureTable(n_cells={self.n_cells}, n_features={self.n_features}, "
            f"masks={self.masks})"
        )


def _mask_column(name: str) -> str:
    return f"{_MASK_PREFIX}{name}"
