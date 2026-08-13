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

import hashlib
import warnings
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from cellpax.space import FittedSpace

_DEFAULT_MASK = "all"
_MASK_PREFIX = "_mask_"


def _jsonable(value: Any) -> bool:
    """Whether a recorded parameter survives a JSON manifest round trip."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_jsonable(v) for v in value)
    return False


@dataclass
class FittedScaler:
    """A fitted scaler paired with the per-feature transforms decided at fit time.

    In step 1 ``transforms`` is all ``None`` (identity); the unified ``preprocess``
    layer will populate it with ``"ihs"`` / ``"log"`` / ``"sqrt"`` per feature.
    """

    scaler: Any
    transforms: list[str | None]
    shifts: list[float] | None = None
    """Per-feature ``log`` shifts, decided once at fit time.

    A ``log`` feature's shift must come from the fit, not from whatever batch is
    being transformed — otherwise the same raw value lands at a different
    transformed value per batch, and ``project`` silently re-fits the one thing
    it promises to hold fixed. ``None`` means the scaler has not been fit yet.
    """

    def fit(self, features: np.ndarray) -> "FittedScaler":
        """Decide the per-feature shifts on ``features``, then fit the scaler."""
        features = np.asarray(features, dtype=float)
        self.shifts = [
            0.0
            if transform != "log" or features[:, index].min() > 0
            else float(1e-9 - features[:, index].min())
            for index, transform in enumerate(self.transforms)
        ]
        self.scaler.fit(self.apply_transforms(features))
        return self

    def apply_transforms(self, features: np.ndarray) -> np.ndarray:
        if all(transform is None for transform in self.transforms):
            return features
        if self.shifts is None and "log" in self.transforms:
            raise ValueError(
                "this scaler carries a 'log' transform but no fitted shifts; the "
                "log shift is a fitted parameter and cannot be derived from the "
                "batch — call fit() (or re-save the analysis to record shifts)"
            )
        features = features.astype(float, copy=True)
        for index, transform in enumerate(self.transforms):
            if transform is None:
                continue
            column = features[:, index]
            if transform == "ihs":
                features[:, index] = np.arcsinh(column)
            elif transform == "sqrt":
                if column.min() < 0:
                    raise ValueError(
                        f"feature {index} has negative values, outside the domain "
                        f"of its fitted 'sqrt' transform; these cells cannot be "
                        f"projected into this scaling"
                    )
                features[:, index] = np.sqrt(column)
            elif transform == "log":
                shifted = column + self.shifts[index]
                if shifted.min() <= 0:
                    raise ValueError(
                        f"feature {index} has values at or below the fitted 'log' "
                        f"shift ({-self.shifts[index]:g}); these cells are outside "
                        f"the domain the transform was fit on"
                    )
                features[:, index] = np.log(shifted)
            else:
                raise ValueError(f"Unknown per-feature transform {transform!r}")
        return features

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.apply_transforms(features))


@dataclass
class FittedEmbedding:
    """The estimator behind a stored embedding, plus the space it was fit in.

    ``model`` is the scikit-learn ``PCA`` or ``umap.UMAP`` that produced the
    coordinates ``embedding`` returns, kept so later cells can be pushed *into* that
    embedding rather than fitting a new one that lands somewhere else entirely.
    ``columns`` records which features it saw, already scaled by the mask's
    :class:`FittedScaler` — ``transform`` expects that same scaled space, which is
    what ``FeatureTable.project`` assembles for you.

    Lives for the session only: it is deliberately not persisted, since coordinates
    can be re-derived but a UMAP fit is neither small nor reproducible across
    versions. After a ``load`` you have the coordinates and must re-``embed`` to
    project anything new.
    """

    model: Any
    method: str
    columns: tuple[str, ...]
    n_components: int
    space: str = "scaled"
    fitted_space: Any = None
    internal_reduction: str | None = None
    """What the backend did to the input on its own, if anything.

    ``None`` for a backend that embeds what it is handed. ``pacmap`` and ``localmap``
    truncated-SVD to 100 dimensions when given more than that, so the space they
    actually built pairs in is not the space they were passed — which matters only
    because :meth:`FeatureTable.graph_provenance` would otherwise report a space that
    was not used.
    """
    params: dict[str, Any] | None = None
    """The call that produced this fit — method, seed, and backend kwargs.

    Recorded so a stored embedding is re-derivable rather than merely present;
    persisted with the coordinates even though the model itself is not.
    """

    def transform(self, scaled: np.ndarray) -> np.ndarray:
        """Coordinates for already-scaled rows over ``columns``."""
        if self.fitted_space is not None:
            scaled = self.fitted_space.transform_scaled(scaled)
        # pacmap/localmap query their saved faiss index here, and a faiss *search* is
        # an OpenMP parallel region just as the build is — so ``project`` needs the
        # same guard ``embed`` uses. See :func:`_faiss_single_thread`.
        guard = (
            _faiss_single_thread() if self.method in _FAISS_BACKED else nullcontext()
        )
        with guard:
            return np.asarray(self.model.transform(scaled))


#: Backends that reach faiss and therefore need :func:`_faiss_single_thread`.
_FAISS_BACKED = frozenset({"pacmap", "localmap"})


@contextmanager
def _faiss_single_thread() -> Iterator[None]:
    """Pin faiss to one OpenMP thread for the duration, then restore the old count.

    Guards against an uninterruptible hang, not a slow path. pacmap and localmap build
    their neighbour index with faiss, whose wheel vendors a private ``libomp``; sklearn
    vendors a second one; and xgboost's macOS wheel links ``@rpath`` against Homebrew's.
    A process that has imported all three is holding three OpenMP runtimes. When one of
    them opens a parallel region, the master can end up waiting on a barrier owned by a
    *different* runtime than the worker pool it is waiting for — neither side ever
    signals the other, and the process parks at 0% CPU forever. ``KeyboardInterrupt``
    cannot rescue it, because the main thread is blocked in native code and never
    returns to the interpreter to notice the signal.

    A team size of one makes faiss run the region on the calling thread with no barrier,
    so there is nothing to deadlock. Measured cost is nil at the sizes this is used on
    (30k cells x 80 features embeds in ~5s either way), which is a good trade against a
    hang that can only be cleared by killing the kernel.

    Restoring the previous count is the reason this is a context manager rather than a
    one-shot call: pacmap sets the same knob itself whenever it is given a
    ``random_state`` and never puts it back, silently leaving faiss single-threaded for
    the rest of the session.

    A no-op when faiss is not installed, which is the case for every backend except
    pacmap and localmap.
    """
    try:
        import faiss
    except ImportError:
        yield
        return
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


@dataclass(frozen=True)
class EmbeddingView:
    """A stored embedding's coordinates together with the columns that address them.

    Plotting an embedding needs two things that used to be fetched separately: the tidy
    frame, and the names of the coordinate columns inside it. Since those columns are
    prefixed with the embedding's name — ``umap_core0``, ``umap_core1`` — reconstructing
    them by hand meant writing that name three times per call, twice as a string prefix and
    once as ``dataframe(embedding=…)``. This carries both, so it is written once.

    Every consumer of these coordinates passes column *names* to a plotting call with
    ``data=`` rather than passing arrays, and several use them as polars predicates rather
    than as axes at all, so names are what this hands back:

    >>> v = ft.embedding_view("l23", name="umap_core", labels=lbl)   # doctest: +SKIP
    >>> sns.scatterplot(**v.xy, data=v.frame, hue=lbl.name,          # doctest: +SKIP
    ...                 palette=lbl.color_map())
    >>> v.frame.filter(pl.col(v.x) > 10)                             # doctest: +SKIP

    On dimensionality: :attr:`coords`, indexing and :attr:`n_components` describe however
    many components the embedding actually has, while :attr:`xy` names its own
    two-dimensionality rather than pretending the rest do not exist. A three-component
    embedding plotted through :attr:`xy` shows the first two axes; :meth:`pair` makes any
    other projection a stated choice.

    Attributes
    ----------
    name : str
        The embedding's stored name.
    mask : str
        The mask whose cells it covers.
    frame : pl.DataFrame
        The tidy view — metadata, features, coordinates, and any ``labels`` joined on.
    coords : tuple of str
        The coordinate column names within ``frame``, in component order.
    """

    name: str
    mask: str
    frame: pl.DataFrame
    coords: tuple[str, ...]

    @property
    def n_components(self) -> int:
        """How many components the stored embedding has."""
        return len(self.coords)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> str:
        """The column addressing component ``index``."""
        return self._axis(index)

    def __iter__(self) -> Iterator[str]:
        return iter(self.coords)

    @property
    def x(self) -> str:
        """The first component's column — an alias for ``view[0]``."""
        return self._axis(0)

    @property
    def y(self) -> str:
        """The second component's column — an alias for ``view[1]``."""
        return self._axis(1)

    @property
    def xy(self) -> dict[str, str]:
        """The first two components, ready to splat into a plotting call.

        Named for the pair it returns, so reaching for it on a higher-dimensional
        embedding is visibly a 2-D projection rather than an oversight. Use :meth:`pair`
        for any other two axes.

        Returns
        -------
        dict of str to str
            ``{"x": coords[0], "y": coords[1]}``. Raises when the embedding has fewer
            than two components.
        """
        return {"x": self._axis(0), "y": self._axis(1)}

    def pair(self, i: int = 0, j: int = 1) -> dict[str, str]:
        """``{"x": …, "y": …}`` for any two components.

        Parameters
        ----------
        i, j : int, default 0 and 1
            Which components to put on each axis.

        Returns
        -------
        dict of str to str
            Splats into any call taking ``x=`` and ``y=``.
        """
        return {"x": self._axis(i), "y": self._axis(j)}

    def _axis(self, index: int) -> str:
        if not -self.n_components <= index < self.n_components:
            raise IndexError(
                f"embedding {self.name!r} has {self.n_components} component(s), so "
                f"component {index} does not exist. Re-run ft.embed(..., "
                f"n_components={index + 1}) if you meant to compute it."
            )
        return self.coords[index]

    def __repr__(self) -> str:
        return (
            f"EmbeddingView(name={self.name!r}, mask={self.mask!r}, "
            f"n_components={self.n_components}, coords={self.coords})"
        )


def _pacmap_reduction(n_features: int, kwargs: dict[str, Any]) -> str | None:
    """Whether pacmap/localmap will reduce the input before constructing pairs.

    Mirrors their own condition — ``distance != "hamming" and n_features > 100 and
    apply_pca`` — so the recorded provenance says what actually happened rather than what
    was requested. Below 100 features the flag has no effect at all, which is worth
    knowing before treating it as a knob.
    """
    if not kwargs.get("apply_pca", True):
        return None
    if kwargs.get("distance", "euclidean") == "hamming":
        return None
    return "tsvd(100)" if n_features > 100 else None


def _default_scaler_factory():
    from sklearn.preprocessing import StandardScaler

    return StandardScaler()


def _validate_scaler_factory(factory: Any) -> Any:
    """Check ``scaler_factory`` is a zero-arg callable yielding a fresh scaler.

    Guards the easy mistake of passing a scaler *instance* (e.g.
    ``make_clipped_scaler()``) instead of the factory itself
    (``make_clipped_scaler``), which would otherwise fail obscurely at scale time.
    """
    if not callable(factory):
        raise TypeError(
            "scaler_factory must be a zero-argument callable returning a fresh "
            f"scaler, not a {type(factory).__name__} instance. Pass the factory "
            "itself (e.g. make_clipped_scaler) rather than calling it "
            "(make_clipped_scaler()); use clipped_scaler_factory(...) to set "
            "custom percentiles."
        )
    try:
        scaler = factory()
    except TypeError as error:
        raise TypeError(
            "scaler_factory must be callable with no arguments; "
            f"calling it raised: {error}"
        ) from error
    if not (hasattr(scaler, "fit") and hasattr(scaler, "transform")):
        raise TypeError(
            "scaler_factory must return an object with fit/transform methods "
            f"(a scikit-learn scaler or Pipeline), got {type(scaler).__name__}"
        )
    return factory


def _dedup(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))


def _columns_label(
    columns: str | FeatureCollection | Sequence[str] | None,
) -> str | None:
    """A human name for ``columns``, if it carries one (a str or FeatureCollection)."""
    if isinstance(columns, FeatureCollection):
        return columns.name
    if isinstance(columns, str):
        return columns
    return None


def _labelled(stored: Any, cluster_id: int, name: str) -> Any:
    """A ``Label`` for ``cluster_id``, keeping a stored color/description if any.

    The table column is authoritative for the name; everything else comes from
    what ``attach`` recorded, since a column can't hold it.
    """
    from cellpax.labels import Label

    if stored is None:
        return Label(id=cluster_id, name=name)
    return Label(
        id=cluster_id,
        name=name,
        color=stored.color,
        description=stored.description,
    )


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
        seed: int = 0,
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
        self._scaler_factory = _validate_scaler_factory(
            scaler_factory or _default_scaler_factory
        )
        self._scaler_cache: dict[tuple[str, tuple[str, ...]], FittedScaler] = {}
        # the scaling+PCA a clustering is computed in, keyed on the *fit* only: alpha and
        # eigenvalue_floor are views over it, so a whitening sweep cannot refit the space
        self._space_cache: dict[
            tuple[str, tuple[str, ...], float, Any], FittedSpace
        ] = {}
        self._collections: dict[str, FeatureCollection] = {}
        self._transforms: dict[str, str | None] = {}
        self._clusterings: dict[str, Any] = {}
        self._embeddings: dict[tuple[str, str], pl.DataFrame] = {}
        # the estimators behind those coordinates, so new cells can be projected in;
        # session-only, unlike the coordinates, which persist
        self._embedding_models: dict[tuple[str, str], FittedEmbedding] = {}
        # coordinates registered via add_embedding: (space label, n_components). Session
        # only, like the models — the coordinates themselves persist either way.
        self._external_embeddings: dict[tuple[str, str], tuple[str, int]] = {}
        # the call behind each embedding's coordinates (method, seed, kwargs);
        # persists in the manifest even though the fitted model does not
        self._embedding_params: dict[tuple[str, str], dict[str, Any]] = {}
        # attached label column -> {"mask": str | None, "labels": {id: Label}},
        # the cluster identity a name + id column pair can't carry on its own
        self._label_meta: dict[str, dict[str, Any]] = {}
        # feature -> mask name: the cells this feature is *informative* for.
        # Absent means valid everywhere. See set_validity.
        self._validity: dict[str, str] = {}
        self._seed = int(seed)

    @property
    def seed(self) -> int:
        """The table-level seed every stochastic verb derives its own seed from."""
        return self._seed

    def _derive_seed(self, op: str, mask: str | None, name: str | None) -> int:
        """A deterministic per-call seed from the table seed and the call's identity.

        Derived from what the *user* names — the verb, the mask, and the run name —
        and deliberately not from the data or the feature list: comparing two
        feature sets under the same name should hold the seed fixed so the features
        are the only thing that changed. Uses ``crc32`` rather than ``hash()``,
        which is salted per process and would break across sessions.
        """
        parts = [self._seed & 0xFFFFFFFF] + [
            zlib.crc32(str(part).encode())
            for part in (op, mask or _DEFAULT_MASK, name or "")
        ]
        return int(np.random.SeedSequence(parts).generate_state(1)[0])

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

    @property
    def labels(self) -> list[str]:
        """Names of label columns previously joined in via ``attach``.

        Attached labels are recorded directly, so this never mistakes a
        coincidental ``{name}`` + ``{name}_id`` column pair (``root`` /
        ``root_id``, say) for a label. The name-pair convention is used only as
        a fallback for tables whose label metadata predates the record, and
        then only where the name column is string-typed and the id column is
        integer-typed — the shape ``attach`` actually writes.
        """
        cols = set(self.columns)
        recorded = [c for c in self._label_meta if c in cols and f"{c}_id" in cols]
        if recorded:
            return recorded
        return [
            c
            for c in self.columns
            if f"{c}_id" in cols
            and self._df.schema[c] in (pl.Utf8, pl.Categorical, pl.Enum)
            and self._df.schema[f"{c}_id"].is_integer()
        ]

    def mask_series(self, mask: str | None = None) -> pl.Series:
        """The boolean membership Series for a mask."""
        name = mask or _DEFAULT_MASK
        column = _mask_column(name)
        if column not in self._df.columns:
            raise KeyError(f"Unknown mask {name!r}; available: {self.masks}")
        return self._df[column]

    def _cell_ids(self, mask: str | None = None) -> np.ndarray:
        """Id-column values for a mask's cells, in ``features(mask)`` row order."""
        return self._df.filter(self.mask_series(mask))[self._id_column].to_numpy()

    # -- construction / mutation ----------------------------------------------

    def add_column(
        self,
        data: Sequence[Any] | np.ndarray | pl.Series,
        name: str,
        *,
        mask: str | None = None,
        fill_value: Any = None,
        overwrite: bool = False,
    ) -> "FeatureTable":
        """Add or replace a column, writing ``data`` into a masked subset.

        ``data`` must have one value per True entry of ``mask``; the remaining
        rows are filled with ``fill_value``. Dtype is inferred by polars.

        Feature columns and the id column cannot be replaced this way: a feature
        write would leave every scaler, space, and model fit on the old values
        serving stale numbers, and an id write would bypass the uniqueness
        invariant. Pass ``overwrite=True`` to replace a *feature* deliberately —
        the fits that saw the old values are dropped. The id column always
        refuses; use ``set_id_column``.
        """
        if name.startswith(_MASK_PREFIX):
            raise ValueError(f"Column names cannot start with {_MASK_PREFIX!r}")
        if name == self._id_column:
            raise ValueError(
                f"{name!r} is the id column; use set_id_column to re-key the table"
            )
        if name in self._features and not overwrite:
            raise ValueError(
                f"{name!r} is a feature column; pass overwrite=True to replace its "
                f"values (fits that saw the old values will be dropped)"
            )
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
        if name in self._features:
            self._invalidate_feature_fits(f"feature {name!r} was overwritten")
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

        A null result counts as ``False`` — a cell the predicate can't decide isn't
        in the subset. That's what makes masking on a label column work directly
        (``pl.col("subclass_nn") == "L23IT"``) even though unassigned cells compare
        null, which is the move that carves the next round of clustering out of a
        propagated label.
        """
        if not name or name.startswith(_MASK_PREFIX) or name == _DEFAULT_MASK:
            raise ValueError(f"Invalid mask name {name!r}")
        if "/" in name:
            raise ValueError(
                f"Invalid mask name {name!r}: '/' would corrupt the item paths a "
                f"saved analysis stores embeddings under"
            )
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
            series = series.fill_null(False)
        if based_on is not None:
            series = series & self.mask_series(based_on)
        self._df = self._df.with_columns(series.alias(_mask_column(name)))
        self._invalidate_scaler_cache(name)
        return self

    def drop_mask(self, name: str) -> "FeatureTable":
        """Remove a named mask, the inverse of ``add_mask``.

        Drops the mask's boolean column and any scalers cached for it. The
        implicit ``"all"`` mask can't be dropped. Masks defined ``based_on``
        this one were already flattened to their own boolean column at
        creation time, so they're unaffected.
        """
        if name == _DEFAULT_MASK:
            raise ValueError(f"Cannot drop the implicit {_DEFAULT_MASK!r} mask")
        column = _mask_column(name)
        if column not in self._df.columns:
            raise KeyError(f"Unknown mask {name!r}; available: {self.masks}")
        backed = sorted(f for f, m in self._validity.items() if m == name)
        if backed:
            raise ValueError(
                f"mask {name!r} is the validity domain of {backed}; clear or move "
                f"those domains (set_validity) before dropping it"
            )
        self._df = self._df.drop(column)
        self._invalidate_scaler_cache(name, reason=f"mask {name!r} was dropped")
        return self

    def _invalidate_scaler_cache(self, mask: str, *, reason: str | None = None) -> None:
        self._scaler_cache = {
            key: fitted for key, fitted in self._scaler_cache.items() if key[0] != mask
        }
        # an embedding model fit in the old scaled space would silently project new
        # cells into coordinates the stored ones no longer share, so it goes too
        self._embedding_models = {
            key: model
            for key, model in self._embedding_models.items()
            if key[0] != mask
        }
        # same argument for the clustering space: it holds the old scaler, so keeping it
        # would freeze preprocessing that no longer exists
        self._space_cache = {
            key: space for key, space in self._space_cache.items() if key[0] != mask
        }
        # ...and for stored results: coordinates and clusterings computed on the old
        # membership/scaling would be served as current. External embeddings stay —
        # they were never derived from the scaled features.
        stale_embeddings = [
            key
            for key in self._embeddings
            if key[0] == mask and key not in self._external_embeddings
        ]
        stale_clusterings = [
            name for name, clus in self._clusterings.items() if clus.mask == mask
        ]
        self._drop_stale_results(
            stale_embeddings,
            stale_clusterings,
            reason or f"mask {mask!r} was redefined",
        )

    def _invalidate_feature_fits(self, reason: str) -> None:
        """Drop every fit and stored result whose scaled space no longer exists."""
        self._scaler_cache = {}
        self._embedding_models = {}
        self._space_cache = {}
        stale_embeddings = [
            key for key in self._embeddings if key not in self._external_embeddings
        ]
        self._drop_stale_results(stale_embeddings, list(self._clusterings), reason)

    def _drop_stale_results(
        self,
        embedding_keys: list[tuple[str, str]],
        clustering_names: list[str],
        reason: str,
    ) -> None:
        for key in embedding_keys:
            del self._embeddings[key]
            self._embedding_params.pop(key, None)
        for name in clustering_names:
            del self._clusterings[name]
        dropped = [f"embedding {m!r}/{n!r}" for m, n in embedding_keys]
        dropped += [f"clustering {n!r}" for n in clustering_names]
        if dropped:
            warnings.warn(
                f"{reason}: dropped {', '.join(dropped)} — they were computed in a "
                f"space that no longer exists; re-run embed()/cluster() (a saved "
                f"analysis is unaffected until you save over it)",
                stacklevel=3,
            )

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
        previous = self._id_column
        self._id_column = name
        if previous != name:
            self._rekey_stored_results(previous, name)
        return self

    def _rekey_stored_results(self, old: str, new: str) -> None:
        """Carry stored embeddings and clusterings across an id-column change.

        Both hold cell ids — embedding frames as a column, clusterings as an
        array — so leaving them keyed by the old ids would break every read
        that joins on the new ones.
        """
        if not self._embeddings and not self._clusterings:
            return
        mapping_frame = self._df.select(old, new)
        rekeyed = {}
        for key, frame in self._embeddings.items():
            if old in frame.columns:
                coordinate_columns = [c for c in frame.columns if c != old]
                frame = frame.join(mapping_frame, on=old, how="left").select(
                    new, *coordinate_columns
                )
            rekeyed[key] = frame
        self._embeddings = rekeyed
        mapping = dict(zip(mapping_frame[old].to_list(), mapping_frame[new].to_list()))
        for clus in self._clusterings.values():
            clus._cell_ids = np.asarray([mapping[c] for c in clus._cell_ids])

    def add_features(
        self,
        source: pl.DataFrame,
        features: Sequence[str] | None = None,
        *,
        on: str | None = None,
        feature_metadata: pl.DataFrame | None = None,
        allow_missing: bool = False,
        collection: str | None = None,
    ) -> "FeatureTable":
        """Join additional feature columns from another source and register them.

        ``source`` is keyed on ``on`` (default: the id column). ``features``
        defaults to every column of ``source`` except ``on``; pass a list to
        select a subset. They are left-joined onto the table and added to the
        feature set; ``feature_metadata`` (a ``feature_id`` + attribute frame)
        extends ``var`` for them. By default every cell must be covered
        (``allow_missing=True`` permits nulls, which then can't be scaled). Pass
        ``collection`` to also define a feature collection of exactly these
        features in the same call.
        """
        if not isinstance(source, pl.DataFrame):
            source = pl.from_pandas(source)
        on = on or self._id_column
        if on not in source.columns or on not in self._df.columns:
            raise ValueError(f"join key {on!r} must be in both the table and source")
        if features is None:
            features = [c for c in source.columns if c != on]
        else:
            features = list(features)
        if not features:
            raise ValueError("no feature columns to add")
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
        reserved = [f for f in features if f.startswith(_MASK_PREFIX)]
        if reserved:
            raise ValueError(
                f"feature names cannot start with {_MASK_PREFIX!r}: {reserved}"
            )

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
        self._embedding_models.clear()
        self._space_cache.clear()
        if collection is not None:
            self.define_features(collection, columns=features)
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
        valid_where: str | None = None,
    ) -> "FeatureTable":
        """Define a named, composable feature collection.

        Provide exactly one selector: an explicit ``columns`` list, a ``family`` or
        ``modality`` value(s) (requires that column in ``feature_metadata``), or a
        polars ``predicate`` over the feature metadata.

        ``valid_where`` names a mask and declares the selected features' validity
        domain in the same call — shorthand for a following ``set_validity``. See
        :meth:`set_validity` for what a validity domain means.
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
        if valid_where is not None:
            self.set_validity(columns=chosen, where=valid_where)
        return self

    # -- validity domains --------------------------------------------------------

    def set_validity(
        self,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        *,
        where: str | None,
    ) -> "FeatureTable":
        """Declare the validity domain of features: the mask where they inform.

        A feature can be measurable for a cell and still be meaningless for it —
        a truncated reconstruction yields an axon length, it just says nothing
        about the cell's type. That distinction cannot be recovered from the
        values (the numbers look fine), so it has to be declared, and this is
        the declaration: ``where`` names a mask, and the selected features are
        informative *only for cells in it*. Everywhere else they are garbage
        that happens to parse.

        The domain is a first-class input, not advice: ``propagate_labels``
        warns when asked to propagate on features whose domain doesn't cover
        the target, and ``ladder=`` uses the domains to give each cell the
        richest features that are actually valid for it. ``validity()`` /
        ``fully_valid()`` / ``validity_patterns()`` are the read side.

        ``where=None`` clears the selected features back to valid-everywhere.
        The mask must already exist (``add_mask`` first) — a domain is
        provenance, and a named mask is the only form of it that persists,
        composes with ``based_on``, and shows in ``ft.masks``. A mask that
        backs a validity domain can't be dropped while it does.
        """
        chosen = self._resolve_columns(columns)
        if where is None:
            for column in chosen:
                self._validity.pop(column, None)
            return self
        if where == _DEFAULT_MASK:
            raise ValueError(
                "the 'all' mask is the default domain; pass where=None to clear"
            )
        self.mask_series(where)  # raises with the available masks if unknown
        for column in chosen:
            self._validity[column] = where
        return self

    @property
    def validity_domains(self) -> dict[str, str]:
        """Feature → mask name, for features with a restricted validity domain."""
        return dict(self._validity)

    def validity(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> np.ndarray:
        """Boolean ``(n_cells, n_features)`` validity, aligned with ``features()``.

        Entry ``(i, j)`` is whether column *j* is informative for cell *i* of
        ``mask`` — membership of the cell in the column's validity mask, or
        ``True`` everywhere for a column with no declared domain.
        """
        cols = self._resolve_columns(columns)
        member = self.mask_series(mask).to_numpy()
        out = np.ones((int(member.sum()), len(cols)), dtype=bool)
        for j, column in enumerate(cols):
            domain = self._validity.get(column)
            if domain is not None:
                out[:, j] = self.mask_series(domain).to_numpy()[member]
        return out

    def fully_valid(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> np.ndarray:
        """Per-cell boolean: every selected column is valid for this cell.

        The coverage question a collection answers as a whole — ``mask`` row
        order, ready to combine with ``features(mask)``.
        """
        return self.validity(mask, columns=columns).all(axis=1)

    def validity_patterns(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """The distinct per-cell validity patterns and how many cells hold each.

        One row per pattern, largest first: ``n_cells``, ``n_invalid``, and
        ``invalid_features`` (the columns that are *not* valid under it).
        Truncation is positional, so in practice a table collapses to a handful
        of patterns — the shape that makes per-pattern models viable later.
        """
        cols = self._resolve_columns(columns)
        matrix = self.validity(mask, columns=cols)
        patterns, counts = np.unique(matrix, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        return pl.DataFrame(
            {
                "n_cells": counts[order].astype(np.int64),
                "n_invalid": (~patterns[order]).sum(axis=1).astype(np.int64),
                "invalid_features": [
                    [cols[j] for j in np.flatnonzero(~row)] for row in patterns[order]
                ],
            }
        )

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
        method: str | None = "ihs",
        threshold: float = 1.5,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> "FeatureTable":
        """Resolve per-feature transforms applied before scaling (unified layer).

        With ``skew_screen`` (default), each selected feature whose right-skewness
        exceeds ``threshold`` is transformed with ``method`` (``"ihs"`` by default —
        it handles zeros and negatives; ``"log"``/``"sqrt"`` are skipped for
        features with negative values). With ``skew_screen=False`` the screen is
        bypassed and ``method`` is applied to *every* selected feature (still
        skipping ``"log"``/``"sqrt"`` on negatives) — the way to set a transform
        explicitly. ``method=None`` clears the selected features back to identity.
        Transforms are recorded per feature and applied whenever features are
        scaled; changing them drops every fit and stored result computed under
        the old transforms.
        """
        if method is not None and method not in {"ihs", "log", "sqrt"}:
            raise ValueError("method must be 'ihs', 'log', 'sqrt', or None")
        for column in self._resolve_columns(columns):
            transform: str | None = None
            if method is not None:
                sample = self._df[column].drop_nulls().to_numpy()
                eligible = not (
                    method in {"log", "sqrt"}
                    and sample.size
                    and float(sample.min()) < 0
                )
                if skew_screen:
                    if sample.size and _skew(sample) > threshold and eligible:
                        transform = method
                elif eligible:
                    transform = method
            self._transforms[column] = transform
        self._invalidate_feature_fits("preprocess() changed the per-feature transforms")
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
        graph_type: str | Sequence[str] = "knn",
        prune: float = 0.0,
        metric: str = "minkowski",
        normalize: bool = True,
        method: str = "average",
        pca: bool | float = 0.95,
        alpha: float = 0.0,
        eigenvalue_floor: float = 0.0,
        seed: int | None = None,
        n_jobs: int = -1,
        name: str | None = None,
        order_by: str | None = None,
        order_agg: Literal["mean", "median"] = "mean",
        order_ascending: bool = True,
    ) -> Any:
        """Consensus-cluster a mask's scaled features into a ``Clustering``.

        Parameters
        ----------
        mask : str, optional
            Which cells to cluster. ``None`` uses every cell.
        columns : str or FeatureCollection or sequence of str, optional
            Which features to compare on. ``None`` uses every feature.
        n_neighbors : int or sequence of int, default 30
            Neighbourhood size for the kNN graph. A sequence sweeps it as a consensus
            axis; a population smaller than ``n_neighbors`` cannot form its own community.
        resolution : float or sequence of float, default 1.0
            Leiden resolution. A sequence sweeps it. See the notes — this is the parameter
            that bounds how many clusters are reachable at all.
        n_times : int, default 1
            Repeats per ``(graph_type, n_neighbors, resolution)`` combination, each with
            its own seed.
        min_cluster_size : int, default 1
            Clusters smaller than this become ``-1`` in each run.
        mutual_only : bool, default False
            Keep only mutual kNN edges. Applies to ``graph_type="knn"`` only; the weighted
            forms are symmetrised by their own rule.
        graph_type : str or sequence of str, default 'knn'
            Edge weighting: ``'knn'`` (unweighted), ``'knn_distance'``, ``'snn_jaccard'``,
            or ``'umap_fuzzy'``. A sequence sweeps it as a third consensus axis. See
            :func:`~cellpax.clustering.kneighbor_graph` for how they differ.
        prune : float, default 0.0
            Jaccard floor for ``graph_type="snn_jaccard"``: edges at or below it are
            dropped. Ignored by every other weighting. Seurat uses ``1/15``. Raising it
            denoises dense regions but can strand cells in sparse ones, which
            ``kneighbor_graph`` counts as ``n_isolated`` and warns about.
        normalize : bool, default True
            Divide co-clustering counts by the runs that kept both cells, giving
            similarities in ``[0, 1]`` rather than raw counts.
        method : {'average', 'single', 'complete'}, default 'average'
            Linkage for the hierarchy built over the consensus.
        pca : bool or float, default 0.95
            Cumulative explained variance to keep before building the graph, or ``False``
            for the raw scaled features at full dimensionality. ``True`` means ``0.95``.
        alpha : float, default 0.0
            Whitening strength: component *j* is scaled by ``λ_j ** (-alpha/2)``. ``0.0``
            leaves PCA's own scaling alone, ``1.0`` equalises the retained components.
            Requires ``pca`` to be on.
        eigenvalue_floor : float, default 0.0
            Added to each eigenvalue before the ``alpha`` scaling, bounding how much a
            near-degenerate component can be amplified. ``ft.space(mask).noise_floor`` is
            the recommended value whenever ``alpha > 0``.
        seed : int, optional
            Seeds every run reproducibly.
        n_jobs : int, default -1
            Parallel workers for the Leiden runs.
        name : str, optional
            Store the result under this name for later ``label`` / ``clustering`` /
            ``compare`` use, and for persistence.
        order_by : str, optional
            Column whose per-cluster aggregate orders the cluster ids at every cut.
        order_agg : {'mean', 'median'}, default 'mean'
            How ``order_by`` is aggregated within a cluster.
        order_ascending : bool, default True
            Direction of that ordering, so cluster 0 is the smallest by default.

        Returns
        -------
        Clustering
            A :class:`~cellpax.clustering.Clustering` — a ``SimilarityMatrix`` that also
            remembers the mask, cell ids, columns and space, so
            ``clus.label(distance_threshold=…)`` needs no mask restated, and that carries
            the individual runs as ``clus.partitions``.

        Notes
        -----
        Clustering runs on the mask's *scaled* features, so ``preprocess`` transforms and
        the per-mask scaler apply, reduced by PCA before the kNN graph is built.

        ``resolution`` is the one parameter that decides how many clusters you can
        possibly get, because the consensus is never coarser than the runs it
        averages. It defaults to Leiden's own ``1.0``, which on a kNN graph of a few
        thousand cells splits into a dozen or more communities per run.

        Sweeping wide does not buy you every grain at once. Pooled runs of differing
        grain average together, so two cells that only the coarse runs group land at
        the fraction of runs that were coarse enough — a middling similarity that then
        needs exactly the threshold you were hoping not to have to guess. The way out
        is to sweep geometrically, ``np.geomspace(0.02, 1.5, 12)``, so the low end is
        sampled densely enough to resolve the coarse structure; read
        ``clus.partitions.by_setting()``, where cluster count plateaus against
        resolution and each plateau is a grain the data supports; then cut one band at
        a time with ``clus.restrict(resolution_max=…)``. Within a band the runs agree,
        the consensus is near-binary, and every threshold gives the same answer.

        ``pca`` is the explained variance kept — ``0.95`` by default, matching
        ``propagate_labels``, so clusters and the propagation of those clusters live
        in one space. This is the space phenograph-style clustering is conventionally
        run in: correlated features stop each counting separately toward the neighbor
        distances, and the graph is built on a denoised, cheaper matrix. Pass
        ``False`` for the raw scaled features at full dimensionality.

        A reduced space can split more finely than the full one — components that
        contributed little variance also contributed little separation, so groups
        that held together on their strength come apart. Re-read
        ``threshold_scan()`` when changing ``pca`` rather than reusing a threshold
        chosen under the other setting.

        ``alpha`` partially whitens the retained components, scaling component *j* by
        ``λ_j ** (-alpha/2)``. ``0.0`` is the default and leaves component scales as PCA
        produced them. Worth sweeping because truncating PCA is a rotation plus a
        truncation and not a reweighting: a block of features measuring one thing several
        ways collapses into a single high-eigenvalue component that then dominates
        Euclidean distance exactly as much as the raw block did. With dozens of
        engineered features rather than thousands of genes that does not average out.
        ``1.0`` equalises the retained components entirely.

        Two things to watch with ``alpha``. Whitening amplifies the *smallest retained*
        component most, so a truncation chosen by cumulative variance becomes a
        discontinuity — the last kept component gets full weight and the first dropped
        one gets none. ``eigenvalue_floor=ft.space(mask).noise_floor`` bounds that. And
        the clustering space diverges from the space ``embed`` builds a UMAP in, so at
        ``alpha > 0`` labels will look *worse* on an existing UMAP even when the
        clustering improved; score the change on ``cellpax.validate``, not on the figure.

        ``graph_type`` selects the edge weighting, and may be a list to sweep it as a
        third consensus axis alongside resolution and seed — see
        :func:`~cellpax.clustering.kneighbor_graph` for the three weightings and their
        different failure modes. Sweeping it marginalises over a choice the consensus
        otherwise takes on faith. But resolution is not comparable across weightings, so
        pool on realised grain (``clus.restrict(n_clusters_min=…, n_clusters_max=…)``)
        and check ``axis_stability(clus.partitions, reference)`` (a module-level
        function) to see that no one weighting is being averaged in against the rest.

        ``order_by`` names a column — typically ``"soma_depth_um"`` — whose values
        are carried onto the ``Clustering`` so every ``label()`` cut of it comes back
        with cluster ids already ordered by that column's per-cluster ``order_agg``
        (``ascending`` by default, so cluster 0 is the shallowest). Ordering at cut
        time rather than after the fact is what keeps two thresholds of the same
        clustering numbered consistently; without it each cut numbers clusters in
        whatever order the dendrogram happens to produce.
        """
        from cellpax.clustering import Clustering, fauxnograph_coclustering

        cols = self._resolve_columns(columns)
        # the space seed keeps the user's value (None = deterministic full-SVD fit,
        # shared across runs); only the ensemble gets a derived per-call seed, so a
        # sweep of named runs still shares one PCA rather than refitting it
        resolved_seed = (
            seed if seed is not None else self._derive_seed("cluster", mask, name)
        )
        if pca is False:
            if alpha:
                raise ValueError(
                    "alpha weights PCA components, so it needs pca to be on; pass "
                    "pca=0.95 (or a component count) alongside alpha"
                )
            data = self.features(mask, scaled=True, columns=cols)
            space = "scaled"
        else:
            variance = 0.95 if pca is True else float(pca)
            fitted = self.space(
                mask,
                columns=cols,
                explained_variance=variance,
                alpha=alpha,
                eigenvalue_floor=eigenvalue_floor,
                seed=seed,
            )
            data = fitted.transform_scaled(
                self.features(mask, scaled=True, columns=cols)
            )
            space = fitted.label
        matrix, partitions = fauxnograph_coclustering(
            data,
            return_partitions=True,
            n_neighbors=list(n_neighbors)
            if isinstance(n_neighbors, (list, tuple))
            else n_neighbors,
            resolution_parameter=list(resolution)
            if isinstance(resolution, (list, tuple))
            else resolution,
            graph_type=graph_type,
            prune=prune,
            metric=metric,
            n_times=n_times,
            min_cluster_size=min_cluster_size,
            mutual_only=mutual_only,
            normalize=normalize,
            seed=resolved_seed,
            n_jobs=n_jobs,
        )
        params = {
            "columns": list(cols),
            "n_neighbors": list(np.atleast_1d(n_neighbors).tolist()),
            "resolution": list(np.atleast_1d(resolution).tolist()),
            "n_times": n_times,
            "min_cluster_size": min_cluster_size,
            "mutual_only": mutual_only,
            "graph_type": list(np.atleast_1d(graph_type).tolist()),
            "prune": prune,
            "metric": metric,
            "normalize": normalize,
            "method": method,
            "pca": pca,
            "alpha": alpha,
            "eigenvalue_floor": eigenvalue_floor,
            "seed": resolved_seed,
        }
        order_values = None
        if order_by is not None:
            if order_by not in self._df.columns:
                raise KeyError(f"Unknown column {order_by!r} for order_by")
            frame = self._df.filter(self.mask_series(mask))
            order_values = frame[order_by].cast(pl.Float64).to_numpy()
        result = Clustering(
            matrix,
            cell_ids=self._cell_ids(mask),
            mask=mask or _DEFAULT_MASK,
            columns=tuple(cols),
            space=space,
            normalized=normalize,
            method=method,
            order_by=order_by,
            order_values=order_values,
            order_agg=order_agg,
            order_ascending=order_ascending,
            partitions=partitions,
            params=params,
        )
        if name is not None:
            self._clusterings[name] = result
        return result

    def neighborhood_purity(
        self,
        labels: Any,
        *,
        mask: str | None = None,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        n_neighbors: int = 20,
    ) -> pl.DataFrame:
        """Per-cell label purity among self-excluded nearest feature-space neighbors.

        ``labels`` is a ``LabelSet`` or the name of an (attached) label column
        (see ``labelset``). For each cell, looks at its ``n_neighbors`` nearest
        *other* cells in scaled feature space and reports the fraction that
        share its label — 1.0 means a cell's whole neighborhood agrees with
        it, 0.0 means none of it does. A quick check of how well a clustering
        respects local structure (ported from dfc's neighborhood purity).
        """
        from cellpax.clustering import neighborhood_purity as _neighborhood_purity

        labels = self._resolve_labels(labels, mask=mask)
        name = mask or (labels.mask if labels is not None else None) or _DEFAULT_MASK
        cell_ids = self._cell_ids(name)
        label_array = self._align_over_clustering(labels, cell_ids)
        features = self.features(name, scaled=True, columns=columns)
        purity = _neighborhood_purity(features, label_array, n_neighbors=n_neighbors)
        return pl.DataFrame({self._id_column: cell_ids, "purity": purity})

    def propagate_labels(
        self,
        labels: Any,
        *,
        to: str | None = None,
        method: Literal["vote", "spread"] = "vote",
        columns: str | FeatureCollection | Sequence[str] | None = None,
        ladder: Sequence[Any] | None = None,
        n_neighbors: int = 30,
        pca: bool | float = 0.95,
        weights: Literal["uniform", "distance"] | None = None,
        preserve_labeled: bool = True,
        min_confidence: float | None = None,
        mutual: bool = True,
        alpha: float = 0.8,
        agreement_folds: int | None = None,
        on_invalid: Literal["warn", "raise", "ignore"] = "warn",
        name: str | None = None,
        seed: int | None = None,
    ) -> Any:
        """Carry a curated subset's labels out to a larger population.

        The dfc workflow: cluster a high-quality core, then label everything that
        looks like it. ``labels`` is a ``LabelSet`` (or an attached column name)
        over a subset of mask ``to``. Returns a
        :class:`~cellpax.propagate.Propagation` — the new ``LabelSet`` (same
        cluster ids, names and colors as ``labels``, named ``{labels.name}_nn`` by
        default), per-cell confidence, and ``self_agreement()``. Nothing is
        attached; ``ft.attach`` it when you're happy.

        ``method="vote"`` (default) gives each cell the majority label of its
        ``n_neighbors`` nearest labeled cells — cheap, and every cell gets a label.
        ``method="spread"`` diffuses labels along a mutual-nearest-neighbor graph
        instead: evidence scales with how much labeled signal is actually nearby,
        and a cell with no mutual path to any labeled cell stays unassigned, which
        is the right answer for a cell unlike anything in the reference. ``mutual``
        and ``alpha`` apply to ``"spread"``; ``weights`` defaults to ``"uniform"``
        for the vote and ``"distance"`` for diffusion.

        One feature space is fit over all of ``to`` — scaling and PCA are
        mask-relative, so the reference must live inside ``to`` (this raises
        otherwise) rather than being scaled on its own. ``pca`` is the explained
        variance kept (``0.95`` by default, ``False`` for raw scaled features).
        ``columns`` narrows to a feature collection, which is how you restrict
        propagation to features that are valid for every cell rather than only for
        the core — compare ``self_agreement()`` across column sets to see what a
        restricted set can still carry. That number is exact leave-one-out for the
        vote and 5-fold for diffusion; pass ``agreement_folds`` to put both on the
        same footing before comparing methods (see
        :class:`~cellpax.propagate.Recovery`). When the chosen columns carry
        declared validity domains (:meth:`set_validity`) that don't cover the
        whole target, this warns — the votes for the uncovered cells would be
        built from uninformative values; ``on_invalid`` escalates that to a
        raise or silences it.

        ``ladder`` replaces the single global column choice with a per-cell
        fallback: an ordered list of collections, richest first, and each cell
        is labeled with the first one whose columns are all *valid for it*.
        Each rung runs as its own propagation — its own scaler and PCA over
        only the participating cells — so a well-reconstructed cell keeps the
        distinctions the full feature set supports while a truncated one falls
        back to what its features can honestly say, and a cell no rung covers
        stays unassigned. The result records which rung labeled each cell
        (``result.rungs`` / a ``{name}_rung`` column in ``result.frame()``) and
        per-rung recovery (``result.rung_recovery``); confidence is comparable
        within a rung, not across rungs.

        With ``preserve_labeled`` (the default) the reference cells keep their own
        labels and only unlabeled cells are filled in. Pass ``False`` to relabel
        every cell from its neighborhood, which smooths a noisy clustering — dfc's
        ``preserve_original_labels=False``.

        ``min_confidence`` unassigns cells whose ``confidence`` — the winning
        label's share of the support that reached them, in ``[0, 1]`` — falls below
        the cut, the reference exempt under ``preserve_labeled``. There is no
        scale-free good value, so don't guess one: the winner among ``c`` locally
        competing clusters can't score below ``1/c``, and with ``weights="uniform"``
        the shares are quantized to ``1/n_neighbors``, so many cuts are exact
        no-ops. Calibrate instead — propagate once with ``preserve_labeled=False``
        (which puts the reference's confidence on the same footing as the rest) and
        read the kept-set error rate against the reference's known labels at each
        candidate cut; the guide's "Choosing ``min_confidence``" walks through it.
        Note it gates *ambiguity*, not *distance*: under ``"vote"`` a cell far from
        the entire reference still comes back unanimous at ``1.0``, and abstaining
        on those is what ``"spread"`` and ``mutual`` are for.
        """
        from cellpax.propagate import Propagation, propagate_knn, propagate_spread

        if method not in ("vote", "spread"):
            raise ValueError(f"method must be 'vote' or 'spread', got {method!r}")
        labels = self._resolve_labels(labels)
        target = to or _DEFAULT_MASK
        target_ids = self._cell_ids(target)
        reference_ids = labels.cell_ids[labels.assigned]
        outside = np.setdiff1d(reference_ids, target_ids)
        if outside.size:
            raise ValueError(
                f"{outside.size} of {reference_ids.size} reference cells are outside "
                f"mask {target!r}; propagation needs the reference inside the target "
                f"mask, so one feature space covers both"
            )
        if ladder is not None:
            if columns is not None:
                raise ValueError(
                    "pass either columns= (one feature set for every cell) or "
                    "ladder= (per-cell fallback through several), not both"
                )
            return self._propagate_ladder(
                labels,
                target=target,
                target_ids=target_ids,
                ladder=list(ladder),
                method=method,
                n_neighbors=n_neighbors,
                pca=pca,
                weights=weights,
                preserve_labeled=preserve_labeled,
                min_confidence=min_confidence,
                mutual=mutual,
                alpha=alpha,
                agreement_folds=agreement_folds,
                name=name,
                seed=seed,
            )
        invalid = ~self.fully_valid(target, columns=columns)
        if invalid.any() and on_invalid != "ignore":
            cols = self._resolve_columns(columns)
            offending = sorted(
                {
                    c
                    for c in cols
                    if self._validity.get(c) is not None
                    and not self.mask_series(self._validity[c])
                    .to_numpy()[self.mask_series(target).to_numpy()]
                    .all()
                }
            )
            message = (
                f"{int(invalid.sum())} of {len(target_ids)} cells in mask "
                f"{target!r} are outside the validity domain of propagation "
                f"columns {offending}; their votes and labels will be built from "
                f"uninformative values. Restrict columns= to a valid collection, "
                f"pass ladder= to fall back per cell, or on_invalid='ignore' to "
                f"accept it"
            )
            if on_invalid == "raise":
                raise ValueError(message)
            warnings.warn(message)
        if pca is False:
            features = self.features(target, scaled=True, columns=columns)
            space = "scaled"
        else:
            variance = 0.95 if pca is True else float(pca)
            features = self.features_pca(
                target, columns=columns, explained_variance=variance, seed=seed
            )
            space = f"pca({variance:g})"
        reference_codes = labels.codes_for(target_ids)
        if method == "vote":
            codes, confidence, recovery = propagate_knn(
                features,
                reference_codes,
                n_neighbors=n_neighbors,
                weights=weights or "uniform",
                preserve_labeled=preserve_labeled,
                min_confidence=min_confidence,
                agreement_folds=0 if agreement_folds is None else agreement_folds,
                seed=seed,
            )
        else:
            codes, confidence, recovery = propagate_spread(
                features,
                reference_codes,
                n_neighbors=n_neighbors,
                mutual=mutual,
                weights=weights or "distance",
                preserve_labeled=preserve_labeled,
                alpha=alpha,
                min_confidence=min_confidence,
                agreement_folds=5 if agreement_folds is None else agreement_folds,
                seed=seed,
            )
        propagated = labels.with_codes(
            target_ids, codes, name=name or f"{labels.name}_nn", mask=target
        )
        return Propagation(
            propagated,
            labels,
            confidence,
            recovery,
            method=method,
            n_neighbors=n_neighbors,
            space=space,
            params={
                "method": method,
                "columns": self._resolve_columns(columns),
                "n_neighbors": n_neighbors,
                "pca": pca,
                "weights": weights or ("uniform" if method == "vote" else "distance"),
                "preserve_labeled": preserve_labeled,
                "min_confidence": min_confidence,
                "mutual": mutual,
                "alpha": alpha,
                "agreement_folds": agreement_folds,
                "seed": seed,
            },
        )

    def _propagate_ladder(
        self,
        labels: Any,
        *,
        target: str,
        target_ids: np.ndarray,
        ladder: list[Any],
        method: str,
        n_neighbors: int,
        pca: bool | float,
        weights: str | None,
        preserve_labeled: bool,
        min_confidence: float | None,
        mutual: bool,
        alpha: float,
        agreement_folds: int | None,
        name: str | None,
        seed: int | None,
    ) -> Any:
        """Per-cell fallback propagation: each cell gets the richest valid rung.

        A global ``columns=`` choice coarsens every cell to the worst cell's
        validity. Here each rung is a collection, listed richest first; a cell
        is assigned the first *usable* rung whose columns are all valid for it
        (usable: at least two reference cells are valid on it too), and each
        rung runs as its own propagation — its own scaler and PCA fit over only
        the cells participating in it, so invalid values never contaminate a
        fit. Cells no usable rung covers stay unassigned: abstention is the
        honest answer for a cell whose informative features don't exist.
        """
        from cellpax.propagate import Propagation, propagate_knn, propagate_spread

        if not ladder:
            raise ValueError("ladder needs at least one collection")
        rung_columns = [self._resolve_columns(rung) for rung in ladder]
        rung_names = [
            rung
            if isinstance(rung, str)
            else getattr(rung, "name", None) or f"rung_{position}"
            for position, rung in enumerate(ladder)
        ]
        n = len(target_ids)
        reference_codes = labels.codes_for(target_ids)
        is_reference = reference_codes != -1

        valid = [self.fully_valid(target, columns=cols) for cols in rung_columns]
        usable = [(v & is_reference).sum() >= 2 for v in valid]
        if not any(usable):
            raise ValueError(
                "no ladder rung has at least two valid reference cells; check the "
                "validity domains against the reference"
            )
        skipped = [rung_names[r] for r in range(len(ladder)) if not usable[r]]
        if skipped:
            warnings.warn(
                f"ladder rungs {skipped} have fewer than two valid reference "
                f"cells and were skipped; cells fall through to the next rung"
            )

        rung = np.full(n, -1, dtype=np.int64)
        for position in range(len(ladder)):
            if not usable[position]:
                continue
            unclaimed = (rung == -1) & valid[position]
            rung[unclaimed] = position

        frame = self._df.filter(self.mask_series(target))
        codes = np.full(n, -1, dtype=np.int64)
        confidence = np.zeros(n, dtype=float)
        recoveries: dict[str, Any] = {}
        for position in range(len(ladder)):
            if not usable[position]:
                recoveries[rung_names[position]] = None
                continue
            voters = valid[position] & is_reference
            queried = rung == position
            rows = np.flatnonzero(voters | queried)
            matrix = frame.select(rung_columns[position]).to_numpy()[rows]
            transforms = [self._transforms.get(c) for c in rung_columns[position]]
            fitted = FittedScaler(
                scaler=self._scaler_factory(), transforms=transforms
            ).fit(matrix)
            data = fitted.transform(matrix)
            if pca is not False:
                from sklearn.decomposition import PCA

                variance = 0.95 if pca is True else pca
                if isinstance(variance, (int, np.integer)) and not isinstance(
                    variance, bool
                ):
                    components: int | float = min(
                        int(variance), data.shape[0], data.shape[1]
                    )
                else:
                    components = float(variance)
                data = PCA(n_components=components, svd_solver="full").fit_transform(
                    data
                )
            row_codes = np.where(voters[rows], reference_codes[rows], -1)
            # a rung whose queried cells are all reference cells (typical for the
            # richest rung: only the well-reconstructed core is fully valid on it)
            # runs as a probe — preserve_labeled would leave it nothing to do —
            # and the global preserve restores their codes below. Confidence for
            # those cells is therefore the probe's winner-share.
            run_preserve = preserve_labeled and bool((queried & ~is_reference).any())
            if method == "vote":
                out_codes, out_confidence, recovery = propagate_knn(
                    data,
                    row_codes,
                    n_neighbors=n_neighbors,
                    weights=weights or "uniform",
                    preserve_labeled=run_preserve,
                    min_confidence=min_confidence,
                    agreement_folds=0 if agreement_folds is None else agreement_folds,
                    seed=seed,
                )
            else:
                out_codes, out_confidence, recovery = propagate_spread(
                    data,
                    row_codes,
                    n_neighbors=n_neighbors,
                    mutual=mutual,
                    weights=weights or "distance",
                    preserve_labeled=run_preserve,
                    alpha=alpha,
                    min_confidence=min_confidence,
                    agreement_folds=5 if agreement_folds is None else agreement_folds,
                    seed=seed,
                )
            recoveries[rung_names[position]] = recovery
            take = queried[rows]
            codes[rows[take]] = out_codes[take]
            confidence[rows[take]] = out_confidence[take]

        if preserve_labeled:
            codes[is_reference] = reference_codes[is_reference]

        propagated = labels.with_codes(
            target_ids, codes, name=name or f"{labels.name}_nn", mask=target
        )
        return Propagation(
            propagated,
            labels,
            confidence,
            None,
            method=method,
            n_neighbors=n_neighbors,
            space=f"ladder({', '.join(rung_names)})",
            params={
                "method": method,
                "ladder": [list(cols) for cols in rung_columns],
                "ladder_names": rung_names,
                "n_neighbors": n_neighbors,
                "pca": pca,
                "weights": weights or ("uniform" if method == "vote" else "distance"),
                "preserve_labeled": preserve_labeled,
                "min_confidence": min_confidence,
                "mutual": mutual,
                "alpha": alpha,
                "agreement_folds": agreement_folds,
                "seed": seed,
            },
            rungs=rung,
            rung_names=rung_names,
            rung_recovery=recoveries,
        )

    def overcluster(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        resolution: float = 2.0,
        n_neighbors: int = 30,
        mutual_only: bool = False,
        graph_type: str = "knn",
        prune: float = 0.0,
        min_cluster_size: int = 1,
        space: FittedSpace | None = None,
        seed: int | None = None,
        name: str = "leiden",
    ) -> Any:
        """A single Leiden partition of a mask's scaled features, as a ``LabelSet``.

        Parameters
        ----------
        mask : str, optional
            Which cells to partition. ``None`` uses every cell.
        columns : str or FeatureCollection or sequence of str, optional
            Which features to compare on. ``None`` uses every feature.
        resolution : float, default 2.0
            Leiden resolution. The default is tuned for *over*-clustering — many small
            clusters — the deliberate over-split downstream merging starts from.
            Pass a lower value for a standalone, one-shot clustering.
        n_neighbors : int, default 30
            Neighbourhood size for the kNN graph.
        mutual_only : bool, default False
            Keep only mutual kNN edges. ``graph_type="knn"`` only.
        graph_type : str, default 'knn'
            Edge weighting — see :func:`~cellpax.clustering.kneighbor_graph`.
        prune : float, default 0.0
            Jaccard floor for ``graph_type="snn_jaccard"``; ignored otherwise.
        min_cluster_size : int, default 1
            Clusters smaller than this become ``-1``.
        space : FittedSpace, optional
            The representation to partition, from :meth:`space`. ``None`` — the default —
            uses the **raw scaled features at full dimensionality**, unlike ``cluster``,
            which reduces to ``pca(0.95)``. That asymmetry used to be implicit; pass
            ``ft.space(mask, columns=…)`` to put this in the clustering space, or a space
            built with an ``alpha`` to share a whitened one. The two are not
            interchangeable, so which is in use should be a decision rather than a default
            nobody noticed.
        seed : int, optional
            Leiden seed.
        name : str, default 'leiden'
            Name for the returned ``LabelSet``.

        Returns
        -------
        LabelSet
            Mask-aligned labels.
        """
        from cellpax.clustering import cluster_leiden, kneighbor_graph
        from cellpax.labels import LabelSet

        seed = (
            seed if seed is not None else self._derive_seed("overcluster", mask, name)
        )
        scaled = self.features(mask, scaled=True, columns=columns)
        data = scaled if space is None else space.transform_scaled(scaled)
        graph = kneighbor_graph(
            data,
            n_neighbors=n_neighbors,
            mutual_only=mutual_only,
            graph_type=graph_type,  # type: ignore[arg-type]
            prune=prune,
        )
        labels = cluster_leiden(
            graph,
            resolution_parameter=resolution,
            seed=seed,
            min_cluster_size=min_cluster_size,
        )
        return LabelSet(
            self._cell_ids(mask), labels, name=name, mask=mask or _DEFAULT_MASK
        )

    def _align_over_clustering(
        self, over_clustering: Any, cell_ids: np.ndarray
    ) -> np.ndarray:
        """Coerce an over-clustering (array or LabelSet) to feature-row order."""
        if hasattr(over_clustering, "codes_for"):  # a LabelSet
            # codes_for aligns by id and returns -1 for uncovered cells, with no
            # assumption that ids are integers or that column order is stable
            return over_clustering.codes_for(cell_ids)
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

        ``similarity`` is a ``Clustering``, a bare ``SimilarityMatrix``, or the name
        of a stored one. Prefer ``clus.label(distance_threshold=…)``, which takes the
        mask from the clustering itself; this form exists for bare
        ``SimilarityMatrix`` objects computed outside the table.

        With a ``Clustering`` and no ``mask``, its own mask is used. Passing a
        ``mask`` that disagrees with it raises rather than pairing the matrix with
        the wrong cells — the mistake this whole signature invites, since rows are
        matched to mask members by position, not by id.
        """
        from cellpax.clustering import Clustering
        from cellpax.labels import LabelSet

        if isinstance(similarity, str):
            similarity = self.clustering(similarity)
        if isinstance(similarity, Clustering):
            if mask is not None and mask != similarity.mask:
                raise ValueError(
                    f"clustering was computed on mask {similarity.mask!r}, not "
                    f"{mask!r}; its rows describe {similarity.mask!r}'s cells in "
                    f"order, so cutting it against another mask would mislabel them. "
                    f"Omit mask= to use {similarity.mask!r}."
                )
            # the Clustering knows its own mask and ordering; don't reimplement them
            return similarity.label(
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                name=name,
            )
        cell_ids = self._cell_ids(mask)
        return LabelSet.from_clustering(
            similarity,
            cell_ids,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            name=name,
            mask=mask or _DEFAULT_MASK,
        )

    def reorder_labels(
        self,
        labels: Any,
        column: str,
        *,
        mask: str | None = None,
        agg: Literal["mean", "median"] = "mean",
        ascending: bool = True,
    ) -> Any:
        """Reorder a ``LabelSet`` by an aggregate of one of this table's columns.

        ``labels`` is a ``LabelSet`` or the name of an (attached) label column
        (see ``labelset``). ``column`` may be any column (feature or metadata,
        e.g. ``"soma_depth_um"``) — values are matched to ``labels`` by cell id
        via :meth:`~cellpax.labels.LabelSet.reorder_by`, so row order doesn't
        matter.
        """
        labels = self._resolve_labels(labels, mask=mask)
        frame = self._df.filter(self.mask_series(mask))
        mapping = dict(zip(frame[self._id_column].to_list(), frame[column].to_list()))
        return labels.reorder_by(mapping, agg=agg, ascending=ascending)

    def attach(
        self, labels: Any, *, name: str | None = None, overwrite: bool = False
    ) -> "FeatureTable":
        """Attach a ``LabelSet`` as name + id columns, joined on the id column.

        Cells outside the label set get a null name and id — e.g. ``"subclass"``
        and ``"subclass_id"``, the latter for ``pl.col(...) == an_int_enum_member``
        comparisons (see ``LabelSet.to_enum``).

        Colors, descriptions and the label set's mask ride along out-of-band (a
        table column can only hold the name), so ``labelset`` gives back an
        equivalent ``LabelSet`` and ``save`` keeps them.

        Attaching over an existing label column raises, because replacing a labelling
        the rest of a session is already reading is worth being deliberate about.
        ``overwrite=True`` says you mean it and swaps the pair in place — the way to
        re-attach after re-cutting a clustering, which is otherwise a ``detach`` every
        time round the loop. ``LabelSet``'s default name is ``"label"``, so an
        unnamed cut collides with the last unnamed cut; ``name=`` on either
        ``label(...)`` or here keeps them apart.

        A :class:`~cellpax.propagate.Propagation` attaches directly: its labels
        go on as usual and its per-cell confidence lands alongside as
        ``{name}_confidence``, so the number that says how much to trust each
        label is no longer dropped on the floor at exactly the moment the label
        becomes a column.
        """
        confidence: np.ndarray | None = None
        if hasattr(labels, "labels") and hasattr(labels, "confidence"):
            propagation = labels
            labels = propagation.labels
            confidence = np.asarray(propagation.confidence, dtype=float)
        column = name or labels.name
        clash = [
            c
            for c in (column, f"{column}_id", f"{column}_confidence")
            if c in self._df.columns
        ]
        if clash and overwrite:
            self._df = self._df.drop(clash)
            self._label_meta.pop(column, None)
        elif clash:
            # a lone clashing column is not attach's own leftovers, so it may well be
            # the caller's data — say so before pointing at anything that deletes it
            remedy = (
                f"pass overwrite=True to replace it, detach({column!r}) first, "
                f"or pass name="
                if column in self.labels
                else f"not as an attached label pair, so this may be your own data — "
                f"pass name= to attach elsewhere, or detach({column!r}) to drop "
                f"{clash} if you do want them gone"
            )
            raise ValueError(f"{clash} already in the table; {remedy}")
        frame = labels.to_frame(id_column=self._id_column).rename(
            {labels.name: column, f"{labels.name}_id": f"{column}_id"}
        )
        keep = [self._id_column, column, f"{column}_id"]
        if confidence is not None:
            frame = frame.with_columns(pl.Series(f"{column}_confidence", confidence))
            keep.append(f"{column}_confidence")
        self._df = self._df.join(frame.select(keep), on=self._id_column, how="left")
        self._label_meta[column] = {"mask": labels.mask, "labels": labels.meta}
        return self

    def detach(self, name: str) -> "FeatureTable":
        """Drop an attached label's ``name`` + ``name_id`` columns, the inverse of ``attach``.

        Drops whichever of the pair is present rather than insisting on both, so a
        half-written pair — the state that made ``attach``'s own advice impossible to
        follow — is still removable. Raises only when neither column exists.
        """
        present = [
            c
            for c in (name, f"{name}_id", f"{name}_confidence")
            if c in self._df.columns
        ]
        if not present:
            raise KeyError(f"Unknown label {name!r}; attached: {self.labels}")
        self._df = self._df.drop(present)
        self._label_meta.pop(name, None)
        return self

    def labelset(self, column: str, *, mask: str | None = None) -> Any:
        """Reconstruct a ``LabelSet`` from a column, the reverse of ``attach``.

        Lets any method that wants a ``LabelSet`` (``compare``, ``dataframe``,
        ``reorder_labels``, ``neighborhood_purity``) be handed a plain column
        name instead — no need to keep the original ``LabelSet`` object around
        once it's been attached, or to build one at all for a column that
        arrived some other way (e.g. an externally supplied cell type column).

        If a companion ``{column}_id`` integer column exists (as ``attach``
        produces), it's used directly and ``column``'s values become the
        cluster names. Otherwise ``column``'s own values are factorized via
        :meth:`~cellpax.labels.LabelSet.from_labels`. Null/missing values
        become unassigned (``-1``).

        For a column that came from ``attach``, the colors, descriptions and mask
        recorded then are restored too — so ``ft.attach(labels)`` followed by
        ``ft.labelset(...)`` round-trips, here or after a ``save``/``load``. The
        column itself still decides the names, since that's what the table shows.
        An explicit ``mask`` overrides the recorded one.
        """
        from cellpax.labels import Label, LabelSet

        stored = self._label_meta.get(column, {})
        if mask is None:
            recorded = stored.get("mask")
            if recorded in self.masks:
                mask = recorded
        identities: dict[int, Label] = stored.get("labels", {})
        frame = self._df.filter(self.mask_series(mask))
        cell_ids = frame[self._id_column].to_numpy()
        id_column = f"{column}_id"
        if id_column in frame.columns:
            raw_ids = frame[id_column].to_list()
            raw_names = frame[column].to_list()
            labels = np.array(
                [-1 if i is None else int(i) for i in raw_ids], dtype=np.int64
            )
            meta = {
                int(i): _labelled(identities.get(int(i)), int(i), str(n))
                for i, n in zip(raw_ids, raw_names)
                if i is not None and int(i) >= 0
            }
            return LabelSet(
                cell_ids, labels, meta=meta, name=column, mask=mask or _DEFAULT_MASK
            )
        return LabelSet.from_labels(
            cell_ids, frame[column].to_list(), name=column, mask=mask or _DEFAULT_MASK
        )

    def _resolve_labels(self, labels: Any, *, mask: str | None = None) -> Any:
        """Coerce a ``LabelSet`` or column-name string into a ``LabelSet``."""
        return self.labelset(labels, mask=mask) if isinstance(labels, str) else labels

    def flatten_labels(
        self,
        labels: Sequence[Any],
        *,
        mask: str | None = None,
        name: str = "label",
        fill: str | None = None,
        source_name: str | None = None,
    ) -> Any:
        """Collapse labels built over different masks into one, in priority order.

        Each cell takes its label from the **first** entry of ``labels`` that
        assigned it; an entry that covers a cell but left it unassigned falls
        through to the next. That's the way out of the recursive descent, where
        every round leaves its own column and a cell has a ``subtype_nn`` only
        where ``family_nn`` sent it down that branch — list them most-specific
        first and the coarse labels fill the holes the fine ones left.

        ``labels`` are attached column names (see ``labelset``) or ``LabelSet``
        objects, mixed freely. Clusters are matched by name across entries, so
        colors and descriptions merge along with the values and the first entry
        to define a name wins — this is
        :meth:`~cellpax.labels.LabelSet.combine` in ``mode="priority"``, then
        :meth:`~cellpax.labels.LabelSet.reindex` onto ``mask``.

        ``mask`` scopes the result: the label set covers exactly that mask's
        cells, in table order. Cells outside it are dropped even where an entry
        labels them, and cells inside it that nothing labels are unassigned —
        or land in the cluster named ``fill``, if you give one.

        With ``source_name`` a second ``LabelSet`` of that name comes back
        alongside, recording which entry won each cell (by the entries' own
        names), for checking how much each mask actually contributed. Neither is
        attached: ``ft.attach(flat)`` when you're happy with it.
        """
        from cellpax.labels import Label, LabelSet

        entries = list(labels)
        if not entries:
            raise ValueError("flatten_labels needs at least one label")
        unknown = [
            item
            for item in entries
            if isinstance(item, str) and item not in self._df.columns
        ]
        if unknown:
            raise KeyError(f"Unknown label {unknown[0]!r}; attached: {self.labels}")
        sources = [self._resolve_labels(item) for item in entries]
        cell_ids = self._cell_ids(mask)
        flat = (
            sources[0]
            .combine(*sources[1:], mode="priority")
            .reindex(cell_ids, fill=fill, name=name, mask=mask or _DEFAULT_MASK)
        )
        if source_name is None:
            return flat

        origin = np.full(cell_ids.shape[0], -1, dtype=np.int64)
        for position, source in enumerate(sources):
            unclaimed = (origin == -1) & (source.codes_for(cell_ids) != -1)
            origin[unclaimed] = position
        taken: set[str] = set()
        meta: dict[int, Label] = {}
        for position, source in enumerate(sources):
            label_name = source.name
            if label_name in taken:  # two entries can share a name; keep them apart
                label_name = f"{label_name} ({position})"
            taken.add(label_name)
            meta[position] = Label(id=position, name=label_name)
        return flat, LabelSet(
            cell_ids, origin, meta=meta, name=source_name, mask=mask or _DEFAULT_MASK
        )

    # -- embeddings ------------------------------------------------------------

    def embed(
        self,
        mask: str | None = None,
        *,
        method: str = "pca",
        columns: str | FeatureCollection | Sequence[str] | None = None,
        n_components: int = 2,
        name: str | None = None,
        space: FittedSpace | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Compute a low-dimensional embedding of a mask's scaled features.

        Parameters
        ----------
        mask : str, optional
            Which cells to embed. ``None`` uses every cell.
        method : {'pca', 'umap', 'pacmap', 'localmap'}, default 'pca'
            Backend. ``'pca'`` uses scikit-learn and is always available; the other
            three are optional installs, imported lazily
            (``pip install 'cellpax[embeddings]'``).
        columns : str or FeatureCollection or sequence of str, optional
            Which features to embed. ``None`` uses every feature.
        n_components : int, default 2
            Output dimensionality.
        name : str, optional
            Key the coordinates are stored under, and the prefix of the returned
            coordinate columns. Defaults to ``method``, with ``columns``' own name folded
            in when it has one (a collection name, or ``FeatureCollection.name``) so that
            embedding two column sets with the same ``method`` does not silently overwrite
            one another. A raw column list carries no name, so pass ``name`` explicitly
            when embedding one.
        space : FittedSpace, optional
            The representation to embed, from :meth:`space`. ``None`` — the default —
            embeds the mask's **raw scaled features at full dimensionality**. Passing one
            embeds ``space.transform_scaled(...)`` instead, so the picture is of a reduced
            or partially whitened space rather than the raw one. The choice is recorded on
            the stored :class:`FittedEmbedding` and reused by :meth:`project`, so cells
            added later are pushed through the *same* space rather than the raw features.
        seed : int, optional
            Random state, forwarded to whichever backend is used. All four accept one;
            without it, no figure can be regenerated.
        **kwargs
            Passed to the backend, so ``n_neighbors=``, ``min_dist=`` (umap),
            ``MN_ratio=``/``FP_ratio=`` (pacmap) or ``low_dist_thres=`` (localmap) are all
            available. Note pacmap and localmap default to ``n_neighbors=10`` against
            umap's 15, so they are not a like-for-like swap.

        Returns
        -------
        pl.DataFrame
            ``cell_id`` plus ``{name}0 … {name}{n_components-1}``, one row per cell in the
            mask. Also stored, so :meth:`embedding` returns it again and
            ``dataframe(embedding=name)`` joins it onto the tidy view.

        Notes
        -----
        Why ``space`` defaults to ``None`` rather than to the clustering space: ``cluster``
        reduces to ``pca(0.95)``, so leaving this alone means **the embedding is not a
        picture of the space you clustered**. That is deliberate. Comparing
        embedding-space neighbourhoods against feature-space neighbourhoods is only
        informative while the two are built separately — make the embedding a view of the
        clustering space and the comparison becomes circular while still producing
        agreeable-looking numbers. So reach for ``space`` when you want to *see* the
        clustering space, and keep a separately-built embedding around if you also want
        that comparison. :meth:`graph_provenance` reports which each one got, and warns
        when a clustering and an embedding on the same mask share a space.

        One interaction worth knowing: ``space`` also switches pacmap and localmap's own
        ``apply_pca`` off, since reducing an already-reduced-and-whitened space would
        partly undo the weighting.

        The backends differ in whether they reduce the input themselves, which is the
        thing that makes that check non-trivial:

        - ``umap-learn`` does **not**. The neighbour graph is built on whatever matrix it
          is handed; PCA enters only as an optional layout initialisation
          (``init="pca"``, not the default). The contrary impression comes from scanpy,
          where ``sc.pp.neighbors`` reduces first.
        - ``pacmap`` and ``localmap`` **do**, conditionally: with ``apply_pca=True``
          (their default) and more than 100 input features, they truncated-SVD down to 100
          before constructing pairs. Below 100 features nothing happens. Passing
          ``space=`` sets ``apply_pca=False`` automatically, since reducing an already
          reduced-and-whitened space would partly undo the weighting; whatever ends up in
          force is recorded and shows in :meth:`graph_provenance`.

        ``pacmap`` and ``localmap`` are built with ``save_tree=True`` so ``project(...,
        embedding=name)`` works — their ``transform`` otherwise requires the original
        training matrix to be handed back. That keeps the neighbour index in memory.

        Both build that index with faiss, and faiss is pinned to a single OpenMP thread
        for the duration of the call (restored afterwards). That is a deadlock guard
        rather than a performance choice: faiss, sklearn and xgboost can each bring a
        *different* ``libomp`` into one process, and a parallel region opened in one
        runtime can block forever on a barrier owned by another, at 0% CPU and immune to
        ``KeyboardInterrupt``. See :func:`_faiss_single_thread`. The cost is
        unmeasurable at these sizes; if a whole session needs the same protection for
        sklearn and xgboost too, set ``OMP_NUM_THREADS=1`` before importing anything.

        Extra keyword arguments go to the backend, so ``n_neighbors=`` and
        ``MN_ratio=``/``FP_ratio=`` (pacmap) or ``low_dist_thres=`` (localmap) are
        available. Note pacmap's ``n_neighbors`` defaults to 10 against UMAP's 15, so the
        two are not a like-for-like swap out of the box.
        """
        columns_label = _columns_label(columns)
        label = name or (f"{method}_{columns_label}" if columns_label else method)
        cols = self._resolve_columns(columns)
        seed = seed if seed is not None else self._derive_seed("embed", mask, label)
        scaled = self.features(mask, scaled=True, columns=cols)
        data = scaled if space is None else space.transform_scaled(scaled)
        internal_reduction: str | None = None
        if method == "pca":
            from sklearn.decomposition import PCA

            model: Any = PCA(n_components=n_components, random_state=seed, **kwargs)
        elif method == "umap":
            try:
                import umap
            except ImportError as error:
                raise ImportError(
                    "method='umap' requires the optional 'umap-learn' package; "
                    "pip install 'cellpax[umap]'"
                ) from error
            model = umap.UMAP(n_components=n_components, random_state=seed, **kwargs)
        elif method in _FAISS_BACKED:
            try:
                import pacmap
            except ImportError as error:
                raise ImportError(
                    f"method={method!r} requires the optional 'pacmap' package; "
                    "pip install 'cellpax[pacmap]'"
                ) from error
            # Reducing a space that is already a reduction would partly undo its
            # weighting, so a caller-supplied space wins unless they say otherwise.
            kwargs.setdefault("apply_pca", space is None)
            kwargs.setdefault("save_tree", True)
            factory = pacmap.PaCMAP if method == "pacmap" else pacmap.LocalMAP
            model = factory(n_components=n_components, random_state=seed, **kwargs)
            internal_reduction = _pacmap_reduction(data.shape[1], kwargs)
        else:
            raise ValueError(
                f"Unknown embedding method {method!r}; expected one of "
                "'pca', 'umap', 'pacmap', 'localmap'"
            )
        guard = _faiss_single_thread() if method in _FAISS_BACKED else nullcontext()
        with guard:
            coords = model.fit_transform(data)
        self._embedding_models[(mask or _DEFAULT_MASK, label)] = FittedEmbedding(
            model=model,
            method=method,
            columns=tuple(cols),
            n_components=n_components,
            space="scaled" if space is None else space.label,
            fitted_space=space,
            internal_reduction=internal_reduction,
            params={
                "method": method,
                "n_components": n_components,
                "seed": seed,
                **{k: v for k, v in kwargs.items() if _jsonable(v)},
            },
        )
        self._embedding_params[(mask or _DEFAULT_MASK, label)] = self._embedding_models[
            (mask or _DEFAULT_MASK, label)
        ].params
        cell_ids = self._cell_ids(mask)
        frame = pl.DataFrame(
            {
                self._id_column: cell_ids,
                **{f"{label}{i}": coords[:, i] for i in range(n_components)},
            }
        )
        self._embeddings[(mask or _DEFAULT_MASK, label)] = frame
        return frame

    def graph_provenance(self) -> pl.DataFrame:
        """Which space and graph every clustering and embedding on this table received.

        One row per consumer: ``consumer`` (``"clustering"`` or ``"embedding"``),
        ``name``, ``mask``, ``space``, ``graph_type``, ``n_neighbors``, ``n_columns``.

        Worth reading because one comparison in this workflow depends on two objects
        staying distinct. Checking feature-space neighbourhoods against embedding-space
        neighbourhoods is a real diagnostic while the embedding is built separately from
        the graph the clustering used; build both from one graph and the comparison
        becomes a tautology that still produces agreeable-looking numbers. This makes the
        answer a table rather than an assumption, and a clustering sharing both space and
        graph construction with an embedding on the same mask warns.

        Embeddings are listed from their stored coordinates, so a reloaded table still
        reports them — but ``space``, ``graph_type`` and ``n_columns`` are null there,
        because the fitted model that knew those is session-only. Null is the honest
        answer; re-run ``embed`` to recover it.
        """
        rows: list[dict[str, Any]] = []
        for name, clustering in self._clusterings.items():
            partitions = getattr(clustering, "partitions", None)
            graph_types = (
                sorted(set(partitions.graph_type.tolist()))
                if partitions is not None
                else []
            )
            neighbors = (
                sorted(set(partitions.n_neighbors.tolist()))
                if partitions is not None
                else []
            )
            rows.append(
                {
                    "consumer": "clustering",
                    "name": name,
                    "mask": getattr(clustering, "mask", None),
                    "space": getattr(clustering, "space", "") or "",
                    "graph_type": ",".join(graph_types) if graph_types else None,
                    "n_neighbors": ",".join(str(k) for k in neighbors)
                    if neighbors
                    else None,
                    "n_columns": len(getattr(clustering, "columns", ()) or ()),
                }
            )
        # Iterate the *coordinates*, which persist, not the fitted models, which are
        # session-only — otherwise a reloaded table reports no embeddings at all while
        # holding every one of their coordinate frames.
        for mask, name in sorted(self._embeddings):
            model = self._embedding_models.get((mask, name))
            external = self._external_embeddings.get((mask, name))
            space: str | None = None
            graph_type: str | None = None
            n_columns: int | None = None
            if external is not None:
                space, graph_type = external[0], "external"
            if model is not None:
                space = model.space
                if model.internal_reduction is not None:
                    # the backend reduced the input itself, so the space it built pairs in
                    # is not the one it was handed; saying only the latter would be a lie
                    space = f"{space} -> {model.internal_reduction}"
                # an embedding builds its own neighbour structure internally; it is never
                # one of the Leiden graph_type objects, which is the point
                graph_type = f"internal:{model.method}"
                n_columns = len(model.columns)
            rows.append(
                {
                    "consumer": "embedding",
                    "name": name,
                    "mask": mask,
                    # null rather than a guess: after a reload the coordinates are all
                    # that survived, so how they were made is genuinely unknown
                    "space": space,
                    "graph_type": graph_type,
                    "n_neighbors": None,
                    "n_columns": n_columns,
                }
            )
        frame = pl.DataFrame(
            rows,
            schema={
                "consumer": pl.Utf8,
                "name": pl.Utf8,
                "mask": pl.Utf8,
                "space": pl.Utf8,
                "graph_type": pl.Utf8,
                "n_neighbors": pl.Utf8,
                "n_columns": pl.Int64,
            },
        )
        self._warn_shared_graphs(frame)
        return frame

    @staticmethod
    def _warn_shared_graphs(frame: pl.DataFrame) -> None:
        """Warn when a clustering and an embedding on one mask share a representation."""
        clusterings = frame.filter(pl.col("consumer") == "clustering")
        embeddings = frame.filter(pl.col("consumer") == "embedding")
        for row in clusterings.iter_rows(named=True):
            shared = embeddings.filter(
                (pl.col("mask") == row["mask"]) & (pl.col("space") == row["space"])
            )
            for other in shared.iter_rows(named=True):
                warnings.warn(
                    f"clustering {row['name']!r} and embedding {other['name']!r} on "
                    f"mask {row['mask']!r} were both built in space {row['space']!r}; "
                    "comparing their neighbourhoods is close to circular, so treat "
                    "agreement between them as uninformative",
                    stacklevel=3,
                )

    def triage_labels(
        self,
        labels: Any,
        *,
        mask: str | None = None,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        n_neighbors: int = 30,
        majority: float = 0.5,
        space: FittedSpace | None = None,
    ) -> pl.DataFrame:
        """Classify each cell by what its feature-space neighbours are labelled.

        The follow-up to spotting a cell sitting inside a differently-coloured cloud on a
        UMAP: pull its nearest neighbours in the *clustering* space and see what they
        are. Three outcomes, and they call for different things —

        ============================ ==========================================
        neighbours mostly...         reading
        ============================ ==========================================
        its own label                embedding artifact; the UMAP misplaced it
        the surrounding label        the label is wrong, or it is a real outlier
        mixed, no majority           the consensus was ambiguous there
        ============================ ==========================================

        Parameters
        ----------
        labels : LabelSet or str
            The labels to triage, or the name of an attached label column.
        mask : str, optional
            Which cells to consider. Defaults to the labels' own mask.
        columns : str or FeatureCollection or sequence of str, optional
            Which features define the space, when ``space`` is not given.
        n_neighbors : int, default 30
            How many nearest neighbours each cell's verdict is based on. Small values are
            noisy per cell; large ones blur genuinely local disagreement.
        majority : float, default 0.5
            Fraction of assigned neighbours a label must reach to decide the verdict.
            Below it on every label, the cell is ``"mixed"``. Raising it moves cells from
            ``"own"``/``"other"`` into ``"mixed"``, i.e. demands more agreement before
            calling a cell explained.
        space : FittedSpace, optional
            The representation neighbours are found in. ``None`` uses the mask's
            ``pca(0.95)`` fit, matching ``cluster``'s default, so the verdict is about the
            space the labels came from. Pass one built with an ``alpha`` to triage in a
            whitened space.

        Returns
        -------
        pl.DataFrame
            One row per cell with ``own_fraction``, ``top_other_label``,
            ``top_other_fraction`` and ``verdict`` in ``{"own", "other", "mixed",
            "unassigned"}`` — so ``.group_by("verdict").len()`` turns "dots scattered here
            and there" into three countable groups. For the mixed ones,
            ``Clustering.cell_stability`` is the cross-check.
        """
        from cellpax.clustering import neighbor_label_composition

        resolved = self._resolve_labels(labels, mask=mask)
        name = (
            mask or (resolved.mask if resolved is not None else None) or _DEFAULT_MASK
        )
        cell_ids = self._cell_ids(name)
        codes = self._align_over_clustering(resolved, cell_ids)
        fitted = space if space is not None else self.space(name, columns=columns)
        data = fitted.transform_scaled(
            self.features(name, scaled=True, columns=fitted.columns)
        )
        return neighbor_label_composition(
            data,
            codes,
            cell_ids=cell_ids,
            n_neighbors=n_neighbors,
            majority=majority,
            names=dict(zip(resolved.ids, resolved.names)),
            id_column=self._id_column,
        )

    def score_cells(
        self,
        mask: str | None = None,
        *,
        scorer: Any = None,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        name: str = "outlier_score",
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Fit an outlier detector on a mask's scaled features; store per-cell scores.

        The adapter between the table and the existing detector ecosystem:
        ``scorer`` is any estimator with ``fit(X)`` plus a per-sample score
        (``score_samples``, ``decision_function``, or scikit-learn LOF's
        ``negative_outlier_factor_``) — sklearn's detectors work directly and
        PyOD's follow the same shape. ``None`` uses an
        ``IsolationForest`` seeded from the table (see the guide's seeds
        section). Higher scores read as *more normal* under all three
        protocols, so masking the tail is ``pl.col(name) < threshold``.

        Scores land as an ordinary metadata column (null off-mask), so they are
        immediately maskable, plottable via ``embedding_view``, usable as an
        audit label in ``label_purity`` — and persist with the table. Scoring
        never filters: an extreme cell is either a reconstruction problem or
        the most interesting thing in the data, and which one it is deserves a
        look rather than a default. Fits are mask-relative like everything
        else, so a score means "unusual within this mask".

        Comparing scores between the full feature set and a truncation-safe
        collection is itself a validity diagnostic: a cell extreme under the
        full set but ordinary under the safe one is flagged *by its invalid
        features* — truncation talking, not biology.

        Returns the ``[id, score]`` frame for immediate thresholding; the
        column is already on the table either way (``overwrite`` semantics
        follow ``add_column``: re-scoring under the same name replaces it).
        """
        cols = self._resolve_columns(columns)
        matrix = self.features(mask, scaled=True, columns=cols)
        if scorer is None:
            from sklearn.ensemble import IsolationForest

            resolved_seed = (
                seed if seed is not None else self._derive_seed("score", mask, name)
            )
            scorer = IsolationForest(random_state=resolved_seed)
        scorer.fit(matrix)
        if hasattr(scorer, "score_samples"):
            values = np.asarray(scorer.score_samples(matrix), dtype=float)
        elif hasattr(scorer, "decision_function"):
            values = np.asarray(scorer.decision_function(matrix), dtype=float)
        elif hasattr(scorer, "negative_outlier_factor_"):
            values = np.asarray(scorer.negative_outlier_factor_, dtype=float)
        else:
            raise TypeError(
                f"{type(scorer).__name__} exposes none of score_samples / "
                f"decision_function / negative_outlier_factor_; wrap it to "
                f"provide a per-sample score"
            )
        if name in self._df.columns and name not in self._features:
            self._df = self._df.drop(name)
        self.add_column(values, name, mask=mask)
        return pl.DataFrame({self._id_column: self._cell_ids(mask), name: values})

    def space(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        explained_variance: float | int = 0.95,
        alpha: float = 0.0,
        eigenvalue_floor: float = 0.0,
        feature_weights: np.ndarray | None = None,
        seed: int | None = None,
    ) -> FittedSpace:
        """The :class:`~cellpax.space.FittedSpace` a mask's cells are compared in.

        Parameters
        ----------
        mask : str, optional
            Which cells the fit is over. ``None`` uses every cell.
        columns : str or FeatureCollection or sequence of str, optional
            Which features the fit is over. ``None`` uses every feature.
        explained_variance : float or int, default 0.95
            A fraction in ``(0, 1]`` keeps the smallest number of components reaching that
            cumulative explained variance, matching scikit-learn's float ``n_components``
            rule; an integer keeps exactly that many. Note ``1`` and ``1.0`` differ — one
            component versus all the variance.
        alpha : float, default 0.0
            Whitening strength applied as a *view* over the cached fit: component *j* is
            scaled by ``λ_j ** (-alpha/2)``. ``0.0`` leaves PCA's own scaling untouched
            (and short-circuits, so it is bitwise the plain projection), ``1.0`` gives
            every retained component unit variance.
        eigenvalue_floor : float, default 0.0
            Added to each eigenvalue before the ``alpha`` scaling. Use
            ``ft.space(mask).noise_floor`` when ``alpha > 0``.
        feature_weights : numpy.ndarray, optional
            Per-feature multipliers applied *before* the PCA, from
            :func:`~cellpax.diagnostics.block_weights`. Unlike ``alpha`` this is part of
            the fit, so it participates in the cache key — and it is frozen into the
            returned space, so ``embed(space=…)`` and ``project`` apply it rather than
            silently skipping it.
        seed : int, optional
            Random state for the PCA fit.

        Returns
        -------
        FittedSpace
            The frozen scaling + PCA. Reusable on new cells via ``transform``, inspectable
            via ``spectrum()`` and ``condition_number``, and persistable — none of which
            the old fit-and-forget ``features_pca`` allowed.

        Notes
        -----
        Cached per ``(mask, columns, explained_variance, seed)``. **``alpha`` and
        ``eigenvalue_floor`` are deliberately not part of that key** — they are applied as
        a view over the cached fit, so a sweep across whitening strengths is guaranteed by
        construction to share one PCA rather than refitting it underneath itself. That is
        what makes "only the swept axis differs" true rather than merely intended.

        Cleared alongside the scalers whenever the mask or the preprocessing changes; a
        space fit to features that have since been redefined is worse than no space.
        """
        cols = self._resolve_columns(columns)
        # feature_weights change the fit, so unlike alpha they belong in the key. Digested
        # because an array cannot be a key, and with blake2b rather than hash() because
        # the latter is salted per process — the key has to survive a save/load.
        weight_key = (
            None
            if feature_weights is None
            else hashlib.blake2b(
                np.asarray(feature_weights, dtype=float).tobytes(), digest_size=8
            ).hexdigest()
        )
        # the int/float distinction is meaningful ("1" = one component, "1.0" = all
        # the variance), so the key must carry the type as well as the value
        variance_key = (
            ("n", int(explained_variance))
            if isinstance(explained_variance, (int, np.integer))
            else ("var", float(explained_variance))
        )
        key = (
            mask or _DEFAULT_MASK,
            tuple(cols),
            variance_key,
            seed,
            weight_key,
        )
        if key not in self._space_cache:
            self._space_cache[key] = FittedSpace.fit(
                self.features(mask, scaled=True, columns=cols),
                columns=cols,
                scaler=self._scaler(mask or _DEFAULT_MASK, cols),
                explained_variance=explained_variance,
                feature_weights=feature_weights,
                seed=seed,
            )
        fitted = self._space_cache[key]
        if alpha or eigenvalue_floor:
            fitted = fitted.with_alpha(alpha, eigenvalue_floor=eigenvalue_floor)
        return fitted

    def features_pca(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        explained_variance: float = 0.95,
        alpha: float = 0.0,
        eigenvalue_floor: float = 0.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """PCA-reduced scaled features, keeping just enough components.

        Parameters
        ----------
        mask : str, optional
            Which cells to project. ``None`` uses every cell.
        columns : str or FeatureCollection or sequence of str, optional
            Which features to project. ``None`` uses every feature.
        explained_variance : float, default 0.95
            Cumulative explained variance to keep.
        alpha : float, default 0.0
            Whitening strength — see :meth:`space`.
        eigenvalue_floor : float, default 0.0
            Eigenvalue offset bounding the ``alpha`` amplification — see :meth:`space`.
        seed : int, optional
            Random state for the PCA fit.

        Returns
        -------
        numpy.ndarray
            An ``(n_cells, k)`` array, ready to feed straight into
            ``fauxnograph_coclustering`` / ``SimilarityMatrix``.

        Notes
        -----
        A thin projection through :meth:`space` — reach for that directly when the fit
        itself is wanted, for the eigenvalue spectrum, or to push new cells into the same
        space.
        """
        fitted = self.space(
            mask,
            columns=columns,
            explained_variance=explained_variance,
            alpha=alpha,
            eigenvalue_floor=eigenvalue_floor,
            seed=seed,
        )
        return fitted.transform_scaled(
            self.features(mask, scaled=True, columns=columns)
        )

    def add_embedding(
        self,
        coords: np.ndarray | pl.DataFrame,
        mask: str | None = None,
        *,
        name: str,
        space: str = "external",
        overwrite: bool = False,
    ) -> pl.DataFrame:
        """Register coordinates computed elsewhere as an embedding of this table.

        Parameters
        ----------
        coords : numpy.ndarray or pl.DataFrame
            An ``(n_cells, k)`` array **in the mask's row order** — the same order
            ``features(mask)`` and ``embedding_view`` use — or a frame carrying the id
            column plus one column per dimension, in which case order does not matter and
            the ids are matched.
        mask : str, optional
            Which cells these cover. ``None`` means every cell.
        name : str
            Key to store under, and the prefix of the coordinate columns
            (``{name}0 … {name}{k-1}``).
        space : str, default 'external'
            A short label recording where the coordinates came from, surfaced by
            :meth:`graph_provenance`. Worth setting to something meaningful
            (``"anatomical"``, ``"pacmap on block-weighted space"``) since nothing else
            records it.
        overwrite : bool, default False
            Replace an existing embedding of the same name.

        Returns
        -------
        pl.DataFrame
            ``cell_id`` plus ``{name}0 … {name}{k-1}``, exactly as :meth:`embed` returns —
            so :meth:`embedding_view`, ``dataframe(embedding=…)`` and the persistence layer
            all treat it identically.

        Notes
        -----
        The obvious use is coordinates from a backend ``embed`` does not wrap, or from a
        representation it cannot build — a space with ``feature_weights`` applied outside
        the table, say. But it is just as useful for coordinates that were never an
        "embedding" at all: **soma position makes a perfectly good one**, which lets an
        anatomical plot reuse the same labels, joins and colour maps as a UMAP.

        >>> xy = ft.dataframe(mask).select(["soma_x_um", "soma_depth_um"]).to_numpy()
        >>> ft.add_embedding(xy, mask, name="soma_xz", space="anatomical")  # doctest: +SKIP
        >>> v = ft.embedding_view(mask, name="soma_xz")                     # doctest: +SKIP

        There is no fitted model behind these, so :meth:`embedding_model` raises and
        :meth:`project` cannot push new cells in — the same position a reloaded embedding
        is in. Coverage is checked here rather than at read time, so a mismatch fails when
        you add it instead of surfacing later as null coordinates.
        """
        resolved = mask or _DEFAULT_MASK
        key = (resolved, name)
        if key in self._embeddings and not overwrite:
            raise ValueError(
                f"embedding {name!r} already exists for mask {resolved!r}; pass "
                f"overwrite=True to replace it"
            )
        cell_ids = self._cell_ids(mask)

        if isinstance(coords, pl.DataFrame):
            if self._id_column not in coords.columns:
                raise ValueError(
                    f"a coordinate frame must carry the id column {self._id_column!r}; "
                    f"got {coords.columns}"
                )
            missing = np.setdiff1d(cell_ids, coords[self._id_column].to_numpy())
            if missing.size:
                raise ValueError(
                    f"coordinates cover {coords.height} cells but mask {resolved!r} has "
                    f"{len(cell_ids)}, and {missing.size} of them are absent — every cell "
                    "in the mask needs coordinates, or plots would silently drop it"
                )
            frame = (
                pl.DataFrame({self._id_column: cell_ids})
                .join(coords, on=self._id_column, how="left")
                .select(
                    [
                        self._id_column,
                        *[c for c in coords.columns if c != self._id_column],
                    ]
                )
            )
            values = frame.drop(self._id_column).to_numpy()
        else:
            values = np.asarray(coords, dtype=float)
            if values.ndim != 2:
                raise ValueError(f"coords must be 2-D, got shape {values.shape}")
            if values.shape[0] != len(cell_ids):
                raise ValueError(
                    f"coords has {values.shape[0]} rows but mask {resolved!r} has "
                    f"{len(cell_ids)} cells; pass a frame with the id column if the "
                    "rows are not in mask order"
                )

        n_components = values.shape[1]
        frame = pl.DataFrame(
            {
                self._id_column: cell_ids,
                **{f"{name}{i}": values[:, i] for i in range(n_components)},
            }
        )
        self._embeddings[key] = frame
        # no fitted model: record what we do know so graph_provenance does not report a
        # null space for coordinates whose origin the caller told us
        self._embedding_models.pop(key, None)
        self._external_embeddings[key] = (space, n_components)
        return frame

    @property
    def embeddings(self) -> list[tuple[str, str]]:
        """Every stored embedding, as ``(mask, name)`` pairs.

        Worth having because ``embed`` derives a name when one is not given — folding in
        the feature collection's name, so ``embed(method="umap", columns="analysis")``
        stores ``"umap_analysis"``. Without this, that name has to be guessed from how
        ``embed`` built it before anything can read the coordinates back.

        Returns
        -------
        list of tuple of (str, str)
            ``(mask, name)`` for each stored embedding, sorted. Lists the *coordinates*,
            which persist, rather than the fitted models, which do not — so a reloaded
            table still reports everything it holds.
        """
        return sorted(self._embeddings)

    def embedding(self, mask: str | None = None, *, name: str = "pca") -> pl.DataFrame:
        """Return a stored embedding's coordinates."""
        key = (mask or _DEFAULT_MASK, name)
        if key not in self._embeddings:
            raise KeyError(
                f"No embedding {name!r} for mask {mask or _DEFAULT_MASK!r}; "
                f"stored: {list(self._embeddings)}"
            )
        return self._embeddings[key]

    def _resolve_embedding_name(self, mask: str, name: str | None) -> str:
        """The embedding to use for ``mask``, defaulting when there is only one."""
        if name is not None:
            return name
        candidates = sorted(n for m, n in self._embeddings if m == mask)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            elsewhere = sorted({m for m, _ in self._embeddings})
            hint = f" Masks that do have embeddings: {elsewhere}." if elsewhere else ""
            raise ValueError(
                f"mask {mask!r} has no embeddings, so there is no name to default to; "
                f"run ft.embed({mask!r}, ...) first.{hint}"
            )
        raise ValueError(
            f"mask {mask!r} has {len(candidates)} embeddings ({candidates}); pass "
            f"name= to say which one"
        )

    def embedding_view(
        self,
        mask: str | None = None,
        *,
        name: str | None = None,
        labels: Any = None,
        scaled: bool = False,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> EmbeddingView:
        """A stored embedding's frame and coordinate column names together.

        Parameters
        ----------
        mask : str, optional
            Which cells. ``None`` uses every cell.
        name : str, optional
            Which embedding. ``None`` resolves it when the mask holds exactly one, and
            raises listing the candidates when it holds several — so a defaulted name is
            never a silent guess.
        labels : LabelSet, optional
            Joined onto the frame, so ``hue=labels.name`` works directly.
        scaled : bool, default False
            Whether feature columns in the frame carry scaled values.
        columns : str or FeatureCollection or sequence of str, optional
            Which features ``scaled`` applies to — it selects what gets scaled, not what
            the frame carries, so every feature column is present either way. Never
            affects the coordinates, which were fixed when the embedding was computed.

        Returns
        -------
        EmbeddingView
            Carrying ``frame``, ``coords``, and the ``xy`` / ``pair`` accessors — so the
            embedding's name is written once rather than once per coordinate column plus
            once for the frame.

        Notes
        -----
        The coordinate names come from the stored frame rather than being rebuilt as
        ``f"{name}{i}"``. That frame is the source of truth: it cannot drift from what
        ``embed`` actually wrote, and it survives a reload, where the fitted model does
        not. Delegates to :meth:`dataframe`, so the guard that refuses to emit null
        coordinates for a mask redefined since ``embed`` still applies.
        """
        resolved_mask = mask or _DEFAULT_MASK
        resolved = self._resolve_embedding_name(resolved_mask, name)
        coordinates = self.embedding(resolved_mask, name=resolved)
        coords = tuple(c for c in coordinates.columns if c != self._id_column)
        frame = self.dataframe(
            mask, scaled=scaled, columns=columns, embedding=resolved, labels=labels
        )
        return EmbeddingView(
            name=resolved, mask=resolved_mask, frame=frame, coords=coords
        )

    def embedding_model(
        self, mask: str | None = None, *, name: str = "pca"
    ) -> FittedEmbedding:
        """The :class:`FittedEmbedding` behind a stored embedding's coordinates.

        Use it to push cells into an existing embedding instead of re-fitting one —
        ``project(..., embedding=name)`` is the version that scales for you. Raises
        if the model isn't available, which happens in three ways worth telling
        apart: the embedding was never computed, it was computed but the mask or the
        preprocessing has since been redefined (the model is dropped rather than
        left to project into a space the stored coordinates no longer share), or the
        table came back from ``load``, which restores coordinates but not fits.
        Re-run ``embed`` in the latter two cases.
        """
        key = (mask or _DEFAULT_MASK, name)
        if key not in self._embedding_models:
            known = "" if key in self._embeddings else " (nor its coordinates)"
            raise KeyError(
                f"No fitted model for embedding {name!r} on mask "
                f"{mask or _DEFAULT_MASK!r}{known}; models are session-only and are "
                f"dropped when the mask or preprocessing changes — re-run embed(). "
                f"Available: {list(self._embedding_models)}"
            )
        return self._embedding_models[key]

    # -- scaling / views -------------------------------------------------------

    def _scaler(self, mask: str, columns: Sequence[str]) -> FittedScaler:
        key = (mask, tuple(columns))
        if key not in self._scaler_cache:
            matrix = self._df.filter(self.mask_series(mask)).select(columns).to_numpy()
            transforms = [self._transforms.get(c) for c in columns]
            fitted = FittedScaler(scaler=self._scaler_factory(), transforms=transforms)
            fitted.fit(matrix)
            self._scaler_cache[key] = fitted
        return self._scaler_cache[key]

    def scaler(
        self,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
    ) -> FittedScaler:
        """The :class:`FittedScaler` for a ``(mask, columns)`` pair, fitting if needed.

        Scaling is mask-relative and lazy, so this is the fit that ``features(mask,
        scaled=True)`` used — the same cached object, not a copy. Reach for it to
        freeze a mask's scaling and apply it elsewhere: ``.transform(matrix)`` runs
        the per-feature transforms then the scaler, and ``.scaler`` is the underlying
        scikit-learn estimator if you want its ``mean_``/``scale_``.

        ``columns`` must name the same feature set that was scaled, since each
        ``(mask, columns)`` pair gets its own fit; ``project`` is the higher-level
        way in and keeps that bookkeeping for you.
        """
        return self._scaler(mask or _DEFAULT_MASK, self._resolve_columns(columns))

    def project(
        self,
        data: pl.DataFrame | np.ndarray,
        mask: str | None = None,
        *,
        columns: str | FeatureCollection | Sequence[str] | None = None,
        embedding: str | None = None,
    ) -> np.ndarray:
        """Push new cells through a mask's existing fits, without re-fitting anything.

        The point of the whole scaler/embedding-model machinery: cells that weren't
        in the table when a space was built still land in *that* space. Nothing is
        re-fit and nothing is stored, so this is safe to call on new data repeatedly.

        ``data`` is a polars frame carrying the feature columns by name (extra
        columns ignored, order irrelevant) or a raw ``(n, len(columns))`` array in
        the resolved column order. Returns the scaled matrix, or the embedding's
        coordinates when ``embedding`` names one — in which case ``columns`` comes
        from the model's own record and passing a conflicting one raises, since the
        model can only accept the space it was fit in.

        Because the fits are mask-relative, the values mean "where these cells sit
        relative to ``mask``" — that's the intended reading for carrying a model or
        an embedding onto new cells, and the reason it isn't the same as adding the
        rows to the table and re-scaling.
        """
        name = mask or _DEFAULT_MASK
        if embedding is not None:
            fitted_embedding = self.embedding_model(name, name=embedding)
            cols = list(fitted_embedding.columns)
            if columns is not None and self._resolve_columns(columns) != cols:
                raise ValueError(
                    f"embedding {embedding!r} was fit on {cols}, so it cannot accept "
                    f"the requested columns; omit columns= to use its own space"
                )
        else:
            fitted_embedding = None
            cols = self._resolve_columns(columns)

        if isinstance(data, pl.DataFrame):
            missing = [c for c in cols if c not in data.columns]
            if missing:
                raise ValueError(f"data is missing feature columns: {missing}")
            matrix = data.select(cols).to_numpy()
        else:
            matrix = np.asarray(data, dtype=float)
            if matrix.ndim != 2:
                raise ValueError(f"data must be 2-dimensional, got {matrix.ndim}d")
            if matrix.shape[1] != len(cols):
                raise ValueError(
                    f"data has {matrix.shape[1]} columns but the fit covers "
                    f"{len(cols)} ({cols}); pass a polars frame to match by name"
                )
        scaled = self._scaler(name, cols).transform(matrix)
        return (
            scaled if fitted_embedding is None else fitted_embedding.transform(scaled)
        )

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
        labels: Any = None,
    ) -> pl.DataFrame:
        """Return the masked table with feature columns raw or scaled.

        The primary interactive surface: one row per masked cell, all
        metadata/label columns intact, feature columns raw (``scaled=False``) or
        normalized (``scaled=True``). With ``columns`` only that collection's
        features are scaled; with ``embedding`` a stored embedding's coordinates
        are joined on. Internal ``_mask_*`` columns are dropped.

        ``labels`` joins an unattached ``LabelSet``'s name/id columns onto the
        view (no ``attach`` needed) — handy for one-off plotting. It may also
        be the name of an already-attached label column (see ``labelset``). If
        ``mask`` is omitted, it defaults to ``labels.mask``, the mask the
        LabelSet was computed on, so passing just ``labels`` is enough on its
        own.

        What you pass wins its own name. Previewing a new cut under a name already
        attached to the table is the normal way to work — cut, look, re-cut — and
        the view shows the ``LabelSet`` you handed it, shadowing the attached column
        for this frame only. The table itself is unchanged; ``attach`` still refuses
        to overwrite. The same holds for ``embedding``.
        """
        labels = self._resolve_labels(labels, mask=mask)
        name = mask or (labels.mask if labels is not None else None) or _DEFAULT_MASK
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
            coords = self.embedding(name, name=embedding)
            uncovered = (
                frame.height
                - frame.join(
                    coords.select(self._id_column), on=self._id_column, how="semi"
                ).height
            )
            if uncovered:
                # a left join would give these cells null coordinates, and every
                # plotting path drops null coordinates without saying so — the
                # result is a figure quietly missing most of its cells
                raise ValueError(
                    f"embedding {embedding!r} covers {coords.height} cells but mask "
                    f"{name!r} has {frame.height}, so {uncovered} would get null "
                    f"coordinates. The embedding was computed over a different set of "
                    f"cells than the mask holds now — the mask was redefined, the "
                    f"table gained rows, or the id column changed since. Re-run "
                    f"ft.embed({name!r}, name={embedding!r}, …) to cover it."
                )
            frame = self._join_view(frame, coords)
        if labels is not None:
            frame = self._join_view(frame, labels.to_frame(id_column=self._id_column))
        return frame

    def _join_view(self, frame: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
        """Left-join onto a view, letting the incoming columns win a name clash.

        Polars would suffix a colliding column ``_right``, which is the worst of the
        three options here: the caller asked for *these* labels or *this* embedding
        and gets a frame where that name still holds the attached column, silently,
        with the thing they passed hidden one suffix away. Anything reading
        ``labels.name`` — every plotting helper — then draws the attached column and
        drops the cells it doesn't cover. Shadowing is safe because this is a view;
        the table's own columns are untouched.
        """
        clash = [
            column
            for column in incoming.columns
            if column != self._id_column and column in frame.columns
        ]
        if clash:
            frame = frame.drop(clash)
        return frame.join(incoming, on=self._id_column, how="left")

    # -- comparison ------------------------------------------------------------

    def compare(self, a: Any, b: Any) -> Any:
        """Compare two label sets over shared cells (see :mod:`cellpax.compare`).

        ``a``/``b`` may each be a ``LabelSet`` or the name of an (attached)
        label column (see ``labelset``).
        """
        from cellpax.compare import compare

        return compare(self._resolve_labels(a), self._resolve_labels(b))

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
