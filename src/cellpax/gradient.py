"""A continuous coordinate along a manifold — ``LabelSet``'s honest sibling.

When the boundary report calls a pair of clusters ``"continuous"``, the modes
were never there: the names were compass directions on one block. The honest
object for that structure is not a better cut but a *coordinate* — where along
the thing each cell sits — and ``Gradient`` is that object: a per-cell arc
length in ``[0, 1]``, with provenance, attachable to the table like any label,
and convertible back to named labels as *declared interval cuts* rather than
pretended modes.

The fit is a Hastie–Stuetzle principal curve (Hastie & Stuetzle, JASA 1989),
implemented in-library because the Python ecosystem offers no maintained
implementation (verified Aug 2026; R's ``princurve`` is the reference).
Iterate two steps until the projections stop moving: smooth each feature as a
function of the current arc-length ordering, then re-project every cell onto
the smoothed curve. Branching topologies are out of scope here — that is
elastic-principal-graph territory (``elpigraph-python``), a future optional
backend.

Two guards are built in rather than documented as advice:

- **The intrinsic-dimension gate.** A 1-D coordinate through genuinely
  higher-dimensional structure compresses it into an artifact. ``parametrize``
  estimates the intrinsic dimension first (TwoNN — Facco et al., Scientific
  Reports 2017; ~15 lines, so implemented here; dadapy has the serious
  version) and warns when the manifold does not look like a curve.
- **The nuisance tripwire.** Truncation manufactures fake gradients: partial
  reconstruction is *graded* feature loss, so a completeness artifact reads as
  a smooth biological axis — more insidious than a fake cluster, because a
  continuum is what you are now primed to accept. Pass ``nuisance=`` columns
  (completeness metrics, depth if depth is nuisance for the question) and the
  fit warns when the coordinate tracks any of them.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from cellpax.labels import LabelSet

__all__ = ["Gradient", "fit_principal_curve", "twonn_dimension", "twonn_profile"]


def twonn_profile(
    features: np.ndarray, *, decimation_levels: int = 4
) -> list[dict[str, float]]:
    """TwoNN intrinsic dimension at each decimation scale, not just the minimum.

    The estimate every level computes on the way to :func:`twonn_dimension`'s
    single number, kept instead of discarded. It is the more informative
    object: dimension *as a function of scale* is what actually distinguishes a
    manifold from noise, and the scalar collapses exactly that.

    Read the shape, not any one value:

    * **flat and low (≈1)** — a curve at every scale, and ``parametrize`` is
      the right tool.
    * **flat and high** — genuinely that many dimensions. A ball stays a ball
      however far out you look; a 1-D coordinate through it is an artifact.
    * **falling with scale** — noise-dominated at small separations. The
      nearest-neighbour distance at full sampling sits *inside* the noise,
      where even a clean curve reads as ambient-dimensional; once the
      subsample thins enough for the scale to clear it, the estimate drops
      toward the real dimension.
    * **rising with scale** — curvature. The structure is locally 1-D but
      folds, so a longer ruler sees more dimensions than a short one.

    Each row is one decimation level: ``n`` cells sampled, the mean ``dimension``
    over the draws at that level, and ``spread`` (the standard deviation across
    draws, ``0.0`` at full sampling where there is only one). A wide spread is
    its own finding — it means the estimate is not determined by the data at
    that scale.

    This is a decimation proxy, not a scale-dependent estimator. For the real
    thing (GRIDE) use the optional ``dadapy`` package.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` coordinate matrix.
    decimation_levels : int, default 4
        Number of successively halved sample sizes to evaluate.

    Returns
    -------
    list of dict
        One record per scale containing ``n``, ``dimension``, ``spread``, and
        ``draws``.
    """
    from sklearn.neighbors import NearestNeighbors

    features = np.asarray(features, dtype=float)
    rng = np.random.default_rng(0)

    def _estimate(points: np.ndarray) -> float:
        finder = NearestNeighbors(n_neighbors=3)
        finder.fit(points)
        distances, _ = finder.kneighbors(points)
        r1 = np.maximum(distances[:, 1], np.finfo(float).tiny)
        ratios = np.maximum(distances[:, 2] / r1, 1.0 + 1e-12)
        keep = np.isfinite(ratios)
        return float(keep.sum() / np.log(ratios[keep]).sum())

    profile: list[dict[str, float]] = []
    n = features.shape[0]
    for level in range(decimation_levels):
        size = n // (2**level)
        if size < 20:
            break
        draws = 1 if level == 0 else 3
        level_values = []
        for _ in range(draws):
            rows = (
                np.arange(n) if size == n else rng.choice(n, size=size, replace=False)
            )
            level_values.append(_estimate(features[rows]))
        profile.append(
            {
                "n": float(size),
                "dimension": float(np.mean(level_values)),
                "spread": float(np.std(level_values)) if draws > 1 else 0.0,
                "draws": float(draws),
            }
        )
    return profile


def twonn_dimension(features: np.ndarray, *, decimation_levels: int = 4) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017), decimated.

    Uses only each point's two nearest-neighbour distances: under a locally
    uniform density the ratios ``mu = r2/r1`` are Pareto with shape equal to
    the intrinsic dimension, so ``d = n / sum(log(mu))``. At full sampling the
    nearest-neighbour scale sits *inside the noise*, where a noisy curve is
    genuinely ambient-dimensional — so the estimate is repeated on random
    decimations (n, n/2, n/4, …), which probe progressively larger scales,
    and the **minimum** across levels is returned: the cleanest-manifold
    reading, which is the right side to err on for a gate (a true ball stays
    high at every scale, a noisy curve drops toward 1 once the scale clears
    the noise).

    Because it takes the minimum it under-reports, so a gate built on it fires
    rarely — and when it does fire that is strong evidence rather than a
    borderline call. See :func:`twonn_profile` for the per-scale estimates this
    collapses, which are the more informative read.

    For real scale-dependent analysis (GRIDE) use the optional ``dadapy``
    package; this exists to gate ``parametrize``, not to settle dimensionality
    questions.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` coordinate matrix.
    decimation_levels : int, default 4
        Number of scales considered by :func:`twonn_profile`.

    Returns
    -------
    float
        Minimum TwoNN dimension estimate across the evaluated scales.
    """
    profile = twonn_profile(features, decimation_levels=decimation_levels)
    return min(level["dimension"] for level in profile)


def _binned_smoother(
    order_values: np.ndarray, targets: np.ndarray, positions: np.ndarray, span: float
) -> np.ndarray:
    """Running-mean smoother of ``targets`` against ``order_values`` at ``positions``.

    A uniform kernel over a fixed fraction of the coordinate range — the
    dependency-free workhorse the curve iteration needs (splines want strictly
    increasing knots, which projected arc lengths routinely violate).
    """
    window = span * (order_values.max() - order_values.min() + np.finfo(float).tiny)
    smoothed = np.empty((positions.size, targets.shape[1]))
    for index, position in enumerate(positions):
        weights = np.abs(order_values - position) <= window
        if not weights.any():
            nearest = np.argmin(np.abs(order_values - position))
            smoothed[index] = targets[nearest]
        else:
            smoothed[index] = targets[weights].mean(axis=0)
    return smoothed


def fit_principal_curve(
    features: np.ndarray,
    *,
    span: float = 0.1,
    n_samples: int = 200,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Hastie–Stuetzle principal curve; returns ``(arc_lengths, curve_samples)``.

    ``arc_lengths`` is per-row arc length normalized to ``[0, 1]`` along the
    fitted curve; ``curve_samples`` is the ``(n_samples, n_dims)`` polyline.
    Initialized from the first principal component, iterated
    smooth-then-project until the mean squared projection distance moves less
    than ``tol`` relatively, or ``max_iter``. ``span`` is the smoother's
    window as a fraction of the coordinate range — the bias/variance knob:
    small spans follow wiggles (and noise), large spans straighten the curve
    back toward the principal component.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_dimensions)`` feature-space coordinates.
    span : float, default 0.1
        Smoothing-window width as a fraction of the coordinate range.
    n_samples : int, default 200
        Number of points in the fitted curve polyline.
    max_iter : int, default 30
        Maximum smooth-and-project iterations.
    tol : float, default 1e-4
        Relative convergence tolerance for projection error.

    Returns
    -------
    arc_lengths : numpy.ndarray
        Per-cell normalized position in ``[0, 1]``.
    curve_samples : numpy.ndarray
        ``(n_samples, n_dimensions)`` fitted polyline.

    Raises
    ------
    ValueError
        If fewer than ten rows are supplied or ``features`` is not two-dimensional.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 10:
        raise ValueError("fit_principal_curve needs a (n >= 10, d) matrix")
    center = features.mean(axis=0)
    centered = features - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    lam = centered @ vt[0]

    previous_error = np.inf
    curve = None
    for _ in range(max_iter):
        positions = np.linspace(lam.min(), lam.max(), n_samples)
        curve = _binned_smoother(lam, features, positions, span)
        # re-project every cell onto the polyline; new lambda = arc length there
        segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        # cdist keeps this at (n, n_samples) doubles; broadcasting through the
        # feature axis would multiply that by the dimensionality
        from scipy.spatial.distance import cdist

        distances = cdist(features, curve)
        nearest = distances.argmin(axis=1)
        error = float(np.mean(distances[np.arange(features.shape[0]), nearest] ** 2))
        lam = arc[nearest]
        if previous_error - error < tol * max(previous_error, np.finfo(float).tiny):
            break
        previous_error = error

    total = lam.max() - lam.min()
    normalized = (lam - lam.min()) / (total if total > 0 else 1.0)
    return normalized, curve


class Gradient:
    """A per-cell coordinate along a fitted 1-D manifold, with provenance.

    Created by ``FeatureTable.parametrize``. ``coordinate`` is arc length in
    ``[0, 1]`` over exactly the cells the fit covered; ``bin()`` converts
    intervals of it back into a ``LabelSet`` whose names are declared cuts of
    a persisted coordinate — revisable and honest — rather than modes the
    data never had. ``ft.attach(gradient)`` writes it as a float column.

    Parameters
    ----------
    cell_ids : numpy.ndarray
        Cell identifiers in coordinate order.
    coordinate : numpy.ndarray
        Normalized arc length for each cell.
    name : str, default 'gradient'
        Column name used when attaching the result.
    mask : str, optional
        Source mask name.
    method : str, default 'principal_curve'
        Fitting method recorded for provenance.
    params : dict, optional
        Parameters of the originating call.
    loadings : polars.DataFrame, optional
        Per-feature association with the coordinate.
    intrinsic_dimension : float, optional
        Scalar TwoNN dimensionality summary.
    dimension_profile : list of dict, optional
        TwoNN estimate at each decimation scale.
    """

    def __init__(
        self,
        cell_ids: np.ndarray,
        coordinate: np.ndarray,
        *,
        name: str = "gradient",
        mask: str | None = None,
        method: str = "principal_curve",
        params: dict[str, Any] | None = None,
        loadings: pl.DataFrame | None = None,
        intrinsic_dimension: float | None = None,
        dimension_profile: list[dict[str, float]] | None = None,
    ) -> None:
        cell_ids = np.asarray(cell_ids)
        coordinate = np.asarray(coordinate, dtype=float)
        if cell_ids.shape[0] != coordinate.shape[0]:
            raise ValueError("cell_ids and coordinate must have the same length")
        self._cell_ids = cell_ids
        self._coordinate = coordinate
        self.name = name
        self.mask = mask
        self.method = method
        self._params = dict(params) if params else None
        self._loadings = loadings
        self.intrinsic_dimension = intrinsic_dimension
        #: TwoNN estimate per decimation scale — see :func:`twonn_profile`.
        #: The shape of this is what says whether a curve was the right model;
        #: ``intrinsic_dimension`` is only its minimum.
        self.dimension_profile = list(dimension_profile or [])

    @property
    def cell_ids(self) -> np.ndarray:
        """The covered cells, aligned with ``coordinate``. A copy."""
        return self._cell_ids.copy()

    @property
    def coordinate(self) -> np.ndarray:
        """Arc length per cell in ``[0, 1]``. A copy."""
        return self._coordinate.copy()

    @property
    def params(self) -> dict[str, Any] | None:
        """The ``parametrize`` call that produced this, seed included."""
        return dict(self._params) if self._params else None

    def loadings(self) -> pl.DataFrame:
        """Spearman correlation of each input feature with the coordinate.

        Which features vary along the axis (|rho| high), which are flat, and —
        read together with a nuisance check — which are carrying artifact.
        Computed at fit time on the features the curve was fit through.

        Returns
        -------
        polars.DataFrame
            Features and Spearman correlations, strongest absolute values first.
        """
        if self._loadings is None:
            raise ValueError("this Gradient was built without loadings")
        return self._loadings

    def __len__(self) -> int:
        return int(self._cell_ids.shape[0])

    def to_frame(self, *, id_column: str = "cell_id") -> pl.DataFrame:
        """Return identifiers and coordinates as a tidy frame.

        Parameters
        ----------
        id_column : str, default 'cell_id'
            Name of the identifier column.

        Returns
        -------
        polars.DataFrame
            ``[id_column, name]`` with one row per covered cell.
        """
        return pl.DataFrame({id_column: self._cell_ids, self.name: self._coordinate})

    def coordinate_for(self, cell_ids: np.ndarray) -> np.ndarray:
        """Align coordinates to an arbitrary identifier order.

        Parameters
        ----------
        cell_ids : numpy.ndarray
            Identifiers to look up, in desired output order.

        Returns
        -------
        numpy.ndarray
            Coordinate per requested identifier; ``nan`` when uncovered.
        """
        by_cell = dict(zip(self._cell_ids, self._coordinate))
        return np.array(
            [by_cell.get(c, np.nan) for c in np.asarray(cell_ids).reshape(-1)],
            dtype=float,
        )

    def bin(
        self,
        edges: int | list[float],
        *,
        names: list[str] | None = None,
        name: str | None = None,
    ) -> "LabelSet":
        """Cut the coordinate into a ``LabelSet`` — named intervals, not modes.

        ``edges`` is an integer (that many equal-*quantile* bins, so sizes are
        balanced) or an explicit ascending list of interior cut points in
        coordinate units. ``names`` labels the bins in order (defaults to the
        interval bounds, so the name *says* it is a cut). The point of going
        back to labels this way: the labels stay honest, because the
        coordinate they cut persists alongside them and the cuts are declared
        numbers anyone can revise.

        Parameters
        ----------
        edges : int or list of float
            Number of equal-quantile bins or ascending interior cut points.
        names : list of str, optional
            Bin names in ascending coordinate order.
        name : str, optional
            Name of the returned label set.

        Returns
        -------
        LabelSet
            Interval codes aligned to the gradient's cells.

        Raises
        ------
        ValueError
            If the bin count, cut ordering, or number of names is invalid.
        """
        from cellpax.labels import LabelSet

        if isinstance(edges, int):
            if edges < 2:
                raise ValueError("bin needs at least two bins")
            quantiles = np.linspace(0, 1, edges + 1)[1:-1]
            cuts = np.quantile(self._coordinate, quantiles)
        else:
            cuts = np.asarray(edges, dtype=float)
            if cuts.size and not np.all(np.diff(cuts) > 0):
                raise ValueError("edges must be strictly ascending")
        codes = np.searchsorted(cuts, self._coordinate, side="right")
        n_bins = cuts.size + 1
        if names is None:
            bounds = np.concatenate([[0.0], cuts, [1.0]])
            names = [
                f"{self.name}[{bounds[i]:.2f}-{bounds[i + 1]:.2f}]"
                for i in range(n_bins)
            ]
        if len(names) != n_bins:
            raise ValueError(f"expected {n_bins} names, got {len(names)}")
        return LabelSet(
            self._cell_ids,
            codes.astype(np.int64),
            names=list(names),
            name=name or self.name,
            mask=self.mask,
        )

    def __repr__(self) -> str:
        dimension = (
            f", intrinsic_dimension={self.intrinsic_dimension:.2f}"
            if self.intrinsic_dimension is not None
            else ""
        )
        return (
            f"Gradient(name={self.name!r}, n_cells={len(self)}, "
            f"mask={self.mask!r}, method={self.method!r}{dimension})"
        )
