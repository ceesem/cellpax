"""The representation cells are compared in: scaling, PCA, and how PCs are weighted.

``FeatureTable`` already freezes two fits and reuses them — :class:`FittedScaler` for
scaling and :class:`FittedEmbedding` for a stored embedding. The PCA that clustering
actually runs in was the missing third: ``features_pca`` refit it on every call and threw
it away, so the space a clustering was computed in could neither be reapplied to new
cells nor persisted, and a sweep over a downstream parameter silently refit the space
underneath itself.

:class:`FittedSpace` closes that. It fits **all** components once and treats both the
truncation and the PC weighting as cheap views over that one fit, which is what makes
"nothing upstream refits inside the sweep" a structural property rather than a
convention:

- the full eigenvalue spectrum is available, so the scree plot and the excess kurtosis of
  *discarded* components cost nothing;
- :meth:`FittedSpace.with_alpha` and :meth:`FittedSpace.with_components` share the
  fitted arrays, so five whitening strengths are five views of one PCA.

Why weight PCs at all: truncating PCA is a rotation plus a truncation, not a
reweighting. A block of features that measures one thing several ways (soma volume,
soma area, soma radius; or eleven percentiles of one depth profile) collapses into a
single high-eigenvalue component that then dominates Euclidean distance exactly as much
as the raw block did. With dozens of engineered features rather than thousands of genes
this does not average out. Scaling component *j* by ``λ_j**(-alpha/2)`` interpolates
between leaving that alone (``alpha=0``, the historical behaviour) and equalising it
entirely (``alpha=1``, full whitening).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

# Robust sigma per IQR: for a Gaussian, IQR = 2 * Phi^-1(0.75) * sigma.
_IQR_TO_SIGMA = 1.3489795003921634


@dataclass(frozen=True)
class FittedSpace:
    """A frozen scaling + PCA fit, plus how strongly its components are equalised.

    The object ``ft.space(...)`` returns and ``ft.cluster`` compares cells in. Built by
    :meth:`fit` on one mask's scaled features, then reused: :meth:`transform` pushes raw
    rows through the same scaling, centring, rotation, and PC weighting, so cells that
    were not part of the fit land in the same space rather than in a new one.

    ``components_`` and ``eigenvalues_`` hold **every** component, not just the retained
    ones; ``n_components`` records where the truncation falls. Nothing is recomputed when
    that boundary or ``alpha`` moves — see :meth:`with_alpha` and
    :meth:`with_components`.

    Parameters
    ----------
    scaler : object, optional
        The fitted scaler applied before PCA — a :class:`~cellpax.FittedScaler`, or
        anything with a ``transform``. ``None`` means rows arrive already scaled.
    columns : tuple of str
        Feature names, in the column order the fit saw. ``transform`` assumes this order
        and cannot check it, which is why ``FeatureTable.space`` keeps the bookkeeping.
    mean_ : numpy.ndarray
        Per-feature mean of the scaled data, subtracted before projection.
    components_ : numpy.ndarray
        ``(n_total_components, n_features)`` rotation, rows ordered by descending
        eigenvalue.
    eigenvalues_ : numpy.ndarray
        Variance along each component (scikit-learn's ``explained_variance_``).
    explained_variance_ratio_ : numpy.ndarray
        The same, as a fraction of the total.
    n_components : int
        How many leading components :meth:`transform` keeps.
    alpha : float, default 0.0
        PC weighting exponent. ``0.0`` leaves component scales untouched, ``1.0`` gives
        every retained component unit variance. Intermediate values partially equalise.
    eigenvalue_floor : float, default 0.0
        Added to each eigenvalue before the ``alpha`` scaling. Guards the sharp edge in
        whitening: the *smallest retained* component is amplified most, so a truncation
        chosen by cumulative variance turns into a discontinuity — the last kept
        component gets full weight and the first dropped one gets none. A floor of
        :attr:`noise_floor` keeps near-degenerate directions from being inflated into
        the metric. ``0.0`` is exact whitening.
    explained_variance : float, optional
        The cumulative-variance target ``n_components`` came from, kept for provenance.
        ``None`` when the count was set directly.
    feature_weights : numpy.ndarray, optional
        Per-feature multipliers applied before PCA, in ``columns`` order.
    """

    scaler: Any
    columns: tuple[str, ...]
    mean_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray
    explained_variance_ratio_: np.ndarray
    n_components: int
    alpha: float = 0.0
    eigenvalue_floor: float = 0.0
    explained_variance: float | None = None
    feature_weights: np.ndarray | None = None
    """Per-feature multipliers applied to the scaled features *before* PCA.

    The other way to stop a redundant block of features dominating Euclidean distance,
    and a better-targeted one than ``alpha``: it acts on the features, so it never touches
    the low-variance component directions that whitening amplifies. See
    :func:`cellpax.diagnostics.block_weights`, which divides each correlated block by the
    scale of its own leading direction so a block of eleven near-identical columns counts
    once rather than eleven times.

    Frozen with the rest of the fit — applied inside :meth:`transform` and
    :meth:`transform_scaled`, and carried through :meth:`to_records` — so an embedding or
    a projection built on a weighted space cannot silently skip the weighting.
    """

    # -- construction ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        scaled: np.ndarray,
        *,
        columns: Any,
        scaler: Any = None,
        explained_variance: float | int = 0.95,
        alpha: float = 0.0,
        eigenvalue_floor: float = 0.0,
        feature_weights: np.ndarray | None = None,
        seed: int | None = None,
    ) -> "FittedSpace":
        """Fit PCA on already-scaled features, keeping every component.

        Parameters
        ----------
        scaled : numpy.ndarray
            ``(n_cells, n_features)`` **already-scaled** matrix, e.g.
            ``ft.features(mask, scaled=True)``.
        columns : sequence of str
            Feature names, in the column order of ``scaled``.
        scaler : object, optional
            The fitted scaler that produced ``scaled``, stored so :meth:`transform` can run
            the whole chain on raw rows later. ``None`` means rows always arrive scaled.
        explained_variance : float or int, default 0.95
            A fraction in ``(0, 1]`` keeps the smallest number of components whose
            cumulative explained-variance ratio reaches it, matching scikit-learn's float
            ``n_components`` semantics; an integer keeps exactly that many. ``1`` and
            ``1.0`` therefore differ.
        alpha : float, default 0.0
            Initial whitening strength — see :meth:`with_alpha`, which is the cheaper way
            to vary it.
        eigenvalue_floor : float, default 0.0
            Initial eigenvalue offset — see :attr:`noise_floor`.
        feature_weights : numpy.ndarray, optional
            Per-feature multipliers applied to ``scaled`` before the PCA is fit, and
            re-applied by :meth:`transform` thereafter. See
            :func:`cellpax.diagnostics.block_weights`.
        seed : int, optional
            Random state for the PCA.

        Returns
        -------
        FittedSpace

        Notes
        -----
        The PCA always keeps *all* components regardless of ``explained_variance``, so the
        truncation is only ever a view. ``svd_solver="full"`` is pinned rather than left to
        ``"auto"`` so the rotation does not change with matrix shape.
        """
        from sklearn.decomposition import PCA

        scaled = np.asarray(scaled, dtype=float)
        if scaled.ndim != 2:
            raise ValueError(f"scaled features must be 2-D, got shape {scaled.shape}")
        if scaled.shape[0] < 2:
            raise ValueError("PCA needs at least two rows")
        if feature_weights is not None:
            feature_weights = np.asarray(feature_weights, dtype=float).reshape(-1)
            if feature_weights.shape[0] != scaled.shape[1]:
                raise ValueError(
                    f"feature_weights has {feature_weights.shape[0]} entries but "
                    f"{scaled.shape[1]} features were given"
                )
            # weight first: mean_ and the rotation are both fit on the weighted data, so
            # transform_scaled has to apply the weights in the same order
            scaled = scaled * feature_weights

        model = PCA(n_components=None, svd_solver="full", random_state=seed)
        model.fit(scaled)

        space = cls(
            scaler=scaler,
            columns=tuple(columns),
            mean_=np.asarray(model.mean_, dtype=float),
            components_=np.asarray(model.components_, dtype=float),
            eigenvalues_=np.asarray(model.explained_variance_, dtype=float),
            explained_variance_ratio_=np.asarray(
                model.explained_variance_ratio_, dtype=float
            ),
            n_components=1,
            alpha=float(alpha),
            eigenvalue_floor=float(eigenvalue_floor),
            explained_variance=None,
            feature_weights=feature_weights,
        )
        if isinstance(explained_variance, (int, np.integer)) and not isinstance(
            explained_variance, bool
        ):
            n = int(explained_variance)
            if not 1 <= n <= space.n_total_components:
                raise ValueError(
                    f"n_components={n} outside 1..{space.n_total_components}"
                )
            return replace(space, n_components=n, explained_variance=None)
        target = float(explained_variance)
        if not 0.0 < target <= 1.0:
            raise ValueError(
                f"explained_variance must be in (0, 1] or an integer count, "
                f"got {explained_variance!r}"
            )
        return replace(
            space,
            n_components=space.n_components_for(target),
            explained_variance=target,
        )

    # -- views over the same fit ----------------------------------------------

    def with_alpha(
        self, alpha: float, *, eigenvalue_floor: float | None = None
    ) -> "FittedSpace":
        """This space at a different PC weighting, sharing the same fit.

        Parameters
        ----------
        alpha : float
            Weighting exponent: component *j* is scaled by ``λ_j ** (-alpha/2)``. ``0.0``
            leaves PCA's scaling untouched and short-circuits entirely, ``1.0`` gives unit
            variance.
        eigenvalue_floor : float, optional
            New eigenvalue offset. ``None`` keeps the current one.

        Returns
        -------
        FittedSpace
            A view, not a copy: it holds the *same* arrays. That is what lets a sweep over
            ``alpha`` be honest — every representation differs only in the weighting,
            because there is only one PCA behind all of them.
        """
        return replace(
            self,
            alpha=float(alpha),
            eigenvalue_floor=self.eigenvalue_floor
            if eigenvalue_floor is None
            else float(eigenvalue_floor),
        )

    def with_components(self, n_components: int) -> "FittedSpace":
        """Return this fitted space at a different component count.

        Parameters
        ----------
        n_components : int
            Number of leading components to retain.

        Returns
        -------
        FittedSpace
            A view sharing the same fitted arrays.

        Raises
        ------
        ValueError
            If the count is outside the fitted component range.
        """
        n = int(n_components)
        if not 1 <= n <= self.n_total_components:
            raise ValueError(
                f"n_components={n} outside 1..{self.n_total_components} available"
            )
        return replace(self, n_components=n, explained_variance=None)

    def n_components_for(self, explained_variance: float = 0.95) -> int:
        """Components needed to reach a cumulative explained-variance fraction.

        Matches scikit-learn's float ``n_components`` rule, so switching a
        ``PCA(n_components=0.95)`` call over to this space keeps the same width.

        Parameters
        ----------
        explained_variance : float, default 0.95
            Cumulative explained-variance target in ``(0, 1]``.

        Returns
        -------
        int
            Smallest leading-component count reaching the target.
        """
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError(
                f"explained_variance must be in (0, 1], got {explained_variance}"
            )
        cumulative = np.cumsum(self.explained_variance_ratio_)
        # capped: at exactly 1.0 (or under accumulated round-off) searchsorted lands
        # one past the end, which would report n_total + 1 components
        return int(
            min(
                np.searchsorted(cumulative, explained_variance, side="right") + 1,
                cumulative.size,
            )
        )

    # -- properties -----------------------------------------------------------

    @property
    def n_total_components(self) -> int:
        """Components the fit produced, before truncation."""
        return int(self.components_.shape[0])

    @property
    def n_features(self) -> int:
        """Number of input features in the fitted space."""
        return int(self.components_.shape[1])

    @property
    def retained(self) -> np.ndarray:
        """Boolean mask over all components: which ones :meth:`transform` keeps."""
        mask = np.zeros(self.n_total_components, dtype=bool)
        mask[: self.n_components] = True
        return mask

    @property
    def cumulative_explained_variance(self) -> float:
        """Fraction of variance the retained components actually carry."""
        return float(self.explained_variance_ratio_[: self.n_components].sum())

    @property
    def noise_floor(self) -> float:
        """Median eigenvalue among the discarded components.

        The natural value for :attr:`eigenvalue_floor`: an estimate of the variance
        scale that the truncation has already judged to be noise. Adding it before the
        ``alpha`` scaling stops whitening from inflating retained directions that are
        barely above that scale. ``0.0`` when nothing was discarded.
        """
        discarded = self.eigenvalues_[self.n_components :]
        return float(np.median(discarded)) if discarded.size else 0.0

    @property
    def condition_number(self) -> float:
        """Condition number of the centred, scaled feature matrix (pre-PCA).

        ``sqrt(λ_max / λ_min)`` over *all* components, i.e. the ratio of the largest to
        smallest singular value — the sense in which "the matrix is ill-conditioned" is
        usually meant. See :attr:`covariance_condition_number` for the other one.
        ``inf`` when the matrix is rank-deficient.
        """
        return float(np.sqrt(self.covariance_condition_number))

    @property
    def covariance_condition_number(self) -> float:
        """``λ_max / λ_min`` over all components — the covariance's condition number.

        The square of :attr:`condition_number`. Reported separately because both
        quantities get called "the condition number" and they differ by a square root,
        which is a factor of thousands at the values redundant feature blocks produce.
        """
        smallest = float(self.eigenvalues_[-1])
        if smallest <= 0:
            return float("inf")
        return float(self.eigenvalues_[0] / smallest)

    @property
    def label(self) -> str:
        """Provenance string, e.g. ``'pca(0.95, alpha=0.5)'``.

        What ``Clustering.space`` records, so a reloaded clustering still says which
        representation produced it.
        """
        if self.explained_variance is not None:
            parts = [f"pca({self.explained_variance:g}"]
        else:
            parts = [f"pca(n={self.n_components}"]
        if self.feature_weights is not None:
            parts.append("weighted")
        if self.alpha != 0.0:
            parts.append(f"alpha={self.alpha:g}")
            if self.eigenvalue_floor:
                parts.append(f"floor={self.eigenvalue_floor:g}")
        return ", ".join(parts) + ")"

    # -- transforms -----------------------------------------------------------

    def _alpha_scale(self) -> np.ndarray | None:
        """Per-component multiplier implementing the ``alpha`` weighting, or ``None``.

        ``None`` at ``alpha == 0`` — the identity is skipped rather than applied as
        ``λ**0``, so an unweighted space is bitwise the plain projection.
        """
        if self.alpha == 0.0:
            return None
        eigenvalues = self.eigenvalues_[: self.n_components] + self.eigenvalue_floor
        # A rank-deficient direction has an eigenvalue near zero rather than at zero, and
        # dividing by its square root amplifies pure round-off by many orders of
        # magnitude. Use the standard relative rank tolerance so that is caught rather
        # than silently promoted into the metric.
        tolerance = (
            float(self.eigenvalues_[0]) * self.n_total_components * np.finfo(float).eps
        )
        if np.any(eigenvalues <= tolerance):
            raise ValueError(
                f"cannot weight components whose eigenvalues are numerically zero: the "
                f"retained spectrum reaches {eigenvalues.min():.3g}, at or below the "
                f"rank tolerance {tolerance:.3g}, so whitening it would amplify "
                f"round-off. Pass eigenvalue_floor=space.noise_floor, or truncate "
                f"further with with_components()."
            )
        return eigenvalues ** (-self.alpha / 2.0)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Coordinates for raw (unscaled) rows over :attr:`columns`.

        Runs the frozen scaler, then centres, projects onto the retained components,
        and applies the ``alpha`` weighting.

        Parameters
        ----------
        raw : numpy.ndarray
            ``(n_rows, n_features)`` raw feature matrix in :attr:`columns` order.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_components)`` fitted-space coordinates.
        """
        raw = np.asarray(raw, dtype=float)
        scaled = raw if self.scaler is None else self.scaler.transform(raw)
        return self.transform_scaled(scaled)

    def transform_scaled(self, scaled: np.ndarray) -> np.ndarray:
        """Project rows that are already scaled.

        Parameters
        ----------
        scaled : numpy.ndarray
            ``(n_rows, n_features)`` scaled matrix in :attr:`columns` order.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_components)`` fitted-space coordinates.
        """
        scaled = self._weighted(scaled)
        coords = (scaled - self.mean_) @ self.components_[: self.n_components].T
        scale = self._alpha_scale()
        return coords if scale is None else coords * scale

    def _weighted(self, scaled: np.ndarray) -> np.ndarray:
        """Validate width and apply :attr:`feature_weights`, as the fit did."""
        scaled = np.asarray(scaled, dtype=float)
        if scaled.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features ({len(self.columns)} columns "
                f"were fit), got {scaled.shape[1]}"
            )
        if self.feature_weights is None:
            return scaled
        return scaled * self.feature_weights

    def scores(
        self, scaled: np.ndarray, *, n_components: int | None = None
    ) -> np.ndarray:
        """Unweighted projection onto components, retained *and* discarded.

        Parameters
        ----------
        scaled : numpy.ndarray
            Already-scaled rows over :attr:`columns`.
        n_components : int, optional
            How many components to project onto. ``None`` — the default — gives every
            component, including the discarded ones, which is the point.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_components)`` scores, with no ``alpha`` weighting applied.

        Notes
        -----
        The diagnostic entry point: the discarded columns are what :meth:`spectrum` reads to
        tell a near-Gaussian dropped component (noise) from a sharply leptokurtic one (a
        small group of cells the truncation is throwing away).
        """
        scaled = self._weighted(scaled)
        k = self.n_total_components if n_components is None else int(n_components)
        return (scaled - self.mean_) @ self.components_[:k].T

    # -- diagnostics ----------------------------------------------------------

    def spectrum(
        self, scaled: np.ndarray | None = None, *, tail_sigma: float = 5.0
    ) -> pl.DataFrame:
        """The eigenvalue spectrum, one row per component.

        Parameters
        ----------
        scaled : numpy.ndarray, optional
            Already-scaled rows. Without them only the eigenvalue columns are available;
            with them each row also gets ``excess_kurtosis`` and ``n_tail_cells``, which
            are the informative ones.
        tail_sigma : float, default 5.0
            Tail cutoff, in **Gaussian-comparable** units: the IQR is converted with the
            1.349 factor first, so ``5.0`` means five robust standard deviations rather
            than five IQRs.

        Returns
        -------
        pl.DataFrame
            One row per component: ``component``, ``eigenvalue``, ``ev_ratio``,
            ``cumulative_ev``, ``retained``, plus ``excess_kurtosis`` and
            ``n_tail_cells`` when ``scaled`` is given.

        Notes
        -----
        The kurtosis columns are the point of keeping the discarded components. A
        dropped component with excess kurtosis near ``0`` is noise and losing it costs
        nothing. One with high excess kurtosis is a small group of cells separating from
        the bulk along a low-variance direction, and a cumulative-variance cut discards
        it precisely *because* few cells are involved — which is backwards if those cells
        are a rare type. ``n_tail_cells`` counts how many sit beyond
        ``median ± tail_sigma × IQR/1.349`` on that component, i.e. how many cells the
        cut would be giving up on.

        ``tail_sigma`` is in Gaussian-comparable units: the IQR is converted with the
        1.349 factor first, so ``5.0`` means five robust standard deviations rather than
        five IQRs.
        """
        n = self.n_total_components
        frame = pl.DataFrame(
            {
                "component": np.arange(n, dtype=np.int64),
                "eigenvalue": self.eigenvalues_,
                "ev_ratio": self.explained_variance_ratio_,
                "cumulative_ev": np.cumsum(self.explained_variance_ratio_),
                "retained": self.retained,
            }
        )
        if scaled is None:
            return frame
        coords = self.scores(scaled)
        return frame.with_columns(
            pl.Series("excess_kurtosis", _excess_kurtosis(coords)),
            pl.Series("n_tail_cells", _tail_counts(coords, tail_sigma)),
        )

    # -- serialization --------------------------------------------------------

    def to_records(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """JSON-safe metadata and named arrays, for persistence.

        Deliberately explicit parameters rather than a pickled estimator. The point of
        freezing a space is to reapply it *unchanged* to a future dataset, possibly
        under a different scikit-learn; a pickle couples that to a library version, and
        an opaque blob cannot be inspected to see what the frozen scaling actually was.

        Returns
        -------
        metadata : dict
            JSON-safe space description.
        arrays : dict of str to numpy.ndarray
            Named numeric arrays required to restore the fit.
        """
        meta: dict[str, Any] = {
            "columns": list(self.columns),
            "n_components": int(self.n_components),
            "alpha": float(self.alpha),
            "eigenvalue_floor": float(self.eigenvalue_floor),
            "explained_variance": self.explained_variance,
        }
        arrays: dict[str, np.ndarray] = {
            "mean_": self.mean_,
            "components_": self.components_,
            "eigenvalues_": self.eigenvalues_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
        }
        if self.feature_weights is not None:
            arrays["feature_weights"] = self.feature_weights
        scaler_meta, scaler_arrays = _scaler_records(self.scaler)
        meta["scaler"] = scaler_meta
        arrays.update(scaler_arrays)
        return meta, arrays

    @classmethod
    def from_records(
        cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> "FittedSpace":
        """Rebuild a space from serialized records.

        Parameters
        ----------
        meta : dict
            JSON-safe metadata returned by :meth:`to_records`.
        arrays : dict of str to numpy.ndarray
            Named arrays returned by :meth:`to_records`.

        Returns
        -------
        FittedSpace
            Reconstructed frozen space.
        """
        return cls(
            scaler=_scaler_from_records(meta.get("scaler"), arrays),
            columns=tuple(meta["columns"]),
            mean_=np.asarray(arrays["mean_"], dtype=float),
            components_=np.asarray(arrays["components_"], dtype=float),
            eigenvalues_=np.asarray(arrays["eigenvalues_"], dtype=float),
            explained_variance_ratio_=np.asarray(
                arrays["explained_variance_ratio_"], dtype=float
            ),
            n_components=int(meta["n_components"]),
            alpha=float(meta.get("alpha", 0.0)),
            eigenvalue_floor=float(meta.get("eigenvalue_floor", 0.0)),
            explained_variance=meta.get("explained_variance"),
            feature_weights=(
                np.asarray(arrays["feature_weights"], dtype=float)
                if "feature_weights" in arrays
                else None
            ),
        )

    def __repr__(self) -> str:
        return (
            f"FittedSpace({self.label}, {self.n_components}/"
            f"{self.n_total_components} components, {self.n_features} features, "
            f"cumulative_ev={self.cumulative_explained_variance:.3f})"
        )


# --------------------------------------------------------------------------- #
# diagnostics helpers
# --------------------------------------------------------------------------- #


def _excess_kurtosis(coords: np.ndarray) -> np.ndarray:
    """Fisher excess kurtosis per column: ``m4 / m2**2 - 3``, 0 for a Gaussian."""
    centered = coords - coords.mean(axis=0)
    m2 = (centered**2).mean(axis=0)
    m4 = (centered**4).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(m2 > 0, m4 / np.square(m2) - 3.0, 0.0)
    return np.asarray(out, dtype=float)


def _tail_counts(coords: np.ndarray, tail_sigma: float) -> np.ndarray:
    """Cells beyond ``median ± tail_sigma × IQR/1.349`` on each column."""
    median = np.median(coords, axis=0)
    q75, q25 = np.percentile(coords, [75, 25], axis=0)
    sigma = (q75 - q25) / _IQR_TO_SIGMA
    counts = np.zeros(coords.shape[1], dtype=np.int64)
    usable = sigma > 0
    if usable.any():
        deviation = np.abs(coords[:, usable] - median[usable]) / sigma[usable]
        counts[usable] = (deviation > tail_sigma).sum(axis=0)
    return counts


# --------------------------------------------------------------------------- #
# scaler parameters, extracted rather than pickled
# --------------------------------------------------------------------------- #

_STANDARD = "standard"
_ROBUST = "robust"
_ROBUST_PERCENTILE = "robust_percentile"
_ROBUST_SIGMA = "robust_sigma"
_QUANTILE = "quantile"
_PERCENTILE_QUANTILE = "percentile_quantile"


def _quantile_records(
    transformer: Any, meta: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    """Record a fitted ``QuantileTransformer``'s landmarks and configuration."""
    meta["output_distribution"] = str(transformer.output_distribution)
    meta["n_quantiles"] = int(transformer.n_quantiles_)
    meta["subsample"] = (
        None if transformer.subsample is None else int(transformer.subsample)
    )
    meta["random_state"] = (
        transformer.random_state
        if transformer.random_state is None or isinstance(transformer.random_state, int)
        else None
    )
    meta["ignore_implicit_zeros"] = bool(transformer.ignore_implicit_zeros)
    arrays["scaler__quantiles_"] = np.asarray(transformer.quantiles_, dtype=float)
    arrays["scaler__references_"] = np.asarray(transformer.references_, dtype=float)


def _quantile_from_records(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    """Rebuild a fitted ``QuantileTransformer`` from :func:`_quantile_records`."""
    from sklearn.preprocessing import QuantileTransformer

    quantiles = np.atleast_2d(np.asarray(arrays["scaler__quantiles_"], dtype=float))
    references = np.asarray(arrays["scaler__references_"], dtype=float).ravel()
    if quantiles.shape[0] != references.size:
        # a single-feature fit round-trips as one row; landmarks are the rows
        quantiles = quantiles.T
    transformer = QuantileTransformer(
        n_quantiles=int(meta["n_quantiles"]),
        output_distribution=meta["output_distribution"],
        subsample=meta.get("subsample"),
        random_state=meta.get("random_state"),
        ignore_implicit_zeros=bool(meta.get("ignore_implicit_zeros", False)),
    )
    transformer.quantiles_ = quantiles
    transformer.references_ = references
    transformer.n_quantiles_ = int(meta["n_quantiles"])
    transformer.n_features_in_ = quantiles.shape[1]
    return transformer


def _scaler_records(fitted: Any) -> tuple[dict[str, Any] | None, dict[str, np.ndarray]]:
    """Extract a fitted scaler's parameters as metadata plus named arrays.

    Raises on a scaler shape it does not recognise. That is deliberate: the older
    persistence layer silently recorded an unrecognised scaler as ``"standard"``, so a
    reload quietly changed the preprocessing. Failing loudly is the lesser harm — pass
    the fitted space through ``DataFolio.add_model`` if a genuinely custom scaler has to
    be carried.
    """
    if fitted is None:
        return None, {}
    inner = getattr(fitted, "scaler", fitted)
    transforms = list(getattr(fitted, "transforms", []) or [])
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"transforms": transforms}
    shifts = getattr(fitted, "shifts", None)
    if shifts is not None:
        meta["shifts"] = [float(s) for s in shifts]

    steps = getattr(inner, "named_steps", None)
    if steps is not None:
        scaler, clipper = steps.get("scaler"), steps.get("clipper")
        order = list(steps)
        if (
            order == ["clipper", "scaler"]
            and type(clipper).__name__ == "PercentileClipper"
            and type(scaler).__name__ == "QuantileTransformer"
        ):
            meta["kind"] = _PERCENTILE_QUANTILE
            meta["lower"] = float(clipper.lower)
            meta["upper"] = float(clipper.upper)
            arrays["clipper__lower_bounds_"] = np.asarray(
                clipper.lower_bounds_, dtype=float
            )
            arrays["clipper__upper_bounds_"] = np.asarray(
                clipper.upper_bounds_, dtype=float
            )
            _quantile_records(scaler, meta, arrays)
            return meta, arrays
        if order != ["scaler", "clipper"] or type(scaler).__name__ != "RobustScaler":
            raise TypeError(
                "only Pipeline([('scaler', RobustScaler()), ('clipper', ...)]) and "
                "Pipeline([('clipper', PercentileClipper()), ('scaler', "
                f"QuantileTransformer())]) pipelines can be frozen, got steps {order}"
            )
        arrays["scaler__center_"] = np.asarray(scaler.center_, dtype=float)
        arrays["scaler__scale_"] = np.asarray(scaler.scale_, dtype=float)
        name = type(clipper).__name__
        if name == "PercentileClipper":
            meta["kind"] = _ROBUST_PERCENTILE
            meta["lower"] = float(clipper.lower)
            meta["upper"] = float(clipper.upper)
            arrays["clipper__lower_bounds_"] = np.asarray(
                clipper.lower_bounds_, dtype=float
            )
            arrays["clipper__upper_bounds_"] = np.asarray(
                clipper.upper_bounds_, dtype=float
            )
        elif name == "SigmaClipper":
            # Nothing fitted to store, which is the reason to prefer it: a frozen
            # space carries no clip bounds for a future dataset to shift.
            meta["kind"] = _ROBUST_SIGMA
            meta["n_sigma"] = float(clipper.n_sigma)
        else:
            raise TypeError(f"cannot freeze clipper of type {name}")
        return meta, arrays

    name = type(inner).__name__
    if name == "StandardScaler":
        meta["kind"] = _STANDARD
        arrays["scaler__mean_"] = np.asarray(inner.mean_, dtype=float)
        arrays["scaler__scale_"] = np.asarray(inner.scale_, dtype=float)
    elif name == "RobustScaler":
        meta["kind"] = _ROBUST
        arrays["scaler__center_"] = np.asarray(inner.center_, dtype=float)
        arrays["scaler__scale_"] = np.asarray(inner.scale_, dtype=float)
    elif name == "QuantileTransformer":
        meta["kind"] = _QUANTILE
        _quantile_records(inner, meta, arrays)
    else:
        raise TypeError(
            f"cannot freeze a scaler of type {name}; supported: StandardScaler, "
            "RobustScaler, QuantileTransformer, RobustScaler + "
            "PercentileClipper/SigmaClipper pipelines, and PercentileClipper + "
            "QuantileTransformer pipelines"
        )
    return meta, arrays


def _scaler_from_records(
    meta: dict[str, Any] | None, arrays: dict[str, np.ndarray]
) -> Any:
    """Rebuild a fitted scaler from :func:`_scaler_records` output."""
    if meta is None:
        return None
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    from cellpax.clustering import PercentileClipper, SigmaClipper
    from cellpax.featuretable import FittedScaler

    kind = meta["kind"]
    inner: Any
    if kind == _QUANTILE:
        inner = _quantile_from_records(meta, arrays)
    elif kind == _PERCENTILE_QUANTILE:
        quantile_clipper = PercentileClipper(meta["lower"], meta["upper"])
        quantile_clipper.lower_bounds_ = np.asarray(
            arrays["clipper__lower_bounds_"], dtype=float
        )
        quantile_clipper.upper_bounds_ = np.asarray(
            arrays["clipper__upper_bounds_"], dtype=float
        )
        inner = Pipeline(
            [
                ("clipper", quantile_clipper),
                ("scaler", _quantile_from_records(meta, arrays)),
            ]
        )
    elif kind == _STANDARD:
        inner = StandardScaler()
        inner.mean_ = np.asarray(arrays["scaler__mean_"], dtype=float)
        inner.scale_ = np.asarray(arrays["scaler__scale_"], dtype=float)
        inner.var_ = inner.scale_**2
        inner.n_features_in_ = inner.mean_.size
    else:
        robust = RobustScaler()
        robust.center_ = np.asarray(arrays["scaler__center_"], dtype=float)
        robust.scale_ = np.asarray(arrays["scaler__scale_"], dtype=float)
        robust.n_features_in_ = robust.center_.size
        if kind == _ROBUST:
            inner = robust
        elif kind == _ROBUST_PERCENTILE:
            clipper: Any = PercentileClipper(meta["lower"], meta["upper"])
            clipper.lower_bounds_ = np.asarray(
                arrays["clipper__lower_bounds_"], dtype=float
            )
            clipper.upper_bounds_ = np.asarray(
                arrays["clipper__upper_bounds_"], dtype=float
            )
            inner = Pipeline([("scaler", robust), ("clipper", clipper)])
        elif kind == _ROBUST_SIGMA:
            sigma_clipper = SigmaClipper(meta["n_sigma"])
            sigma_clipper.n_features_in_ = robust.center_.size
            inner = Pipeline([("scaler", robust), ("clipper", sigma_clipper)])
        else:
            raise ValueError(f"unknown frozen scaler kind {kind!r}")
    shifts = meta.get("shifts")
    return FittedScaler(
        scaler=inner,
        transforms=list(meta.get("transforms", [])),
        shifts=None if shifts is None else [float(s) for s in shifts],
    )
