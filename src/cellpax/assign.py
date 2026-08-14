"""Conformal label assignment: sets with coverage certificates, not vote shares.

``propagate_labels`` answers "what do my neighbours say?"; its confidence is a
plurality margin with no calibration semantics (the guide is blunt about
this). ``ft.assign`` answers a different question with a guarantee attached:
*which labels can this cell defensibly be given?* A classifier is trained on
part of the curated reference, its surprise at the held-out rest becomes the
calibration yardstick (split conformal prediction), and every cell then gets a
**p-value per label**: how ordinary this cell would look among the calibration
cells if that label were true. Everything else is derived from those p-values
at read time —

- ``prediction_set(alpha)``: the labels not rejected at level ``alpha``. Under
  exchangeability with the calibration cells, the true label is in the set
  with probability at least ``1 - alpha`` — finite-sample, regardless of the
  classifier's quality. The *size* is the honest uncertainty readout:
  singleton where the answer is clear, several labels where it genuinely
  isn't, empty where the cell fits nothing at the stated level.
- ``to_labelset(alpha)``: the conformal-honest hard labelling — singletons
  keep their label, everything else stays unassigned.

Calibration is **Mondrian by class** by default (Vovk's term): each label's
threshold comes from calibration cells *of that label*, so the guarantee
holds per class rather than on average — without it, a 90% marginal guarantee
is happily satisfied by 96% on the abundant type and 40% on the rare one,
which hides the failure exactly where it matters. The cost is that each class
needs enough calibration cells to support its own quantile; classes too small
for the requested ``alpha`` are named in a warning at read time rather than
silently borrowing another class's threshold.

Two honesty notes. Calibrated *probabilities* (sigmoid-calibrated on the same
held-out fold) ride along for ranking and expectation-style uses, but they
are a different object from the sets: probabilities can be wrong together,
sets carry the certificate. And an empty set is **not** out-of-distribution
detection — the guarantee says nothing about cells that aren't exchangeable
with the reference (the truncated periphery, above all). ``plausibility()``
(each cell's best p-value) is a useful screen, but ``score_cells`` and the
validity machinery are the tools actually aimed at novelty; the shift-aware
extension of these guarantees is tracked separately.

The conformal engine is the ``crepes`` package (Boström; standard and
Mondrian conformal classifiers); the design choice cellpax adds is storing
the p-value matrix as *evidence* and making ``alpha`` a read-time parameter —
the same stance the boundary report takes, where columns are the result and
verdicts are summaries.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl

__all__ = ["Assignment", "conditional_prediction_set", "weighted_p_values"]


def density_ratio_weights(
    covariates_calibration: np.ndarray,
    covariates_target: np.ndarray,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Likelihood-ratio weights ``target / calibration`` from shift covariates.

    The standard density-ratio-by-classification estimate: a logistic
    regression learns to tell calibration cells from target cells using only
    the shift covariates, and each cell's odds of being "target" estimate the
    ratio up to a constant (which weighted conformal normalizes away
    per test point). Returns ``(weights_calibration, weights_target)``.

    The estimate is only as good as the covariates: weights computed from a
    completeness metric correct a completeness shift and nothing else. And no
    weighting can conjure support that isn't there — a target regime with *no*
    comparable calibration cells gets enormous weights on a handful of cells,
    which shows up as exploded set sizes rather than restored coverage.
    """
    from sklearn.linear_model import LogisticRegression

    covariates_calibration = np.atleast_2d(
        np.asarray(covariates_calibration, dtype=float)
    )
    covariates_target = np.atleast_2d(np.asarray(covariates_target, dtype=float))
    if covariates_calibration.shape[0] == 1 and covariates_calibration.ndim == 2:
        covariates_calibration = covariates_calibration.reshape(-1, 1)
    if covariates_target.shape[0] == 1 and covariates_target.ndim == 2:
        covariates_target = covariates_target.reshape(-1, 1)
    stacked = np.vstack([covariates_calibration, covariates_target])
    membership = np.concatenate(
        [
            np.zeros(covariates_calibration.shape[0]),
            np.ones(covariates_target.shape[0]),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(stacked, membership)
    probability = np.clip(model.predict_proba(stacked)[:, 1], 1e-6, 1 - 1e-6)
    odds = probability / (1 - probability)
    return (
        odds[: covariates_calibration.shape[0]],
        odds[covariates_calibration.shape[0] :],
    )


def weighted_p_values(
    calibration_scores: np.ndarray,
    calibration_codes: np.ndarray,
    test_scores: np.ndarray,
    *,
    class_ids: np.ndarray,
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
) -> np.ndarray:
    """Class-conditional weighted conformal p-values (Tibshirani et al. 2019).

    For candidate class *k* at test cell *x* with nonconformity ``s``:

    ``p = (Σ_{i ∈ cal_k} w_i · 1{s_i ≥ s} + w(x)) / (Σ_{i ∈ cal_k} w_i + w(x))``

    — ordinary class-conditional conformal when every weight is 1, and valid
    under covariate shift when the weights are the true likelihood ratio.
    ``calibration_scores`` is per-cell (each scored against its own class);
    ``test_scores`` is ``(n_test, n_classes)`` (each cell scored against every
    candidate class). Composes Mondrian-by-class with the shift weighting, as
    in group-weighted conformal prediction.
    """
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    out = np.empty_like(test_scores)
    for j, class_id in enumerate(class_ids):
        members = calibration_codes == class_id
        scores_k = calibration_scores[members]
        weights_k = calibration_weights[members]
        total_k = weights_k.sum()
        # broadcast: for each test cell, weighted mass of calibration scores
        # at least as nonconforming as the cell's own
        at_least = scores_k[None, :] >= test_scores[:, j : j + 1]
        numerator = (at_least * weights_k[None, :]).sum(axis=1) + test_weights
        out[:, j] = numerator / (total_k + test_weights)
    return out


def conditional_prediction_set(
    classifier: Any,
    data_calibration: np.ndarray,
    codes_calibration: np.ndarray,
    data_test: np.ndarray,
    *,
    covariates_calibration: np.ndarray,
    covariates_test: np.ndarray,
    alpha: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Prediction sets with conditional coverage over the covariate class.

    The MAPIE adapter (Gibbs, Cherian & Candès conditional conformal;
    ``ConditionalSplitConformalClassifier``): coverage is guaranteed not just
    per group but for every subpopulation expressible as a linear function of
    the supplied covariates — completeness metrics being the intended use, so
    "the guarantee holds at every reconstruction quality" rather than on
    average. Needs the optional ``conditional`` extra (``mapie`` + ``cvxpy``).

    Two costs against the in-library weighted path: ``alpha`` must be chosen
    here (an LP is solved per test cell per level — there is no read-time
    p-value matrix), and the runtime is minutes rather than milliseconds at
    tens of thousands of cells. Returns a boolean ``(n_test, n_classes)`` set
    matrix over the classifier's classes.
    """
    try:
        from mapie.conditional_conformal_prediction import (
            ConditionalSplitConformalClassifier,
        )
    except ImportError as error:
        raise ImportError(
            "conditional coverage needs the optional 'conditional' extra: "
            "pip install 'cellpax[conditional]'"
        ) from error

    n_features = data_calibration.shape[1]

    class _StripCovariates:
        """Present the classifier with features only; MAPIE sees the stack."""

        classes_ = classifier.classes_

        def predict_proba(self, stacked: Any) -> np.ndarray:
            return classifier.predict_proba(np.asarray(stacked)[:, :n_features])

        def predict(self, stacked: Any) -> np.ndarray:
            return classifier.predict(np.asarray(stacked)[:, :n_features])

        def __sklearn_is_fitted__(self) -> bool:
            return True

    def _feature_map(stacked: Any) -> np.ndarray:
        covariates = np.asarray(stacked)[:, n_features:]
        return np.hstack([np.ones((covariates.shape[0], 1)), covariates])

    stacked_calibration = np.hstack(
        [data_calibration, np.atleast_2d(covariates_calibration.T).T]
    )
    stacked_test = np.hstack([data_test, np.atleast_2d(covariates_test.T).T])
    conformal = ConditionalSplitConformalClassifier(
        feature_map=_feature_map,
        estimator=_StripCovariates(),
        confidence_level=1 - alpha,
        prefit=True,
        seed=seed,
    )
    conformal.conformalize(stacked_calibration, codes_calibration)
    # predict_set returns (point_predictions, sets); the sets are what we want
    _, sets = conformal.predict_set(stacked_test)
    sets = np.asarray(sets)
    if sets.ndim == 3:  # (n, n_classes, n_levels) with a single level
        sets = sets[:, :, 0]
    return sets.astype(bool)


class Assignment:
    """Per-cell, per-label conformal p-values plus calibrated probabilities.

    Built by ``FeatureTable.assign``. The p-value matrix is the evidence;
    sets, hard labels, and coverage claims are derived from it at whatever
    ``alpha`` the reading calls for.
    """

    def __init__(
        self,
        cell_ids: np.ndarray,
        p_values: np.ndarray,
        probabilities: np.ndarray | None,
        *,
        class_ids: np.ndarray,
        reference: Any,
        calibration_counts: dict[int, int],
        name: str = "assign",
        mask: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        cell_ids = np.asarray(cell_ids)
        p_values = np.asarray(p_values, dtype=float)
        if p_values.shape[0] != cell_ids.shape[0]:
            raise ValueError("p_values and cell_ids must have the same length")
        if p_values.shape[1] != len(class_ids):
            raise ValueError("p_values must have one column per class")
        self._cell_ids = cell_ids
        self._p_values = p_values
        self._probabilities = (
            None if probabilities is None else np.asarray(probabilities, dtype=float)
        )
        self._class_ids = np.asarray(class_ids, dtype=np.int64)
        self._reference = reference
        self._calibration_counts = dict(calibration_counts)
        self.name = name
        self.mask = mask
        self._params = dict(params) if params else None

    # -- evidence ---------------------------------------------------------------

    @property
    def cell_ids(self) -> np.ndarray:
        return self._cell_ids.copy()

    @property
    def class_ids(self) -> np.ndarray:
        """Reference cluster ids, in p-value column order."""
        return self._class_ids.copy()

    @property
    def class_names(self) -> list[str]:
        """Reference cluster names, in p-value column order."""
        by_id = dict(zip(self._reference.ids, self._reference.names))
        return [str(by_id.get(int(i), i)) for i in self._class_ids]

    @property
    def p_values(self) -> np.ndarray:
        """The ``(n_cells, n_classes)`` conformal p-value matrix. A copy."""
        return self._p_values.copy()

    @property
    def probabilities(self) -> np.ndarray | None:
        """Sigmoid-calibrated class probabilities, or ``None`` when
        calibration wasn't possible (a class too small to hold out). A copy.
        """
        return None if self._probabilities is None else self._probabilities.copy()

    @property
    def calibration_counts(self) -> dict[int, int]:
        """Calibration cells per class id — what backs each class's guarantee."""
        return dict(self._calibration_counts)

    @property
    def reference(self) -> Any:
        """The curated ``LabelSet`` the assignment was calibrated against."""
        return self._reference

    @property
    def params(self) -> dict[str, Any] | None:
        """The ``ft.assign`` call that produced this, seed included."""
        return dict(self._params) if self._params else None

    def plausibility(self) -> np.ndarray:
        """Each cell's best p-value — how ordinary it looks under its best label.

        A low value means no label makes this cell look like the calibration
        cells. A useful screen, but not out-of-distribution detection: the
        guarantee behind these numbers assumes exchangeability, which is
        precisely what a truncated or foreign cell violates. Cross-check with
        ``score_cells`` before reading low plausibility as novelty.
        """
        return self._p_values.max(axis=1)

    # -- read-time derivations ----------------------------------------------------

    def _check_alpha(self, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        # a class's (1 - alpha) calibration quantile needs at least
        # ceil(1/alpha) - 1 points to be a real number rather than the max
        needed = int(np.ceil(1.0 / alpha)) - 1
        by_id = dict(zip(self._reference.ids, self._reference.names))
        starved = [
            str(by_id.get(int(k), k))
            for k, count in self._calibration_counts.items()
            if count < needed
        ]
        if starved:
            warnings.warn(
                f"at alpha={alpha:g} the per-class guarantee needs at least "
                f"{needed} calibration cells; classes {starved} have fewer, so "
                f"their coverage claim is not backed — coarsen the labels, "
                f"raise alpha, or curate more of those cells"
            )

    def prediction_set(self, alpha: float = 0.1) -> np.ndarray:
        """Boolean ``(n_cells, n_classes)``: labels not rejected at ``alpha``.

        Under exchangeability, each cell's true label is inside its set with
        probability at least ``1 - alpha``, per class (Mondrian). Row sums are
        the uncertainty readout; see ``set_sizes``.
        """
        self._check_alpha(alpha)
        return self._p_values > alpha

    def set_sizes(self, alpha: float = 0.1) -> np.ndarray:
        """Per-cell prediction-set size at ``alpha`` — the uncertainty readout."""
        return self.prediction_set(alpha).sum(axis=1)

    def to_labelset(self, alpha: float = 0.1, *, name: str | None = None) -> Any:
        """The conformal-honest hard labelling at ``alpha``.

        Singleton sets keep their label (names and colors from the
        reference); multi-label and empty sets stay unassigned — ambiguity
        and no-fit are both reasons to abstain, not to guess. The trade is
        explicit: a lower ``alpha`` covers more but abstains more.
        """
        sets = self.prediction_set(alpha)
        sizes = sets.sum(axis=1)
        codes = np.full(self._cell_ids.shape[0], -1, dtype=np.int64)
        singleton = sizes == 1
        codes[singleton] = self._class_ids[sets[singleton].argmax(axis=1)]
        return self._reference.with_codes(
            self._cell_ids,
            codes,
            name=name or f"{self.name}@{alpha:g}",
            mask=self.mask,
        )

    def frame(
        self, *, alpha: float | None = None, id_column: str = "cell_id"
    ) -> pl.DataFrame:
        """The evidence as a tidy frame: p-values, probabilities, plausibility.

        With ``alpha``, adds ``set_size`` and ``assigned`` (the singleton's
        name, null otherwise) — the read-time verdict alongside the evidence.
        """
        names = self.class_names
        data: dict[str, Any] = {id_column: self._cell_ids}
        for j, class_name in enumerate(names):
            data[f"p_{class_name}"] = self._p_values[:, j]
        if self._probabilities is not None:
            for j, class_name in enumerate(names):
                data[f"prob_{class_name}"] = self._probabilities[:, j]
        data["plausibility"] = self.plausibility()
        frame = pl.DataFrame(data)
        if alpha is not None:
            sets = self.prediction_set(alpha)
            sizes = sets.sum(axis=1)
            singleton = sizes == 1
            assigned = [
                names[int(sets[i].argmax())] if singleton[i] else None
                for i in range(sets.shape[0])
            ]
            frame = frame.with_columns(
                pl.Series("set_size", sizes.astype(np.int64)),
                pl.Series("assigned", assigned, dtype=pl.Utf8),
            )
        return frame

    def coverage(self, alpha: float = 0.1) -> pl.DataFrame:
        """Empirical per-class coverage on the reference cells, at ``alpha``.

        The self-check: among cells whose curated label is known, how often
        does the prediction set contain it? Read against ``1 - alpha`` and
        ``calibration_counts`` — a small class's empirical coverage is noisy
        in exactly the way its guarantee is weak.
        """
        truth = self._reference.codes_for(self._cell_ids)
        sets = self.prediction_set(alpha)
        column_of = {int(k): j for j, k in enumerate(self._class_ids)}
        rows = []
        by_id = dict(zip(self._reference.ids, self._reference.names))
        for class_id, j in column_of.items():
            members = truth == class_id
            if not members.any():
                continue
            rows.append(
                {
                    "class": str(by_id.get(class_id, class_id)),
                    "n_reference": int(members.sum()),
                    "n_calibration": self._calibration_counts.get(class_id, 0),
                    "coverage": float(sets[members, j].mean()),
                }
            )
        return pl.DataFrame(rows).sort("coverage")

    def __repr__(self) -> str:
        return (
            f"Assignment(name={self.name!r}, n_cells={self._cell_ids.shape[0]}, "
            f"n_classes={len(self._class_ids)}, mask={self.mask!r}, "
            f"calibration={sorted(self._calibration_counts.values())})"
        )
