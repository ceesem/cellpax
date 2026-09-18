"""Label propagation from a curated subset into a larger population.

The workhorse of the old dfc pipeline: cluster a high-quality "core" subset, then
carry those labels out to every cell that looks like them. Two methods, both
array-level primitives that treat ``-1`` as unlabeled:

``propagate_knn``
    A vote of each cell's ``k`` nearest labeled cells. Cheap, and the standard
    baseline (the single-cell field's label transfer, e.g. scanpy's ``ingest``, is
    this in PCA space). Every cell gets a label whether or not it has any business
    having one: ``k`` votes are always cast, so a cell far from every labeled cell
    still returns a unanimous verdict.

``propagate_spread``
    Label diffusion over a mutual-nearest-neighbor graph. Evidence scales with how
    much real signal is nearby rather than being fixed at ``k``, and a cell with no
    mutual path to any labeled cell stays unassigned — which is the right answer
    for a cell that isn't like anything in the reference. Mutuality does the
    rejecting, so no distance threshold has to be calibrated against a population
    whose density genuinely varies.

Both are self-excluding: a cell never votes for itself, so ``self_agreement`` is an
honest recovery rate rather than an inflated one (dfc's ``predict_from_neighborhood``
fit and predicted on the same rows, making every cell its own nearest neighbor).

Both also return a ``confidence`` per row — the winning label's share of the support
that reached it, in ``[0, 1]`` — which ``min_confidence`` cuts on. It is a plurality
margin, not a probability, and its scale is not portable between runs: the winner
among ``c`` labels competing in a neighborhood cannot fall below ``1/c``, and under
``weights="uniform"`` the shares are quantized to multiples of ``1/n_neighbors``, so
cuts falling between grid points do nothing at all. Pick the value by calibrating
against the reference rows, whose labels are known, rather than a priori.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from cellpax.labels import LabelSet

_UNASSIGNED = -1
_TINY = np.finfo(float).tiny

Weights = Literal["uniform", "distance"]


@dataclass(frozen=True)
class Recovery:
    """How well a method re-derives reference labels that were withheld from it.

    ``agreement`` and ``abstained`` are fractions of the reference cells: recovered
    correctly, and left with no evidence to decide on at all. They're reported apart
    because the remedies differ — misassignment says the clusters aren't separable in
    this feature space, abstention says the cells aren't connected to the reference.

    ``estimator`` names how it was measured, since the numbers aren't interchangeable:
    ``"leave-one-out"`` (exact, the kNN vote's, free) or ``"{k}-fold"`` (withholding a
    stratified share at a time, which also thins the reference and so reads slightly
    pessimistic). Compare across feature sets with one estimator, not across estimators.

    Attributes
    ----------
    agreement : float
        Fraction of reference cells recovered correctly.
    abstained : float
        Fraction left without evidence to decide.
    estimator : str
        Recovery protocol, such as ``"leave-one-out"`` or ``"5-fold"``.
    """

    agreement: float
    abstained: float
    estimator: str

    @property
    def misassigned(self) -> float:
        """Reference cells that had evidence and got the wrong label from it."""
        return max(0.0, 1.0 - self.agreement - self.abstained)


def _check(features: np.ndarray, codes: np.ndarray, preserve_labeled: bool):
    features = np.asarray(features, dtype=float)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if features.shape[0] != codes.shape[0]:
        raise ValueError("features and codes must have the same number of rows")
    below = np.unique(codes[codes < _UNASSIGNED])
    if below.size:
        raise ValueError(
            f"codes below -1 have no meaning here ({below.tolist()}); -1 is the "
            "only unassigned sentinel"
        )
    fit_rows = np.flatnonzero(codes != _UNASSIGNED)
    if fit_rows.size < 2:
        raise ValueError("propagation needs at least two labeled cells to vote")
    if fit_rows.size == codes.shape[0] and preserve_labeled:
        raise ValueError(
            "every cell is already labeled; pass preserve_labeled=False to smooth"
        )
    return features, codes, fit_rows


def _one_hot(codes: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    labels = np.zeros((codes.shape[0], int(codes.max()) + 1), dtype=float)
    labels[fit_rows, codes[fit_rows]] = 1.0
    return labels


def _decide(
    support: np.ndarray,
    codes: np.ndarray,
    fit_rows: np.ndarray,
    *,
    preserve_labeled: bool,
    min_confidence: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-class support into (codes, confidence), applying the shared rules.

    ``support`` is whatever evidence the method gathered per class, excluding a
    cell's own label. A row with no support at all is unassigned — for the vote
    that can't happen, for diffusion it's the unreachable cells.
    """
    total = support.sum(axis=1)
    reachable = total > 0
    out = np.where(reachable, support.argmax(axis=1), _UNASSIGNED)
    confidence = np.where(
        reachable, support.max(axis=1) / np.maximum(total, _TINY), 0.0
    )
    if preserve_labeled:
        out[fit_rows] = codes[fit_rows]
        own = support[fit_rows, codes[fit_rows]]
        confidence[fit_rows] = own / np.maximum(total[fit_rows], _TINY)
    if min_confidence is not None:
        weak = confidence < min_confidence
        if preserve_labeled:
            weak[fit_rows] = False  # a curated label is not the vote's to discard
        out[weak] = _UNASSIGNED
    return out, confidence


# -- k nearest neighbor vote -----------------------------------------------------


def propagate_knn(
    features: np.ndarray,
    codes: np.ndarray,
    *,
    n_neighbors: int = 30,
    weights: Weights = "uniform",
    preserve_labeled: bool = True,
    min_confidence: float | None = None,
    agreement_folds: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Recovery]:
    """Spread cluster codes by a vote of each row's ``n_neighbors`` labeled neighbors.

    ``features`` and ``codes`` are row-aligned over *both* populations: rows coded
    ``-1`` are the unlabeled ones to fill in, the rest are the reference voted from.
    A row never counts itself, so a reference row's outcome is leave-one-out.

    ``weights="distance"`` weights each vote by inverse distance, so a nearby
    labeled cell counts for more than one at the edge of the neighborhood — worth
    reaching for when clusters differ a lot in size, since uniform votes favor the
    larger one near a boundary.

    Returns ``(codes, confidence, recovery)``: the propagated code per row, the share
    of (weighted) neighbor support behind the code it ended up with, and a
    :class:`Recovery`. Recovery is exact leave-one-out by default — dropping the
    self-match removes a row's influence completely, so it costs nothing. Pass
    ``agreement_folds=5`` to measure it fold-wise instead, which is what makes it
    comparable with ``propagate_spread``'s.

    ``min_confidence`` unassigns rows whose winning share falls below it, the
    reference exempt under ``preserve_labeled``. Because ``k`` votes are always cast
    it catches rows caught *between* clusters, never rows far from all of them —
    those come back unanimous at ``1.0`` however distant they are.

    Parameters
    ----------
    features : numpy.ndarray
        Row-aligned ``(n_cells, n_features)`` coordinate matrix.
    codes : numpy.ndarray
        Integer labels per row; ``-1`` marks cells to predict.
    n_neighbors : int, default 30
        Number of labelled neighbours voting for each row.
    weights : {'uniform', 'distance'}, default 'uniform'
        Vote weighting rule.
    preserve_labeled : bool, default True
        Keep supplied reference codes in the output.
    min_confidence : float, optional
        Unassign predictions below this winning support share.
    agreement_folds : int, default 0
        Use fold-wise reference recovery when at least two; zero uses exact
        leave-one-out recovery.
    seed : int, optional
        Seed for stratified recovery folds.

    Returns
    -------
    codes : numpy.ndarray
        Propagated integer code for each row.
    confidence : numpy.ndarray
        Winning support share for each row.
    recovery : Recovery
        Held-out recovery of the supplied reference labels.
    """
    features, codes, fit_rows = _check(features, codes, preserve_labeled)
    k = int(min(n_neighbors, fit_rows.size - 1))
    support = _vote_support(features, codes, fit_rows, k, weights)
    out, confidence = _decide(
        support,
        codes,
        fit_rows,
        preserve_labeled=preserve_labeled,
        min_confidence=min_confidence,
    )
    if agreement_folds >= 2 and fit_rows.size >= agreement_folds:
        recovery = _vote_recovery(
            features,
            codes,
            fit_rows,
            folds=agreement_folds,
            n_neighbors=n_neighbors,
            weights=weights,
            seed=seed,
        )
    else:
        agreement = float((support.argmax(axis=1)[fit_rows] == codes[fit_rows]).mean())
        # a vote always finds k neighbors, so it can never abstain
        recovery = Recovery(agreement, 0.0, "leave-one-out")
    return out, confidence, recovery


def _vote_recovery(
    features: np.ndarray,
    codes: np.ndarray,
    fit_rows: np.ndarray,
    *,
    folds: int,
    n_neighbors: int,
    weights: Weights,
    seed: int | None,
) -> Recovery:
    """Fold-wise recovery for the vote, for comparison against diffusion's.

    Costs a neighbor search per fold, unlike the free leave-one-out, and reads
    slightly lower because each fold votes from a thinner reference.
    """
    recovered = 0
    for held in _stratified_folds(codes, fit_rows, folds, seed):
        if held.size == 0:
            continue
        kept = np.setdiff1d(fit_rows, held)
        k = int(min(n_neighbors, kept.size))
        support = _vote_support(features, codes, kept, k, weights)
        recovered += int((support[held].argmax(axis=1) == codes[held]).sum())
    return Recovery(recovered / fit_rows.size, 0.0, f"{folds}-fold")


def _vote_support(
    features: np.ndarray,
    codes: np.ndarray,
    fit_rows: np.ndarray,
    k: int,
    weights: Weights,
) -> np.ndarray:
    """``(n_rows, n_codes)`` weight each row's ``k`` nearest labeled neighbors give.

    The kNN index holds only the labeled rows; every row is queried against it, and
    a row that is itself in the index has that self-match dropped before the ``k``
    nearest are counted.
    """
    from sklearn.neighbors import NearestNeighbors

    index = NearestNeighbors(n_neighbors=min(k + 1, fit_rows.size))
    index.fit(features[fit_rows])
    distances, found = index.kneighbors(features)
    neighbor_rows = fit_rows[found]

    is_self = neighbor_rows == np.arange(codes.shape[0])[:, None]
    # a stable sort keeps distance order among the non-self neighbors
    keep = np.argsort(is_self, axis=1, kind="stable")[:, :k]
    chosen = np.take_along_axis(neighbor_rows, keep, axis=1)
    chosen_distance = np.take_along_axis(distances, keep, axis=1)

    if weights == "distance":
        vote = 1.0 / (chosen_distance + _distance_floor(chosen_distance))
    elif weights == "uniform":
        vote = np.ones_like(chosen_distance)
    else:
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")

    support = np.zeros((codes.shape[0], int(codes.max()) + 1), dtype=float)
    np.add.at(support, (np.arange(codes.shape[0])[:, None], codes[chosen]), vote)
    return support


def _distance_floor(distances: np.ndarray) -> float:
    """A small offset so coincident cells get a large weight, not an infinite one."""
    positive = distances[distances > 0]
    return float(np.median(positive) * 1e-3) if positive.size else 1.0


# -- diffusion over a mutual nearest neighbor graph ------------------------------


def propagate_spread(
    features: np.ndarray,
    codes: np.ndarray,
    *,
    n_neighbors: int = 30,
    mutual: bool = True,
    weights: Weights = "distance",
    preserve_labeled: bool = True,
    alpha: float = 0.8,
    min_confidence: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    agreement_folds: int = 5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Recovery | None]:
    """Diffuse cluster codes along a neighbor graph until the label field settles.

    Label spreading in the Zhou/Zhu sense: each row's label distribution is
    repeatedly replaced by the average of its neighbors', with the reference rows
    held fixed (``preserve_labeled``) or pulled back toward their own labels with
    strength ``1 - alpha`` (smoothing). Two things follow that a fixed-``k`` vote
    can't give:

    - Evidence accumulates over as much nearby signal as exists, rather than
      exactly ``n_neighbors`` votes, so a cell in a dense well-labeled region draws
      on all of it.
    - A row with no path to any labeled row receives nothing and stays ``-1``. With
      ``mutual`` (the default), a cell that names labeled neighbors which don't name
      it back is pruned out of the graph, so cells unlike anything in the reference
      abstain instead of being assigned by their nearest — and the rejection follows
      local density rather than a global distance cutoff.

    Returns ``(codes, confidence, recovery)``. Confidence is the share of arriving
    label mass behind the winning code — continuous here rather than quantized by
    ``n_neighbors``, since the mass accumulated is not a fixed count of votes, so
    ``min_confidence`` cuts on a smooth distribution. :class:`Recovery` is measured over
    ``agreement_folds`` stratified folds — reference labels withheld a fold at a time
    and re-derived, reporting misassignment and abstention apart (pass 0 to skip,
    in which case the recovery is ``None``).
    There's no free leave-one-out here: a clamped row's label reaches its unlabeled
    neighbors and comes back on the next hop, so removing a row's influence means
    removing its label. The graph is built once and reused across folds, so this
    costs extra diffusions, not extra neighbor searches.

    Parameters
    ----------
    features : numpy.ndarray
        Row-aligned ``(n_cells, n_features)`` coordinate matrix.
    codes : numpy.ndarray
        Integer labels per row; ``-1`` marks cells to predict.
    n_neighbors : int, default 30
        Neighbourhood size of the diffusion graph.
    mutual : bool, default True
        Retain only mutually selected neighbour edges.
    weights : {'uniform', 'distance'}, default 'distance'
        Edge-weighting rule.
    preserve_labeled : bool, default True
        Clamp supplied reference codes during diffusion.
    alpha : float, default 0.8
        Weight given to propagated rather than initial label mass.
    min_confidence : float, optional
        Unassign predictions below this winning support share.
    max_iter : int, default 100
        Maximum diffusion iterations.
    tol : float, default 1e-6
        Convergence tolerance on the label field.
    agreement_folds : int, default 5
        Stratified folds used to assess reference recovery; zero disables it.
    seed : int, optional
        Seed for recovery folds.

    Returns
    -------
    codes : numpy.ndarray
        Propagated integer code for each row.
    confidence : numpy.ndarray
        Winning support share for each row.
    recovery : Recovery, optional
        Fold-wise recovery, or ``None`` when disabled.
    """
    features, codes, fit_rows = _check(features, codes, preserve_labeled)
    transition = _transition_matrix(features, n_neighbors, mutual, weights)
    labels = _one_hot(codes, fit_rows)

    field = _diffuse(
        transition,
        labels,
        fit_rows,
        preserve_labeled=preserve_labeled,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
    # one more hop, so every row is judged on what arrives from its neighbors: for a
    # clamped reference row that's its neighborhood instead of the label it was
    # handed, and for the rest it's the settled field (no self-loops to discount)
    support = transition @ field

    out, confidence = _decide(
        support,
        codes,
        fit_rows,
        preserve_labeled=preserve_labeled,
        min_confidence=min_confidence,
    )
    recovery = (
        _spread_recovery(
            transition,
            codes,
            fit_rows,
            folds=agreement_folds,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        if agreement_folds
        else None
    )
    return out, confidence, recovery


def _transition_matrix(
    features: np.ndarray, n_neighbors: int, mutual: bool, weights: Weights
) -> Any:
    """Row-normalized affinity over a (mutual) kNN graph; empty rows stay empty."""
    from scipy.sparse import coo_matrix, diags
    from sklearn.neighbors import kneighbors_graph

    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
    k = int(min(n_neighbors, features.shape[0] - 1))
    # the adjacency pattern comes from connectivity, not distance: a coincident
    # neighbor is an explicit zero under mode="distance", and sparse algebra
    # silently drops explicit zeros, which would disconnect exact duplicates
    directed = kneighbors_graph(features, k, mode="connectivity", include_self=False)

    # an edge survives only if both cells name each other (mutual) or either does
    keep = (
        directed.multiply(directed.T) if mutual else directed.maximum(directed.T)
    ).tocoo()

    if weights == "distance":
        gaps = np.linalg.norm(features[keep.row] - features[keep.col], axis=1)
        # the floor keeps a zero-distance edge at a large weight, not an infinite one
        affinity = coo_matrix(
            (1.0 / (gaps + _distance_floor(gaps)), (keep.row, keep.col)),
            shape=keep.shape,
        ).tocsr()
    else:
        affinity = keep.tocsr().astype(float)

    row_total = np.asarray(affinity.sum(axis=1)).ravel()
    inverse = np.where(row_total > 0, 1.0 / np.maximum(row_total, _TINY), 0.0)
    return diags(inverse) @ affinity


def _diffuse(
    transition: Any,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    *,
    preserve_labeled: bool,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Iterate the label field to convergence."""
    field = labels.copy()
    for _ in range(max_iter):
        updated = transition @ field
        if preserve_labeled:
            updated[fit_rows] = labels[fit_rows]
        else:
            updated = alpha * updated + (1.0 - alpha) * labels
        shift = float(np.abs(updated - field).max())
        field = updated
        if shift < tol:
            break
    return field


def _stratified_folds(
    codes: np.ndarray, fit_rows: np.ndarray, folds: int, seed: int | None
) -> list[np.ndarray]:
    """Split the reference rows into folds with each cluster spread across all of them.

    Unstratified folds let a small tight cluster land mostly in one fold, lose its
    labels together and fail together — correlated failures that make a perfectly
    propagatable cluster look hopeless. Dealing each cluster's shuffled members
    round-robin (continuing where the previous cluster left off, so singletons don't
    all pile into fold 0) keeps every fold's composition close to the whole.
    """
    rng = np.random.default_rng(seed)
    assignment = np.empty(fit_rows.size, dtype=np.int64)
    reference_codes = codes[fit_rows]
    start = 0
    for code in np.unique(reference_codes):
        members = rng.permutation(np.flatnonzero(reference_codes == code))
        assignment[members] = (np.arange(members.size) + start) % folds
        start = int((start + members.size) % folds)
    return [fit_rows[assignment == fold] for fold in range(folds)]


def _spread_recovery(
    transition: Any,
    codes: np.ndarray,
    fit_rows: np.ndarray,
    *,
    folds: int,
    alpha: float,
    max_iter: int,
    tol: float,
    seed: int | None,
) -> "Recovery":
    """Withhold each fold of the reference labels and see if diffusion re-derives them.

    The graph is built once and reused, so this costs a few extra diffusions rather
    than extra neighbor searches. Diffusion has no free leave-one-out equivalent: a
    clamped row's label reaches its unlabeled neighbors and returns on the next hop,
    so the only way to remove a row's own influence is to remove its label.
    """
    if folds < 2 or fit_rows.size < folds:
        return Recovery(float("nan"), float("nan"), "none")
    recovered = unreachable = 0
    for held in _stratified_folds(codes, fit_rows, folds, seed):
        if held.size == 0:
            continue
        kept = np.setdiff1d(fit_rows, held)
        labels = _one_hot(codes, kept)
        field = _diffuse(
            transition,
            labels,
            kept,
            preserve_labeled=True,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        arriving = field[held]
        reached = arriving.sum(axis=1) > 0
        recovered += int(((arriving.argmax(axis=1) == codes[held]) & reached).sum())
        unreachable += int((~reached).sum())
    total = fit_rows.size
    return Recovery(recovered / total, unreachable / total, f"{folds}-fold")


# -- calibrating min_confidence ----------------------------------------------------


def confidence_curve(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
) -> pl.DataFrame:
    """Coverage against error at every ``min_confidence`` cut that changes anything.

    The calibration read for choosing ``min_confidence``: run a *probe* propagation
    with ``preserve_labeled=False`` first — that relabels the reference rows from
    their neighborhoods too, so their confidence is a winner-share comparable with
    everyone else's — then pass the reference codes as ``truth`` (``-1`` where the
    answer isn't known), the probe's propagated codes as ``predicted``, and the
    probe's per-row ``confidence``, all row-aligned.

    Returns one row per distinct confidence value among the known-truth cells — the
    only cuts at which the kept set actually changes, since under
    ``weights="uniform"`` confidence is quantized and a cut between grid points does
    nothing. Columns: ``cut``, ``n_kept`` (known-truth cells at or above it),
    ``kept_fraction`` (of all known-truth cells) and ``error_rate`` (wrong
    predictions among the kept). Read it as a purity-versus-coverage curve and pick
    the cut whose trade you can live with. Cells with no known truth never enter
    it — and remember the cut measures ambiguity, not novelty: a cell far outside
    the reference is unanimous at 1.0, which is what ``method="spread"`` is for.

    Parameters
    ----------
    truth : numpy.ndarray
        Known integer codes, with ``-1`` for unknown rows.
    predicted : numpy.ndarray
        Propagated code for each row.
    confidence : numpy.ndarray
        Winning support share for each row.

    Returns
    -------
    polars.DataFrame
        One row per effective cut with kept count, coverage, and error rate.

    Raises
    ------
    ValueError
        If inputs have different lengths or no row has known truth.
    """
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    if not (truth.shape[0] == predicted.shape[0] == confidence.shape[0]):
        raise ValueError("truth, predicted and confidence must have the same length")
    known = truth != _UNASSIGNED
    n_known = int(known.sum())
    if n_known == 0:
        raise ValueError("no cells have a known truth code to calibrate against")
    wrong = predicted != truth
    rows = []
    for cut in np.unique(confidence[known]):
        kept = known & (confidence >= cut)
        n_kept = int(kept.sum())
        rows.append(
            {
                "cut": float(cut),
                "n_kept": n_kept,
                "kept_fraction": n_kept / n_known,
                "error_rate": float(wrong[kept].mean()),
            }
        )
    return pl.DataFrame(rows)


# -- result ----------------------------------------------------------------------


class Propagation:
    """A propagated ``LabelSet`` and the confidence behind each assignment.

    Returned by :meth:`~cellpax.featuretable.FeatureTable.propagate_labels`. The
    labels aren't attached to the table — ``ft.attach(result.labels)`` when you're
    happy with them, and ``result.frame()`` if you want the confidence alongside.

    Parameters
    ----------
    labels : LabelSet
        Propagated labels over the target population.
    reference : LabelSet
        Curated labels propagated from.
    confidence : numpy.ndarray
        Winning support share for each target cell.
    recovery : Recovery, optional
        Held-out reference recovery diagnostics.
    method : str
        Propagation method name.
    n_neighbors : int
        Neighbourhood size used by the method.
    space : str
        Representation in which neighbours were found.
    params : dict, optional
        Provenance for the originating call.
    rungs : numpy.ndarray, optional
        Ladder rung that labelled each cell.
    rung_names : sequence of str, optional
        Ladder collection names, richest first.
    rung_recovery : dict, optional
        Recovery diagnostics keyed by rung name.
    """

    def __init__(
        self,
        labels: "LabelSet",
        reference: "LabelSet",
        confidence: np.ndarray,
        recovery: Recovery | None,
        *,
        method: str,
        n_neighbors: int,
        space: str,
        params: dict[str, Any] | None = None,
        rungs: np.ndarray | None = None,
        rung_names: Sequence[str] | None = None,
        rung_recovery: dict[str, Recovery | None] | None = None,
    ) -> None:
        self._labels = labels
        self._reference = reference
        self._confidence = np.asarray(confidence, dtype=float)
        self._recovery = recovery
        self._method = method
        self._n_neighbors = int(n_neighbors)
        self._space = space
        self._params = params
        self._rungs = None if rungs is None else np.asarray(rungs, dtype=np.int64)
        self._rung_names = None if rung_names is None else list(rung_names)
        self._rung_recovery = rung_recovery

    @property
    def labels(self) -> "LabelSet":
        """The propagated ``LabelSet``, carrying the reference's names and colors."""
        return self._labels

    @property
    def reference(self) -> "LabelSet":
        """The ``LabelSet`` propagated from."""
        return self._reference

    @property
    def confidence(self) -> np.ndarray:
        """Support share behind each cell's assigned label, in ``cell_ids`` order.

        Not one population, so don't quantile it whole. For a propagated cell this is
        the *winning* label's share, bounded below by ``1/c`` for the ``c`` labels
        competing locally. For a reference cell held fixed by ``preserve_labeled`` it
        is the share behind the label it was *given*, which its neighborhood need not
        favor at all — a curated cell sitting deep inside another cluster reports
        ``0.0`` and keeps its label regardless of ``min_confidence``. Low values there
        flag the core for review; they don't describe a weak assignment.
        """
        return self._confidence.copy()

    @property
    def method(self) -> str:
        """Which propagation was run — ``"vote"`` or ``"spread"``."""
        return self._method

    @property
    def space(self) -> str:
        """The representation the neighbours were found in, e.g. ``'pca(0.95)'``.

        A ladder propagation reports ``'ladder(<rungs>)'`` — each rung was its
        own space, so no single label describes the geometry.
        """
        return self._space

    @property
    def recovery(self) -> Recovery | None:
        """The full recovery read: agreement, abstention, and which estimator ran.

        ``None`` when it was skipped (``agreement_folds=0``).
        """
        return self._recovery

    @property
    def params(self) -> dict[str, Any] | None:
        """The call that produced this propagation, as passed by the caller.

        ``FeatureTable.propagate_labels`` records its arguments here for
        provenance; a Propagation built by hand carries ``None``.
        """
        return self._params

    @property
    def rungs(self) -> np.ndarray | None:
        """Which ladder rung labeled each cell, in ``cell_ids`` order.

        ``None`` unless the propagation ran with ``ladder=``. Values index
        ``rung_names``; ``-1`` marks a cell no usable rung's features were valid
        for (its label is unassigned — the honest answer, not a fallback vote).
        Confidence is comparable *within* a rung, not across rungs: each rung is
        its own feature space with its own local label competition.
        """
        return None if self._rungs is None else self._rungs.copy()

    @property
    def rung_names(self) -> list[str] | None:
        """The ladder's collection names, richest first, indexed by ``rungs``."""
        return None if self._rung_names is None else list(self._rung_names)

    @property
    def rung_recovery(self) -> dict[str, Recovery | None] | None:
        """Per-rung :class:`Recovery`, keyed by rung name.

        Each rung's recovery is computed over the reference cells that voted in
        it, so comparing rungs answers the question the ladder poses: how much
        does each narrower feature set still carry? ``None`` for a rung whose
        recovery was skipped or that had too few valid reference cells to run.
        """
        return None if self._rung_recovery is None else dict(self._rung_recovery)

    def self_agreement(self) -> float:
        """How often the reference's own labels are recovered without being given.

        Leave-one-out for ``"vote"`` (a cell never votes for itself), fold-wise
        recovery for ``"spread"``. Near 1.0 means the labels are locally coherent in
        this feature space and propagation is interpolating; a low value means it is
        guessing, which usually points at the feature set rather than at
        ``n_neighbors``. Comparing it between feature sets is the cheap way to see
        what a restricted set (say, only the features valid for every cell) can
        still carry.

        This is only the fraction recovered — see ``recovery`` for how the rest
        failed, since a misassigned cell and an unreachable one call for different
        fixes, and for which estimator produced the number. ``nan`` when recovery
        was skipped (``agreement_folds=0``).

        Returns
        -------
        float
            Reference recovery fraction, or ``nan`` when not assessed.
        """
        return float("nan") if self._recovery is None else self._recovery.agreement

    def n_reference(self) -> int:
        """Count cells supplied by the reference.

        Returns
        -------
        int
            Number of reference cells.
        """
        return int(self._reference.assigned.sum())

    def frame(self, *, id_column: str = "cell_id") -> pl.DataFrame:
        """``to_frame`` of the propagated labels plus a ``{name}_confidence`` column.

        A ladder propagation adds a ``{name}_rung`` column naming the collection
        that labeled each cell (null where no rung was valid).

        Parameters
        ----------
        id_column : str, default 'cell_id'
            Name of the identifier column.

        Returns
        -------
        polars.DataFrame
            Propagated names and ids, confidence, and optional ladder rung.
        """
        frame = self._labels.to_frame(id_column=id_column).with_columns(
            pl.Series(f"{self._labels.name}_confidence", self._confidence)
        )
        if self._rungs is not None and self._rung_names is not None:
            names = [self._rung_names[r] if r >= 0 else None for r in self._rungs]
            frame = frame.with_columns(
                pl.Series(f"{self._labels.name}_rung", names, dtype=pl.Utf8)
            )
        return frame

    def __repr__(self) -> str:
        if self._recovery is None:
            agreement = "self_agreement=skipped"
        else:
            agreement = (
                f"self_agreement={self._recovery.agreement:.3f} "
                f"({self._recovery.estimator})"
            )
        return (
            f"Propagation(method={self._method!r}, labels={self._labels.name!r}, "
            f"n_cells={len(self._labels)}, reference={self.n_reference()}, "
            f"unassigned={self._labels.n_unassigned}, "
            f"{agreement}, "
            f"n_neighbors={self._n_neighbors}, space={self._space!r})"
        )
