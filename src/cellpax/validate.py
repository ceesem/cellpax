"""Scoring a representation, so the choice between them is not made by eye.

Whitening strength, graph weighting, neighbourhood size and clipping rule are all
parameters with no principled default, and a UMAP is the worst available way to choose
between them: it is a lossy 2-D projection, it is usually built in a different space from
the one being compared, and it rewards whichever setting produces visually tidy blobs
rather than reproducible groups. Two criteria here, plus one tripwire:

:func:`subsample_stability` — **the primary one.**
    Cluster repeated subsamples under a *frozen* representation and measure how much the
    labels agree with the full-data labels. Needs no ground truth, and measures the
    property actually wanted: that the structure is a property of the population rather
    than of the particular cells in hand.

:func:`loo_knn_recovery` / :func:`graph_knn_recovery` — **the external cross-check.**
    Leave-one-out kNN recovery of labels the clustering never saw. Any population-wide
    external call works; note that agreeing with another classifier's output rewards
    reproducing its decision boundary, errors included, so this ranks representations
    rather than certifying them.

:func:`label_purity` — **the tripwire.**
    Per-cluster composition under an audit label. A cluster straddling a boundary the
    clustering had no access to is a red flag, not a tuning signal. Note it can only fire
    where such straddling is *possible*: cluster a cohort that was pre-split on the audit
    label and purity is 1.0 by construction and says nothing.

:func:`cross_dataset_classification` — **the transfer check.**
    Train on one dataset's labels, predict another's. For datasets joined into one space
    (:func:`~cellpax.datasets.join_datasets`), where the errors land says which types the
    two still disagree about.

:func:`paired_recovery` turns any set of scores into differences with a paired test.
Absolute accuracies are rarely comparable across datasets — and are meaningless when the
labelled cells were drawn under a stratified design — while the paired difference between
two representations on the same cells is exactly what the question needs.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl

from cellpax.clustering import (
    Clustering,
    SimilarityMatrix,
    fauxnograph_coclustering,
)

_logger = logging.getLogger("cellpax.validate")
_UNASSIGNED = -1


# --------------------------------------------------------------------------- #
# 1. subsample reproducibility
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stability:
    """How reproducible a representation's clustering is under resampling.

    ``draw_ari`` is one adjusted Rand index per draw, comparing that draw's labels with
    the full-data labels restricted to the drawn cells. ``cell_stability`` is per cell:
    the average fraction of its full-data cluster-mates that stayed with it, over the
    draws it appeared in.

    ``assigned_fractions`` is one number per draw: of the drawn cells assigned in the
    full-data labels, the share the draw assigned at all. It exists because the ARI is
    computed over cells assigned in *both* labellings, so a draw where
    ``min_cluster_size`` wiped out an entire cluster can still score a perfect ARI on
    what remains — ``min_assigned`` is where that loss is visible. A cell assigned in
    the reference but unassigned in the draw also counts as not retained in
    ``cell_stability``.

    Absolute ARI is not interpretable on its own — it depends on the inner ensemble size,
    the cut, and how separable the data happens to be. The comparison between two
    representations scored with identical settings is what carries meaning, which is why
    :func:`subsample_stability` fixes the ensemble rather than letting it follow the
    parameter under test.

    Attributes
    ----------
    draw_ari : numpy.ndarray
        Adjusted Rand index for each subsample draw.
    cell_stability : numpy.ndarray
        Per-cell mean retention with its full-data cluster-mates.
    full_labels : numpy.ndarray
        Reference labels from clustering the complete dataset.
    assigned_fractions : numpy.ndarray
        Assigned fraction for each draw.
    n_draws : int
        Number of subsamples.
    fraction : float
        Fraction of cells included per draw.
    settings : dict
        Clustering settings held fixed across draws.
    """

    draw_ari: np.ndarray
    cell_stability: np.ndarray
    full_labels: np.ndarray
    assigned_fractions: np.ndarray
    n_draws: int
    fraction: float
    settings: dict[str, Any]

    @property
    def mean_ari(self) -> float:
        """Mean adjusted Rand index across draws."""
        return float(np.mean(self.draw_ari))

    @property
    def median_ari(self) -> float:
        """Median adjusted Rand index across draws."""
        return float(np.median(self.draw_ari))

    @property
    def min_ari(self) -> float:
        """Worst adjusted Rand index across draws."""
        return float(np.min(self.draw_ari))

    @property
    def mean_assigned(self) -> float:
        """Mean assigned-cell fraction across draws."""
        return float(np.mean(self.assigned_fractions))

    @property
    def min_assigned(self) -> float:
        """The worst draw's assigned fraction — a wiped-out cluster shows up here."""
        return float(np.min(self.assigned_fractions))

    def summary(self) -> pl.DataFrame:
        """Summarize headline stability values and settings.

        Returns
        -------
        polars.DataFrame
            One-row summary.
        """
        return pl.DataFrame(
            {
                "n_draws": [self.n_draws],
                "fraction": [self.fraction],
                "mean_ari": [self.mean_ari],
                "median_ari": [self.median_ari],
                "min_ari": [self.min_ari],
                "mean_assigned": [self.mean_assigned],
                "min_assigned": [self.min_assigned],
                "n_clusters_full": [
                    int(np.unique(self.full_labels[self.full_labels >= 0]).size)
                ],
                **{key: [value] for key, value in self.settings.items()},
            }
        )

    def __repr__(self) -> str:
        return (
            f"Stability(n_draws={self.n_draws}, fraction={self.fraction:g}, "
            f"mean_ari={self.mean_ari:.3f}, min_ari={self.min_ari:.3f})"
        )


def subsample_stability(
    data: np.ndarray,
    *,
    distance_threshold: float,
    n_draws: int = 10,
    fraction: float = 0.8,
    graph_type: str | list[str] = "knn",
    n_neighbors: int | list[int] = 30,
    resolution: float | list[float] = 1.0,
    n_times: int = 2,
    min_cluster_size: int = 1,
    method: str = "average",
    prune: float = 0.0,
    seed: int | None = 0,
    n_jobs: int = -1,
) -> Stability:
    """Cluster repeated subsamples and measure agreement with the full-data labels.

    Parameters
    ----------
    data : numpy.ndarray
        **Already-transformed coordinates** — the output of
        ``FittedSpace.transform_scaled`` or ``ft.features_pca(...)``. Taking coordinates
        rather than a table is deliberate and load-bearing: the representation must be fit
        outside this function and merely *indexed* per draw. Refitting a scaler or a PCA
        inside a draw would measure the stability of the preprocessing confounded with the
        stability of the clustering, and those answer different questions.
    distance_threshold : float
        Where to cut each draw's consensus, and the full-data one.
    n_draws : int, default 10
        How many subsamples. More reduces the noise on ``mean_ari``; cost is linear.
    fraction : float, default 0.8
        Share of cells per draw, in ``(0, 1)``. Lower is a harsher test.
    graph_type : str or list of str, default 'knn'
        Edge weighting(s) for the inner ensemble.
    n_neighbors : int or list of int, default 30
        Neighbourhood size(s) for the inner ensemble.
    resolution : float or list of float, default 1.0
        Leiden resolution(s) for the inner ensemble.
    n_times : int, default 2
        Repeats per setting inside each draw. Keep this small — the point is a *relative*
        number, and the total cost is ``n_draws`` times the ensemble.
    min_cluster_size : int, default 1
        Clusters smaller than this become ``-1``.
    method : {'average', 'single', 'complete'}, default 'average'
        Linkage over each consensus.
    prune : float, default 0.0
        Jaccard floor, for ``graph_type="snn_jaccard"``.
    seed : int, optional, default 0
        Seeds both the draws and the inner runs.
    n_jobs : int, default -1
        Parallel workers.

    Returns
    -------
    Stability
        Per-draw ARI, a per-cell retention score, per-draw assigned fractions, and
        the full-data labels. The ARI is computed over cells assigned in both
        labellings, so read ``min_assigned`` alongside it — a draw that discarded a
        whole cluster is invisible to the ARI and plain in the assigned fraction.

    Notes
    -----
    Every draw runs the same ensemble settings, and **those settings must be held fixed
    across the representations being compared** — absolute ARI depends on them, so only the
    paired comparison carries meaning.

    Interpretation: a representation whose groups survive dropping a fifth of the cells is
    describing the population; one whose groups rearrange was describing these cells. On
    well-separated data expect ARI near 1.0, and on a genuine continuum expect it well
    below — which is the property that makes this a usable criterion rather than a
    formality, since a continuum *should* score badly.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    draw_size = int(round(fraction * n))
    if draw_size < 3:
        raise ValueError(
            f"fraction={fraction} of {n} cells leaves {draw_size} per draw, too few "
            "to cluster"
        )

    settings = {
        "graph_type": ",".join(graph_type)
        if isinstance(graph_type, list)
        else graph_type,
        "n_neighbors": str(n_neighbors),
        "n_times": n_times,
        "distance_threshold": distance_threshold,
        "min_cluster_size": min_cluster_size,
    }
    _logger.info(
        "subsample_stability: %d draws at fraction=%g over %d cells, inner ensemble "
        "%s x n_times=%d (held fixed so representations stay comparable)",
        n_draws,
        fraction,
        n,
        settings["graph_type"],
        n_times,
    )

    def _labels(rows: np.ndarray | None) -> np.ndarray:
        subset = data if rows is None else data[rows]
        matrix = fauxnograph_coclustering(
            subset,
            graph_type=graph_type,
            n_neighbors=n_neighbors,
            resolution_parameter=resolution,
            n_times=n_times,
            min_cluster_size=min_cluster_size,
            prune=prune,
            normalize=True,
            seed=seed,
            n_jobs=n_jobs,
        )
        similarity = SimilarityMatrix(matrix, normalized=True, method=method)
        return similarity.cluster_labels(
            distance_threshold, min_cluster_size=min_cluster_size
        )

    full_labels = _labels(None)

    rng = np.random.default_rng(seed)
    draw_ari = np.empty(n_draws, dtype=float)
    assigned_fractions = np.empty(n_draws, dtype=float)
    retention_sum = np.zeros(n, dtype=float)
    retention_count = np.zeros(n, dtype=float)

    for draw in range(n_draws):
        rows = np.sort(rng.choice(n, size=draw_size, replace=False))
        drawn = _labels(rows)
        reference = full_labels[rows]
        draw_ari[draw] = _ari(reference, drawn)
        in_reference = reference >= 0
        assigned_fractions[draw] = (
            float((drawn[in_reference] >= 0).mean())
            if in_reference.any()
            else float("nan")
        )
        retention_sum[rows] += _retention(reference, drawn)
        retention_count[rows] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_stability = np.where(
            retention_count > 0, retention_sum / retention_count, np.nan
        )
    return Stability(
        draw_ari=draw_ari,
        cell_stability=cell_stability,
        full_labels=full_labels,
        assigned_fractions=assigned_fractions,
        n_draws=n_draws,
        fraction=fraction,
        settings=settings,
    )


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index over cells assigned in both labellings."""
    from sklearn.metrics import adjusted_rand_score

    usable = (a >= 0) & (b >= 0)
    if usable.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(a[usable], b[usable]))


def _retention(reference: np.ndarray, drawn: np.ndarray) -> np.ndarray:
    """Per cell, the fraction of its reference cluster-mates that stayed with it.

    Computed as the size of a cell's ``(reference, drawn)`` group over the size of its
    reference group within the draw — so it reads as "how much of my cluster came with
    me", the per-cell version of the ARI. A cell assigned in the reference but
    unassigned in the draw scores 0.0 — being dropped is not being retained, however
    many cluster-mates were dropped alongside it.
    """
    pairs = np.stack([reference, drawn], axis=1)
    _, pair_index, pair_counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    _, ref_index, ref_counts = np.unique(
        reference, return_inverse=True, return_counts=True
    )
    out = pair_counts[pair_index.ravel()] / ref_counts[ref_index.ravel()]
    out = np.where(drawn >= 0, out, 0.0)
    # A cell nobody could be grouped with has no retention to report.
    return np.where(reference >= 0, out, np.nan)


def clustering_stability(
    clustering: Clustering,
    data: np.ndarray,
    *,
    distance_threshold: float,
    **kwargs: Any,
) -> Stability:
    """:func:`subsample_stability` with the ensemble settings taken from a ``Clustering``.

    Parameters
    ----------
    clustering : Clustering
        Whose ``partitions`` supply ``graph_type``, ``n_neighbors``, ``resolution`` and
        ``method``. Raises when it has none.
    data : numpy.ndarray
        The coordinates that clustering saw — this cannot be recovered from the clustering
        itself, so passing the wrong matrix silently scores the wrong thing.
    distance_threshold : float
        Where to cut.
    **kwargs
        Override any setting read off the clustering, plus the rest of
        :func:`subsample_stability`'s parameters (``n_draws``, ``fraction``, …).

    Returns
    -------
    Stability
    """
    partitions = clustering.partitions
    if partitions is None:
        raise ValueError(
            "clustering has no partitions to read settings from (they are session-only); "
            "pass the settings to subsample_stability directly"
        )
    defaults: dict[str, Any] = {
        "graph_type": sorted(set(partitions.graph_type.tolist())),
        "n_neighbors": sorted(set(int(k) for k in partitions.n_neighbors)),
        "resolution": sorted(set(float(r) for r in partitions.resolution)),
        "method": clustering.method,
    }
    defaults.update(kwargs)
    return subsample_stability(data, distance_threshold=distance_threshold, **defaults)


# --------------------------------------------------------------------------- #
# 2. recovery of external labels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecoveryScore:
    """Leave-one-out recovery of known labels in one candidate representation.

    ``correct`` is per labelled cell, which is the field that matters: a scalar accuracy
    cannot be compared between representations with any rigour, but the same cells scored
    twice can — see :func:`paired_recovery`.

    ``weights`` carries inverse-probability design weights when the labelled cells came
    from a stratified sample. Read :attr:`accuracy_ipw` as an estimate of population
    accuracy, and be aware of what the weighting does to precision: a design that
    deliberately oversamples hard cases gives those cells small weights, so the weighted
    estimate is dominated by the easy stratum and its standard error can swamp the effect
    being measured. When that is the case, :meth:`by_stratum` is the informative view.

    ``n_abstained`` counts labelled cells with no reachable labelled neighbour to vote
    for them — only :func:`graph_knn_recovery` produces those, on a disconnected graph.
    Abstentions are excluded from the accuracy denominators (an unreachable cell says
    nothing about the decision boundary), while ``correct`` holds ``False`` for them,
    which is how :func:`paired_recovery` sees them: abstained is not correct.

    Attributes
    ----------
    correct : numpy.ndarray
        Boolean recovery result for each labelled cell.
    truth, predicted : numpy.ndarray
        Aligned true and recovered integer codes.
    n_neighbors : int
        Neighbourhood size used for recovery.
    name : str
        Representation name used in comparisons.
    weights : numpy.ndarray, optional
        Per-cell inverse-probability weights.
    strata : numpy.ndarray, optional
        Per-cell audit strata.
    n_abstained : int
        Labelled cells for which no prediction was possible.
    """

    correct: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray
    n_neighbors: int
    name: str = ""
    weights: np.ndarray | None = None
    strata: np.ndarray | None = None
    n_abstained: int = 0

    @property
    def n_labeled(self) -> int:
        """Number of labelled cells scored."""
        return int(self.correct.shape[0])

    @property
    def accuracy(self) -> float:
        """Unweighted share of labelled cells recovered, abstentions excluded."""
        scored = self.correct.shape[0] - self.n_abstained
        return float(self.correct.sum() / scored) if scored else float("nan")

    @property
    def accuracy_ipw(self) -> float:
        """Design-weighted accuracy, or the unweighted one when no weights were given."""
        if self.weights is None:
            return self.accuracy
        voted = self.predicted != _UNASSIGNED
        total = float(self.weights[voted].sum())
        if total <= 0:
            return float("nan")
        return float((self.correct[voted] * self.weights[voted]).sum() / total)

    def by_stratum(self) -> pl.DataFrame:
        """Accuracy within each stratum, with counts — the view weighting hides.

        A stratified design exists to put labelling effort where the classifiers disagree,
        and those are the cells that distinguish representations. Population weighting
        then shrinks them back down. This shows them at face value.

        Returns
        -------
        polars.DataFrame
            Per-stratum counts and accuracy.
        """
        if self.strata is None:
            return pl.DataFrame(
                {
                    "stratum": ["all"],
                    "n": [self.n_labeled],
                    "accuracy": [self.accuracy],
                    "weight": [None],
                }
            )
        frame = pl.DataFrame(
            {
                "stratum": np.asarray(self.strata).astype(str),
                "correct": self.correct.astype(float),
                "weight": self.weights
                if self.weights is not None
                else np.ones(self.n_labeled),
            }
        )
        return (
            frame.group_by("stratum")
            .agg(
                pl.len().alias("n"),
                pl.col("correct").mean().alias("accuracy"),
                pl.col("weight").first().alias("weight"),
            )
            .sort("stratum")
        )

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return (
            f"RecoveryScore({label}n_labeled={self.n_labeled}, "
            f"accuracy={self.accuracy:.4f}, accuracy_ipw={self.accuracy_ipw:.4f})"
        )


def loo_knn_recovery(
    coords: np.ndarray,
    truth: np.ndarray,
    *,
    n_neighbors: int = 15,
    weights: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    vote_weights: str = "uniform",
    name: str = "",
) -> RecoveryScore:
    """Leave-one-out kNN recovery of ``truth`` within a candidate representation.

    Parameters
    ----------
    coords : numpy.ndarray
        The candidate representation's coordinates, ``(n_cells, k)``.
    truth : numpy.ndarray
        Row-aligned label codes, ``-1`` for cells carrying no known label. Only labelled
        cells vote, and only they are scored.
    n_neighbors : int, default 15
        Votes per cell. Capped at the number of other labelled cells.
    weights : numpy.ndarray, optional
        Per-cell inverse-probability design weights (``N_stratum / n_stratum``) when the
        labelled set is a stratified sample. Accepts either full-length or labelled-only;
        any other length raises rather than being aligned by guesswork.
        Read :attr:`RecoveryScore.accuracy_ipw` with the caveat in the notes.
    strata : numpy.ndarray, optional
        Each cell's stratum, so :meth:`RecoveryScore.by_stratum` can break the result down.
    vote_weights : {'uniform', 'distance'}, default 'uniform'
        Whether nearer labelled cells count for more.
    name : str, optional
        Label for this representation in :func:`paired_recovery` output and ``repr``.

    Returns
    -------
    RecoveryScore
        Carrying the per-cell ``correct`` vector, which is what makes paired comparison
        possible.

    Notes
    -----
    Each labelled cell is classified by its ``n_neighbors`` nearest *other* labelled cells,
    so nothing leaks — this is :func:`~cellpax.propagate.propagate_knn` with
    ``preserve_labeled=False``, which already excludes the self-match, rather than a second
    implementation of the same vote.

    Two things to watch. Neighbourhood density among a stratified sample is distorted
    relative to the population, so absolute accuracy means little — but the distortion is
    identical across representations, so paired comparison remains valid. And an external
    label that is itself a classifier's output makes this a measure of agreement with that
    classifier, boundary and errors alike.
    """
    from cellpax.propagate import propagate_knn

    coords = np.asarray(coords, dtype=float)
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    if coords.shape[0] != truth.shape[0]:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but truth covers {truth.shape[0]} cells"
        )
    labeled = np.flatnonzero(truth != _UNASSIGNED)
    if labeled.size < 3:
        raise ValueError(f"need at least three labelled cells, got {labeled.size}")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] == truth.shape[0]:
            weights = weights[labeled]
        elif weights.shape[0] != labeled.size:
            raise ValueError(
                f"weights has {weights.shape[0]} entries but there are "
                f"{truth.shape[0]} cells and {labeled.size} labelled ones"
            )
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape[0] == truth.shape[0]:
            strata = strata[labeled]
        elif strata.shape[0] != labeled.size:
            raise ValueError(
                f"strata has {strata.shape[0]} entries but there are "
                f"{truth.shape[0]} cells and {labeled.size} labelled ones"
            )

    predicted, _confidence, _recovery = propagate_knn(
        coords,
        truth,
        n_neighbors=n_neighbors,
        weights=vote_weights,  # type: ignore[arg-type]
        preserve_labeled=False,
    )
    return RecoveryScore(
        correct=(predicted[labeled] == truth[labeled]),
        truth=truth[labeled],
        predicted=predicted[labeled],
        n_neighbors=n_neighbors,
        name=name,
        weights=weights,
        strata=strata,
    )


def graph_knn_recovery(
    graph: Any,
    truth: np.ndarray,
    *,
    n_neighbors: int = 15,
    cost: str = "neg_log",
    weights: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    max_sources: int = 2000,
    seed: int | None = 0,
    name: str = "",
) -> RecoveryScore:
    """The same recovery, but with distance measured along a neighbour graph.

    Parameters
    ----------
    graph : igraph.Graph
        The graph to score, from :func:`~cellpax.clustering.kneighbor_graph`. Must have one
        vertex per row of ``truth``.
    truth : numpy.ndarray
        Label codes per vertex, ``-1`` for unlabelled.
    n_neighbors : int, default 15
        Graph-nearest labelled cells voting per cell.
    cost : {'neg_log', 'complement'}, default 'neg_log'
        How edge *similarities* become path *costs*. ``'neg_log'`` uses ``-log(w)``, under
        which path cost adds like independent log-probabilities; ``'complement'`` uses
        ``1 - w``. The choice changes the ranking, which is why it is a parameter rather
        than a constant. Ignored for an unweighted graph, which uses hop count.
    weights, strata : numpy.ndarray, optional
        As in :func:`loo_knn_recovery`.
    max_sources : int, default 2000
        Cap on labelled cells used as shortest-path sources, since the all-pairs
        computation is quadratic in that count. Sampled, and both the cap and the realised
        count are logged rather than left implicit.
    seed : int, optional, default 0
        Seeds that sampling.
    name : str, optional
        Label for this representation.

    Returns
    -------
    RecoveryScore
        A vertex with no labelled cell reachable from it abstains rather than being
        scored wrong: it is excluded from the accuracy denominator and counted in
        ``n_abstained``. Its ``correct`` entry is ``False``, which is what
        :func:`paired_recovery` pairs on.
    """
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    if graph.vcount() != truth.shape[0]:
        raise ValueError(
            f"graph has {graph.vcount()} vertices but truth covers {truth.shape[0]} cells"
        )
    labeled = np.flatnonzero(truth != _UNASSIGNED)
    if labeled.size < 3:
        raise ValueError(f"need at least three labelled cells, got {labeled.size}")

    if labeled.size > max_sources:
        rng = np.random.default_rng(seed)
        labeled = np.sort(rng.choice(labeled, size=max_sources, replace=False))
        _logger.info(
            "graph_knn_recovery: sampled %d of the labelled cells as sources "
            "(max_sources=%d); accuracy is estimated on that sample",
            labeled.size,
            max_sources,
        )
    else:
        _logger.info("graph_knn_recovery: using all %d labelled cells", labeled.size)

    if "weight" in graph.es.attributes():
        similarity = np.asarray(graph.es["weight"], dtype=float)
        if cost == "neg_log":
            costs = -np.log(np.clip(similarity, 1e-12, None))
        elif cost == "complement":
            costs = 1.0 - similarity
        else:
            raise ValueError(f"cost must be 'neg_log' or 'complement', got {cost!r}")
        costs = np.clip(costs, 0.0, None).tolist()
    else:
        costs = None

    distances = np.asarray(
        graph.distances(
            source=labeled.tolist(), target=labeled.tolist(), weights=costs
        ),
        dtype=float,
    )
    np.fill_diagonal(distances, np.inf)  # leave-one-out: never vote for yourself
    labels = truth[labeled]
    k = int(min(n_neighbors, labeled.size - 1))
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]

    predicted = np.empty(labeled.size, dtype=np.int64)
    for row in range(labeled.size):
        reachable = labels[nearest[row]][np.isfinite(distances[row, nearest[row]])]
        predicted[row] = (
            np.bincount(reachable).argmax() if reachable.size else _UNASSIGNED
        )

    return RecoveryScore(
        correct=(predicted == labels),
        truth=labels,
        predicted=predicted,
        n_neighbors=k,
        name=name,
        weights=None if weights is None else np.asarray(weights, dtype=float)[labeled],
        strata=None if strata is None else np.asarray(strata)[labeled],
        n_abstained=int((predicted == _UNASSIGNED).sum()),
    )


def paired_recovery(results: dict[str, RecoveryScore]) -> pl.DataFrame:
    """Pairwise differences between representations, tested on the same cells.

    Parameters
    ----------
    results : dict of str to RecoveryScore
        Named scores to compare. All must have been scored on the same cells in the same
        order, which is checked — comparing scores over different labelled sets would be
        meaningless.

    Returns
    -------
    pl.DataFrame
        One row per pair with ``delta_accuracy``, ``delta_accuracy_ipw``, the two
        discordance counts, and ``mcnemar_p`` — an exact binomial test on the discordant
        pairs.

    Notes
    -----
    Differences rather than absolute numbers, because absolute recovery is not the
    quantity of interest and is often not even well defined: with a stratified labelled
    set the population accuracy depends on the design, while the paired difference does
    not. And the paired test is what keeps a difference of a handful of cells from being
    read as a result — with a few hundred labelled cells, most differences between
    reasonable representations will not clear it, and that is the honest answer.

    An abstaining cell (see :func:`graph_knn_recovery`) counts as *not correct* here: a
    representation whose graph cannot reach a cell has not recovered it, so an opponent
    that does reach and recover it scores a discordant win on that cell.
    """
    from scipy.stats import binomtest

    if len(results) < 2:
        raise ValueError("need at least two representations to compare")
    sizes = {name: score.n_labeled for name, score in results.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"all representations must be scored on the same cells, got sizes {sizes}"
        )
    reference = next(iter(results.values())).truth
    for name, score in results.items():
        if not np.array_equal(score.truth, reference):
            raise ValueError(
                f"{name!r} was scored against different truth labels; paired comparison "
                "needs the same cells in the same order"
            )

    rows = []
    for left, right in combinations(results, 2):
        a, b = results[left], results[right]
        only_a = int((a.correct & ~b.correct).sum())
        only_b = int((~a.correct & b.correct).sum())
        discordant = only_a + only_b
        p = float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
        rows.append(
            {
                "a": left,
                "b": right,
                "accuracy_a": a.accuracy,
                "accuracy_b": b.accuracy,
                "delta_accuracy": a.accuracy - b.accuracy,
                "delta_accuracy_ipw": a.accuracy_ipw - b.accuracy_ipw,
                "n_correct_a_only": only_a,
                "n_correct_b_only": only_b,
                "n_discordant": discordant,
                "mcnemar_p": p,
            }
        )
    return pl.DataFrame(rows).sort("delta_accuracy", descending=True)


# --------------------------------------------------------------------------- #
# 3. purity against an audit label
# --------------------------------------------------------------------------- #


def label_purity(
    labels: Any,
    audit: Any,
    *,
    level_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Per-cluster composition under a label the clustering never saw.

    Parameters
    ----------
    labels : pl.DataFrame or LabelSet or numpy.ndarray
        A ``nested_labels`` frame — in which case every ``level_*`` column is scored and the
        result carries a ``level`` column — or a single flat labelling.
    audit : sequence or pl.Series
        Row-aligned audit values, strings or codes. Nulls are counted apart rather than
        treated as a category.
    level_columns : list of str, optional
        Which columns of a frame to score. Defaults to every ``level_*`` column.

    Returns
    -------
    pl.DataFrame
        One row per cluster, sorted worst-purity first: ``n_cells``, ``n_audit_labels``,
        ``dominant``, ``purity`` (the dominant label's share of the non-null values),
        ``n_audit_null``, and ``n_unassigned`` — how many cells that labelling left
        unassigned (``-1``), repeated per row (per level for a nested frame). The
        unassigned pile gets no row of its own: it is mixed by construction, so a row
        for it would top every worst-first read while saying nothing about clusters.

    Notes
    -----
    A cluster spanning a boundary the features had no access to is a genuine red flag
    rather than something to tune away. But the check only has teeth where straddling is
    possible: score a cohort that was itself selected on the audit label and every purity
    is 1.0 by construction, which is not evidence of anything. It bites when the audit
    label comes from a different source than the one that defined the cohort, because then
    the two genuinely disagree about some cells.

    For the full cross-tabulation rather than the per-cluster summary, use
    :func:`cellpax.compare` — ``compare(labels, audit).contingency()``.
    """
    audit_values = np.asarray(
        audit.to_numpy() if hasattr(audit, "to_numpy") else audit, dtype=object
    ).reshape(-1)

    if isinstance(labels, pl.DataFrame):
        columns = level_columns or [c for c in labels.columns if c.startswith("level_")]
        if not columns:
            raise ValueError(
                f"no level_* columns in the frame (has {labels.columns}); pass "
                "level_columns explicitly"
            )
        frames = []
        for level, column in enumerate(columns):
            frame = _purity_one(labels[column].to_numpy(), audit_values)
            frames.append(
                frame.with_columns(pl.lit(level).cast(pl.Int64).alias("level"))
            )
        return pl.concat(frames).select(
            ["level", *[c for c in frames[0].columns if c != "level"]]
        )

    codes = np.asarray(
        labels.codes if hasattr(labels, "codes") else labels, dtype=np.int64
    ).reshape(-1)
    return _purity_one(codes, audit_values)


def _purity_one(codes: np.ndarray, audit: np.ndarray) -> pl.DataFrame:
    """Per-cluster purity for one flat labelling; unassigned cells counted apart."""
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if codes.shape[0] != audit.shape[0]:
        raise ValueError(
            f"labels cover {codes.shape[0]} cells but audit covers {audit.shape[0]}"
        )
    n_unassigned = int((codes == _UNASSIGNED).sum())
    rows = []
    for code in np.unique(codes):
        if int(code) == _UNASSIGNED:
            continue
        members = audit[codes == code]
        known = [v for v in members if v is not None and v == v]
        if known:
            # tallied on the values themselves, so 1 and "1" stay two labels
            tally = Counter(known)
            value, count = min(tally.items(), key=lambda item: (-item[1], str(item[0])))
            dominant = str(value)
            purity = float(count / len(known))
            n_labels = len(tally)
        else:
            dominant, purity, n_labels = None, float("nan"), 0
        rows.append(
            {
                "cluster": int(code),
                "n_cells": int(members.size),
                "n_audit_labels": n_labels,
                "dominant": dominant,
                "purity": purity,
                "n_audit_null": int(members.size - len(known)),
                "n_unassigned": n_unassigned,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "cluster": pl.Int64,
            "n_cells": pl.Int64,
            "n_audit_labels": pl.Int64,
            "dominant": pl.String,
            "purity": pl.Float64,
            "n_audit_null": pl.Int64,
            "n_unassigned": pl.Int64,
        },
    ).sort("purity")


# --------------------------------------------------------------------------- #
# 5. transfer between datasets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TransferScore:
    """A classifier trained on one dataset's labels, scored on another dataset's.

    The cross-dataset counterpart of :class:`RecoveryScore`. When two datasets have been
    put in one space, a classifier fit on either should predict the other's labels; its
    errors say where they still disagree. Those errors are only informative when they are
    read per label — a boundary the two taxonomies drew in different places (an L2/L3
    split, one dataset's ITC being the other's BPC + MPC) looks like a misalignment in a
    scalar accuracy and is obvious in :meth:`confusion`.

    Attributes
    ----------
    truth : numpy.ndarray
        Test-dataset label names of the scored cells.
    predicted : numpy.ndarray
        Predicted label names, aligned to ``truth``.
    train_classes : tuple of str
        Labels the classifier saw during training.
    train : str
        Dataset the classifier was trained on.
    test : str
        Dataset it was scored on.
    cell_ids : numpy.ndarray, optional
        Ids of the scored cells, aligned to ``truth``.
    """

    truth: np.ndarray
    predicted: np.ndarray
    train_classes: tuple[str, ...]
    train: str
    test: str
    cell_ids: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        """Number of labelled test cells scored."""
        return int(self.truth.shape[0])

    @property
    def shared(self) -> np.ndarray:
        """Per-cell mask of test cells whose true label exists in the training set."""
        return np.isin(self.truth, np.asarray(self.train_classes, dtype=object))

    @property
    def accuracy(self) -> float:
        """Share of all scored test cells predicted correctly."""
        if not self.n_cells:
            return float("nan")
        return float(np.mean(self.truth == self.predicted))

    @property
    def accuracy_shared(self) -> float:
        """Accuracy over test cells whose label the classifier could have predicted.

        A label present only in the test dataset cannot be predicted by construction, so
        it drags :attr:`accuracy` down without saying anything about alignment.
        """
        shared = self.shared
        if not shared.any():
            return float("nan")
        return float(np.mean(self.truth[shared] == self.predicted[shared]))

    def confusion(self) -> pl.DataFrame:
        """Long-form confusion table, normalized within each true label.

        Returns
        -------
        polars.DataFrame
            ``truth, predicted, n_cells, fraction`` for every observed pair, where
            ``fraction`` is the share of that true label's cells given that prediction.
        """
        frame = pl.DataFrame(
            {
                "truth": self.truth.astype(str).tolist(),
                "predicted": self.predicted.astype(str).tolist(),
            },
            schema={"truth": pl.String, "predicted": pl.String},
        )
        return (
            frame.group_by("truth", "predicted")
            .len("n_cells")
            .with_columns(
                (pl.col("n_cells") / pl.col("n_cells").sum().over("truth")).alias(
                    "fraction"
                )
            )
            .with_columns(pl.col("n_cells").cast(pl.Int64))
            .sort(["truth", "n_cells"], descending=[False, True])
        )

    def by_label(self) -> pl.DataFrame:
        """Recall per true label, with its most common prediction.

        Returns
        -------
        polars.DataFrame
            ``truth, n_cells, in_train, recall, top_prediction, top_fraction``.
        """
        confusion = self.confusion()
        train = set(self.train_classes)
        rows = []
        for (label,), part in confusion.group_by(["truth"], maintain_order=True):
            correct = part.filter(pl.col("predicted") == label)["n_cells"].sum()
            top = part.sort("n_cells", descending=True).row(0, named=True)
            total = int(part["n_cells"].sum())
            rows.append(
                {
                    "truth": label,
                    "n_cells": total,
                    "in_train": label in train,
                    "recall": float(correct / total),
                    "top_prediction": top["predicted"],
                    "top_fraction": float(top["fraction"]),
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "truth": pl.String,
                "n_cells": pl.Int64,
                "in_train": pl.Boolean,
                "recall": pl.Float64,
                "top_prediction": pl.String,
                "top_fraction": pl.Float64,
            },
        ).sort("recall")

    def __repr__(self) -> str:
        return (
            f"TransferScore({self.train} → {self.test}: n={self.n_cells}, "
            f"accuracy={self.accuracy:.3f}, shared={self.accuracy_shared:.3f})"
        )


def cross_dataset_classification(
    features: np.ndarray,
    labels: Any,
    datasets: Any,
    *,
    train: str,
    test: str,
    classifier: Any = None,
    seed: int = 0,
    cell_ids: np.ndarray | None = None,
) -> TransferScore:
    """Fit a classifier on one dataset's labelled cells and predict another's.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix covering both datasets.
    labels : array-like
        ``(n_cells,)`` label names; ``None`` marks unlabelled cells, which are left out
        of both training and scoring.
    datasets : array-like
        ``(n_cells,)`` dataset of each cell.
    train : str
        Dataset whose labelled cells train the classifier.
    test : str
        Dataset whose labelled cells are scored.
    classifier : scikit-learn classifier, optional
        Unfitted estimator; it is cloned, never fitted in place. Defaults to a
        300-tree ``RandomForestClassifier``.
    seed : int, default 0
        ``random_state`` of the default classifier.
    cell_ids : numpy.ndarray, optional
        Ids aligned to ``features``, carried onto the score.

    Returns
    -------
    TransferScore
        Truth and prediction for every labelled test cell.

    Raises
    ------
    ValueError
        On mismatched lengths, identical or unknown datasets, or no labelled cells.
    """
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier

    matrix = np.asarray(features, dtype=float)
    label_values = np.asarray(
        [None if v is None else str(v) for v in np.asarray(labels, dtype=object)],
        dtype=object,
    )
    dataset_values = np.asarray([str(d) for d in np.asarray(datasets).tolist()])
    n_cells = matrix.shape[0]
    if label_values.shape[0] != n_cells or dataset_values.shape[0] != n_cells:
        raise ValueError("features, labels and datasets must have the same length")
    if train == test:
        raise ValueError(f"train and test must be different datasets, got {train!r}")
    known = set(dataset_values.tolist())
    for role, name in (("train", train), ("test", test)):
        if name not in known:
            raise ValueError(f"{role} dataset {name!r} not found; have {sorted(known)}")
    labelled = np.asarray([v is not None for v in label_values])
    train_rows = labelled & (dataset_values == train)
    test_rows = labelled & (dataset_values == test)
    if not train_rows.any() or not test_rows.any():
        raise ValueError(
            f"need labelled cells in both datasets; {train!r} has "
            f"{int(train_rows.sum())}, {test!r} has {int(test_rows.sum())}"
        )
    model = (
        RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        if classifier is None
        else clone(classifier)
    )
    model.fit(matrix[train_rows], label_values[train_rows].astype(str))
    predicted = np.asarray(model.predict(matrix[test_rows]), dtype=object)
    return TransferScore(
        truth=label_values[test_rows],
        predicted=predicted,
        train_classes=tuple(sorted(set(label_values[train_rows].tolist()))),
        train=train,
        test=test,
        cell_ids=None if cell_ids is None else np.asarray(cell_ids)[test_rows],
    )
