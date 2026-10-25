"""Cross-dataset reproducibility of per-dataset clusterings, judged through a bridge space.

Two datasets measure the same features, but batch effects keep those features from
being compared directly. Each dataset is therefore clustered in its own space, where
separation is sharpest, and a *bridge* — an embedding in which the datasets are
aligned, such as a shared autoencoder latent — is used only to ask whether a
partition found in one dataset reappears in the other. This is prediction strength
(Tibshirani & Walther 2005) and classifier-transfer stability (Lange et al. 2004),
with a kNN vote in the bridge as the transfer model.

A raw transfer score has three biases, and every number here is reported next to
the reading that corrects it:

``ceiling``
    The same partition carried through the bridge *within its own dataset*
    (leave-one-out, so no batch effect). A bridge that blurs fine structure loses
    fine splits here too, so ``efficiency = cross / ceiling`` separates "the bridge
    cannot resolve this" from "this does not reproduce". Read it per cluster: a
    cluster whose ``ceiling_purity`` is high but whose cross ``efficiency`` is low
    is *refuted*; one whose ceiling is itself low is *unverifiable* through this
    bridge, and its efficiency near one is a statement about the bridge, not
    support for the split. Global AMI hides both, because it is carried by
    whatever coarse structure the two partitions share.
``null``
    Random tessellations at the same realized grain, carried and scored by exactly
    the same estimator. ``excess = ami - null_ami`` is agreement beyond what a
    smooth partition with no correspondence earns on these cells. It is taken on
    AMI, not on efficiency: a tessellation cuts through populations, so its
    ceiling is low and its ratio inflated — efficiency is the blur diagnostic,
    not the reproducibility score.
``coverage``
    Mutual-only cross edges let a cell with no counterpart abstain, so a type one
    dataset lacks is not forced onto the other. Agreement is computed over the
    co-assigned cells only, and coverage says how many that is.

Carrying a partition is one sparse one-hot product against a vote matrix built once,
so every stored Leiden run can be scored without rebuilding a consensus matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix, diags
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from cellpax.clustering import _neighbor_arrays

if TYPE_CHECKING:
    from cellpax.clustering import Clustering
    from cellpax.labels import LabelSet

_UNASSIGNED = -1
_TINY = 1e-12

Null = Literal["voronoi", "kmeans"]


def _row_normalized(matrix: csr_matrix) -> csr_matrix:
    """Rows scaled to sum to one; empty rows stay empty (they abstain)."""
    total = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.where(total > 0, 1.0 / np.maximum(total, _TINY), 0.0)
    return csr_matrix(diags(inverse) @ matrix)


def _connectivity(indices: np.ndarray, n_columns: int) -> csr_matrix:
    """Sparse ``(n_rows, n_columns)`` 0/1 matrix with a one on each neighbour edge."""
    n_rows, k = indices.shape
    rows = np.repeat(np.arange(n_rows), k)
    data = np.ones(n_rows * k, dtype=float)
    return coo_matrix(
        (data, (rows, indices.ravel())), shape=(n_rows, n_columns)
    ).tocsr()


def _query(source: np.ndarray, target: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Indices of each target row's ``n_neighbors`` nearest source rows."""
    k = int(min(n_neighbors, source.shape[0]))
    finder = NearestNeighbors(n_neighbors=k).fit(source)
    return finder.kneighbors(target, return_distance=False)


def _one_hot(codes: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Sparse ``(n, k)`` indicator over the assigned cells, and the code of each column.

    Unassigned cells get an empty row, so they cast no vote.
    """
    clusters, columns = np.unique(codes[codes >= 0], return_inverse=True)
    rows = np.flatnonzero(codes >= 0)
    indicator = coo_matrix(
        (np.ones(rows.size), (rows, columns)), shape=(codes.size, clusters.size)
    ).tocsr()
    return indicator, clusters


def _ami(truth: np.ndarray, predicted: np.ndarray) -> float:
    """AMI over the cells assigned on both sides; NaN when none are."""
    both = (truth >= 0) & (predicted >= 0)
    if not both.any():
        return float("nan")
    return float(adjusted_mutual_info_score(truth[both], predicted[both]))


def _n_clusters(codes: np.ndarray) -> int:
    return int(np.unique(codes[codes >= 0]).size)


def _image_floor(n_cells: np.ndarray) -> np.ndarray:
    """Smallest image that counts as one: a few stray cells are not a counterpart."""
    return np.maximum(3.0, 0.1 * np.asarray(n_cells, dtype=float))


#: Verdict labels, in the order counts are reported.
VERDICTS = (
    "reproduces",
    "weak",
    "nested",
    "split",
    "refuted",
    "unverifiable",
    "absent",
)

#: One-sided verdicts that count as a passing image, before the match is checked back.
_PASSING = ("reproduces", "weak")


@dataclass(frozen=True)
class VerdictRule:
    """Thresholds that turn a cluster's purity readings into a verdict.

    A cluster is judged on two numbers: ``ceiling_purity``, how cleanly the
    bridge recovers it inside its own dataset, and ``efficiency``, its purity in
    the other dataset over that ceiling. :meth:`classify` applies the thresholds
    to one direction; the verdicts reported by :class:`BridgeScore` and
    :meth:`TransferBridge.score_runs` then check the match *back*, because a
    one-sided purity rewards over-splitting — two halves of a type the other
    dataset keeps whole each land cleanly inside it.

    ``reproduces``
        Ceiling at least ``ceiling_floor`` and efficiency at least
        ``reproduce_at`` — and its best match across is a mutual best match
        that passes the same test the other way. A one-to-one anchor.
    ``weak``
        As ``reproduces``, but one or both directions only clear
        ``refute_below``.
    ``nested``
        Its image lands cleanly in one cluster across, but that cluster's best
        match is not it: it is part of something the other dataset keeps
        whole (or draws differently). An anchor *into* that cluster.
    ``split``
        Its image is impure, but mostly covered — at ``reproduce_at`` of its
        ceiling — by two or more clusters across that each pass and point
        back at it: the other dataset subdivides it. A one-to-many anchor.
    ``refuted``
        Ceiling at least ``ceiling_floor`` and efficiency below
        ``refute_below``, and not a split: the bridge resolves the cluster and
        the other dataset does not have it.
    ``unverifiable``
        Ceiling below ``ceiling_floor`` (or none): the bridge cannot see the
        cluster even within its own dataset. Not evidence against it, but not
        support either — the bridge discards dataset-specific variation, so
        artifact-driven splits land here, and nothing here should be treated
        as shared.
    ``absent``
        Visible to the bridge, but no cell of the other dataset was carried
        into it — typically a type the other dataset lacks.

    Clusters with fewer than ``min_cells`` cells are left out of the counts.

    Attributes
    ----------
    ceiling_floor : float, default 0.6
        Lowest within-dataset purity at which the bridge can judge a cluster.
    reproduce_at : float, default 0.85
        Efficiency at or above which a judgeable cluster reproduces.
    refute_below : float, default 0.7
        Efficiency below which a judgeable cluster is refuted.
    min_cells : int, default 20
        Smallest cluster counted.
    """

    ceiling_floor: float = 0.6
    reproduce_at: float = 0.85
    refute_below: float = 0.7
    min_cells: int = 20

    def __post_init__(self) -> None:
        if not self.refute_below <= self.reproduce_at:
            raise ValueError(
                f"refute_below ({self.refute_below}) must not exceed reproduce_at "
                f"({self.reproduce_at})"
            )

    def classify(self, ceiling_purity: np.ndarray, purity: np.ndarray) -> np.ndarray:
        """One-sided verdict per cluster from its within-dataset and cross purity.

        The first pass only: ``reproduces`` / ``weak`` here mean the image is
        clean in this direction, before the match is checked back (see the
        class notes); ``nested`` and ``split`` never come out of this.

        Parameters
        ----------
        ceiling_purity, purity : numpy.ndarray
            Per-cluster purity inside the source dataset and in the other one;
            NaN where the cluster has no image.

        Returns
        -------
        numpy.ndarray
            ``reproduces``, ``weak``, ``refuted``, ``unverifiable`` or
            ``absent`` per cluster.
        """
        ceiling_purity = np.asarray(ceiling_purity, dtype=float)
        purity = np.asarray(purity, dtype=float)
        visible = np.nan_to_num(ceiling_purity, nan=0.0) >= self.ceiling_floor
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = purity / ceiling_purity
        return np.select(
            [
                ~visible,
                np.isnan(purity),
                efficiency >= self.reproduce_at,
                efficiency < self.refute_below,
            ],
            ["unverifiable", "absent", "reproduces", "refuted"],
            default="weak",
        )


def _strata_ami(
    truth: np.ndarray, predicted: np.ndarray, strata: np.ndarray, min_cells: int = 30
) -> float:
    """Size-weighted AMI within each stratum: the subdivision, not the strata.

    Global AMI between two fine partitions is carried by the coarse structure
    they share. Restricting to the cells of one stratum (a subclass, say) asks
    only whether the split *inside* it reproduces. Strata with fewer than
    ``min_cells`` cells, or that ``truth`` does not split, are skipped.
    """
    values, weights = [], []
    for stratum in np.unique(strata[strata != None]):  # noqa: E711 — elementwise
        rows = strata == stratum
        if rows.sum() < min_cells or _n_clusters(truth[rows]) < 2:
            continue
        value = _ami(truth[rows], predicted[rows])
        if np.isfinite(value):
            values.append(value)
            weights.append(rows.sum())
    return float(np.average(values, weights=weights)) if values else float("nan")


@dataclass(frozen=True)
class Carry:
    """One partition carried through a vote matrix onto a set of target cells.

    Attributes
    ----------
    soft : numpy.ndarray
        ``(n_target, k)`` share of each target cell's assigned voters per source
        cluster. Rows of abstaining cells are zero.
    codes : numpy.ndarray
        Hard source-cluster code per target cell, ``-1`` where it abstained.
    confidence : numpy.ndarray
        The winning share, ``0`` where it abstained.
    clusters : numpy.ndarray
        The source cluster code of each ``soft`` column.
    """

    soft: np.ndarray
    codes: np.ndarray
    confidence: np.ndarray
    clusters: np.ndarray


def _soft_votes(
    votes: csr_matrix, codes: np.ndarray, min_confidence: float | None
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """``(soft, hard, confidence, clusters)`` for one carry, ``soft`` kept sparse.

    A row has at most ``n_neighbors`` voters, so the shares stay sparse however
    many clusters there are — what lets :meth:`TransferBridge.score_runs` keep
    every run's soft carry at once.
    """
    indicator, clusters = _one_hot(codes)
    tally = csr_matrix(votes @ indicator)
    # shares are over the voters that held a label, so an unassigned neighbour
    # neither dilutes the vote nor counts as a vote for anything
    total = np.asarray(tally.sum(axis=1)).ravel()
    soft = _row_normalized(tally)
    n_target = votes.shape[0]
    if clusters.size == 0:
        return soft, np.full(n_target, _UNASSIGNED), np.zeros(n_target), clusters
    winner = np.asarray(soft.argmax(axis=1)).ravel()
    confidence = soft.max(axis=1).toarray().ravel()
    abstain = total <= 0
    if min_confidence is not None:
        abstain |= confidence < min_confidence
    hard = np.where(abstain, _UNASSIGNED, clusters[winner])
    return soft, hard, np.where(abstain, 0.0, confidence), clusters


def _as_array(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _cross_mass(soft: Any, target_codes: np.ndarray) -> np.ndarray:
    """``(k_target, k_source)`` soft vote mass of each source cluster per target cluster.

    Rows follow ``_one_hot(target_codes)``, so they are the target partition's
    clusters in sorted order — the same order that partition's own carry uses
    for its columns, which is what lets the two directions be read together.
    """
    target_onehot, _ = _one_hot(target_codes)
    return _as_array(target_onehot.T @ soft)


def _ceiling_purity(
    soft: Any, source_codes: np.ndarray, n_source: np.ndarray
) -> np.ndarray:
    """Per source cluster, the share of its within-dataset image that is itself.

    The soft carry, its clusters and this indicator all come from
    ``_one_hot(source_codes)``, so their columns name the same clusters in the
    same order and the diagonal is "carried back into itself".
    """
    self_onehot, _ = _one_hot(source_codes)
    within = _as_array(self_onehot.T @ soft)
    mass = within.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass >= _image_floor(n_source), np.diag(within) / mass, np.nan)


def _one_side(
    cross: np.ndarray, sizes: np.ndarray, ceiling: np.ndarray, rule: VerdictRule
) -> dict[str, np.ndarray]:
    """One direction's purity, best match (index), image and one-sided verdict."""
    image = cross.sum(axis=0) if cross.size else np.zeros(sizes.size)
    has_image = image >= _image_floor(sizes)
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(
            has_image, (cross.max(axis=0) if cross.size else 0.0) / image, np.nan
        )
    best = (
        np.where(has_image, cross.argmax(axis=0), -1)
        if cross.shape[0]
        else np.full(sizes.size, -1)
    )
    return {
        "image": image,
        "purity": purity,
        "best": best,
        "base": rule.classify(ceiling, purity),
    }


def _two_sided(
    cross_ab: np.ndarray,
    cross_ba: np.ndarray,
    side_a: dict[str, np.ndarray],
    side_b: dict[str, np.ndarray],
    rule: VerdictRule,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verdicts for both partitions, each checked against the other's.

    ``cross_ab`` is ``(k_b, k_a)``: a's clusters' vote mass in b's clusters, and
    ``cross_ba`` the reverse. ``side_*`` carry ``sizes`` and ``ceiling`` per
    cluster, in sorted-cluster order. Returns per side the one-sided readings
    plus ``verdict``, ``mutual`` and ``split_into`` (indices across).
    """
    ab = _one_side(cross_ab, side_a["sizes"], side_a["ceiling"], rule)
    ba = _one_side(cross_ba, side_b["sizes"], side_b["ceiling"], rule)
    for here, there, cross, ceiling in (
        (ab, ba, cross_ab, side_a["ceiling"]),
        (ba, ab, cross_ba, side_b["ceiling"]),
    ):
        n = here["best"].size
        verdict = here["base"].astype(object).copy()
        mutual = np.zeros(n, dtype=bool)
        split_into: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            j = here["best"][i]
            if j >= 0:
                mutual[i] = there["best"][j] == i
            if here["base"][i] in _PASSING:
                if mutual[i] and there["base"][j] in _PASSING:
                    both = here["base"][i] == there["base"][j] == "reproduces"
                    verdict[i] = "reproduces" if both else "weak"
                else:
                    verdict[i] = "nested"
            elif here["base"][i] == "refuted" and cross.shape[0]:
                parts = [
                    j2
                    for j2 in range(there["best"].size)
                    if there["best"][j2] == i and there["base"][j2] in _PASSING
                ]
                if len(parts) >= 2 and here["image"][i] > 0:
                    covered = cross[parts, i].sum() / here["image"][i]
                    if covered / ceiling[i] >= rule.reproduce_at:
                        verdict[i] = "split"
                        split_into[i] = parts
        here["verdict"] = verdict.astype(str)
        here["mutual"] = mutual
        here["split_into"] = split_into
    return ab, ba


def _cluster_frame(
    side: dict[str, Any],
    *,
    clusters: np.ndarray,
    sizes: np.ndarray,
    ceiling: np.ndarray,
    target_clusters: np.ndarray,
    direction: str,
    names: dict[int, str] | None,
    target_names: dict[int, str] | None,
    rule: VerdictRule,
) -> pl.DataFrame:
    """Per-source-cluster readings for one direction, verdicts already two-sided.

    The ceiling reads the same quantity within the source dataset, where the
    "target partition" is the source partition itself, so the ratio is how much
    of the bridge's resolving power the cross-dataset match retains.
    """
    best = np.where(
        side["best"] >= 0,
        target_clusters[np.clip(side["best"], 0, max(target_clusters.size - 1, 0))]
        if target_clusters.size
        else _UNASSIGNED,
        _UNASSIGNED,
    )
    frame = pl.DataFrame(
        {
            "direction": [direction] * clusters.size,
            "cluster": clusters.astype(np.int64),
            "n_cells": sizes.astype(np.int64),
            "image_mass": side["image"],
            "best_match": best.astype(np.int64),
            "mutual": side["mutual"],
            "purity": side["purity"],
            "ceiling_purity": ceiling,
            "verdict": side["verdict"],
            "split_into": [
                [int(target_clusters[j]) for j in parts] for parts in side["split_into"]
            ],
        },
        schema_overrides={"split_into": pl.List(pl.Int64)},
    ).with_columns(
        (pl.col("purity") / pl.col("ceiling_purity")).alias("efficiency"),
        pl.when(pl.col("n_cells") < rule.min_cells)
        .then(pl.lit("small"))
        .otherwise(pl.col("verdict"))
        .alias("verdict"),
    )
    if names:
        frame = frame.with_columns(
            pl.col("cluster").replace_strict(names, default=None).alias("cluster_name")
        )
    if target_names:
        frame = frame.with_columns(
            pl.col("best_match")
            .replace_strict(target_names, default=None)
            .alias("best_match_name"),
            pl.col("split_into")
            .list.eval(pl.element().replace_strict(target_names, default=None))
            .alias("split_into_names"),
        )
    return frame


@dataclass(frozen=True)
class BridgeScore:
    """How well two per-dataset partitions reproduce each other through the bridge.

    Attributes
    ----------
    summary : dict
        Per direction (``a->b``, ``b->a``): ``ami`` of the carried labels against
        the target's own partition, ``ceiling`` (the source partition's
        within-dataset recovery through the bridge), ``efficiency`` (their
        ratio) and ``coverage`` (co-assigned share of the target's assigned
        cells), and ``worst_efficiency`` (lowest per-cluster efficiency). The
        unsuffixed ``ami`` / ``efficiency`` average the two directions;
        ``worst_efficiency`` is the lower of the two. The headline shares
        pool both directions over *every* cluster of ``min_cells`` or more:
        ``reproduce_share`` (one-to-one reproductions), ``anchored_share``
        (any anchor: reproduces, weak, nested or split) and
        ``anchored_cells`` (the share of cells in anchored clusters).
    clusters : polars.DataFrame
        One row per source cluster per direction: its image's ``purity`` in
        the other dataset, the within-dataset ``ceiling_purity``, their
        ratio, the ``best_match`` cluster across, and the ``verdict`` under
        the :class:`VerdictRule` used (``"small"`` below its ``min_cells``).

    Notes
    -----
    With strata on the bridge, ``summary`` also carries ``strata_ami`` and
    ``strata_ceiling`` (per direction and averaged): the same readings
    restricted to the cells of one stratum at a time, so they measure whether
    the subdivision *inside* each stratum reproduces.
    """

    summary: dict[str, float]
    clusters: pl.DataFrame

    def verdicts(self) -> pl.DataFrame:
        """Cluster counts per verdict and direction.

        Returns
        -------
        polars.DataFrame
            One row per direction, one column per verdict.
        """
        counts = (
            self.clusters.group_by("direction", "verdict")
            .len()
            .pivot(on="verdict", index="direction", values="len")
            .fill_null(0)
        )
        for verdict in (*VERDICTS, "small"):
            if verdict not in counts.columns:
                counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(verdict))
        return counts.select("direction", *VERDICTS, "small").sort("direction")

    def anchors(self) -> pl.DataFrame:
        """Proposed correspondences between the two partitions, one row per anchor.

        The output this machinery exists for: which clusters of one dataset
        the bridge supports as the same thing in the other, ready to carry an
        aligned name. Three kinds:

        ``one-to-one``
            A mutual best match passing both ways (``support`` is
            ``reproduces`` or ``weak``).
        ``split``
            One cluster that the other dataset subdivides: the listed clusters
            across each nest in it and together cover its image.
        ``nested``
            A cluster that lands cleanly inside one across, where that one is
            not a split — it pairs with something else, or draws its edges
            differently. An anchor into it, weaker than the other two.

        Clusters that are ``refuted``, ``unverifiable``, ``absent`` or
        ``small`` are not anchors — they are the dataset-specific remainder.
        These are proposals for naming, read alongside priors; they are not
        tests of discreteness (see the guide).

        Returns
        -------
        polars.DataFrame
            ``kind``, ``support``, ``a`` and ``b`` (lists of cluster names on
            each side), ``n_a`` / ``n_b`` (cells in the clusters counted),
            and ``efficiency_ab`` / ``efficiency_ba``.
        """
        directions = self.clusters["direction"].unique(maintain_order=True).to_list()
        if len(directions) != 2:
            return pl.DataFrame()
        name_col = "cluster_name" if "cluster_name" in self.clusters.columns else None

        def side(direction):
            rows = self.clusters.filter(pl.col("direction") == direction)
            label = (
                rows[name_col].cast(pl.String)
                if name_col
                else rows["cluster"].cast(pl.String)
            )
            return {
                int(c): dict(r, label=n)
                for c, r, n in zip(rows["cluster"], rows.iter_rows(named=True), label)
            }

        fwd, back = side(directions[0]), side(directions[1])
        rows = []
        split_parts_a: set[int] = set()
        split_parts_b: set[int] = set()
        for here, there, flip in ((fwd, back, False), (back, fwd, True)):
            for code, r in here.items():
                if r["verdict"] != "split":
                    continue
                parts = [there[p] for p in r["split_into"] if p in there]
                (split_parts_a if flip else split_parts_b).update(r["split_into"])
                one = ([r["label"]], int(r["n_cells"]), r["efficiency"])
                many = (
                    [p["label"] for p in parts],
                    int(sum(p["n_cells"] for p in parts)),
                    float(np.nanmean([p["efficiency"] for p in parts]))
                    if parts
                    else np.nan,
                )
                a_side, b_side = (many, one) if flip else (one, many)
                rows.append(
                    {
                        "kind": "split",
                        "support": "split",
                        "a": a_side[0],
                        "b": b_side[0],
                        "n_a": a_side[1],
                        "n_b": b_side[1],
                        "efficiency_ab": a_side[2],
                        "efficiency_ba": b_side[2],
                    }
                )
        for code, r in fwd.items():
            partner = back.get(int(r["best_match"]))
            if r["verdict"] in ("reproduces", "weak") and partner is not None:
                rows.append(
                    {
                        "kind": "one-to-one",
                        "support": r["verdict"],
                        "a": [r["label"]],
                        "b": [partner["label"]],
                        "n_a": int(r["n_cells"]),
                        "n_b": int(partner["n_cells"]),
                        "efficiency_ab": r["efficiency"],
                        "efficiency_ba": partner["efficiency"],
                    }
                )
        for here, there, done, flip in (
            (fwd, back, split_parts_a, False),
            (back, fwd, split_parts_b, True),
        ):
            for code, r in here.items():
                partner = there.get(int(r["best_match"]))
                if r["verdict"] != "nested" or code in done or partner is None:
                    continue
                inner = ([r["label"]], int(r["n_cells"]), r["efficiency"])
                outer = ([partner["label"]], int(partner["n_cells"]), np.nan)
                a_side, b_side = (outer, inner) if flip else (inner, outer)
                rows.append(
                    {
                        "kind": "nested",
                        "support": "nested",
                        "a": a_side[0],
                        "b": b_side[0],
                        "n_a": a_side[1],
                        "n_b": b_side[1],
                        "efficiency_ab": a_side[2],
                        "efficiency_ba": b_side[2],
                    }
                )
        if not rows:
            return pl.DataFrame(
                schema={
                    "kind": pl.String,
                    "support": pl.String,
                    "a": pl.List(pl.String),
                    "b": pl.List(pl.String),
                    "n_a": pl.Int64,
                    "n_b": pl.Int64,
                    "efficiency_ab": pl.Float64,
                    "efficiency_ba": pl.Float64,
                }
            )
        order = {"one-to-one": 0, "split": 1, "nested": 2}
        return pl.DataFrame(rows).sort(
            pl.col("kind").replace_strict(order),
            pl.col("n_a"),
            descending=[False, True],
        )

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"BridgeScore(ami={s['ami']:.3f}, efficiency={s['efficiency']:.3f}, "
            f"reproduce_share={s.get('reproduce_share', float('nan')):.2f}, "
            f"anchored_cells={s.get('anchored_cells', float('nan')):.2f}, "
            f"coverage a->b={s['coverage_ab']:.2f}, b->a={s['coverage_ba']:.2f})"
        )


@dataclass
class TransferBridge:
    """Frozen kNN vote matrices between two datasets in an aligned bridge space.

    Build with :meth:`build` or :meth:`cellpax.FeatureTable.transfer_bridge`.

    Attributes
    ----------
    names : tuple of str
        The two dataset names, ``(a, b)``.
    cell_ids : dict
        Each dataset's cell ids, in the row order of its vote matrices.
    cross : dict
        ``cross[(source, target)]`` is the row-normalized ``(n_target,
        n_source)`` vote matrix carrying labels from ``source`` to ``target``.
    within : dict
        ``within[name]`` is the leave-one-out ``(n, n)`` vote matrix inside one
        dataset, the estimator behind every ``ceiling``.
    null_features : dict
        Each dataset's own (unbridged) feature matrix, where the null
        tessellations are drawn. Empty disables the null.
    strata : dict
        Each dataset's per-cell stratum (``None`` for none), in row order, for
        the within-strata readings. Empty disables them.
    refine : dict
        Each dataset's row-normalized transition matrix over its *own*
        features, over which every vote landing on that dataset is diffused
        (see :meth:`build`). Empty disables refinement.
    refine_alpha : float
        Weight of the diffused field against the bridge vote at each step.
    unbridged : dict
        Per dataset, ids of population cells left out of the bridge because they
        have no bridge coordinates (missing embedding values). Their codes are
        ignored when a clustering is aligned to the bridge, rather than raising;
        any other unknown id still raises.
    held_out : dict
        Per dataset, ids of population cells outside the ``mask`` the bridge was
        built over — the other half of a selection / evaluation split. Ignored
        when aligning, like :attr:`unbridged`, but not counted against coverage.
    n_neighbors : int
        Votes per cell.
    mutual : bool
        Whether only mutual neighbour edges vote.
    seed : int
        Seeds the null draws.
    """

    names: tuple[str, str]
    cell_ids: dict[str, np.ndarray]
    cross: dict[tuple[str, str], csr_matrix]
    within: dict[str, csr_matrix]
    null_features: dict[str, np.ndarray] = field(default_factory=dict)
    strata: dict[str, np.ndarray] = field(default_factory=dict)
    refine: dict[str, csr_matrix] = field(default_factory=dict)
    refine_alpha: float = 0.5
    unbridged: dict[str, np.ndarray] = field(default_factory=dict)
    held_out: dict[str, np.ndarray] = field(default_factory=dict)
    n_neighbors: int = 30
    mutual: bool = True
    seed: int = 0

    @classmethod
    def build(
        cls,
        coords: dict[str, np.ndarray],
        cell_ids: dict[str, np.ndarray],
        *,
        n_neighbors: int = 30,
        mutual: bool = True,
        null_features: dict[str, np.ndarray] | None = None,
        strata: dict[str, np.ndarray] | None = None,
        refine_features: dict[str, np.ndarray] | None = None,
        refine_alpha: float = 0.5,
        unbridged: dict[str, np.ndarray] | None = None,
        held_out: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ) -> "TransferBridge":
        """Build the vote matrices from each dataset's bridge coordinates.

        Parameters
        ----------
        coords : dict of str to numpy.ndarray
            Exactly two datasets' ``(n_cells, d)`` coordinates in the shared
            bridge space, keyed by name.
        cell_ids : dict of str to numpy.ndarray
            Row-aligned cell ids per dataset.
        n_neighbors : int, default 30
            Votes per cell, both across datasets and within each.
        mutual : bool, default True
            Keep only edges both ends name. Across datasets this is what lets a
            cell with no counterpart abstain; within a dataset it keeps the
            ceiling on the same footing.
        null_features : dict of str to numpy.ndarray, optional
            Each dataset's own feature matrix for the null tessellations.
        strata : dict of str to numpy.ndarray, optional
            Each dataset's per-cell stratum, for the within-strata readings.
            Values are compared as strings; ``None`` marks no stratum.
        refine_features : dict of str to numpy.ndarray, optional
            Each dataset's own feature matrix (the space its clustering used).
            When given, every vote landing on a dataset — across, within for
            the ceiling, and for the null alike — is then diffused over that
            dataset's mutual-kNN graph in these features: label spreading
            with the bridge vote as the prior, ``F ← α·T·F + (1 − α)·F₀``.
            The bridge says roughly where a cluster lands; the dataset's own
            sharper geometry decides its edges. Diffusion over the target's
            graph pulls *any* labelling toward its structure, which would
            flatter every partition alike — so it is applied identically to
            the ceiling and the null, and read through them. Cells with no
            bridge vote stay abstained: refinement sharpens, it never extends
            coverage onto a type the other dataset lacks.
        unbridged : dict of str to numpy.ndarray, optional
            Ids of cells in each population that have no bridge coordinates and
            so are left out; see the attribute of the same name.
        held_out : dict of str to numpy.ndarray, optional
            Ids of population cells deliberately left out of this bridge; see
            the attribute of the same name.
        refine_alpha : float, default 0.5
            ``α`` above: how far the diffused field may drift from the vote.
            It sets a diffusion length (about ``α / (1 − α)`` graph steps)
            that does not shrink with grain, so a large ``α`` smooths small
            clusters into their neighbours and pulls the suggested band
            coarser. On MICrONS/V1DD inhibitory cells ``0.5`` raised the
            within-strata excess at every grain while leaving the verdict
            counts at fine grain where they were without refinement;
            ``0.8`` raised it further but cost a quarter of the reproducing
            clusters from k≈23 up.
        seed : int, default 0
            Seeds the null draws.

        Returns
        -------
        TransferBridge
            The frozen vote matrices.

        Raises
        ------
        ValueError
            If not exactly two datasets are given, or ids and rows disagree.
        """
        if len(coords) != 2:
            raise ValueError(f"a bridge joins exactly two datasets, got {list(coords)}")
        a, b = coords
        for name in (a, b):
            if np.asarray(cell_ids[name]).shape[0] != coords[name].shape[0]:
                raise ValueError(
                    f"{name!r}: {len(cell_ids[name])} ids for "
                    f"{coords[name].shape[0]} rows"
                )
        strata = {
            name: np.array(
                [None if v is None else str(v) for v in values], dtype=object
            )
            for name, values in (strata or {}).items()
        }
        if strata and set(strata) != {a, b}:
            raise ValueError(f"strata must cover both datasets {a!r} and {b!r}")
        for name, values in strata.items():
            if values.size != coords[name].shape[0]:
                raise ValueError(
                    f"{name!r}: {values.size} strata for {coords[name].shape[0]} rows"
                )
        if not 0.0 <= refine_alpha < 1.0:
            raise ValueError(f"refine_alpha must be in [0, 1), got {refine_alpha}")
        refine = {}
        if refine_features:
            from cellpax.propagate import _transition_matrix

            if set(refine_features) != {a, b}:
                raise ValueError(
                    f"refine_features must cover both datasets {a!r} and {b!r}"
                )
            for name, features in refine_features.items():
                features = np.asarray(features, dtype=float)
                if features.shape[0] != coords[name].shape[0]:
                    raise ValueError(
                        f"{name!r}: {features.shape[0]} refine rows for "
                        f"{coords[name].shape[0]} cells"
                    )
                refine[name] = csr_matrix(
                    _transition_matrix(features, n_neighbors, True, "uniform")
                )
        za = np.asarray(coords[a], dtype=float)
        zb = np.asarray(coords[b], dtype=float)
        # b's queries into a, and a's into b; an edge is mutual when both name it
        b_into_a = _connectivity(_query(za, zb, n_neighbors), za.shape[0])
        a_into_b = _connectivity(_query(zb, za, n_neighbors), zb.shape[0])
        if mutual:
            edges = csr_matrix(b_into_a.multiply(a_into_b.T))
            to_b, to_a = edges, csr_matrix(edges.T)
        else:
            to_b, to_a = b_into_a, a_into_b
        within = {}
        for name, z in ((a, za), (b, zb)):
            k = int(min(n_neighbors, z.shape[0] - 1))
            _, indices = _neighbor_arrays(z, k, "minkowski")
            directed = _connectivity(indices, z.shape[0])
            pattern = directed.multiply(directed.T) if mutual else directed
            within[name] = _row_normalized(csr_matrix(pattern))
        return cls(
            names=(a, b),
            cell_ids={a: np.asarray(cell_ids[a]), b: np.asarray(cell_ids[b])},
            cross={(a, b): _row_normalized(to_b), (b, a): _row_normalized(to_a)},
            within=within,
            null_features=dict(null_features or {}),
            strata=strata,
            refine=refine,
            refine_alpha=float(refine_alpha),
            unbridged={
                name: np.asarray(ids) for name, ids in (unbridged or {}).items()
            },
            held_out={name: np.asarray(ids) for name, ids in (held_out or {}).items()},
            n_neighbors=int(n_neighbors),
            mutual=bool(mutual),
            seed=int(seed),
        )

    def _other(self, name: str) -> str:
        if name not in self.names:
            raise KeyError(f"unknown dataset {name!r}; this bridge joins {self.names}")
        return self.names[1] if name == self.names[0] else self.names[0]

    def carry(
        self, codes: np.ndarray, source: str, *, min_confidence: float | None = None
    ) -> Carry:
        """Carry one dataset's per-cell codes onto the other dataset's cells.

        Parameters
        ----------
        codes : numpy.ndarray
            Integer codes over ``source``'s cells in bridge row order, ``-1``
            for unassigned (those cells cast no vote).
        source : str
            Dataset the codes belong to.
        min_confidence : float, optional
            Abstain where the winning share falls below this.

        Returns
        -------
        Carry
            Soft shares, hard codes and confidence on the other dataset's cells.
        """
        codes = self._check_codes(codes, source)
        return self._carry(source, self._other(source), codes, min_confidence)

    def ceiling(
        self, codes: np.ndarray, dataset: str, *, min_confidence: float | None = None
    ) -> Carry:
        """Carry codes through the bridge within their own dataset, leaving each cell out.

        The same estimator as :meth:`carry` with no dataset boundary in the way,
        so its agreement with ``codes`` is the most a cross-dataset carry of this
        partition could achieve through this bridge.

        Parameters
        ----------
        codes : numpy.ndarray
            Integer codes over ``dataset``'s cells in bridge row order.
        dataset : str
            Dataset the codes belong to.
        min_confidence : float, optional
            Abstain where the winning share falls below this.

        Returns
        -------
        Carry
            The leave-one-out carry onto the same cells.
        """
        codes = self._check_codes(codes, dataset)
        return self._carry(dataset, dataset, codes, min_confidence)

    def _votes(
        self,
        source: str,
        target: str,
        codes: np.ndarray,
        min_confidence: float | None,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """``(soft, hard, confidence, clusters)`` for one carry, refined if set.

        Every cross reading — real partitions and the null alike — goes through
        here, which is what makes refinement a property of the estimator rather
        than of whichever partition it happens to flatter.

        The within-dataset ceiling (``source == target``) is never refined.
        Diffusion would hand each cell's own label back to it through its
        neighbours — the ceiling would stop being leave-one-out — and over
        nearly the graph its clusters were found on, so it would measure
        re-smoothing rather than what the bridge alone resolves. With
        refinement on, efficiency is therefore "the refined carry across over
        what Z alone recovers within", and can exceed one where the target's
        own geometry sharpens a blurred vote.
        """
        within = source == target
        votes = self.within[source] if within else self.cross[(source, target)]
        soft, hard, confidence, clusters = _soft_votes(votes, codes, min_confidence)
        if within or target not in self.refine or clusters.size == 0:
            return soft, hard, confidence, clusters
        from cellpax.propagate import _diffuse

        prior = soft.toarray()
        voted = prior.sum(axis=1) > 0
        field = _diffuse(
            self.refine[target],
            prior,
            np.zeros(0, dtype=np.int64),
            preserve_labeled=False,
            alpha=self.refine_alpha,
            max_iter=200,
            tol=1e-4,
        )
        # sharpen, never extend: a cell the bridge gave no vote stays abstained
        field[~voted] = 0.0
        total = field.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            field = np.where(total[:, None] > 0, field / total[:, None], 0.0)
        winner = field.argmax(axis=1)
        confidence = field[np.arange(field.shape[0]), winner]
        abstain = ~voted
        if min_confidence is not None:
            abstain |= confidence < min_confidence
        hard = np.where(abstain, _UNASSIGNED, clusters[winner])
        return field, hard, np.where(abstain, 0.0, confidence), clusters

    def _carry(
        self, source: str, target: str, codes: np.ndarray, min_confidence: float | None
    ) -> Carry:
        soft, hard, confidence, clusters = self._votes(
            source, target, codes, min_confidence
        )
        return Carry(_as_array(soft), hard, confidence, clusters)

    def _check_codes(self, codes: np.ndarray, name: str) -> np.ndarray:
        self._other(name)
        codes = np.asarray(codes).reshape(-1).astype(np.int64)
        if codes.size != self.cell_ids[name].size:
            raise ValueError(
                f"{name!r} has {self.cell_ids[name].size} cells but {codes.size} "
                f"codes were given"
            )
        return codes

    def align(
        self, codes: np.ndarray, cell_ids: np.ndarray, dataset: str
    ) -> np.ndarray:
        """Reorder per-cell codes into this bridge's row order for ``dataset``.

        Cells of the dataset missing from ``cell_ids`` become ``-1``. Ids of
        cells left out of the bridge for lack of bridge coordinates
        (:attr:`unbridged`) are dropped — they were clustered, they just cannot
        be carried. Any other id raises, since it means the codes were built on
        another population.

        Parameters
        ----------
        codes : numpy.ndarray
            Codes aligned to ``cell_ids``.
        cell_ids : numpy.ndarray
            Ids the codes are keyed by.
        dataset : str
            Which side of the bridge.

        Returns
        -------
        numpy.ndarray
            Codes in bridge row order.

        Raises
        ------
        ValueError
            If any id is not a cell of ``dataset``.
        """
        own = self.cell_ids[dataset]
        order = np.argsort(own, kind="stable")
        ids = np.asarray(cell_ids)
        slot = np.searchsorted(own[order], ids)
        slot = np.clip(slot, 0, own.size - 1)
        found = own[order][slot] == ids
        if not found.all():
            stranger = ~found
            for known in (self.unbridged.get(dataset), self.held_out.get(dataset)):
                if known is not None and known.size:
                    stranger &= ~np.isin(ids, known)
            if stranger.any():
                raise ValueError(
                    f"{int(stranger.sum())} ids are not cells of {dataset!r} in this "
                    f"bridge, e.g. {ids[stranger][:5].tolist()}"
                )
        aligned = np.full(own.size, _UNASSIGNED, dtype=np.int64)
        aligned[order[slot[found]]] = np.asarray(codes).reshape(-1)[found]
        return aligned

    def bridged_fraction(self) -> dict[str, float]:
        """Per dataset, the share of its population that the bridge covers.

        Returns
        -------
        dict
            Dataset name to bridged cells over bridged plus :attr:`unbridged` —
        the embedding's coverage of the cells this bridge is about; cells
        :attr:`held_out` by a ``mask`` are not counted either way.
        """
        out = {}
        for name in self.names:
            n = self.cell_ids[name].size
            missing = self.unbridged.get(name, np.zeros(0)).size
            out[name] = n / (n + missing) if n + missing else float("nan")
        return out

    def score(
        self,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        *,
        names_a: dict[int, str] | None = None,
        names_b: dict[int, str] | None = None,
        min_confidence: float | None = None,
        rule: VerdictRule | None = None,
    ) -> BridgeScore:
        """Score two per-dataset partitions against each other, both directions.

        Parameters
        ----------
        codes_a, codes_b : numpy.ndarray
            Each dataset's codes in bridge row order, ``-1`` for unassigned.
        names_a, names_b : dict of int to str, optional
            Cluster names for the per-cluster frame.
        min_confidence : float, optional
            Abstention floor for every carry, the ceilings included.
        rule : VerdictRule, optional
            Thresholds for the per-cluster verdicts; the defaults otherwise.

        Returns
        -------
        BridgeScore
            Summary and per-cluster readings.
        """
        rule = rule or VerdictRule()
        a, b = self.names
        codes = {a: self._check_codes(codes_a, a), b: self._check_codes(codes_b, b)}
        names = {a: names_a, b: names_b}
        summary: dict[str, float] = {}
        sides = {}
        for source, target, key in ((a, b, "ab"), (b, a, "ba")):
            src, tgt = codes[source], codes[target]
            carried = self.carry(src, source, min_confidence=min_confidence)
            ceiling = self.ceiling(src, source, min_confidence=min_confidence)
            clusters, sizes = np.unique(src[src >= 0], return_counts=True)
            sides[source] = {
                "clusters": clusters,
                "sizes": sizes,
                "ceiling": _ceiling_purity(ceiling.soft, src, sizes)
                if clusters.size
                else np.zeros(0),
                "cross": _cross_mass(carried.soft, tgt),
            }
            ami = _ami(tgt, carried.codes)
            ceil = _ami(src, ceiling.codes)
            summary[f"ami_{key}"] = ami
            summary[f"ceiling_{key}"] = ceil
            summary[f"efficiency_{key}"] = ami / ceil if ceil > 0 else float("nan")
            summary[f"coverage_{key}"] = _coverage(tgt, carried.codes)
            if self.strata:
                summary[f"strata_ami_{key}"] = _strata_ami(
                    tgt, carried.codes, self.strata[target]
                )
                summary[f"strata_ceiling_{key}"] = _strata_ami(
                    src, ceiling.codes, self.strata[source]
                )
        side_ab, side_ba = _two_sided(
            sides[a]["cross"], sides[b]["cross"], sides[a], sides[b], rule
        )
        frames = [
            _cluster_frame(
                side,
                clusters=sides[source]["clusters"],
                sizes=sides[source]["sizes"],
                ceiling=sides[source]["ceiling"],
                target_clusters=sides[target]["clusters"],
                direction=f"{source}->{target}",
                names=names[source],
                target_names=names[target],
                rule=rule,
            )
            for side, source, target in ((side_ab, a, b), (side_ba, b, a))
        ]
        clusters = pl.concat(frames, how="diagonal_relaxed")
        for key, frame in zip(("ab", "ba"), frames):
            worst = frame["efficiency"].fill_nan(None).min()
            summary[f"worst_efficiency_{key}"] = (
                float("nan") if worst is None else float(worst)
            )
        summary["worst_efficiency"] = float(
            np.nanmin([summary["worst_efficiency_ab"], summary["worst_efficiency_ba"]])
        )
        summary["ami"] = float(np.nanmean([summary["ami_ab"], summary["ami_ba"]]))
        summary["efficiency"] = float(
            np.nanmean([summary["efficiency_ab"], summary["efficiency_ba"]])
        )
        summary.update(_shares(clusters))
        if self.strata:
            for reading in ("strata_ami", "strata_ceiling"):
                summary[reading] = _nanmean(
                    summary[f"{reading}_ab"], summary[f"{reading}_ba"]
                )
        return BridgeScore(summary, clusters)

    def score_labels(
        self,
        labels_a: "LabelSet",
        labels_b: "LabelSet",
        *,
        min_confidence: float | None = None,
        rule: VerdictRule | None = None,
    ) -> BridgeScore:
        """:meth:`score` for two ``LabelSet``\\ s, aligned to the bridge by cell id.

        Parameters
        ----------
        labels_a, labels_b : LabelSet
            One label set per dataset. Cells outside a set count as unassigned.
        min_confidence : float, optional
            Abstention floor for every carry.
        rule : VerdictRule, optional
            Thresholds for the per-cluster verdicts.

        Returns
        -------
        BridgeScore
            Summary and per-cluster readings, with cluster names.
        """
        a, b = self.names
        codes_a = labels_a.codes_for(self.cell_ids[a])
        codes_b = labels_b.codes_for(self.cell_ids[b])
        return self.score(
            codes_a,
            codes_b,
            names_a={i: m.name for i, m in labels_a.meta.items()},
            names_b={i: m.name for i, m in labels_b.meta.items()},
            min_confidence=min_confidence,
            rule=rule,
        )

    def _null_codes(
        self, dataset: str, k: int, draws: int, null: Null, rng: np.random.Generator
    ) -> list[np.ndarray]:
        features = self.null_features[dataset]
        out = []
        for _ in range(draws):
            if null == "kmeans":
                from sklearn.cluster import KMeans

                model = KMeans(n_clusters=k, n_init=1, random_state=rng.integers(2**31))
                out.append(model.fit_predict(features).astype(np.int64))
            else:
                seeds = rng.choice(features.shape[0], size=k, replace=False)
                nearest = NearestNeighbors(n_neighbors=1).fit(features[seeds])
                out.append(
                    nearest.kneighbors(features, return_distance=False)[:, 0].astype(
                        np.int64
                    )
                )
        return out

    def score_runs(
        self,
        clustering_a: "Clustering",
        clustering_b: "Clustering",
        *,
        band_ratio: float = 1.5,
        null: Null | None = "voronoi",
        null_draws: int = 3,
        min_confidence: float | None = None,
        rule: VerdictRule | None = None,
        seed: int | None = None,
    ) -> "GrainCorrespondence":
        """Score every pair of stored runs at matching grain, with no consensus rebuild.

        Each run of each clustering is carried across once (and through its own
        dataset once, for the ceiling); pairs are then scored by AMI. Runs are
        paired by *realized* cluster count — resolution means different things at
        different dataset sizes — keeping pairs whose counts are within
        ``band_ratio`` of each other.

        Parameters
        ----------
        clustering_a, clustering_b : Clustering
            Per-dataset clusterings carrying ``partitions``, one on each side of
            the bridge (matched by cell id).
        band_ratio : float, default 1.5
            Largest ratio of realized cluster counts for a pair to be scored.
        null : {'voronoi', 'kmeans'} or None, default 'voronoi'
            ``'voronoi'`` draws random tessellations (nearest of ``k`` random
            seed cells, independently per dataset) in each dataset's own
            features: the chance agreement of a smooth partition with no
            correspondence. ``'kmeans'`` is a competitor rather than a null —
            on genuinely discrete types it finds them too, so excess over it
            reads "beats the simplest partitioner", not "is real". ``None``
            skips it.
        null_draws : int, default 3
            Null tessellations per dataset per realized count.
        min_confidence : float, optional
            Abstention floor for every carry.
        rule : VerdictRule, optional
            Thresholds for the per-cluster verdict counts.
        seed : int, optional
            Seeds the null; defaults to the bridge's seed.

        Returns
        -------
        GrainCorrespondence
            One row per scored run pair.

        Raises
        ------
        ValueError
            If a clustering has no partitions, or a null is requested without
            null features.
        """
        parts = {}
        for name, clus in zip(self.names, (clustering_a, clustering_b)):
            partitions = clus.partitions
            if partitions is None:
                raise ValueError(
                    f"the {name!r} clustering carries no partitions to score"
                )
            parts[name] = [
                (
                    self.align(partitions.labels[:, column], clus.cell_ids, name),
                    {
                        "run": column,
                        "resolution": float(partitions.resolution[column]),
                        "n_neighbors": int(partitions.n_neighbors[column]),
                        "graph_type": str(partitions.graph_type[column]),
                    },
                )
                for column in range(partitions.n_runs)
            ]
        return self._score_partitions(
            parts,
            band_ratio=band_ratio,
            null=null,
            null_draws=null_draws,
            min_confidence=min_confidence,
            rule=rule,
            seed=seed,
        )

    def score_consensus(
        self,
        clustering_a: "Clustering",
        clustering_b: "Clustering",
        *,
        band: dict[str, Any] | None = None,
        resolution_min: float | tuple[float, float] | None = None,
        resolution_max: float | tuple[float, float] | None = None,
        heights: Any = None,
        n_levels: int = 8,
        min_cluster_size: int = 1,
        band_ratio: float = 1.5,
        null: Null | None = "voronoi",
        null_draws: int = 3,
        min_confidence: float | None = None,
        rule: VerdictRule | None = None,
        seed: int | None = None,
    ) -> "GrainCorrespondence":
        """Score the consensus cuts of each dataset's chosen band, end to end.

        The step after :meth:`score_runs`: narrow each clustering to the runs
        of its band with :meth:`~cellpax.Clustering.restrict` (resolution min
        locked, max from the band), cut the re-consensed hierarchy at every
        :meth:`~cellpax.Clustering.nested_labels` level, and score the level
        pairs exactly as runs are scored — same readings, same null. Consensus
        cuts should reproduce at least as well as the runs they average; this
        is where that is checked before a cut is adopted.

        Each restrict rebuilds a consensus matrix, which at tens of thousands
        of cells is several GB; the two sides are built one at a time and
        each is released once its levels are read.

        Parameters
        ----------
        clustering_a, clustering_b : Clustering
            The same per-dataset clusterings :meth:`score_runs` was given.
        band : dict, optional
            A :meth:`GrainCorrespondence.suggest_band` result; supplies each
            dataset's ``resolution_max`` (the top of its band).
        resolution_min, resolution_max : float or (float, float), optional
            Explicit bounds, one value for both datasets or one per dataset in
            bridge order. ``resolution_max`` overrides ``band``.
        heights : sequence of float, optional
            Cut heights for ``nested_labels``; its defaults otherwise.
        n_levels : int, default 8
            Default cut count when ``heights`` is not given.
        min_cluster_size : int, default 1
            Size floor at every level.
        band_ratio : float, default 1.5
            Largest ratio of realized cluster counts for a level pair.
        null : {'voronoi', 'kmeans'} or None, default 'voronoi'
            The null, as in :meth:`score_runs`.
        null_draws : int, default 3
            Null tessellations per dataset per realized count.
        min_confidence : float, optional
            Abstention floor for every carry.
        rule : VerdictRule, optional
            Thresholds for the per-cluster verdict counts.
        seed : int, optional
            Seeds the null; defaults to the bridge's seed.

        Returns
        -------
        GrainCorrespondence
            One row per level pair: ``level_*``, ``height_*`` and
            ``coclustering_frequency_*`` per side alongside every reading.
        """
        a, b = self.names
        if resolution_max is None and band is not None:
            resolution_max = (
                band[f"resolution_{a}"][1],
                band[f"resolution_{b}"][1],
            )
        lows = _per_side(resolution_min)
        highs = _per_side(resolution_max)
        parts = {}
        for i, (name, clus) in enumerate(zip(self.names, (clustering_a, clustering_b))):
            narrowed = (
                clus.restrict(resolution_min=lows[i], resolution_max=highs[i])
                if lows[i] is not None or highs[i] is not None
                else clus
            )
            levels = narrowed.nested_levels(
                heights, n_levels=n_levels, min_cluster_size=min_cluster_size
            )
            labels = narrowed.nested_labels(
                heights, n_levels=n_levels, min_cluster_size=min_cluster_size
            )
            ids = narrowed.cell_ids
            parts[name] = [
                (
                    self.align(labels[f"level_{row['level']}"].to_numpy(), ids, name),
                    {
                        "level": int(row["level"]),
                        "height": float(row["height"]),
                        "coclustering_frequency": float(row["coclustering_frequency"]),
                        "resolution": float(highs[i])
                        if highs[i] is not None
                        else float("nan"),
                    },
                )
                for row in levels.iter_rows(named=True)
            ]
            del narrowed  # the consensus matrix is the largest object in the session
        return self._score_partitions(
            parts,
            band_ratio=band_ratio,
            null=null,
            null_draws=null_draws,
            min_confidence=min_confidence,
            rule=rule,
            seed=seed,
        )

    def _score_partitions(
        self,
        parts: dict[str, list[tuple[np.ndarray, dict[str, Any]]]],
        *,
        band_ratio: float,
        null: Null | None,
        null_draws: int,
        min_confidence: float | None,
        rule: VerdictRule | None,
        seed: int | None,
    ) -> "GrainCorrespondence":
        """Score every matched-grain pair of two sides' partitions, with the null."""
        rule = rule or VerdictRule()
        a, b = self.names
        if null is not None and not self.null_features:
            raise ValueError(
                "this bridge has no null_features; build it with them or pass null=None"
            )

        def entry(codes: np.ndarray, name: str) -> dict[str, Any]:
            # real partitions and null draws go through one path: the same carry,
            # the same ceiling, the same refinement, so their readings compare
            codes = self._check_codes(codes, name)
            cross = self._votes(name, self._other(name), codes, min_confidence)
            ceiling = self._votes(name, name, codes, min_confidence)
            return _run_entry(
                codes, cross[:2], ceiling[:2], strata=self.strata.get(name)
            )

        scored = {
            name: [
                {"meta": meta, "k": _n_clusters(codes), **entry(codes, name)}
                for codes, meta in parts[name]
            ]
            for name in (a, b)
        }

        rng = np.random.default_rng(self.seed if seed is None else seed)
        null_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

        def nulls(name: str, k: int) -> list[dict[str, Any]]:
            if (name, k) not in null_cache:
                null_cache[(name, k)] = [
                    entry(codes, name)
                    for codes in self._null_codes(name, k, null_draws, null, rng)
                ]
            return null_cache[(name, k)]

        pair_strata = (self.strata[a], self.strata[b]) if self.strata else None
        rows = []
        for ra in scored[a]:
            for rb in scored[b]:
                if min(ra["k"], rb["k"]) < 2:
                    continue
                if max(ra["k"], rb["k"]) / min(ra["k"], rb["k"]) > band_ratio:
                    continue
                row = {
                    **{f"{key}_a": value for key, value in ra["meta"].items()},
                    **{f"{key}_b": value for key, value in rb["meta"].items()},
                    "k_a": ra["k"],
                    "k_b": rb["k"],
                    **_pair(ra, rb, rule, pair_strata),
                }
                if null is not None:
                    draws = [
                        _pair(na, nb, rule, pair_strata)
                        for na, nb in zip(nulls(a, ra["k"]), nulls(b, rb["k"]))
                    ]
                    for column in _NULL_COLUMNS:
                        if column in draws[0]:
                            row[f"null_{column}"] = _nanmean(
                                *(d[column] for d in draws)
                            )
                rows.append(row)
        frame = pl.DataFrame(rows) if rows else pl.DataFrame()
        if rows and null is not None:
            excess = [
                (pl.col("ami") - pl.col("null_ami")).alias("excess"),
                (pl.col("n_reproduces") - pl.col("null_n_reproduces")).alias(
                    "reproduces_excess"
                ),
            ]
            if self.strata:
                excess.append(
                    (pl.col("strata_ami") - pl.col("null_strata_ami")).alias(
                        "strata_excess"
                    )
                )
            frame = frame.with_columns(excess)
        return GrainCorrespondence(
            frame, names=self.names, null=null, band_ratio=band_ratio
        )

    def __repr__(self) -> str:
        a, b = self.names
        refine = "off" if not self.refine else self.refine_alpha
        bridged = ""
        held = sum(ids.size for ids in self.held_out.values())
        if held:
            bridged = f"held out {held}, "
        if any(ids.size for ids in self.unbridged.values()):
            shares = ", ".join(
                f"{k}={v:.2f}" for k, v in self.bridged_fraction().items()
            )
            bridged += f"bridged {shares}, "
        return (
            f"TransferBridge({a}: {self.cell_ids[a].size} cells, "
            f"{b}: {self.cell_ids[b].size} cells, n_neighbors={self.n_neighbors}, "
            f"mutual={self.mutual}, refine={refine}, {bridged}"
            f"coverage {a}->{b}={_reachable(self.cross[(a, b)]):.2f}, "
            f"{b}->{a}={_reachable(self.cross[(b, a)]):.2f})"
        )


def _shares(clusters: pl.DataFrame) -> dict[str, float]:
    """Headline shares over *every* counted cluster, both directions pooled.

    The denominators are all clusters (and their cells) at or above
    ``min_cells`` — never only the judgeable ones — so an analysis whose
    splits the bridge mostly cannot see does not read as one that mostly
    reproduces.
    """
    counted = clusters.filter(pl.col("verdict") != "small")
    if counted.is_empty():
        return {
            "reproduce_share": float("nan"),
            "anchored_share": float("nan"),
            "anchored_cells": float("nan"),
        }
    anchored = pl.col("verdict").is_in(["reproduces", "weak", "nested", "split"])
    cells = counted["n_cells"].sum()
    return {
        "reproduce_share": float((counted["verdict"] == "reproduces").mean()),
        "anchored_share": float(counted.select(anchored.mean()).item()),
        "anchored_cells": float(counted.filter(anchored)["n_cells"].sum() / cells)
        if cells
        else float("nan"),
    }


def _per_side(value: Any) -> tuple[Any, Any]:
    """One bound for both sides, or a ``(a, b)`` pair, as a pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected one value or one per dataset, got {value!r}")
        return value[0], value[1]
    return value, value


def _reachable(votes: csr_matrix) -> float:
    """Share of target cells with at least one cross edge."""
    return float((np.diff(votes.indptr) > 0).mean()) if votes.shape[0] else 0.0


def _coverage(target: np.ndarray, carried: np.ndarray) -> float:
    """Co-assigned share of the target's assigned cells."""
    assigned = target >= 0
    if not assigned.any():
        return float("nan")
    return float(((carried >= 0) & assigned).sum() / assigned.sum())


def _nanmean(*values: float) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


#: Readings :meth:`GrainCorrespondence.curve` pools, when present.
_CURVE_METRICS = (
    "ami",
    "null_ami",
    "excess",
    "efficiency",
    "worst_efficiency",
    "n_counted",
    "n_reproduces",
    "null_n_reproduces",
    "reproduces_excess",
    "n_weak",
    "n_nested",
    "n_split",
    "n_refuted",
    "n_unverifiable",
    "n_absent",
    "reproduce_share",
    "anchored_share",
    "anchored_cells",
    "strata_ami",
    "null_strata_ami",
    "strata_excess",
    "strata_ceiling",
)

#: Pair columns averaged over null draws into ``null_<column>``.
_NULL_COLUMNS = ("ami", "strata_ami", "n_reproduces", "n_refuted", "n_unverifiable")


def _run_entry(
    codes: np.ndarray,
    cross: tuple[csr_matrix, np.ndarray],
    ceiling: tuple[csr_matrix, np.ndarray],
    *,
    strata: np.ndarray | None = None,
) -> dict:
    """What a pair score needs from one partition: its carries, soft and hard."""
    clusters, sizes = np.unique(codes[codes >= 0], return_counts=True)
    soft_ceiling, hard_ceiling = ceiling
    entry = {
        "codes": codes,
        "soft": cross[0],
        "carried": cross[1],
        "ceiling": _ami(codes, hard_ceiling),
        "clusters": clusters,
        "sizes": sizes,
        "ceiling_purity": _ceiling_purity(soft_ceiling, codes, sizes)
        if clusters.size
        else np.zeros(0),
    }
    if strata is not None:
        entry["strata_ceiling"] = _strata_ami(codes, hard_ceiling, strata)
    return entry


def _pair_verdicts(
    ra: dict[str, Any], rb: dict[str, Any], rule: VerdictRule
) -> dict[str, float]:
    """Two-sided verdict counts, shares and worst efficiency for one partition pair.

    The same soft purities and the same :func:`_two_sided` check that
    :class:`BridgeScore` uses, so a run pair and a ``score`` of the same two
    partitions agree.
    """
    cross_ab = _cross_mass(ra["soft"], rb["codes"])
    cross_ba = _cross_mass(rb["soft"], ra["codes"])
    side_ab, side_ba = _two_sided(
        cross_ab,
        cross_ba,
        {"sizes": ra["sizes"], "ceiling": ra["ceiling_purity"]},
        {"sizes": rb["sizes"], "ceiling": rb["ceiling_purity"]},
        rule,
    )
    counts = dict.fromkeys(VERDICTS, 0)
    worst, n_cells, anchored_cells = [], 0, 0
    for side, entry in ((side_ab, ra), (side_ba, rb)):
        counted = entry["sizes"] >= rule.min_cells
        for verdict, size in zip(side["verdict"][counted], entry["sizes"][counted]):
            counts[verdict] += 1
            n_cells += int(size)
            if verdict in ("reproduces", "weak", "nested", "split"):
                anchored_cells += int(size)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (side["purity"] / entry["ceiling_purity"])[counted]
        if np.isfinite(ratio).any():
            worst.append(float(np.nanmin(ratio)))
    n_counted = sum(counts.values())
    anchored = sum(counts[v] for v in ("reproduces", "weak", "nested", "split"))
    return {
        "worst_efficiency": min(worst) if worst else float("nan"),
        "n_counted": n_counted,
        **{f"n_{v}": counts[v] for v in VERDICTS},
        "reproduce_share": counts["reproduces"] / n_counted
        if n_counted
        else float("nan"),
        "anchored_share": anchored / n_counted if n_counted else float("nan"),
        "anchored_cells": anchored_cells / n_cells if n_cells else float("nan"),
    }


def _pair(
    ra: dict[str, Any],
    rb: dict[str, Any],
    rule: VerdictRule,
    strata: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, float]:
    """Both directions of every pair reading, averaged or summed across them."""
    ami_ab = _ami(rb["codes"], ra["carried"])
    ami_ba = _ami(ra["codes"], rb["carried"])
    eff_ab = ami_ab / ra["ceiling"] if ra["ceiling"] > 0 else float("nan")
    eff_ba = ami_ba / rb["ceiling"] if rb["ceiling"] > 0 else float("nan")
    row = {
        "ami_ab": ami_ab,
        "ami_ba": ami_ba,
        "ami": _nanmean(ami_ab, ami_ba),
        "ceiling_a": ra["ceiling"],
        "ceiling_b": rb["ceiling"],
        "efficiency": _nanmean(eff_ab, eff_ba),
        "coverage_ab": _coverage(rb["codes"], ra["carried"]),
        "coverage_ba": _coverage(ra["codes"], rb["carried"]),
        **_pair_verdicts(ra, rb, rule),
    }
    if strata is not None:
        a_strata, b_strata = strata
        row["strata_ami"] = _nanmean(
            _strata_ami(rb["codes"], ra["carried"], b_strata),
            _strata_ami(ra["codes"], rb["carried"], a_strata),
        )
        row["strata_ceiling"] = _nanmean(ra["strata_ceiling"], rb["strata_ceiling"])
    return row


class GrainCorrespondence:
    """Cross-dataset agreement of every matched-grain pair of stored runs.

    Parameters
    ----------
    frame : polars.DataFrame
        One row per scored run pair — the output of
        :meth:`TransferBridge.score_runs`.
    names : tuple of str
        The bridge's dataset names, ``(a, b)``.
    null : str or None
        Which null the ``null_ami`` and ``excess`` columns came from.
    band_ratio : float, optional
        The pairing limit the pairs were scored under: no pair's realized
        cluster counts differ by more than this factor, so no reading here can
        say anything about a pairing beyond it.

    Notes
    -----
    Three readings per pair, each answering a different question:

    ``excess`` (``ami - null_ami``)
        Does this pair agree beyond chance for smooth partitions at this grain?
        The reproducibility score.
    ``efficiency`` (``ami`` over the source partitions' within-dataset ceilings)
        Is a shortfall in ``ami`` the bridge's blur? ``ami`` falls with grain
        whether or not finer splits reproduce, because a blurring bridge
        misplaces more cells the more boundaries there are; efficiency stays
        near one when the cross-dataset carry loses no more than the bridge
        loses within a dataset.
    ``worst_efficiency``
        The same, for the single worst cluster of either partition (soft
        purities, as in :class:`BridgeScore`). Global AMI is carried by shared coarse structure, so one
        refuted fine split barely moves it; this is the column that sees it.

    Global AMI is a poor guide to grain: its excess over the null peaks at
    the coarsest split both datasets share and falls from there, because
    random fine tessellations of the same cells share more and more
    information. Two readings follow fine structure instead:

    ``n_reproduces`` / ``reproduces_excess``
        Clusters (both partitions, counted per :class:`VerdictRule`) whose
        split reappears as cleanly as the bridge can show, and that count
        over the null's. It rises while finer splits keep reproducing and
        stops when they don't — the grain readout :meth:`suggest_band` uses.
    ``strata_ami`` / ``strata_excess``
        With strata on the bridge, AMI within each stratum: whether the
        subdivision *inside* a subclass reproduces, free of the subclasses
        themselves.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        names: tuple[str, str],
        null: str | None,
        band_ratio: float | None = None,
    ) -> None:
        self.frame = frame
        self.names = names
        self.null = null
        self.band_ratio = band_ratio

    def _default_metric(self, *, matching: bool = False) -> str:
        """``ami`` for matching partitions, ``reproduces_excess`` for choosing grain."""
        if matching:
            return "ami"
        return (
            "reproduces_excess"
            if "reproduces_excess" in self.frame.columns
            else "n_reproduces"
        )

    def pairs(
        self, *, metric: str | None = None, frame: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """Every scored pair of nominal resolutions, pooled over seeds.

        Runs repeated at one resolution (``n_times``) differ only by seed, so a
        resolution pair is the unit you actually choose between; this pools
        them. Pivot it on ``resolution_b`` for the heatmap view.

        Parameters
        ----------
        metric : str, optional
            The reading summarized with a standard error. Defaults to
            ``reproduces_excess`` when a null was drawn.
        frame : polars.DataFrame, optional
            A filtered copy of :attr:`frame` to pool instead of all of it.

        Returns
        -------
        polars.DataFrame
            One row per ``(resolution_a, resolution_b)``: run-pair count,
            median realized ``k_a`` / ``k_b``, ``metric`` and ``metric_se``, and
            the means of the other headline readings present. Sorted by
            ``metric``, best first.
        """
        frame = self.frame if frame is None else frame
        if frame.is_empty():
            return frame
        metric = metric or self._default_metric()
        if metric not in frame.columns:
            raise ValueError(f"{metric!r} is not a column of the pair frame")
        extra = [
            m
            for m in (
                "ami",
                "excess",
                "efficiency",
                "strata_excess",
                "n_reproduces",
                "n_refuted",
                "n_unverifiable",
                "n_absent",
            )
            if m in frame.columns and m != metric
        ]
        pooled = frame.group_by("resolution_a", "resolution_b").agg(
            n_pairs=pl.len(),
            k_a=pl.col("k_a").median(),
            k_b=pl.col("k_b").median(),
            **{metric: pl.col(metric).mean()},
            _var=pl.col(metric).var(),
            **{m: pl.col(m).mean() for m in extra},
        )
        fallback = pooled.select(pl.col("_var").fill_nan(None).mean()).item()
        fallback = float(np.sqrt(fallback)) if fallback is not None else 0.0
        se = (
            pl.when(pl.col("n_pairs") > 1)
            .then(pl.col("_var").sqrt())
            .otherwise(pl.lit(fallback))
            .fill_nan(fallback)
            .fill_null(fallback)
            / pl.col("n_pairs").sqrt()
        )
        return (
            pooled.with_columns(se.alias(f"{metric}_se"))
            .drop("_var")
            .select(
                "resolution_a",
                "resolution_b",
                "n_pairs",
                "k_a",
                "k_b",
                metric,
                f"{metric}_se",
                *extra,
            )
            .sort(metric, descending=True, nulls_last=True)
        )

    def complement(
        self,
        side: str | None = None,
        *,
        resolution_min: float | None = None,
        resolution_max: float | None = None,
        n_clusters_min: int | None = None,
        n_clusters_max: int | None = None,
        metric: str | None = None,
    ) -> dict[str, Any]:
        """The other dataset's band that corresponds to a band you chose in one.

        Choose a band in ``side`` — by nominal resolution, realized cluster
        count, or both, the same selectors as
        :meth:`~cellpax.Clustering.restrict` — and for every resolution of
        ``side`` inside it, find the resolution of the other dataset whose
        runs agree with it best. The complementary band spans those best
        partners; the *tolerant* band spans every partner within one standard
        error of each best, which is the honest width when the ridge is flat.

        The match is judged on ``ami`` by default, and deliberately not on
        the grain readings. Choosing *which* partner corresponds and choosing
        *how fine* to cut are different questions, and the grain readings are
        biased for the first: ``excess`` subtracts a null whose AMI rises with
        the partner's cluster count (fine random tessellations share more
        information), so it pulls every match toward the coarsest shared
        split; the verdict counts grow with the partner's cluster count, so
        they pull toward the finest. Chance-adjusted AMI has neither pull. On
        MICrONS/V1DD inhibitory cells it tracked matched structure (partner
        k ≈ own k from 4 to 34 clusters) where ``excess`` stuck at 5, 9 and
        22 and ``reproduces_excess`` chose partners twice as fine.

        Parameters
        ----------
        side : str, optional
            The dataset the band is chosen in; defaults to the first.
        resolution_min, resolution_max : float, optional
            Nominal resolution bounds of the band in ``side``.
        n_clusters_min, n_clusters_max : int, optional
            Realized cluster-count bounds of the band in ``side``.
        metric : str, optional
            The pair reading that decides each best partner. Defaults to
            ``ami``; see above before changing it.

        Returns
        -------
        dict
            ``side``, ``metric``, and for both datasets ``resolution_<name>``
            and ``k_<name>`` ranges — the chosen band's as realized, and the
            complement's — in the same shape as :meth:`suggest_band`, so
            either feeds ``restrict`` the same way;
            ``tolerant_resolution_<other>``; ``mean`` of ``metric`` over the
            best partners; and ``ridge``, a frame of each chosen resolution's
            best partner.

        Raises
        ------
        ValueError
            If no scored pair has ``side`` inside the band.

        Warns
        -----
        UserWarning
            If the complement reaches the edge of the other dataset's sweep, or
            its partners' cluster counts press against the ``band_ratio`` the
            pairs were scored under — either way the true complement may lie
            beyond what was scored.
        """
        a, b = self.names
        side = side or a
        if side not in self.names:
            raise KeyError(
                f"unknown dataset {side!r}; this correspondence joins {self.names}"
            )
        here, there = ("a", "b") if side == a else ("b", "a")
        other = b if side == a else a
        metric = metric or self._default_metric(matching=True)
        res, k = pl.col(f"resolution_{here}"), pl.col(f"k_{here}")
        keep = pl.lit(True)
        if resolution_min is not None:
            keep &= res >= resolution_min
        if resolution_max is not None:
            keep &= res <= resolution_max
        if n_clusters_min is not None:
            keep &= k >= n_clusters_min
        if n_clusters_max is not None:
            keep &= k <= n_clusters_max
        inside = self.frame.filter(keep) if not self.frame.is_empty() else self.frame
        if inside.is_empty():
            raise ValueError(
                f"no scored pair has {side!r} inside that band; its scored "
                f"resolutions are {sorted(self.frame[f'resolution_{here}'].unique().to_list()) if not self.frame.is_empty() else []}"
            )
        pooled = self.pairs(metric=metric, frame=inside)
        best = (
            pooled.sort(metric, descending=True, nulls_last=True)
            .group_by(f"resolution_{here}", maintain_order=True)
            .first()
            .sort(f"resolution_{here}")
        )
        floors = best.select(
            f"resolution_{here}",
            (pl.col(metric) - pl.col(f"{metric}_se")).alias("_floor"),
        )
        tolerant = pooled.join(floors, on=f"resolution_{here}").filter(
            pl.col(metric) >= pl.col("_floor")
        )
        partner = best[f"resolution_{there}"]
        swept = self.frame[f"resolution_{there}"]
        if partner.max() >= swept.max() or partner.min() <= swept.min():
            edge = "finest" if partner.max() >= swept.max() else "coarsest"
            warnings.warn(
                f"the complement in {other!r} reaches the {edge} resolution its "
                f"sweep scored; the best partner may lie beyond it",
                stacklevel=2,
            )
        if self.band_ratio is not None:
            ratio = best.select(
                (
                    pl.max_horizontal("k_a", "k_b") / pl.min_horizontal("k_a", "k_b")
                ).max()
            ).item()
            if ratio is not None and ratio >= 0.9 * self.band_ratio:
                warnings.warn(
                    f"best partners differ in cluster count by up to {ratio:.2f}x, "
                    f"against the band_ratio={self.band_ratio} the pairs were scored "
                    f"under; rescore with a wider band_ratio so pairings beyond it "
                    f"can compete",
                    stacklevel=2,
                )
        return {
            "side": side,
            "metric": metric,
            f"resolution_{side}": (
                float(inside[f"resolution_{here}"].min()),
                float(inside[f"resolution_{here}"].max()),
            ),
            f"k_{side}": (
                int(inside[f"k_{here}"].min()),
                int(inside[f"k_{here}"].max()),
            ),
            f"resolution_{other}": (float(partner.min()), float(partner.max())),
            f"k_{other}": (
                int(best[f"k_{there}"].min()),
                int(best[f"k_{there}"].max()),
            ),
            f"tolerant_resolution_{other}": (
                float(tolerant[f"resolution_{there}"].min()),
                float(tolerant[f"resolution_{there}"].max()),
            ),
            "mean": float(best[metric].mean()),
            "ridge": best.select(
                f"resolution_{here}",
                f"resolution_{there}",
                "k_a",
                "k_b",
                "n_pairs",
                metric,
                f"{metric}_se",
            ),
        }

    def ridge(self, *, by: str = "ami") -> pl.DataFrame:
        """For each run of ``a``, its best-matching run of ``b``, summarized by grain.

        The best match is the ``b`` run with the highest ``by`` for that ``a``
        run — which grain on the other side the partition corresponds to. Rows
        are then grouped by ``a``'s realized cluster count.

        Parameters
        ----------
        by : str, default 'ami'
            Column that picks each run's best partner.

        Returns
        -------
        polars.DataFrame
            Per realized ``k_a``: run count, median matched ``k_b``, the
            resolution ranges on both sides, and the mean and spread of
            ``ami``, ``efficiency`` and (with a null) ``excess``.
        """
        if self.frame.is_empty():
            return self.frame
        best = (
            self.frame.sort(by, descending=True, nulls_last=True)
            .group_by("run_a", maintain_order=True)
            .first()
        )
        metrics = [
            m
            for m in ("ami", "efficiency", "excess", "n_reproduces", "strata_ami")
            if m in best.columns
        ]
        return (
            best.group_by("k_a")
            .agg(
                pl.len().alias("n_runs"),
                pl.col("k_b").median().alias("k_b"),
                pl.col("resolution_a").min().alias("resolution_a_min"),
                pl.col("resolution_a").max().alias("resolution_a_max"),
                pl.col("resolution_b").min().alias("resolution_b_min"),
                pl.col("resolution_b").max().alias("resolution_b_max"),
                *[pl.col(m).mean().alias(m) for m in metrics],
                *[pl.col(m).std().alias(f"{m}_sd") for m in metrics],
            )
            .sort("k_a")
        )

    def matched(self, *, by: str = "ami") -> pl.DataFrame:
        """The pairs that are a best correspondence for at least one of their runs.

        Each run of either dataset keeps only its best partner by ``by``; a
        pair survives if it is that for its ``a`` run, its ``b`` run, or both.
        This is what the grain readings should average over. ``band_ratio``
        has to be wide enough for a dataset that resolves one more type to
        pair at all, and that same width admits mismatched pairings — seven
        clusters against fourteen — which can only nest, never reproduce
        one-to-one, and would drag every bin they land in down.

        Parameters
        ----------
        by : str, default 'ami'
            The pairing reading; see :meth:`complement` for why AMI.

        Returns
        -------
        polars.DataFrame
            The surviving rows of :attr:`frame`.
        """
        if self.frame.is_empty() or by not in self.frame.columns:
            return self.frame
        keys = [
            (f"run_{s}" if f"run_{s}" in self.frame.columns else f"level_{s}")
            for s in ("a", "b")
        ]
        if not all(k in self.frame.columns for k in keys):
            return self.frame
        ranked = self.frame.with_row_index("_row")
        best = set()
        for key in keys:
            top = (
                ranked.sort(by, descending=True, nulls_last=True)
                .group_by(key, maintain_order=True)
                .first()
            )
            best.update(top["_row"].to_list())
        return ranked.filter(pl.col("_row").is_in(sorted(best))).drop("_row")

    def _binned(self, bin_ratio: float, matched: bool = True) -> pl.DataFrame:
        """The pair frame with each pair's log-spaced grain bin attached."""
        if bin_ratio <= 1.0:
            raise ValueError(f"bin_ratio must exceed 1, got {bin_ratio}")
        grain = (pl.col("k_a") * pl.col("k_b")).sqrt()
        step = float(np.log(bin_ratio))
        frame = self.matched() if matched else self.frame
        return frame.with_columns(
            (grain.log() / step).round().cast(pl.Int64).alias("_bin"),
            grain.alias("_grain"),
        )

    def curve(self, *, bin_ratio: float = 1.25, matched: bool = True) -> pl.DataFrame:
        """Pair readings pooled into log-spaced grain bins.

        A pair's grain is the geometric mean of its two realized cluster
        counts; pairs are pooled into bins a factor ``bin_ratio`` wide, so a
        fine end sampled by a few runs still pools enough pairs to have a
        spread. The ``_se`` columns are that spread over runs, seeds and
        partners; a bin holding a single pair gets the pooled within-bin
        standard deviation instead of a spurious zero.

        Parameters
        ----------
        bin_ratio : float, default 1.25
            Width of each grain bin, as a ratio of cluster counts.
        matched : bool, default True
            Pool only each run's best correspondence (:meth:`matched`), not
            every pair within ``band_ratio``.

        Returns
        -------
        polars.DataFrame
            Per bin: ``grain`` (its geometric centre), ``grain_min`` /
            ``grain_max`` of the pairs in it, ``n_pairs``, and the mean and
            standard error of every reading present (``ami``, ``excess``,
            ``n_reproduces``, ``reproduces_excess``, ``strata_ami``, …).
        """
        if self.frame.is_empty():
            return self.frame
        metrics = [m for m in _CURVE_METRICS if m in self.frame.columns]
        binned = self._binned(bin_ratio, matched)
        curve = (
            binned.group_by("_bin")
            .agg(
                pl.len().alias("n_pairs"),
                pl.col("_grain").min().round().cast(pl.Int64).alias("grain_min"),
                pl.col("_grain").max().round().cast(pl.Int64).alias("grain_max"),
                *[pl.col(m).mean().alias(m) for m in metrics],
                *[pl.col(m).var().alias(f"_{m}_var") for m in metrics],
            )
            .sort("_bin")
        )
        step = float(np.log(bin_ratio))
        columns = [
            (pl.col("_bin") * step).exp().round().cast(pl.Int64).alias("grain"),
        ]
        for m in metrics:
            pooled = curve.select(pl.col(f"_{m}_var").fill_nan(None).mean()).item()
            fallback = float(np.sqrt(pooled)) if pooled is not None else 0.0
            sd = pl.col(f"_{m}_var").sqrt()
            columns.append(
                pl.when(pl.col("n_pairs") > 1)
                .then(sd)
                .otherwise(pl.lit(fallback))
                .fill_nan(fallback)
                .fill_null(fallback)
                .truediv(pl.col("n_pairs").sqrt())
                .alias(f"{m}_se")
            )
        return curve.with_columns(columns).select(
            "grain",
            "grain_min",
            "grain_max",
            "n_pairs",
            *metrics,
            *[f"{m}_se" for m in metrics],
        )

    def suggest_band(
        self,
        *,
        metric: str | None = None,
        bin_ratio: float = 1.25,
        matched: bool = True,
    ) -> dict[str, Any]:
        """The grain bins whose ``metric`` is within one standard error of its peak.

        The one-standard-error rule on :meth:`curve`: take the peak bin, then
        every bin whose mean is at least the peak minus the peak's standard
        error. A flat top is a band, not a point, and the band is the honest
        answer — the finest grain that keeps reproducing is its upper end.

        Parameters
        ----------
        metric : str, optional
            The curve column to maximize. Defaults to ``reproduces_excess``
            (reproducing clusters over the null's) when a null was drawn, and
            ``n_reproduces`` otherwise.
        bin_ratio : float, default 1.25
            Grain bin width, as in :meth:`curve`.
        matched : bool, default True
            Read the curve over best correspondences only, as in :meth:`curve`.

        Returns
        -------
        dict
            ``metric``, ``peak_grain``, ``peak``, ``peak_se``,
            ``grain_min`` / ``grain_max`` of the band, ``at_edge`` (the band
            reaches the finest or coarsest grain scored, so the curve may not
            have turned over yet — extend the resolution sweep before reading
            the band as a peak), ``best_pair`` (the single resolution pair
            inside the band, pooled over seeds, with the highest ``ami`` — a
            joint choice, unlike the two ranges, made on AMI because the grain
            readings are biased as pairing criteria; see :meth:`complement`), and per dataset the realized
            ``k_*`` and nominal ``resolution_*`` ranges of the pairs inside it.
            The two ranges are each dataset's span over the band, not a
            pairing: to choose a band in one dataset and get its partner in the
            other, use :meth:`complement`.

        Raises
        ------
        ValueError
            If there are no scored pairs, or ``metric`` is not a curve column.
        """
        curve = self.curve(bin_ratio=bin_ratio, matched=matched)
        if curve.is_empty():
            raise ValueError("no run pairs were scored, so there is no band to suggest")
        if metric is None:
            metric = (
                "reproduces_excess"
                if "reproduces_excess" in curve.columns
                else "n_reproduces"
            )
        if metric not in curve.columns:
            raise ValueError(f"{metric!r} is not a curve column: {curve.columns}")
        ranked = curve.filter(
            pl.col(metric).is_not_nan() & pl.col(metric).is_not_null()
        )
        peak_row = ranked.sort(metric, descending=True).row(0, named=True)
        floor = peak_row[metric] - peak_row[f"{metric}_se"]
        band = ranked.filter(pl.col(metric) >= floor)
        binned = self._binned(bin_ratio, matched)
        step = float(np.log(bin_ratio))
        centres = (pl.col("_bin") * step).exp().round().cast(pl.Int64)
        inside = binned.filter(centres.is_in(band["grain"].implode()))
        if band["grain"].max() == ranked["grain"].max():
            edge: str | None = "finest"
        elif band["grain"].min() == ranked["grain"].min():
            edge = "coarsest"
        else:
            edge = None
        if edge is not None:
            warnings.warn(
                f"the {metric} band reaches the {edge} grain scored, so the curve "
                f"may not have turned over; extend the resolution sweep before "
                f"reading it as a peak",
                stacklevel=2,
            )
        a, b = self.names
        # the band is chosen on the grain reading; the pairing inside it on
        # AMI, which (unlike the grain readings) has no pull toward either end
        matching = self._default_metric(matching=True)
        if matching not in inside.columns:
            matching = metric
        top = self.pairs(metric=matching, frame=inside).row(0, named=True)
        best_pair = {
            f"resolution_{a}": float(top["resolution_a"]),
            f"resolution_{b}": float(top["resolution_b"]),
            f"k_{a}": float(top["k_a"]),
            f"k_{b}": float(top["k_b"]),
            matching: float(top[matching]),
        }
        return {
            "metric": metric,
            "best_pair": best_pair,
            "peak_grain": int(peak_row["grain"]),
            "peak": float(peak_row[metric]),
            "peak_se": float(peak_row[f"{metric}_se"]),
            "at_edge": edge,
            "grain_min": int(band["grain_min"].min()),
            "grain_max": int(band["grain_max"].max()),
            f"k_{a}": (int(inside["k_a"].min()), int(inside["k_a"].max())),
            f"k_{b}": (int(inside["k_b"].min()), int(inside["k_b"].max())),
            f"resolution_{a}": (
                float(inside["resolution_a"].min()),
                float(inside["resolution_a"].max()),
            ),
            f"resolution_{b}": (
                float(inside["resolution_b"].min()),
                float(inside["resolution_b"].max()),
            ),
        }

    def __repr__(self) -> str:
        a, b = self.names
        return (
            f"GrainCorrespondence({a}<->{b}, {self.frame.height} run pairs, "
            f"null={self.null!r})"
        )
