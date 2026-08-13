"""Scoring representations: subsample reproducibility, label recovery, purity."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import Clustering, fauxnograph_coclustering, kneighbor_graph
from cellpax.space import FittedSpace
from cellpax.validate import (
    clustering_stability,
    graph_knn_recovery,
    label_purity,
    loo_knn_recovery,
    paired_recovery,
    subsample_stability,
)

#: Held fixed across representations, which is what makes their ARIs comparable.
_ENSEMBLE = dict(n_neighbors=20, resolution=[0.3, 1.0], n_times=2, seed=0, n_jobs=1)

#: The same ensemble in ``fauxnograph_coclustering``'s own parameter names.
_SWEEP = dict(
    n_neighbors=20, resolution_parameter=[0.3, 1.0], n_times=2, seed=0, n_jobs=1
)


def _blobs(per: int = 100, dim: int = 6, gap: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = np.vstack([rng.normal(i * gap, 0.4, (per, dim)) for i in range(3)])
    return coords, np.repeat([0, 1, 2], per)


def _easy_and_hard_strata(per: int = 120, seed: int = 0):
    """Two well-separated classes plus two heavily overlapping ones.

    Mimics the shape of a stratified labelling design: an agreement stratum that any
    representation gets right, and a disagreement stratum that is the informative part.
    """
    rng = np.random.default_rng(seed)
    easy = np.vstack([rng.normal(0.0, 0.3, (per, 4)), rng.normal(8.0, 0.3, (per, 4))])
    hard = np.vstack([rng.normal(3.6, 1.0, (per, 4)), rng.normal(4.4, 1.0, (per, 4))])
    coords = np.vstack([easy, hard])
    truth = np.repeat([0, 1, 2, 3], per)
    strata = np.repeat(["easy", "easy", "hard", "hard"], per)
    weights = np.where(strata == "easy", 589.6, 4.5)
    return coords, truth, strata, weights


def _continuum(n: int = 300, dim: int = 6, seed: int = 0) -> np.ndarray:
    """One elongated cloud — no groups to find, so nothing reproducible to recover."""
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 1.0, (n, dim))
    coords[:, 0] *= 6.0
    return coords


# -- subsample reproducibility ---------------------------------------------------


def test_separable_data_is_reproducible_and_a_continuum_is_not() -> None:
    """The property that makes this a criterion: a continuum *should* score badly."""
    coords, _ = _blobs()
    separable = subsample_stability(
        coords, distance_threshold=0.5, n_draws=5, **_ENSEMBLE
    )
    smooth = subsample_stability(
        _continuum(), distance_threshold=0.5, n_draws=5, **_ENSEMBLE
    )
    assert separable.mean_ari > 0.95
    assert smooth.mean_ari < 0.9
    assert separable.mean_ari > smooth.mean_ari


def test_reports_per_draw_and_per_cell_detail() -> None:
    coords, _ = _blobs()
    stability = subsample_stability(
        coords, distance_threshold=0.5, n_draws=4, fraction=0.7, **_ENSEMBLE
    )
    assert stability.draw_ari.shape == (4,)
    assert stability.cell_stability.shape == (coords.shape[0],)
    assert stability.min_ari <= stability.median_ari <= 1.0
    finite = stability.cell_stability[np.isfinite(stability.cell_stability)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    # every draw also reports how much of the reference it assigned at all
    assert stability.assigned_fractions.shape == (4,)
    assert (
        (stability.assigned_fractions >= 0.0) & (stability.assigned_fractions <= 1.0)
    ).all()
    assert stability.min_assigned <= stability.mean_assigned <= 1.0


def test_a_cluster_wiped_out_in_a_draw_reads_as_lost_not_retained() -> None:
    """ARI sees only co-assigned cells, so the loss must show up somewhere else."""
    from cellpax.validate import Stability, _ari, _retention

    reference = np.array([0, 0, 0, 1, 1, 1])
    drawn = np.array([0, 0, 0, -1, -1, -1])  # cluster 1 fell below min_cluster_size

    retention = _retention(reference, drawn)
    assert retention[:3].tolist() == [1.0, 1.0, 1.0]
    assert retention[3:].tolist() == [0.0, 0.0, 0.0]  # dropped, not "retained together"

    # the ARI over co-assigned cells is still perfect — which is exactly why the
    # assigned fraction is tracked apart
    assert _ari(reference, drawn) == 1.0
    stability = Stability(
        draw_ari=np.array([1.0, 1.0]),
        cell_stability=retention,
        full_labels=reference,
        assigned_fractions=np.array([1.0, 0.5]),
        n_draws=2,
        fraction=0.8,
        settings={},
    )
    assert stability.min_assigned == 0.5
    assert stability.mean_assigned == 0.75
    summary = stability.summary()
    assert summary["min_assigned"][0] == 0.5
    assert summary["mean_assigned"][0] == 0.75


def test_rejects_a_draw_count_below_one() -> None:
    coords, _ = _blobs(per=10)
    with pytest.raises(ValueError, match="n_draws"):
        subsample_stability(coords, distance_threshold=0.5, n_draws=0)


def test_summary_records_the_settings_it_was_measured_under() -> None:
    """Absolute ARI is only comparable between runs of the same ensemble."""
    coords, _ = _blobs()
    summary = subsample_stability(
        coords, distance_threshold=0.5, n_draws=3, **_ENSEMBLE
    ).summary()
    assert summary.height == 1
    for column in ("mean_ari", "graph_type", "n_neighbors", "distance_threshold"):
        assert column in summary.columns


def test_the_space_is_never_refit_inside_a_draw() -> None:
    """Otherwise this measures preprocessing stability confounded with clustering.

    The discipline is structural: ``subsample_stability`` takes coordinates, so there is
    no fit for it to redo — the fitted arrays are the same objects throughout.
    """
    raw, _ = _blobs()
    space = FittedSpace.fit(raw, columns=[f"c{i}" for i in range(raw.shape[1])])
    components = space.components_
    coords = space.transform_scaled(raw)
    subsample_stability(coords, distance_threshold=0.5, n_draws=3, **_ENSEMBLE)
    assert space.components_ is components
    np.testing.assert_array_equal(coords, space.transform_scaled(raw))


def test_whitening_views_share_one_fit_across_a_sweep() -> None:
    """A sweep over alpha must differ only in alpha, which the views guarantee."""
    raw, _ = _blobs()
    space = FittedSpace.fit(raw, columns=[f"c{i}" for i in range(raw.shape[1])])
    for alpha in (0.0, 0.5, 1.0):
        view = space.with_alpha(alpha)
        subsample_stability(
            view.transform_scaled(raw), distance_threshold=0.5, n_draws=2, **_ENSEMBLE
        )
        assert view.components_ is space.components_


def test_rejects_a_fraction_that_leaves_too_few_cells() -> None:
    coords, _ = _blobs(per=2)
    with pytest.raises(ValueError, match="fraction"):
        subsample_stability(coords, distance_threshold=0.5, fraction=1.5)
    with pytest.raises(ValueError, match="too few"):
        subsample_stability(coords, distance_threshold=0.5, fraction=0.1)


def test_clustering_stability_reads_the_ensemble_off_a_clustering() -> None:
    coords, _ = _blobs()
    matrix, partitions = fauxnograph_coclustering(
        coords, return_partitions=True, **_SWEEP
    )
    clus = Clustering(
        matrix,
        cell_ids=np.arange(1, coords.shape[0] + 1),
        mask="all",
        normalized=True,
        partitions=partitions,
    )
    stability = clustering_stability(
        clus, coords, distance_threshold=0.5, n_draws=3, n_jobs=1, seed=0
    )
    assert stability.mean_ari > 0.9

    bare = Clustering(
        matrix, cell_ids=np.arange(1, coords.shape[0] + 1), mask="all", normalized=True
    )
    with pytest.raises(ValueError, match="no partitions"):
        clustering_stability(bare, coords, distance_threshold=0.5)


# -- recovery of external labels -------------------------------------------------


def test_loo_knn_recovers_planted_labels() -> None:
    coords, truth = _blobs()
    score = loo_knn_recovery(coords, truth, n_neighbors=15)
    assert score.accuracy == 1.0
    assert score.n_labeled == coords.shape[0]
    assert score.correct.shape == (coords.shape[0],)


def test_only_labeled_cells_vote_and_are_scored() -> None:
    coords, truth = _blobs()
    partial = truth.copy()
    partial[::2] = -1
    score = loo_knn_recovery(coords, partial, n_neighbors=5)
    assert score.n_labeled == (partial >= 0).sum()


def test_recovery_degrades_on_overlapping_populations() -> None:
    """A scorer that returns 1.0 on everything cannot rank representations."""
    coords, truth = _blobs(gap=0.25)
    assert loo_knn_recovery(coords, truth, n_neighbors=15).accuracy < 0.9


def test_needs_enough_labeled_cells() -> None:
    coords, truth = _blobs()
    barely = np.full(coords.shape[0], -1)
    barely[:2] = [0, 1]
    with pytest.raises(ValueError, match="three labelled cells"):
        loo_knn_recovery(coords, barely)


# -- design weighting ------------------------------------------------------------


def _stratified_score():
    coords, truth, strata, weights = _easy_and_hard_strata()
    return coords, loo_knn_recovery(
        coords, truth, n_neighbors=15, weights=weights, strata=strata
    )


def test_ipw_weighting_pulls_the_estimate_toward_the_heavy_stratum() -> None:
    """And so away from the cells a stratified design was built to over-sample.

    Not a defect in the weighting — it correctly estimates a population quantity. It is
    the reason a population estimate is a poor instrument for *choosing* a
    representation: the informative stratum carries a small design weight precisely
    because it was over-sampled, so it barely moves the number. Hence ``by_stratum``.
    """
    _coords, score = _stratified_score()
    by_stratum = score.by_stratum()
    accuracy = dict(zip(by_stratum["stratum"], by_stratum["accuracy"]))
    heavy, light = accuracy["easy"], accuracy["hard"]

    # the strata genuinely differ, so there is something for the weighting to hide
    assert heavy > light
    assert abs(score.accuracy_ipw - heavy) < abs(score.accuracy_ipw - light)
    # the unweighted average sits between them; the weighted one is pinned near the easy
    assert score.accuracy_ipw > score.accuracy


def test_by_stratum_reports_counts_and_weights() -> None:
    coords, score = _stratified_score()
    frame = score.by_stratum()
    assert set(frame["stratum"]) == {"easy", "hard"}
    assert frame["n"].sum() == coords.shape[0]
    assert sorted(frame["weight"].to_list()) == [4.5, 589.6]


def test_weights_and_strata_matching_neither_length_are_rejected() -> None:
    coords, truth = _blobs(per=20)
    partial = truth.copy()
    partial[::2] = -1  # 30 labelled of 60, so 7 matches neither count
    with pytest.raises(ValueError, match="weights has 7"):
        loo_knn_recovery(coords, partial, weights=np.ones(7))
    with pytest.raises(ValueError, match="strata has 7"):
        loo_knn_recovery(coords, partial, strata=np.array(["s"] * 7))


def test_unweighted_scores_report_ipw_as_the_plain_accuracy() -> None:
    coords, truth = _blobs()
    score = loo_knn_recovery(coords, truth, n_neighbors=15)
    assert score.accuracy_ipw == score.accuracy
    assert score.by_stratum()["stratum"].to_list() == ["all"]


# -- graph distance --------------------------------------------------------------


def test_graph_recovery_works_under_both_cost_conversions() -> None:
    coords, truth = _blobs()
    graph = kneighbor_graph(coords, 15, graph_type="umap_fuzzy")
    for cost in ("neg_log", "complement"):
        score = graph_knn_recovery(graph, truth, n_neighbors=10, cost=cost)
        assert score.accuracy > 0.9
    with pytest.raises(ValueError, match="cost must be"):
        graph_knn_recovery(graph, truth, cost="inverse")


def test_graph_recovery_handles_an_unweighted_graph() -> None:
    coords, truth = _blobs()
    graph = kneighbor_graph(coords, 15)
    assert graph_knn_recovery(graph, truth, n_neighbors=10).accuracy > 0.9


def test_an_unreachable_vertex_abstains_rather_than_counting_as_wrong() -> None:
    import igraph

    graph = igraph.Graph(6)
    graph.add_edges([(0, 1), (1, 2), (0, 2), (3, 4)])  # vertex 5 is an island
    truth = np.array([0, 0, 0, 1, 1, 1])

    score = graph_knn_recovery(graph, truth, n_neighbors=2)
    assert score.n_abstained == 1
    assert not score.correct[5]  # abstained is not correct, for pairing purposes
    assert score.accuracy == 1.0  # every reachable cell was recovered
    assert score.n_labeled == 6


def test_loo_recovery_never_abstains() -> None:
    coords, truth = _blobs(per=20)
    assert loo_knn_recovery(coords, truth, n_neighbors=5).n_abstained == 0


def test_source_cap_is_applied_and_logged(caplog) -> None:
    coords, truth = _blobs()
    graph = kneighbor_graph(coords, 15, graph_type="umap_fuzzy")
    with caplog.at_level("INFO", logger="cellpax.validate"):
        score = graph_knn_recovery(graph, truth, n_neighbors=5, max_sources=50)
    assert score.n_labeled == 50
    assert "max_sources=50" in caplog.text


# -- paired comparison -----------------------------------------------------------


def test_identical_representations_are_perfectly_concordant() -> None:
    coords, truth = _blobs(gap=1.0)
    a = loo_knn_recovery(coords, truth, n_neighbors=15, name="a")
    b = loo_knn_recovery(coords, truth, n_neighbors=15, name="b")
    frame = paired_recovery({"a": a, "b": b})
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["delta_accuracy"] == 0.0
    assert row["n_discordant"] == 0
    assert row["mcnemar_p"] == 1.0


def test_a_clear_difference_clears_the_paired_test() -> None:
    coords, truth = _blobs(gap=1.0)
    space = FittedSpace.fit(coords, columns=[f"c{i}" for i in range(coords.shape[1])])
    scores = {
        f"alpha={a}": loo_knn_recovery(
            space.with_alpha(a).transform_scaled(coords), truth, n_neighbors=15
        )
        for a in (0.0, 1.0)
    }
    frame = paired_recovery(scores)
    row = frame.row(0, named=True)
    if row["n_discordant"] > 10:
        assert row["mcnemar_p"] < 0.05


def test_pairing_requires_the_same_cells() -> None:
    coords, truth = _blobs()
    full = loo_knn_recovery(coords, truth, n_neighbors=15)
    partial_truth = truth.copy()
    partial_truth[::2] = -1
    partial = loo_knn_recovery(coords, partial_truth, n_neighbors=5)
    with pytest.raises(ValueError, match="same cells"):
        paired_recovery({"full": full, "partial": partial})
    with pytest.raises(ValueError, match="at least two"):
        paired_recovery({"full": full})


def test_pairing_rejects_differently_ordered_truth() -> None:
    coords, truth = _blobs()
    a = loo_knn_recovery(coords, truth, n_neighbors=15)
    b = loo_knn_recovery(coords, truth[::-1].copy(), n_neighbors=15)
    with pytest.raises(ValueError, match="different truth"):
        paired_recovery({"a": a, "b": b})


# -- purity ----------------------------------------------------------------------


def test_pure_clusters_score_one() -> None:
    codes = np.repeat([0, 1], 50)
    audit = np.array(["E"] * 50 + ["I"] * 50, dtype=object)
    frame = label_purity(codes, audit)
    assert frame["purity"].to_list() == [1.0, 1.0]
    assert frame["dominant"].to_list() == ["E", "I"]


def test_a_straddling_cluster_is_surfaced_first() -> None:
    """Sorted by purity, so the red flag is the row you read."""
    codes = np.repeat([0, 1, 2], 40)
    audit = np.array(["E"] * 40 + ["E"] * 20 + ["I"] * 20 + ["I"] * 40, dtype=object)
    frame = label_purity(codes, audit)
    worst = frame.row(0, named=True)
    assert worst["cluster"] == 1
    assert worst["purity"] == pytest.approx(0.5)
    assert worst["n_audit_labels"] == 2


def test_every_level_of_a_nested_frame_is_scored() -> None:
    audit = np.array(["E"] * 100 + ["I"] * 100, dtype=object)
    nested = pl.DataFrame(
        {
            "cell_id": np.arange(1, 201),
            "level_0": np.repeat([0, 1], 100),
            "level_1": np.repeat([0, 1, 2, 3], 50),
        }
    )
    frame = label_purity(nested, audit)
    assert frame["level"].unique().to_list() == [0, 1]
    assert frame.filter(pl.col("level") == 0).height == 2
    assert frame.filter(pl.col("level") == 1).height == 4
    assert frame["purity"].min() == 1.0


def test_null_audit_values_are_counted_apart() -> None:
    codes = np.repeat([0, 1], 20)
    audit = np.array(["E"] * 15 + [None] * 5 + ["I"] * 20, dtype=object)
    frame = label_purity(codes, audit).sort("cluster")
    assert frame["n_audit_null"].to_list() == [5, 0]
    assert frame["purity"].to_list() == [1.0, 1.0]


def test_unassigned_cells_are_counted_apart_not_scored_as_the_worst_cluster() -> None:
    """A mixed unassigned pile is impure by construction, so a row for it would
    reliably top the worst-first report while saying nothing about any cluster."""
    codes = np.array([0] * 10 + [1] * 10 + [-1] * 4)
    audit = np.array(["E"] * 10 + ["I"] * 10 + ["E", "I", "E", "I"], dtype=object)
    frame = label_purity(codes, audit)
    assert set(frame["cluster"]) == {0, 1}
    assert frame["purity"].to_list() == [1.0, 1.0]
    assert frame["n_unassigned"].to_list() == [4, 4]


def test_audit_values_of_different_types_are_not_conflated() -> None:
    codes = np.zeros(4, dtype=np.int64)
    audit = np.array([1, "1", 1, "1"], dtype=object)  # int 1 and string "1"
    frame = label_purity(codes, audit)
    assert frame["n_audit_labels"][0] == 2
    assert frame["purity"][0] == pytest.approx(0.5)


def test_a_frame_without_level_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="no level_"):
        label_purity(pl.DataFrame({"a": [1, 2]}), np.array(["E", "I"], dtype=object))


def test_purity_checks_row_alignment() -> None:
    with pytest.raises(ValueError, match="but audit covers"):
        label_purity(np.array([0, 0, 1]), np.array(["E", "I"], dtype=object))
