"""kNN label propagation from a curated subset into a larger population."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.propagate import confidence_curve, propagate_knn, propagate_spread


def _core_and_periphery(per_type: int = 30, *, errant: int = 0) -> FeatureTable:
    """Three types, each with a tight core and a diffuse periphery around it.

    ``errant`` adds cells far from every type, standing in for reconstructions whose
    features are measurable but meaningless. They get ``true_type`` ``-1``.
    """
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0, 0], [6, 6, 0, 0], [0, 0, 6, 6]], dtype=float)
    blocks, is_core, true_type = [], [], []
    for t, center in enumerate(centers):
        blocks.append(rng.normal(center, 0.35, (per_type, 4)))
        blocks.append(rng.normal(center, 1.1, (per_type, 4)))
        is_core += [True] * per_type + [False] * per_type
        true_type += [t] * (2 * per_type)
    if errant:
        blocks.append(rng.normal(60, 0.3, (errant, 4)))
        is_core += [False] * errant
        true_type += [-1] * errant
    coords = np.vstack(blocks)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, coords.shape[0] + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(4)},
            "is_core": is_core,
            "true_type": true_type,
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(4)])
    ft.add_mask("exc", pl.col("cell_id") > 0)
    ft.add_mask("exc_core", pl.col("is_core"))
    return ft


def _core_labels(ft: FeatureTable) -> object:
    ft.cluster(mask="exc_core", n_neighbors=10, n_times=5, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", mask="exc_core", distance_threshold=0.5, name="subclass")
    return labels.rename([f"C{i}" for i in labels.ids]).set_colors({"C0": "#f00"})


# -- the array primitive ---------------------------------------------------------


def test_propagate_knn_votes_over_unlabeled_rows() -> None:
    # three cells of code 0, one unlabeled, one lone cell of code 1
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    codes = np.array([0, 0, 0, -1, 1])

    filled, confidence, recovery = propagate_knn(features, codes, n_neighbors=3)
    # the unlabeled cell's three nearest labeled cells are codes 1, 0, 0
    assert filled.tolist() == [0, 0, 0, 0, 1]
    assert confidence[3] == pytest.approx(2 / 3)
    # the lone code-1 cell is outvoted by its neighbors, so it doesn't agree
    assert recovery.agreement == 0.75
    assert confidence[4] == 0.0  # no support at all for the label it keeps

    # smoothing lets that vote win instead of preserving the label
    smoothed, _, _ = propagate_knn(
        features, codes, n_neighbors=3, preserve_labeled=False
    )
    assert smoothed.tolist() == [0, 0, 0, 0, 0]


def test_propagate_knn_excludes_a_row_from_its_own_neighborhood() -> None:
    # one labeled outlier sitting inside a run of the other label
    features = np.array([[0.0], [0.1], [0.2], [0.3]])
    codes = np.array([0, 0, 0, 1])

    # k=1 is the sharp case: the outlier's only neighbor is the cell next to it,
    # never itself, so smoothing flips it. dfc's self-vote would have held it at 1
    # (a cell was always its own nearest neighbor, at distance 0).
    smoothed, confidence, recovery = propagate_knn(
        features, codes, n_neighbors=1, preserve_labeled=False
    )
    assert smoothed.tolist() == [0, 0, 0, 0]
    assert recovery.agreement == 0.75  # 3 of the 4 rows agree with their own label
    assert recovery.estimator == "leave-one-out"
    assert recovery.abstained == 0.0  # a vote always finds k neighbors
    assert recovery.misassigned == 0.25
    assert confidence.tolist() == [1.0, 1.0, 1.0, 1.0]  # one neighbor, one vote


def test_propagate_knn_distance_weights_beat_a_larger_cluster() -> None:
    # a small group of two right next to the query cell, a big group further off
    features = np.array([[0.0], [0.1], [0.5], [3.0], [4.0], [5.0], [6.0], [7.0]])
    codes = np.array([1, 1, -1, 0, 0, 0, 0, 0])

    uniform, _, _ = propagate_knn(features, codes, n_neighbors=5)
    assert uniform[2] == 0  # outvoted 3-2 by the bigger group

    weighted, confidence, _ = propagate_knn(
        features, codes, n_neighbors=5, weights="distance"
    )
    assert weighted[2] == 1  # its two close neighbors carry more weight
    assert confidence[2] > 0.5


def test_propagate_argument_errors() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="same number of rows"):
        propagate_knn(features, np.array([0, 1]))
    with pytest.raises(ValueError, match="at least two labeled"):
        propagate_knn(features, np.array([0, -1, -1]))
    with pytest.raises(ValueError, match="every cell is already labeled"):
        propagate_knn(features, np.array([0, 1, 0]))
    for propagate in (propagate_knn, propagate_spread):
        with pytest.raises(ValueError, match="weights must be"):
            propagate(features, np.array([0, -1, 1]), weights="nope")


def test_a_code_below_minus_one_is_rejected_before_it_corrupts_the_vote() -> None:
    # a -2 would wrap around one-hot indexing and land its vote on the last cluster
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    codes = np.array([0, 1, -1, -9])
    for propagate in (propagate_knn, propagate_spread):
        with pytest.raises(ValueError, match=r"\[-9\]"):
            propagate(features, codes)


# -- diffusion -------------------------------------------------------------------


def _core_periphery_and_junk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two labeled clusters, unlabeled diffuse cells around each, and 5 errant cells."""
    rng = np.random.default_rng(0)
    core = np.vstack([rng.normal(0, 0.3, (30, 4)), rng.normal(6, 0.3, (30, 4))])
    periphery = np.vstack([rng.normal(0, 1.0, (20, 4)), rng.normal(6, 1.0, (20, 4))])
    junk = rng.normal(60, 0.3, (5, 4))  # nowhere near anything real
    features = np.vstack([core, periphery, junk])
    codes = np.array([0] * 30 + [1] * 30 + [-1] * 45)
    truth = np.array([0] * 30 + [1] * 30 + [0] * 20 + [1] * 20 + [-1] * 5)
    return features, codes, truth


def test_spread_abstains_where_the_vote_guesses() -> None:
    features, codes, truth = _core_periphery_and_junk()

    voted, vote_confidence, _ = propagate_knn(features, codes, n_neighbors=20)
    # the vote must assign k neighbors' worth of label to every cell, however far
    assert (voted[100:] != -1).all()
    assert vote_confidence[100:].min() == 1.0  # unanimous, and unanimously wrong

    spread, spread_confidence, _ = propagate_spread(features, codes, n_neighbors=20)
    # nothing reaches the errant cells through a mutual graph, so they abstain
    assert (spread[100:] == -1).all()
    assert (spread_confidence[100:] == 0.0).all()
    # while most of the real periphery still gets labeled, and correctly. how much
    # depends on the space -- mutual pruning in raw feature space is more
    # conservative than in the scaled/PCA space ft.propagate_labels builds
    assigned = spread[60:100] != -1
    assert assigned.sum() > 0.7 * 40
    assert (spread[60:100][assigned] == truth[60:100][assigned]).all()


def test_spread_mutuality_is_what_does_the_rejecting() -> None:
    features, codes, _ = _core_periphery_and_junk()

    mutual, _, _ = propagate_spread(features, codes, n_neighbors=20, mutual=True)
    union, _, _ = propagate_spread(features, codes, n_neighbors=20, mutual=False)

    assert (mutual[100:] == -1).all()  # pruned out of the graph entirely
    assert (union[100:] != -1).all()  # reachable, so they get their nearest label
    assert (union[60:100] != -1).all()  # ...at the cost of rejecting nothing


def test_spread_recovery_is_stratified_fold_recovery() -> None:
    features, codes, _ = _core_periphery_and_junk()

    _, _, recovery = propagate_spread(features, codes, n_neighbors=20, seed=0)
    assert recovery.agreement == pytest.approx(1.0)  # clean clusters recover fully
    assert recovery.abstained == 0.0
    assert recovery.misassigned == 0.0
    assert recovery.estimator == "5-fold"

    # the fold count is honoured, and folds are reproducible for a given seed
    ten = propagate_spread(features, codes, n_neighbors=20, seed=0, agreement_folds=10)[
        2
    ]
    assert ten.estimator == "10-fold"
    repeat = propagate_spread(features, codes, n_neighbors=20, seed=0)[2]
    assert repeat == recovery

    skipped = propagate_spread(features, codes, n_neighbors=20, agreement_folds=0)[2]
    assert skipped is None  # 0 skips the fold-wise diffusions entirely


def test_stratified_folds_spread_every_cluster_across_every_fold() -> None:
    from cellpax.propagate import _stratified_folds

    # a big cluster and a cluster of 4 -- unstratified, the small one could land
    # almost entirely in one fold and fail as a block
    codes = np.array([0] * 40 + [1] * 4)
    fit_rows = np.arange(codes.size)

    folds = _stratified_folds(codes, fit_rows, 4, seed=0)
    assert sorted(np.concatenate(folds).tolist()) == fit_rows.tolist()  # a partition
    for fold in folds:
        assert (codes[fold] == 1).sum() == 1  # each fold holds exactly one small cell
        assert (codes[fold] == 0).sum() == 10

    # singleton clusters don't all pile into fold 0
    singles = _stratified_folds(np.arange(4), np.arange(4), 4, seed=0)
    assert sorted(len(fold) for fold in singles) == [1, 1, 1, 1]


def test_recovery_separates_abstention_from_misassignment() -> None:
    # two far-apart pairs, so withholding a label leaves that cell's group with a
    # single mutual neighbour to hear from -- and the isolated pair unreachable
    rng = np.random.default_rng(0)
    together = rng.normal(0, 0.2, (12, 2))
    alone = rng.normal(50, 0.02, (4, 2))  # a tight island of its own cluster
    features = np.vstack([together, alone])
    codes = np.array([0] * 12 + [1] * 4)

    recovery = propagate_spread(
        features,
        codes,
        n_neighbors=3,
        preserve_labeled=False,
        agreement_folds=4,
        seed=0,
    )[2]
    # whatever the split, the three parts account for every reference cell exactly
    assert recovery.agreement + recovery.abstained + recovery.misassigned == 1.0
    # and a cell that heard nothing is counted apart from one that heard wrong
    assert recovery.abstained > 0.0


def test_vote_recovery_can_be_measured_fold_wise_for_comparison() -> None:
    features, codes, _ = _core_periphery_and_junk()

    exact = propagate_knn(features, codes, n_neighbors=20)[2]
    assert exact.estimator == "leave-one-out"

    folded = propagate_knn(features, codes, n_neighbors=20, agreement_folds=5, seed=0)[
        2
    ]
    assert folded.estimator == "5-fold"  # now comparable with spread's
    assert folded.abstained == 0.0  # a vote can't abstain, however far away it is


def test_spread_reaches_a_cell_coincident_with_its_reference() -> None:
    # exact duplicates: each unlabeled copy's only mutual neighbor sits at distance
    # zero, the case low-cardinality integer features produce all the time
    features = np.array([[0.0], [0.0], [5.0], [5.0]])
    codes = np.array([0, -1, 1, -1])

    spread, confidence, _ = propagate_spread(features, codes, n_neighbors=1)
    assert spread.tolist() == [0, 0, 1, 1]  # each duplicate takes its twin's label
    assert (confidence[[1, 3]] > 0).all()


def test_spread_smoothing_relabels_the_reference() -> None:
    # cell 3 is labeled against the group it sits in; cell 8 is unlabeled
    features = np.array([[0.0], [0.1], [0.2], [0.3], [5.0], [5.1], [5.2], [5.3], [5.4]])
    codes = np.array([0, 0, 0, 1, 1, 1, 1, 1, -1])

    kept, confidence, _ = propagate_spread(
        features, codes, n_neighbors=3, preserve_labeled=True
    )
    assert kept[3] == 1  # preserved as given
    assert confidence[3] < 0.5  # but its neighborhood doesn't back that up

    smoothed, _, _ = propagate_spread(
        features, codes, n_neighbors=3, preserve_labeled=False, alpha=0.9
    )
    assert smoothed[3] == 0  # its neighborhood outweighs its own label


# -- calibrating min_confidence ----------------------------------------------------


def test_confidence_curve_trades_coverage_for_purity() -> None:
    truth = np.array([0, 0, 1, 1, -1])
    predicted = np.array([0, 1, 1, 0, 0])
    confidence = np.array([0.9, 0.2, 0.8, 0.4, 1.0])

    curve = confidence_curve(truth, predicted, confidence)
    # one row per cut that changes the kept set — the unknown-truth cell's 1.0
    # never appears, because it can't move the calibration either way
    assert curve["cut"].to_list() == [0.2, 0.4, 0.8, 0.9]
    assert curve["n_kept"].to_list() == [4, 3, 2, 1]  # monotone as the cut rises
    assert curve["kept_fraction"].to_list() == [1.0, 0.75, 0.5, 0.25]
    assert curve["error_rate"].to_list() == pytest.approx([0.5, 1 / 3, 0.0, 0.0])


def test_confidence_curve_matches_a_probe_propagation() -> None:
    features, codes, _ = _core_periphery_and_junk()
    probe, confidence, _ = propagate_knn(
        features, codes, n_neighbors=10, preserve_labeled=False
    )
    curve = confidence_curve(codes, probe, confidence)

    known = int((codes != -1).sum())
    assert curve["n_kept"].max() <= known
    assert curve["n_kept"].is_sorted(descending=True)  # keeping less as the cut rises
    assert ((curve["kept_fraction"] > 0) & (curve["kept_fraction"] <= 1)).all()
    assert ((curve["error_rate"] >= 0) & (curve["error_rate"] <= 1)).all()


def test_confidence_curve_needs_aligned_arrays_and_some_known_truth() -> None:
    with pytest.raises(ValueError, match="same length"):
        confidence_curve(np.array([0, 1]), np.array([0]), np.array([0.5]))
    with pytest.raises(ValueError, match="known truth"):
        confidence_curve(np.array([-1, -1]), np.array([0, 1]), np.array([0.5, 0.6]))


# -- the FeatureTable verb -------------------------------------------------------


def test_propagate_labels_core_to_population() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    assert labels.n_clusters == 3

    result = ft.propagate_labels(labels, to="exc", n_neighbors=20)

    assert result.labels.name == "subclass_nn"
    assert result.labels.mask == "exc"
    assert len(result.labels) == 180  # every cell of the target mask
    assert result.labels.n_unassigned == 0
    assert result.labels.names == labels.names  # ids and names carried over
    assert result.labels.color_map() == {"C0": "#f00"}
    assert result.reference is labels
    assert result.n_reference() == 90
    assert result.self_agreement() > 0.9
    assert ((result.confidence >= 0) & (result.confidence <= 1)).all()

    frame = result.frame()
    assert frame.columns == [
        "cell_id",
        "subclass_nn",
        "subclass_nn_id",
        "subclass_nn_confidence",
    ]
    assert "self_agreement=" in repr(result)

    # the periphery cells land with their own type's core cluster
    table = ft.dataframe()
    got = result.labels.codes_for(table["cell_id"].to_numpy())
    truth = table["true_type"].to_numpy()
    periphery = ~table["is_core"].to_numpy()
    pairs = {(int(t), int(g)) for t, g in zip(truth[periphery], got[periphery])}
    assert len(pairs) == 3  # a clean one-to-one, whatever the cluster numbering

    ft.attach(result.labels)
    assert ft.labelset("subclass_nn").color_map() == {"C0": "#f00"}


def test_propagate_labels_preserves_or_smooths_the_reference() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    core_ids = labels.cell_ids

    kept = ft.propagate_labels(labels, to="exc", n_neighbors=20)
    assert kept.labels.codes_for(core_ids).tolist() == labels.codes.tolist()

    # smoothing relabels the reference cells from their neighborhoods instead
    smoothed = ft.propagate_labels(
        labels, to="exc_core", preserve_labeled=False, n_neighbors=15
    )
    assert len(smoothed.labels) == len(labels)
    assert smoothed.labels.name == "subclass_nn"


def test_propagate_labels_min_confidence_leaves_cells_unassigned() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)

    gated = ft.propagate_labels(labels, to="exc", n_neighbors=20, min_confidence=1.01)
    # nothing can clear an impossible bar -- except the reference, which is exempt
    assert gated.labels.n_unassigned == 180 - 90
    assert gated.labels.codes_for(labels.cell_ids).tolist() == labels.codes.tolist()


def test_propagate_labels_requires_the_reference_inside_the_target() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    ft.add_mask("tiny", pl.col("cell_id") <= 5)

    with pytest.raises(ValueError, match=r"reference cells are outside mask 'tiny'"):
        ft.propagate_labels(labels, to="tiny")


def test_propagate_labels_feature_space_options() -> None:
    ft = _core_and_periphery()
    ft.define_features("first_two", columns=["m0", "m1"])
    labels = _core_labels(ft)

    raw = ft.propagate_labels(labels, to="exc", pca=False, n_neighbors=20)
    assert "space='scaled'" in repr(raw)

    reduced = ft.propagate_labels(labels, to="exc", pca=0.8, n_neighbors=20)
    assert "space='pca(0.8)'" in repr(reduced)

    # m0/m1 alone can't separate the two types that differ only in m2/m3
    partial = ft.propagate_labels(
        labels, to="exc", columns="first_two", pca=False, n_neighbors=20
    )
    assert partial.self_agreement() < raw.self_agreement()


def test_unassign_then_refill_by_propagation() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    junk = labels.copy()

    junk.unassign("C1")
    assert "C1" not in junk.names
    assert junk.n_unassigned == 30

    refilled = ft.propagate_labels(junk, to="exc_core", n_neighbors=15)
    # the emptied cells are redistributed among the clusters that remain
    assert refilled.labels.n_unassigned == 0
    assert set(refilled.labels.names) == {"C0", "C2"}
    assert sum(refilled.labels.counts().values()) == 90


def test_propagate_labels_method_spread() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)

    result = ft.propagate_labels(labels, to="exc", method="spread", n_neighbors=20)
    assert result.method == "spread"
    assert "method='spread'" in repr(result)
    assert result.labels.names == labels.names
    assert result.self_agreement() > 0.9
    # these blobs are well separated, so everything is reachable and gets labeled
    assert result.labels.n_unassigned == 0

    vote = ft.propagate_labels(labels, to="exc", method="vote", n_neighbors=20)
    assert vote.labels.codes.tolist() == result.labels.codes.tolist()

    with pytest.raises(ValueError, match="method must be 'vote' or 'spread'"):
        ft.propagate_labels(labels, to="exc", method="nope")


def test_propagate_labels_spread_leaves_unreachable_cells_unassigned() -> None:
    ft = _core_and_periphery(errant=5)
    labels = _core_labels(ft)
    errant_ids = ft.dataframe().filter(pl.col("true_type") == -1)["cell_id"].to_numpy()

    spread = ft.propagate_labels(labels, to="exc", method="spread", n_neighbors=20)
    vote = ft.propagate_labels(labels, to="exc", method="vote", n_neighbors=20)

    assert (spread.labels.codes_for(errant_ids) == -1).all()  # abstains
    assert (vote.labels.codes_for(errant_ids) != -1).all()  # guesses, confidently
    # the real cells are unaffected by their presence
    assert spread.labels.n_unassigned == len(errant_ids)


def test_recursive_descent_cluster_propagate_mask_repeat() -> None:
    """The pipeline's three moves compose: labels carve the mask for the next round."""
    ft = _core_and_periphery(errant=5)

    coarse = _core_labels(ft)
    first = ft.propagate_labels(coarse, to="exc", method="spread", n_neighbors=20)
    ft.attach(first.labels)
    assert first.labels.n_unassigned == 5  # the errant cells abstain -> null column

    # masking on a label column works even though unassigned cells compare null,
    # which is what lets the next round of clustering be carved out of this one
    biggest = max(first.labels.counts(), key=first.labels.counts().get)
    ft.add_mask("branch", pl.col("subclass_nn") == biggest)
    ft.add_mask("branch_core", pl.col("is_core"), based_on="branch")
    assert ft.mask_series("branch").sum() == first.labels.counts()[biggest]
    assert not ft.mask_series("branch").is_null().any()
    # nesting holds: the branch's core cells are a subset of both parents
    assert (ft.mask_series("branch_core") & ~ft.mask_series("branch")).sum() == 0

    # and the whole cycle runs again inside that branch
    ft.cluster(
        mask="branch_core", n_neighbors=10, n_times=3, seed=0, n_jobs=1, name="b"
    )
    fine = ft.label("b", mask="branch_core", distance_threshold=0.5, name="subtype")
    second = ft.propagate_labels(fine, to="branch", method="spread", n_neighbors=15)
    ft.attach(second.labels)

    # the hierarchy is legible in the table: a subtype only where the branch is
    table = ft.dataframe()
    in_branch = ft.mask_series("branch")
    assert table.filter(~in_branch)["subtype_nn"].null_count() == (~in_branch).sum()
    assert (
        table.filter(in_branch)["subtype_nn"].null_count() == second.labels.n_unassigned
    )


def test_propagate_labels_accepts_an_attached_column_name() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    ft.attach(labels)

    result = ft.propagate_labels("subclass", to="exc", n_neighbors=20)
    assert result.labels.names == labels.names
    assert result.n_reference() == 90  # resolved back to the core mask it came from


# -- projecting labels onto rows that were never in the table --------------------


def test_project_labels_scores_outside_rows_in_the_masks_frozen_space() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)

    # rows from somewhere else entirely: one per type's core centre, plus a stray
    outside = pl.DataFrame(
        {
            "cell_id": [9001, 9002, 9003, 9004],
            **{
                f"m{i}": col
                for i, col in enumerate(
                    np.array(
                        [[0, 0, 0, 0], [6, 6, 0, 0], [0, 0, 6, 6], [60, 60, 60, 60]],
                        dtype=float,
                    ).T
                )
            },
            "unrelated": ["a", "b", "c", "d"],  # extra columns are ignored
        }
    )
    result = ft.project_labels(outside, labels, n_neighbors=20)

    assert len(result.labels) == 4  # just the new cells, not the reference
    assert result.labels.cell_ids.tolist() == [9001, 9002, 9003, 9004]
    assert result.labels.mask is None  # they are in no mask of this table
    assert result.labels.names == labels.names
    assert result.labels.color_map() == {"C0": "#f00"}
    assert result.n_reference() == 90
    assert ft.n_cells == 180  # the table is untouched

    # each lands on the core it was placed at; the vote can't reject the stray
    by_id = dict(zip(result.labels.cell_ids.tolist(), result.labels.to_names()))
    assert len({by_id[9001], by_id[9002], by_id[9003]}) == 3
    assert by_id[9004] is not None


def test_project_labels_uses_the_existing_fit_not_a_new_one() -> None:
    """A reference cell's own feature row must come back with its own label."""
    ft = _core_and_periphery()
    labels = _core_labels(ft)

    core = ft.dataframe("exc_core")
    twins = core.head(20).with_columns(pl.col("cell_id") + 100_000)
    result = ft.project_labels(twins, labels, n_neighbors=5)

    original = labels.codes_for(core.head(20)["cell_id"].to_numpy())
    assert result.labels.codes.tolist() == original.tolist()
    assert (result.confidence > 0.9).all()  # a coincident twin is unambiguous


def test_project_labels_spread_abstains_where_the_vote_cannot() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    stray = np.array([[60.0, 60.0, 60.0, 60.0]])

    voted = ft.project_labels(stray, labels, cell_ids=[9001], n_neighbors=20)
    spread = ft.project_labels(
        stray, labels, cell_ids=[9001], method="spread", n_neighbors=20
    )
    assert voted.labels.n_unassigned == 0  # k votes are always cast
    assert spread.labels.n_unassigned == 1  # nothing is mutually near it


def test_project_labels_argument_errors() -> None:
    ft = _core_and_periphery()
    labels = _core_labels(ft)
    raw = np.zeros((3, 4))

    with pytest.raises(ValueError, match="the new rows need cell ids"):
        ft.project_labels(raw, labels)
    with pytest.raises(ValueError, match="3 cell ids for 2 rows"):
        ft.project_labels(np.zeros((2, 4)), labels, cell_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="but the fit covers 4"):
        ft.project_labels(np.zeros((3, 2)), labels, cell_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="missing feature columns"):
        ft.project_labels(pl.DataFrame({"cell_id": [1], "m0": [0.0]}), labels)
    with pytest.raises(ValueError, match="must be 'vote' or 'spread'"):
        ft.project_labels(raw, labels, cell_ids=[1, 2, 3], method="knn")

    # the reference has to live in the mask whose space is being reused
    with pytest.raises(ValueError, match="reference cells are outside mask"):
        ft.add_mask("half", pl.col("cell_id") <= 30)
        ft.project_labels(raw, labels, cell_ids=[1, 2, 3], mask="half")
