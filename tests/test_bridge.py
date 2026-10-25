"""Cross-dataset reproducibility of per-dataset clusterings, judged through a bridge."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.sparse import issparse

from cellpax import (
    FeatureTable,
    LabelSet,
    TransferBridge,
    VerdictRule,
    clipped_scaler_factory,
    identity_scaler_factory,
    join_datasets,
    quantile_scaler_factory,
)

A_COLS = ["a0", "a1", "a2", "a3"]
Z_COLS = ["z0", "z1", "z2"]
COARSE = {"P": (0.0, 0.0), "Q": (8.0, 0.0), "R": (0.0, 8.0)}
ONLY_D1 = ("S", (12.0, 12.0))


def _dataset(
    n_per: int, *, warp: bool, seed: int, first_id: int, extra: bool
) -> tuple[FeatureTable, np.ndarray, np.ndarray]:
    """Three coarse types, each split into two subtypes along latent axis 2.

    A sees the subtype split at full margin but through a per-dataset warp; Z sees
    it through the same map in both datasets, at a fifth of the margin — blurred
    but present. ``extra`` adds a fourth type only this dataset has.
    """
    rng = np.random.default_rng(seed)
    latent, coarse, fine = [], [], []
    types = dict(COARSE)
    if extra:
        types[ONLY_D1[0]] = ONLY_D1[1]
    for c, (name, (cx, cy)) in enumerate(types.items()):
        for s, offset in enumerate((3.0, -3.0)):
            x = rng.normal(size=(n_per, 4))
            x[:, 0] += cx
            x[:, 1] += cy
            x[:, 2] += offset
            latent.append(x)
            coarse += [c] * n_per
            fine += [2 * c + s] * n_per
    u = np.vstack(latent)
    a = u.copy()
    if warp:
        a[:, 0] = 1.7 * a[:, 0] + 4.0
        a[:, 2] = np.sign(a[:, 2]) * np.abs(a[:, 2]) ** 1.3
        a[:, 3] = np.exp(a[:, 3] / 2)
    z = np.column_stack([u[:, 0], u[:, 1], u[:, 2] / 5.0])
    z += rng.normal(scale=0.3, size=z.shape)
    df = pl.DataFrame(
        {
            "cell": pl.Series(range(first_id, first_id + len(u)), dtype=pl.Int64),
            "coarse": [list(types)[c] for c in coarse],
            **{col: a[:, i] for i, col in enumerate(A_COLS)},
            **{col: z[:, i] for i, col in enumerate(Z_COLS)},
        }
    )
    table = FeatureTable(df, A_COLS + Z_COLS, id_column="cell")
    return table, np.asarray(coarse), np.asarray(fine)


@pytest.fixture(scope="module")
def joined():
    t1, c1, f1 = _dataset(80, warp=False, seed=1, first_id=1, extra=True)
    t2, c2, f2 = _dataset(80, warp=True, seed=2, first_id=10_000, extra=False)
    ft = join_datasets(
        {"d1": t1, "d2": t2}, scaler_factory=identity_scaler_factory(), seed=3
    )
    truth = {"d1": (c1, f1), "d2": (c2, f2)}
    return ft, truth


@pytest.fixture(scope="module")
def bridge(joined):
    ft, _ = joined
    return ft.transfer_bridge(
        "d1", "d2", columns=Z_COLS, null_columns=A_COLS, strata="coarse"
    )


def test_vote_matrices_stay_sparse(bridge):
    for votes in (*bridge.cross.values(), *bridge.within.values()):
        assert issparse(votes)
    assert bridge.cross[("d1", "d2")].shape == (480, 640)
    assert bridge.within["d1"].diagonal().sum() == 0  # leave-one-out


def test_carry_matches_an_explicit_vote():
    rng = np.random.default_rng(0)
    za, zb = rng.normal(size=(60, 2)), rng.normal(size=(40, 2))
    built = TransferBridge.build(
        {"a": za, "b": zb},
        {"a": np.arange(60), "b": np.arange(40)},
        n_neighbors=5,
        mutual=False,
    )
    codes = rng.integers(0, 3, size=60)
    codes[:10] = -1
    carried = built.carry(codes, "a")
    distance = np.linalg.norm(zb[:, None, :] - za[None, :, :], axis=2)
    for i in range(40):
        voters = codes[np.argsort(distance[i])[:5]]
        voters = voters[voters >= 0]
        if voters.size == 0:
            assert carried.codes[i] == -1
            continue
        counts = np.bincount(voters, minlength=3)
        assert counts[carried.codes[i]] == counts.max()
        assert carried.confidence[i] == pytest.approx(counts.max() / voters.size)
        np.testing.assert_allclose(carried.soft[i], counts / voters.size)


def test_unassigned_cells_cast_no_vote():
    za = np.array([[0.0], [0.1], [5.0]])
    zb = np.array([[0.05]])
    built = TransferBridge.build(
        {"a": za, "b": zb},
        {"a": np.arange(3), "b": np.arange(1)},
        n_neighbors=2,
        mutual=False,
    )
    carried = built.carry(np.array([-1, 1, 0]), "a")
    assert carried.codes[0] == 1
    assert carried.confidence[0] == 1.0
    assert built.carry(np.array([-1, -1, -1]), "a").codes[0] == -1


def test_a_type_one_dataset_lacks_abstains_instead_of_being_forced(joined, bridge):
    _, truth = joined
    (c1, _), (c2, _) = truth["d1"], truth["d2"]
    carried = bridge.carry(c2, "d2")
    only_d1 = c1 == 3
    assert (carried.codes[only_d1] == -1).mean() > 0.9
    assert (carried.codes[~only_d1] >= 0).mean() > 0.8

    forced = TransferBridge.build(
        {"d1": np.zeros((2, 1)), "d2": np.ones((2, 1))},
        {"d1": np.arange(2), "d2": np.arange(2)},
        n_neighbors=1,
        mutual=False,
    )
    assert (forced.carry(np.array([0, 1]), "d2").codes >= 0).all()


def test_coarse_types_reproduce(joined, bridge):
    _, truth = joined
    score = bridge.score(truth["d1"][0], truth["d2"][0])
    assert score.summary["ami"] > 0.9
    assert score.summary["efficiency"] > 0.9
    # d2 has no S cells, so not every d1 cell is reachable from it
    assert score.summary["coverage_ba"] < 0.9
    assert score.summary["coverage_ab"] > 0.8


def test_efficiency_separates_bridge_blur_from_irreproducibility(joined, bridge):
    _, truth = joined
    (_, f1), (c2, f2) = truth["d1"], truth["d2"]
    shared = bridge.score(f1, f2)
    # the blurred bridge recovers the fine split imperfectly even within a dataset,
    # and the cross-dataset carry loses no more than that
    assert shared.summary["ceiling_ab"] < 0.92
    assert shared.summary["efficiency"] > 0.9
    assert shared.summary["worst_efficiency"] > 0.85


def test_refuted_and_unverifiable_splits_read_differently(joined, bridge):
    ft, truth = joined
    (_, f1), (c2, _) = truth["d1"], truth["d2"]
    # d2 splits each coarse type along a3, which the bridge does not see
    a3 = ft.features("d2", columns=["a3"]).ravel()
    unrelated = 2 * c2 + (a3 > np.median(a3)).astype(int)
    broken = bridge.score(f1, unrelated)
    # global AMI is carried by the shared coarse structure and hides the break
    assert broken.summary["efficiency"] > 0.75
    clusters = broken.clusters.filter(pl.col("purity").is_not_nan())
    forward = clusters.filter(pl.col("direction") == "d1->d2")
    backward = clusters.filter(pl.col("direction") == "d2->d1")
    # d1's split is visible in the bridge and does not reappear in d2: refuted
    assert forward["ceiling_purity"].min() > 0.8
    assert forward["efficiency"].max() < 0.8
    assert broken.summary["worst_efficiency_ab"] < 0.8
    # d2's split is invisible to the bridge even within d2: unverifiable, so its
    # low cross purity is not held against it
    assert backward["ceiling_purity"].max() < 0.65
    assert backward["efficiency"].min() > 0.85


def test_cluster_frame_reads_per_cluster(joined, bridge):
    ft, truth = joined
    named = np.array(["P", "Q", "R", "S"])
    la = LabelSet.from_labels(bridge.cell_ids["d1"], named[truth["d1"][0]], name="a")
    lb = LabelSet.from_labels(bridge.cell_ids["d2"], named[truth["d2"][0]], name="b")
    score = bridge.score_labels(la, lb)
    frame = score.clusters.filter(pl.col("direction") == "d1->d2")
    matched = dict(zip(frame["cluster_name"], frame["best_match_name"]))
    assert matched["P"] == "P" and matched["Q"] == "Q" and matched["R"] == "R"
    # nothing in d2 sits near S, so S has (almost) no image there
    s_mass = frame.filter(pl.col("cluster_name") == "S")["image_mass"][0]
    assert s_mass < 0.05 * frame["image_mass"].sum()
    assert "BridgeScore" in repr(score)


def test_align_reorders_and_refuses_strangers(bridge):
    ids = bridge.cell_ids["d2"]
    codes = np.arange(ids.size) % 3
    order = np.random.default_rng(0).permutation(ids.size)
    np.testing.assert_array_equal(bridge.align(codes[order], ids[order], "d2"), codes)
    partial = bridge.align(codes[:5], ids[:5], "d2")
    assert (partial[5:] == -1).all()
    with pytest.raises(ValueError, match="not cells of"):
        bridge.align(np.zeros(1), np.array([-99]), "d2")
    with pytest.raises(KeyError):
        bridge.carry(codes, "elsewhere")
    with pytest.raises(ValueError, match="codes were given"):
        bridge.carry(codes[:3], "d2")


@pytest.fixture(scope="module")
def grain(joined, bridge):
    ft, _ = joined
    sweep = dict(columns=A_COLS, resolution=[0.05, 0.2, 0.6, 1.5], n_times=2, seed=0)
    c1 = ft.cluster("d1", **sweep)
    c2 = ft.cluster("d2", **sweep)
    return bridge.score_runs(c1, c2, band_ratio=1.5)


def test_score_runs_pairs_by_realized_grain(grain):
    frame = grain.frame
    assert frame.height > 0
    ratio = pl.max_horizontal("k_a", "k_b") / pl.min_horizontal("k_a", "k_b")
    assert frame.select(ratio.max()).item() <= 1.5
    for column in ("ami", "efficiency", "worst_efficiency", "null_ami", "coverage_ab"):
        assert column in frame.columns
    np.testing.assert_allclose(frame["excess"], frame["ami"] - frame["null_ami"])


def test_score_runs_counts_verdicts_and_strata(grain):
    frame = grain.frame
    from cellpax.bridge import VERDICTS

    verdicts = [f"n_{v}" for v in VERDICTS]
    np.testing.assert_array_equal(
        frame.select(pl.sum_horizontal(verdicts)).to_series(), frame["n_counted"]
    )
    np.testing.assert_allclose(
        frame["reproduces_excess"], frame["n_reproduces"] - frame["null_n_reproduces"]
    )
    assert {"strata_ami", "null_strata_ami", "strata_excess", "strata_ceiling"} <= set(
        frame.columns
    )
    assert frame["reproduces_excess"].max() > 0


def test_curve_and_band(grain):
    curve = grain.curve()
    assert curve["grain"].is_sorted()
    assert {"n_pairs", "reproduces_excess", "reproduces_excess_se"} <= set(
        curve.columns
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        band = grain.suggest_band()
    assert band["metric"] == "reproduces_excess"
    assert band["at_edge"] in (None, "finest", "coarsest")
    assert band["grain_min"] <= band["peak_grain"] <= band["grain_max"] + 1
    assert band["peak_se"] >= 0
    with pytest.raises(ValueError, match="must exceed 1"):
        grain.curve(bin_ratio=1.0)
    lo, hi = band["resolution_d1"]
    assert lo <= hi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert grain.suggest_band(metric="ami")["metric"] == "ami"
    with pytest.raises(ValueError, match="not a curve column"):
        grain.suggest_band(metric="nope")


def test_real_partitions_beat_the_null(grain):
    frame = grain.frame
    assert frame["null_ami"].max() < 0.5
    assert frame["excess"].min() > 0.25
    ridge = grain.ridge()
    assert ridge["k_a"].is_sorted()
    assert {"k_b", "efficiency", "excess", "efficiency_sd"} <= set(ridge.columns)
    assert "GrainCorrespondence" in repr(grain)


def test_kmeans_competitor_and_no_null(joined, bridge):
    ft, _ = joined
    sweep = dict(columns=A_COLS, resolution=[0.2], n_times=1, seed=0)
    c1, c2 = ft.cluster("d1", **sweep), ft.cluster("d2", **sweep)
    competitor = bridge.score_runs(c1, c2, null="kmeans", null_draws=1)
    assert "null_ami" in competitor.frame.columns
    bare = bridge.score_runs(c1, c2, null=None)
    assert "excess" not in bare.frame.columns
    no_null = ft.transfer_bridge("d1", "d2", columns=Z_COLS, null_columns=False)
    with pytest.raises(ValueError, match="no null_features"):
        no_null.score_runs(c1, c2)
    assert "TransferBridge(d1" in repr(no_null)


def test_transfer_bridge_works_on_any_two_masks():
    table, _, _ = _dataset(20, warp=False, seed=0, first_id=1, extra=False)
    table = table.add_mask("left", pl.col("a1") < 0).add_mask(
        "right", pl.col("a1") >= 0
    )
    built = table.transfer_bridge("left", "right", columns=Z_COLS, null_columns=False)
    assert built.names == ("left", "right")
    assert built.cell_ids["left"].size + built.cell_ids["right"].size == table.n_cells


def test_transfer_bridge_checks_its_populations(joined):
    ft, _ = joined
    with pytest.raises(ValueError, match="no mask 'nowhere'"):
        ft.transfer_bridge("d1", "nowhere", columns=Z_COLS)
    with pytest.raises(ValueError, match="share"):
        ft.transfer_bridge("d1", "all", columns=Z_COLS, null_columns=False)
    with pytest.raises(ValueError, match="has no cells inside"):
        ft.transfer_bridge("d1", "d2", columns=Z_COLS, mask="d1", null_columns=False)
    with pytest.raises(ValueError, match="not a column"):
        ft.transfer_bridge("d1", "d2", columns=Z_COLS, strata="missing")


def test_transfer_bridge_accepts_a_space(joined):
    ft, _ = joined
    space = ft.space(columns=Z_COLS, explained_variance=1.0)
    built = ft.transfer_bridge(
        "d1", "d2", columns=Z_COLS, space=space, null_columns=False
    )
    assert built.cross[("d1", "d2")].nnz > 0
    ft_d1 = ft.space("d1", columns=Z_COLS, explained_variance=1.0)
    with pytest.raises(ValueError, match="fit on mask"):
        ft.transfer_bridge("d1", "d2", columns=Z_COLS, space=ft_d1, null_columns=False)


def test_transfer_bridge_can_vote_in_raw_coordinates():
    rng = np.random.default_rng(4)
    n = 120
    half = np.repeat([0, 1], n // 2)
    frames = []
    for first_id in (1, 10_000):
        frames.append(
            pl.DataFrame(
                {
                    "cell": pl.Series(range(first_id, first_id + n), dtype=pl.Int64),
                    "a0": rng.normal(size=n) + 6 * half,
                    # informative, at its own scale
                    "z0": rng.normal(scale=0.5, size=n) + 4 * half,
                    # an inactive latent: tiny, pure noise
                    "z1": rng.normal(scale=1e-3, size=n),
                }
            )
        )
    tables = {
        name: FeatureTable(df, ["a0", "z0", "z1"], id_column="cell")
        for name, df in zip(("d1", "d2"), frames)
    }
    ft = join_datasets(
        tables,
        scaler_factory=clipped_scaler_factory(mode="sigma", base="standard"),
        seed=5,
    )
    raw = ft.transfer_bridge(
        "d1", "d2", columns=["z0", "z1"], scale=False, null_columns=False
    )
    direct = TransferBridge.build(
        {
            name: ft.dataframe(name).select("z0", "z1").to_numpy().astype(float)
            for name in ("d1", "d2")
        },
        {name: ft.dataframe(name)[ft.id_column].to_numpy() for name in ("d1", "d2")},
        n_neighbors=30,
    )
    assert (raw.cross[("d1", "d2")] != direct.cross[("d1", "d2")]).nnz == 0
    scaled = ft.transfer_bridge("d1", "d2", columns=["z0", "z1"], null_columns=False)
    labels = {
        name: LabelSet.from_labels(
            ft.dataframe(name)[ft.id_column].to_numpy(),
            half,
            name=f"{name}_half",
            mask=name,
        )
        for name in ("d1", "d2")
    }

    def purity(bridge):
        return bridge.score_labels(labels["d1"], labels["d2"]).clusters["purity"].mean()

    assert purity(raw) >= purity(scaled)
    with pytest.raises(ValueError, match="one or the other"):
        ft.transfer_bridge(
            "d1",
            "d2",
            columns=["z0", "z1"],
            space=ft.space(columns=["z0", "z1"], explained_variance=1.0),
            scale=False,
        )


def test_transfer_bridge_can_scale_each_population_on_its_own():
    rng = np.random.default_rng(6)
    n = 150
    half = np.repeat([0, 1], n // 2)
    tables = {}
    # the same two types in both, but d2 measures the bridge column offset and
    # stretched -- a jointly scaled bridge sees two shifted clouds
    for name, first_id, (offset, stretch) in (
        ("d1", 1, (0.0, 1.0)),
        ("d2", 10_000, (25.0, 3.0)),
    ):
        b0 = offset + stretch * (rng.normal(scale=0.6, size=n) + 3 * half)
        df = pl.DataFrame(
            {
                "cell": pl.Series(range(first_id, first_id + n), dtype=pl.Int64),
                "a0": rng.normal(size=n) + 5 * half,
                "b0": b0,
            }
        )
        tables[name] = FeatureTable(df, ["a0", "b0"], id_column="cell")
    ft = join_datasets(tables, scaler_factory=identity_scaler_factory(), seed=7)
    labels = {
        name: LabelSet.from_labels(
            ft.dataframe(name)[ft.id_column].to_numpy(),
            half,
            name=f"{name}_half",
            mask=name,
        )
        for name in ("d1", "d2")
    }

    def purity(bridge):
        # a cluster the carry never reaches has no purity: count it as none
        clusters = bridge.score_labels(labels["d1"], labels["d2"]).clusters
        return clusters["purity"].fill_nan(0.0).fill_null(0.0).mean()

    kws = dict(columns=["b0"], null_columns=False)
    joint = ft.transfer_bridge(
        "d1", "d2", scaler_factory=clipped_scaler_factory(base="standard"), **kws
    )
    own = ft.transfer_bridge(
        "d1",
        "d2",
        scale="per_mask",
        scaler_factory=quantile_scaler_factory(),
        **kws,
    )
    assert purity(own) > purity(joint)
    assert purity(own) > 0.9
    with pytest.raises(ValueError, match="scale must be"):
        ft.transfer_bridge("d1", "d2", scale="both", **kws)


def test_build_rejects_bad_inputs():
    with pytest.raises(ValueError, match="exactly two"):
        TransferBridge.build({"a": np.zeros((3, 1))}, {"a": np.arange(3)})
    with pytest.raises(ValueError, match="ids for"):
        TransferBridge.build(
            {"a": np.zeros((3, 1)), "b": np.zeros((3, 1))},
            {"a": np.arange(2), "b": np.arange(3)},
        )
    two = {"a": np.zeros((3, 1)), "b": np.ones((3, 1))}
    ids = {"a": np.arange(3), "b": np.arange(3)}
    with pytest.raises(ValueError, match="cover both"):
        TransferBridge.build(two, ids, strata={"a": ["x"] * 3})
    with pytest.raises(ValueError, match="strata for"):
        TransferBridge.build(two, ids, strata={"a": ["x"] * 2, "b": ["x"] * 3})


def test_verdict_rule_classifies_and_validates():
    rule = VerdictRule()
    verdicts = rule.classify(
        [0.9, 0.9, 0.9, 0.9, 0.3, np.nan], [0.85, 0.7, 0.5, np.nan, 0.3, 0.9]
    )
    assert list(verdicts) == [
        "reproduces",
        "weak",
        "refuted",
        "absent",
        "unverifiable",
        "unverifiable",
    ]
    with pytest.raises(ValueError, match="must not exceed"):
        VerdictRule(reproduce_at=0.5, refute_below=0.7)


def test_verdicts_name_refuted_and_unverifiable_splits(joined, bridge):
    ft, truth = joined
    (_, f1), (c2, f2) = truth["d1"], truth["d2"]
    shared = bridge.score(f1, f2).verdicts()
    forward = shared.filter(pl.col("direction") == "d1->d2").row(0, named=True)
    assert forward["reproduces"] == 6  # every shared subtype
    assert forward["absent"] == 2  # the two S subtypes: d2 has no S cells

    a3 = ft.features("d2", columns=["a3"]).ravel()
    unrelated = 2 * c2 + (a3 > np.median(a3)).astype(int)
    broken = bridge.score(f1, unrelated).verdicts()
    forward = broken.filter(pl.col("direction") == "d1->d2").row(0, named=True)
    backward = broken.filter(pl.col("direction") == "d2->d1").row(0, named=True)
    assert forward["refuted"] >= 4
    assert backward["unverifiable"] == 6


def test_strata_readings_see_the_subdivision(joined, bridge):
    _, truth = joined
    (c1, f1), (c2, f2) = truth["d1"], truth["d2"]
    fine = bridge.score(f1, f2).summary
    assert fine["strata_ami"] > 0.6
    assert fine["strata_ceiling"] > fine["strata_ami"] - 0.1
    # the coarse labels do not subdivide any stratum, so there is nothing to score
    coarse = bridge.score(c1, c2).summary
    assert np.isnan(coarse["strata_ami"])


def test_run_pair_verdicts_match_a_direct_score(joined, bridge):
    ft, _ = joined
    sweep = dict(columns=A_COLS, resolution=[0.6], n_times=1, seed=0)
    c1, c2 = ft.cluster("d1", **sweep), ft.cluster("d2", **sweep)
    row = bridge.score_runs(c1, c2, null=None, band_ratio=10).frame.row(0, named=True)
    codes_a = bridge.align(c1.partitions.labels[:, 0], c1.cell_ids, "d1")
    codes_b = bridge.align(c2.partitions.labels[:, 0], c2.cell_ids, "d2")
    direct = bridge.score(codes_a, codes_b)
    counts = direct.verdicts()
    for verdict in ("reproduces", "weak", "refuted", "unverifiable", "absent"):
        assert row[f"n_{verdict}"] == counts[verdict].sum()
    assert row["ami"] == pytest.approx(direct.summary["ami"])
    assert row["strata_ami"] == pytest.approx(direct.summary["strata_ami"])


def test_band_at_the_sweep_edge_warns():
    from cellpax.bridge import GrainCorrespondence

    rows = [
        {"k_a": k, "k_b": k, "resolution_a": r, "resolution_b": r, "n_reproduces": k}
        for k, r in ((2, 0.1), (4, 0.3), (8, 0.9), (16, 2.0))
    ]
    rising = GrainCorrespondence(pl.DataFrame(rows), names=("a", "b"), null=None)
    with pytest.warns(UserWarning, match="finest grain scored"):
        band = rising.suggest_band()
    assert band["at_edge"] == "finest"
    assert band["resolution_a"] == (2.0, 2.0)
    falling = GrainCorrespondence(
        pl.DataFrame([{**r, "n_reproduces": -r["k_a"]} for r in rows]),
        names=("a", "b"),
        null=None,
    )
    with pytest.warns(UserWarning, match="coarsest"):
        assert falling.suggest_band()["at_edge"] == "coarsest"
    peaked = GrainCorrespondence(
        pl.DataFrame(
            [{**r, "n_reproduces": [1, 5, 2, 1][i]} for i, r in enumerate(rows)]
        ),
        names=("a", "b"),
        null=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert peaked.suggest_band()["at_edge"] is None


@pytest.fixture(scope="module")
def refined(joined):
    ft, _ = joined
    return ft.transfer_bridge(
        "d1",
        "d2",
        columns=Z_COLS,
        null_columns=A_COLS,
        strata="coarse",
        refine_columns=A_COLS,
    )


def test_refinement_sharpens_blurred_splits(joined, bridge, refined):
    _, truth = joined
    (_, f1), (_, f2) = truth["d1"], truth["d2"]
    plain = bridge.score(f1, f2).summary
    sharp = refined.score(f1, f2).summary
    # the ceiling stays the bridge alone, leave-one-out: refinement never touches it
    assert sharp["ceiling_ab"] == plain["ceiling_ab"]
    # the A graph separates the subtypes the blurred bridge mixes across datasets
    assert sharp["ami"] > plain["ami"]
    assert sharp["efficiency"] > plain["efficiency"]
    assert "refine=0.5" in repr(refined)
    carried = refined.carry(f1, "d1")
    np.testing.assert_allclose(carried.soft[carried.codes >= 0].sum(axis=1), 1.0)


def test_refinement_never_extends_coverage(joined, bridge, refined):
    _, truth = joined
    (c1, _), (c2, _) = truth["d1"], truth["d2"]
    plain = bridge.carry(c2, "d2").codes
    sharp = refined.carry(c2, "d2").codes
    # cells the bridge gave no vote (the d1-only type) stay abstained
    assert ((plain == -1) <= (sharp == -1)).all()
    assert (sharp[c1 == 3] == -1).mean() > 0.9


def test_refinement_lifts_the_null_too(joined, refined, grain):
    ft, _ = joined
    sweep = dict(columns=A_COLS, resolution=[0.05, 0.2, 0.6, 1.5], n_times=2, seed=0)
    sharp = refined.score_runs(ft.cluster("d1", **sweep), ft.cluster("d2", **sweep))
    # diffusion over the target graph flatters every partition, chance ones included
    assert sharp.frame["null_ami"].mean() > grain.frame["null_ami"].mean()


def test_build_validates_refinement():
    two = {"a": np.zeros((5, 1)), "b": np.ones((5, 1))}
    ids = {"a": np.arange(5), "b": np.arange(5)}
    with pytest.raises(ValueError, match="refine_alpha"):
        TransferBridge.build(two, ids, refine_alpha=1.0)
    with pytest.raises(ValueError, match="cover both"):
        TransferBridge.build(two, ids, refine_features={"a": np.zeros((5, 2))})
    with pytest.raises(ValueError, match="refine rows"):
        TransferBridge.build(
            two, ids, refine_features={"a": np.zeros((4, 2)), "b": np.zeros((5, 2))}
        )


def test_score_consensus_scores_band_cuts(joined, bridge):
    ft, _ = joined
    sweep = dict(columns=A_COLS, resolution=[0.05, 0.2, 0.6, 1.5], n_times=2, seed=0)
    c1, c2 = ft.cluster("d1", **sweep), ft.cluster("d2", **sweep)
    band = {"resolution_d1": (0.05, 0.6), "resolution_d2": (0.05, 0.6)}
    cuts = bridge.score_consensus(c1, c2, band=band, n_levels=6)
    frame = cuts.frame
    assert frame.height > 0
    assert {"level_a", "height_a", "coclustering_frequency_b", "excess"} <= set(
        frame.columns
    )
    assert (frame["resolution_a"] == 0.6).all()
    # the coarse consensus cut of shared types reproduces well beyond chance
    assert frame["excess"].max() > 0.3
    unbounded = bridge.score_consensus(c1, c2, null=None, n_levels=4)
    assert unbounded.frame["resolution_a"].is_nan().all()
    with pytest.raises(ValueError, match="one per dataset"):
        bridge.score_consensus(c1, c2, resolution_max=(0.2, 0.3, 0.4))


def test_spaces_supply_the_null_and_refinement(joined):
    ft, truth = joined
    spaces = {
        name: ft.space(name, columns=A_COLS, explained_variance=1.0)
        for name in ("d1", "d2")
    }
    built = ft.transfer_bridge(
        "d1", "d2", columns=Z_COLS, spaces=spaces, refine_columns=True
    )
    expected = spaces["d1"].transform(ft.features("d1", columns=A_COLS))
    np.testing.assert_allclose(built.null_features["d1"], expected)
    assert built.refine
    bare = ft.transfer_bridge(
        "d1", "d2", columns=Z_COLS, spaces=spaces, null_columns=False
    )
    assert not bare.null_features and not bare.refine
    with pytest.raises(ValueError, match="keyed by both"):
        ft.transfer_bridge("d1", "d2", columns=Z_COLS, spaces={"d1": spaces["d1"]})
    with pytest.raises(ValueError, match="supplies the null"):
        ft.transfer_bridge(
            "d1", "d2", columns=Z_COLS, spaces=spaces, null_columns=A_COLS
        )
    with pytest.raises(ValueError, match="fit on mask"):
        ft.transfer_bridge(
            "d1", "d2", columns=Z_COLS, spaces={"d1": spaces["d2"], "d2": spaces["d2"]}
        )
    with pytest.raises(ValueError, match="pass them"):
        ft.transfer_bridge("d1", "d2", columns=Z_COLS, refine_columns=True)


def _without_z(ft, share, seed=0):
    """The joined fixture with ``share`` of d1's cells missing every Z value."""
    df = ft.dataframe()
    rng = np.random.default_rng(seed)
    d1 = (df["dataset"] == "d1").to_numpy()
    drop = d1 & (rng.random(df.height) < share)
    df = df.with_columns(
        [
            pl.when(pl.Series(drop)).then(None).otherwise(pl.col(c)).alias(c)
            for c in Z_COLS
        ]
    )
    table = FeatureTable(df, A_COLS + Z_COLS, id_column=ft.id_column)
    for name in ("d1", "d2"):
        table = table.add_mask(name, pl.col("dataset") == name)
    return table, df.filter(pl.Series(drop))[ft.id_column].to_numpy()


def test_cells_without_z_are_clustered_but_not_bridged(joined):
    ft, _ = joined
    table, missing = _without_z(ft, 0.1)
    sweep = dict(columns=A_COLS, resolution=[0.2, 0.6], n_times=1, seed=0)
    c1, c2 = table.cluster("d1", **sweep), table.cluster("d2", **sweep)
    assert c1.cell_ids.size == 640  # every d1 cell clustered, Z or not
    with pytest.warns(UserWarning, match="no bridge coordinates"):
        built = table.transfer_bridge("d1", "d2", columns=Z_COLS, null_columns=A_COLS)
    assert built.cell_ids["d1"].size == 640 - missing.size
    np.testing.assert_array_equal(np.sort(built.unbridged["d1"]), np.sort(missing))
    assert built.bridged_fraction()["d1"] == pytest.approx(1 - missing.size / 640)
    assert "bridged d1=" in repr(built)
    grain = built.score_runs(c1, c2, band_ratio=3)
    assert grain.frame.height > 0 and grain.frame["ami"].max() > 0.5
    # an id the bridge never heard of is still an error, not a quiet drop
    with pytest.raises(ValueError, match="not cells of"):
        built.align(np.zeros(1), np.array([-99]), "d1")


def test_too_little_bridge_coverage_raises(joined):
    ft, _ = joined
    table, _ = _without_z(ft, 0.5)
    with pytest.raises(ValueError, match="min_bridged"):
        table.transfer_bridge("d1", "d2", columns=Z_COLS, null_columns=False)
    with pytest.warns(UserWarning):
        relaxed = table.transfer_bridge(
            "d1", "d2", columns=Z_COLS, null_columns=False, min_bridged=0.3
        )
    assert relaxed.bridged_fraction()["d1"] < 0.6


def _shifted_correspondence(band_ratio=2.0):
    """b needs twice a's resolution for the same grain; agreement peaks on that ridge."""
    from cellpax.bridge import GrainCorrespondence

    rng = np.random.default_rng(0)
    rows = []
    for ra in (0.1, 0.2, 0.4, 0.8):
        for rb in (0.1, 0.2, 0.4, 0.8, 1.6):
            for _ in range(2):  # two seeds per resolution pair
                rows.append(
                    {
                        "resolution_a": ra,
                        "resolution_b": rb,
                        "k_a": int(round(40 * ra)),
                        "k_b": int(round(20 * rb)),
                        "ami": 1 - abs(np.log2(rb / (2 * ra))) + rng.normal(0, 0.01),
                        "excess": 0.5 - abs(np.log2(rb / ra)) + rng.normal(0, 0.01),
                        "reproduces_excess": 10 * ra + rng.normal(0, 0.1),
                    }
                )
    return GrainCorrespondence(
        pl.DataFrame(rows), names=("a", "b"), null="voronoi", band_ratio=band_ratio
    )


def test_complement_follows_the_ridge():
    corr = _shifted_correspondence()
    pairs = corr.pairs(metric="ami")
    assert pairs.height == 20 and (pairs["n_pairs"] == 2).all()
    assert (
        pairs.row(0, named=True)["resolution_b"]
        == 2 * pairs.row(0, named=True)["resolution_a"]
    )

    comp = corr.complement("a", resolution_min=0.1, resolution_max=0.2)
    assert comp["metric"] == "ami"
    assert comp["resolution_a"] == (0.1, 0.2)
    assert comp["resolution_b"] == (0.2, 0.4)
    assert comp["k_a"] == (4, 8) and comp["k_b"] == (4, 8)
    assert comp["ridge"].height == 2
    lo, hi = comp["tolerant_resolution_b"]
    assert lo <= 0.2 and hi >= 0.4

    # the same band, chosen by realized cluster count instead
    by_k = corr.complement("a", n_clusters_min=4, n_clusters_max=8)
    assert by_k["resolution_b"] == (0.2, 0.4)
    # and from the other side
    back = corr.complement("b", resolution_min=0.4, resolution_max=0.8)
    assert back["resolution_a"] == (0.2, 0.4)
    # another reading follows another ridge: this fixture's excess peaks on the diagonal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # its ridge sits at the sweep's coarse edge
        on_excess = corr.complement("a", resolution_max=0.2, metric="excess")
    assert on_excess["resolution_b"] == (0.1, 0.2)


def test_complement_warns_at_the_edges():
    corr = _shifted_correspondence()
    with pytest.warns(UserWarning, match="finest resolution"):
        corr.complement("a", resolution_min=0.8)
    tight = _shifted_correspondence(band_ratio=1.05)
    with pytest.warns(UserWarning, match="wider band_ratio"):
        tight.complement("a", resolution_max=0.2)
    with pytest.raises(ValueError, match="inside that band"):
        corr.complement("a", resolution_min=5.0)
    with pytest.raises(KeyError):
        corr.complement("c")
    with pytest.raises(ValueError, match="not a column"):
        corr.pairs(metric="nope")


def test_band_carries_a_joint_best_pair(grain):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        band = grain.suggest_band()
    best = band["best_pair"]
    assert {
        "resolution_d1",
        "resolution_d2",
        "k_d1",
        "k_d2",
        "ami",
    } <= set(best)
    assert band["resolution_d1"][0] <= best["resolution_d1"] <= band["resolution_d1"][1]
    assert grain.band_ratio == 1.5
    assert grain.pairs().height > 0


def test_a_split_type_reads_nested_and_split_not_reproduced(joined, bridge):
    _, truth = joined
    (_, f1), (c2, _) = truth["d1"], truth["d2"]
    # d1 at subtype grain against d2 at coarse grain: d2 keeps whole what d1 splits
    score = bridge.score(f1, c2)
    counts = score.verdicts()
    forward = counts.filter(pl.col("direction") == "d1->d2").row(0, named=True)
    backward = counts.filter(pl.col("direction") == "d2->d1").row(0, named=True)
    assert forward["nested"] == 6  # each subtype lands inside its coarse type
    assert forward["absent"] == 2  # the d1-only type
    assert backward["split"] == 3  # each coarse type is subdivided across
    # over-splitting earns nothing one-to-one
    assert forward["reproduces"] == backward["reproduces"] == 0
    assert score.summary["reproduce_share"] == 0.0
    assert score.summary["anchored_cells"] > 0.6


def test_anchors_list_one_to_one_and_splits(joined, bridge):
    _, truth = joined
    (c1, f1), (c2, f2) = truth["d1"], truth["d2"]
    named = np.array(["P", "Q", "R", "S"])
    fine = np.array([f"{n}{i}" for n in "PQRS" for i in (0, 1)])
    la = LabelSet.from_labels(bridge.cell_ids["d1"], fine[f1], name="a")
    lb = LabelSet.from_labels(bridge.cell_ids["d2"], named[c2], name="b")
    anchors = bridge.score_labels(la, lb).anchors()
    splits = anchors.filter(pl.col("kind") == "split")
    assert splits.height == 3
    for row in splits.iter_rows(named=True):
        coarse = row["b"][0]
        assert sorted(row["a"]) == [f"{coarse}0", f"{coarse}1"]
    # the d1-only type anchors nowhere
    assert not any("S" in name for names in anchors["a"] for name in names)

    same = bridge.score_labels(
        la, LabelSet.from_labels(bridge.cell_ids["d2"], fine[f2], name="b")
    )
    one = same.anchors().filter(pl.col("kind") == "one-to-one")
    assert one.height == 6
    assert all(row["a"] == row["b"] for row in one.iter_rows(named=True))


def test_a_half_bridge_scores_a_whole_clustering(joined):
    ft, truth = joined
    rng = np.random.default_rng(1)
    table = ft.add_mask("half0", pl.Series(rng.random(ft.n_cells) < 0.5))
    sweep = dict(columns=A_COLS, resolution=[0.2, 0.6], n_times=1, seed=0)
    c1, c2 = table.cluster("d1", **sweep), table.cluster("d2", **sweep)
    half = table.transfer_bridge(
        "d1", "d2", columns=Z_COLS, mask="half0", null_columns=A_COLS
    )
    assert half.cell_ids["d1"].size < 640
    grain = half.score_runs(c1, c2, band_ratio=3)  # whole clusterings, half the cells
    assert grain.frame.height > 0
    # held-out cells are neither bridged nor counted against the embedding's coverage
    assert half.bridged_fraction()["d1"] == 1.0
    assert half.held_out["d1"].size + half.cell_ids["d1"].size == 640
    assert "held out" in repr(half)


def test_matched_pairs_keep_each_runs_best_partner(grain):
    matched = grain.matched()
    assert 0 < matched.height <= grain.frame.height
    for key in ("run_a", "run_b"):
        best = grain.frame.sort("ami", descending=True).group_by(key).first()
        for row in best.iter_rows(named=True):
            hit = matched.filter(
                (pl.col("run_a") == row["run_a"]) & (pl.col("run_b") == row["run_b"])
            )
            assert hit.height == 1
    assert grain.curve(matched=False)["n_pairs"].sum() == grain.frame.height
    assert grain.curve()["n_pairs"].sum() == matched.height
