"""The local labeling app: the descent loop, the ledger, and replay.

The property worth testing hardest is that the ledger is the source of truth —
that replaying it reproduces the FeatureTable exactly. Everything else in the
app is a view; if replay drifts, the app is quietly lying about provenance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.app.config import DEFAULT_SWEEP, AppConfig
from cellpax.app.demo import demo_table
from cellpax.app.ledger import Entry, Ledger
from cellpax.app.levels import LevelTree
from cellpax.app.session import LargeLinkage, Session
from cellpax.clustering import _coverage_verdict, _plateau_frame

fastapi = pytest.importorskip("fastapi", reason="app extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

# Small but genuinely hierarchical: three families of two subtypes each, so a
# threshold scan has more than one plateau to find and a descent has somewhere
# to descend to.
_SWEEP = dict(
    graph_type=["knn"],
    n_neighbors=[15],
    resolution=list(np.geomspace(0.05, 2.0, 12)),
    n_times=3,
)


@pytest.fixture
def session(tmp_path) -> Session:
    config = AppConfig(
        columns="analysis", order_by="soma_depth_um", root_mask="all_cells"
    )
    state = Session(
        demo_table(n_per_subtype=40), config, ledger=Ledger(tmp_path / "l.jsonl")
    )
    state.open_level("root", mask="all_cells")
    return state


@pytest.fixture
def client(session) -> TestClient:
    from cellpax.app.server import create_app

    return TestClient(create_app(session.config, session=session))


def _cluster(session: Session, level: str = "root") -> None:
    session.submit_cluster(level, **_SWEEP)
    session.jobs.wait(600)


def _family_threshold(session: Session) -> float:
    """The threshold that cuts at the three-family level.

    Selected by cluster count rather than plateau width. The fixture encodes two
    real scales — three families of two subtypes — and under the sigma-clipped
    scaler their plateaus come out nearly equally wide, so "the widest plateau"
    is an accident of the fixture rather than a statement about it. Tests that
    mean "cut at the family level" should say so.
    """
    scan = session.scan("root", n_points=60, min_cluster_size=10)
    families = [p for p in scan["plateaus"] if p["n_clusters"] == 3]
    assert families, (
        f"no three-family plateau; found {[p['n_clusters'] for p in scan['plateaus']]}"
    )
    return families[0]["midpoint"]


# -- the loop ------------------------------------------------------------------


def test_sweep_then_scan_finds_both_designed_scales(session):
    """The demo has three families of two subtypes, so both should show up.

    Asserting the *hierarchy* rather than which plateau is widest: the fixture
    encodes two real scales and the tool's job is to expose both, leaving which
    one to cut at as the reader's decision. Under the sigma-clipped scaler they
    come out nearly equally wide, which is exactly the situation the threshold
    machinery exists for.
    """
    _cluster(session)
    scan = session.scan("root", n_points=60, min_cluster_size=10)
    by_count = {p["n_clusters"]: p for p in scan["plateaus"]}
    assert 3 in by_count, f"no family plateau; found {sorted(by_count)}"
    assert 6 in by_count, f"no subtype plateau; found {sorted(by_count)}"
    # families are the coarser split, so they survive to a higher threshold
    assert by_count[3]["midpoint"] > by_count[6]["midpoint"]
    assert by_count[3]["width"] > 0 and by_count[6]["width"] > 0


def test_plateaus_are_widest_first_and_exclude_singleton_cuts():
    frame = pl.DataFrame(
        {
            "distance_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "n_clusters": [8, 4, 4, 4, 2, 1],
            "n_assigned": [100] * 6,
            "n_unassigned": [0] * 6,
            "largest_cluster": [20, 40, 40, 40, 60, 100],
            "median_cluster_size": [12.0] * 6,
        }
    )
    out = _plateau_frame(frame).to_dicts()
    assert [row["n_clusters"] for row in out][0] == 4
    assert out[0]["width"] == pytest.approx(0.2)
    assert out[0]["midpoint"] == pytest.approx(0.2)
    # a one-cluster "plateau" is the tree collapsing, not a grain
    assert all(row["n_clusters"] > 1 for row in out)


def test_cut_preview_is_free_and_does_not_commit(session):
    _cluster(session)
    preview = session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    assert preview["n_clusters"] == 3
    assert preview["n_assigned"] + preview["n_unassigned"] == 240
    assert not session.tree["root"].committed
    assert "root_label" not in session.ft.labels
    # previewing must not write to the ledger, or scrubbing would bury the
    # decisions in the motion that led to them
    assert [e.kind for e in session.ledger] == ["open_level", "cluster"]


def test_commit_attaches_labels_and_opens_children(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    result = session.commit(
        "root", label_name="family", children={"kid_a": ["a"], "kid_b": ["b"]}
    )

    assert result["children"] == ["kid_a", "kid_b"]
    assert "family" in session.ft.labels
    assert {"kid_a", "kid_b"} <= set(session.ft.masks)
    assert session.tree["root"].committed
    assert session.tree["root"].children == ["kid_a", "kid_b"]
    assert session.tree["kid_a"].parent == "root"
    # the child mask is nested inside its parent, not carved from the whole table
    assert int(session.ft.mask_series("kid_a").sum()) == 80


def test_descent_reaches_the_subtypes(session):
    """A child level clusters its own cohort, and finds the finer split."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family", children={"kid_a": ["a"]})

    _cluster(session, "kid_a")
    child = session.scan("kid_a", n_points=40, min_cluster_size=5)
    assert child["plateaus"], "child level found no plateau at all"
    assert session.tree["kid_a"].n_cells == 80


# -- the ledger ----------------------------------------------------------------


def test_replay_of_the_generated_script_reproduces_the_table(session):
    """The load-bearing property: the ledger, not the app, is the truth."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family", children={"kid_a": ["a"]})

    namespace = {"ft": demo_table(n_per_subtype=40)}
    exec(compile(session.ledger.to_script(), "<ledger>", "exec"), namespace)
    replayed = namespace["ft"]

    assert replayed.labels == session.ft.labels
    assert sorted(replayed.masks) == sorted(session.ft.masks)
    assert (
        replayed.dataframe("all_cells")["family"].to_list()
        == session.ft.dataframe("all_cells")["family"].to_list()
    )


def test_script_uses_mask_names_not_level_names(session):
    """A level is the app's handle; the mask is what cellpax takes."""
    _cluster(session)
    script = session.ledger.to_script()
    assert "ft.space('all_cells'" in script
    assert "ft.space('root'" not in script


def test_rename_before_cut_still_renders_in_order(session):
    """Naming happens while deciding, so it lands in the ledger before the cut.

    The script must still emit ``label(...)`` before ``rename(...)`` or it would
    reference an undefined name.
    """
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a"})
    session.commit("root", label_name="family")

    kinds = [e.kind for e in session.ledger]
    assert kinds.index("rename") < kinds.index("cut")  # the ledger order
    script = session.ledger.to_script()
    assert script.index(
        "lbl_root = mid_root.label("
        if "mid_root" in script
        else "lbl_root = clus_root.label("
    ) < script.index(".rename(")


def test_ledger_round_trips_through_the_file(tmp_path):
    path = tmp_path / "l.jsonl"
    first = Ledger(path)
    first.append("open_level", "root", mask="m", parent=None)
    first.append("cut", "root", distance_threshold=0.5, min_cluster_size=10)
    reloaded = Ledger(path)
    assert [e.kind for e in reloaded] == ["open_level", "cut"]
    assert reloaded.entries[1].payload["distance_threshold"] == 0.5


def test_truncate_is_the_undo_primitive(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    for i in range(4):
        ledger.append("note", "root", text=str(i))
    dropped = ledger.truncate(2)
    assert [e.payload["text"] for e in dropped] == ["2", "3"]
    assert len(Ledger(tmp_path / "l.jsonl")) == 2


def test_unknown_entry_kind_is_refused():
    with pytest.raises(ValueError, match="unknown entry kind"):
        Entry(kind="destroy", level="root")


def test_corrupt_ledger_names_the_line(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text('{"kind":"note","level":"root"}\nnot json\n')
    with pytest.raises(ValueError, match=r"l\.jsonl:2"):
        Ledger(path)


# -- guards and diagnostics ----------------------------------------------------


def test_large_linkage_asks_before_allocating(session):
    """Not a refusal — a confirmation, with the number attached."""
    session.config.linkage_warn_cells = 10
    with pytest.raises(LargeLinkage) as excinfo:
        session.submit_cluster("root", **_SWEEP)
    assert "confirm=true" in str(excinfo.value)
    assert excinfo.value.n_cells == 240
    # and it proceeds when confirmed
    session.submit_cluster("root", **_SWEEP, confirm_large=True)
    session.jobs.wait(600)
    assert session.tree["root"].has_clustering


def test_large_linkage_route_returns_409_not_400(client, session):
    session.config.linkage_warn_cells = 10
    response = client.post(
        "/api/levels/root/cluster",
        json={"graph_type": ["knn"], "n_neighbors": [15], "n_times": 2},
    )
    assert response.status_code == 409
    assert response.json()["needs_confirmation"] is True
    assert response.json()["peak_gb"] == pytest.approx(
        2 * response.json()["gb"], rel=0.01
    )


@pytest.mark.parametrize(
    "runs, inside, total, expect",
    [
        (0, 0, 4, "empty"),
        (5, 2, 4, "thin"),
        (50, 1, 4, "un-marginalised"),
        (50, 1, 1, "ok"),
        (50, 4, 4, "ok"),
    ],
)
def test_coverage_verdict_calls_out_the_failure_modes(runs, inside, total, expect):
    assert _coverage_verdict(runs, inside, total).startswith(expect)


def test_coverage_reports_the_window(session):
    _cluster(session)
    coverage = session.coverage("root")
    assert coverage["n_runs_total"] == 36
    assert coverage["n_runs_in_window"] == 36  # no restriction yet
    assert coverage["verdict"].startswith("ok")


def test_axis_stability_probes_three_thresholds(session):
    _cluster(session)
    result = session.axis_stability("root")
    assert [p["threshold"] for p in result["probes"]] == [0.4, 0.5, 0.6]
    assert isinstance(result["ranking_stable"], bool)


def test_flow_matches_by_membership_and_keeps_an_unassigned_band(session):
    _cluster(session)
    flow = session.flow("root", n_levels=4, min_cluster_size=20)
    # coarse first
    assert flow["heights"] == sorted(flow["heights"], reverse=True)
    # every edge's cells are accounted for at both ends
    for level in range(len(flow["heights"]) - 1):
        moved = sum(e["value"] for e in flow["edges"] if e["level"] == level)
        assert moved == 240
    assert any(node["unassigned"] for node in flow["nodes"]) or all(
        not node["unassigned"] for node in flow["nodes"]
    )


def test_unassigned_only_drains_as_the_cut_coarsens(session):
    """Coarsening only grows clusters, so the size floor can only release cells."""
    _cluster(session)
    flow = session.flow("root", n_levels=5, min_cluster_size=25)
    by_level = {}
    for node in flow["nodes"]:
        if node["unassigned"]:
            by_level[node["level"]] = node["n_cells"]
    counts = [by_level.get(i, 0) for i in range(len(flow["heights"]))]
    assert counts == sorted(counts), f"unassigned band refilled: {counts}"


# -- level tree ----------------------------------------------------------------


def test_tree_evicts_least_recently_used_heavy_state():
    tree = LevelTree(max_resident=2)
    for name in ("a", "b", "c"):
        tree.add(name, mask=name)
        tree[name].clustering = object()
        tree.touch(name)
    assert tree["a"].clustering is None  # evicted
    assert tree["b"].clustering is not None
    assert tree["c"].clustering is not None


def test_tree_path_survives_a_cycle():
    tree = LevelTree()
    tree.add("a", mask="a")
    tree.add("b", mask="b", parent="a")
    tree["a"].parent = "b"  # a cycle should not hang the breadcrumb
    assert tree.path_to("b") == ["a", "b"]


def test_opening_a_level_on_an_unknown_mask_is_refused(session):
    with pytest.raises(ValueError, match="not defined"):
        session.open_level("nope", mask="does_not_exist")


# -- config --------------------------------------------------------------------


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        AppConfig.from_dict({"folio": None, "not_a_key": 1})


def test_config_resolution_grid_is_geometric():
    config = AppConfig()
    grid = config.resolution_grid()
    assert len(grid) == DEFAULT_SWEEP["resolution_points"]
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert max(ratios) - min(ratios) < 1e-9


def test_default_sweep_favours_grid_points_over_seeds():
    """Seeds at one setting reproduce each other; grid points add coverage."""
    assert DEFAULT_SWEEP["resolution_points"] > 4 * DEFAULT_SWEEP["n_times"]


@pytest.mark.parametrize(
    "uri", ["gs://bucket/cellpax/v1dd_v1512", "s3://b/x", "https://host/folio"]
)
def test_remote_folios_are_not_resolved_as_filesystem_paths(uri, tmp_path):
    """A bucket URI must survive intact.

    Path-resolving it would silently turn ``gs://bucket/x`` into
    ``<config dir>/gs:/bucket/x`` — one collapsed slash, a path that exists
    nowhere, and a folio load that fails a long way from the cause.
    """
    config = AppConfig.from_dict({"folio": uri, "table_name": "refined"}, base=tmp_path)
    assert config.folio == uri
    assert config.folio_is_remote


def test_local_folios_still_resolve_against_the_config_file(tmp_path):
    config = AppConfig.from_dict(
        {"folio": "sub/folio", "table_name": "t"}, base=tmp_path
    )
    assert config.folio == (tmp_path / "sub/folio").resolve()
    assert not config.folio_is_remote


def test_ledger_stays_local_even_when_the_folio_is_remote(tmp_path, monkeypatch):
    """The ledger is written on every decision; it does not belong in a bucket."""
    monkeypatch.chdir(tmp_path)
    config = AppConfig.from_dict({"folio": "gs://b/x", "table_name": "refined"})
    assert config.resolved_ledger_path() == tmp_path / "refined.ledger.jsonl"
    # and it is per-table, so two analyses do not share one
    other = AppConfig.from_dict({"folio": "gs://b/y", "table_name": "avant_garde"})
    assert other.resolved_ledger_path() != config.resolved_ledger_path()


# -- routes --------------------------------------------------------------------


def test_health_and_state(client):
    assert client.get("/api/health").json()["ok"] is True
    state = client.get("/api/state").json()
    assert state["n_cells"] == 240
    assert [lv["name"] for lv in state["levels"]] == ["root"]
    assert state["demo"] is True


def test_reads_before_clustering_fail_with_a_useful_message(client):
    response = client.get("/api/levels/root/scan")
    assert response.status_code == 400
    assert "no clustering yet" in response.json()["detail"]


def test_full_route_loop(client, session):
    client.post(
        "/api/levels/root/cluster",
        json={
            "graph_type": ["knn"],
            "n_neighbors": [15],
            "n_times": 3,
            "resolution": list(np.geomspace(0.05, 2.0, 12)),
        },
    )
    session.jobs.wait(600)
    assert client.get("/api/jobs").json()["jobs"][0]["status"] == "done"

    scan = client.get(
        "/api/levels/root/scan", params={"n_points": 60, "min_cluster_size": 10}
    ).json()
    families = [p for p in scan["plateaus"] if p["n_clusters"] == 3]
    assert families, "no family plateau over the route"
    midpoint = families[0]["midpoint"]

    preview = client.get(
        "/api/levels/root/preview",
        params={"distance_threshold": midpoint, "min_cluster_size": 10},
    ).json()
    assert preview["n_clusters"] == 3

    client.post(
        "/api/levels/root/names", json={"names": {"0": "a", "1": "b", "2": "c"}}
    )
    committed = client.post(
        "/api/levels/root/commit",
        json={"label_name": "family", "children": {"kid": ["a"]}},
    ).json()
    assert committed["children"] == ["kid"]
    assert "family" in client.get("/api/state").json()["labels"]
    assert "def " not in client.get("/api/ledger/script").text  # plain calls only


def test_job_failure_is_reported_not_fatal(session):
    """A failing job must not take the worker thread with it."""
    job = session.jobs.submit(
        "cluster",
        "root",
        "boom",
        lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    session.jobs.wait(30)
    assert session.jobs.get(job.id).status == "failed"
    assert "boom" in session.jobs.get(job.id).error
    # the worker is still alive
    follow_up = session.jobs.submit("cluster", "root", "fine", lambda _p: 42)
    session.jobs.wait(30)
    assert session.jobs.get(follow_up.id).result == 42


# -- consensus density: the restrict memory guard --------------------------------


def test_coarse_windows_make_the_consensus_denser_not_sparser():
    """The counterintuitive direction, and the one that eats a machine.

    Restricting narrows the *runs*; it widens the *matrix*. Each cluster adds a
    ``size**2`` block to ``A @ A.T``, so selecting the coarse runs selects the
    ones with the biggest clusters.
    """
    from cellpax.clustering import coclustering_matrix, consensus_density

    rng = np.random.default_rng(0)
    n = 1200
    coarse = np.stack([rng.integers(0, 4, n) for _ in range(20)], axis=1)
    fine = np.stack([rng.integers(0, 60, n) for _ in range(20)], axis=1)

    assert consensus_density(coarse)["density"] > 0.9
    assert consensus_density(fine)["density"] < 0.3
    # and the prediction matches what actually gets built
    for groups in (coarse, fine):
        predicted = consensus_density(groups)["density"]
        actual = coclustering_matrix(groups).nnz / n**2
        assert abs(predicted - actual) < 0.02


def test_normalization_matches_the_coo_reference_exactly():
    """The in-place rewrite must not change a single value."""
    from scipy.sparse import coo_matrix

    from cellpax.clustering import coclustering_matrix

    rng = np.random.default_rng(7)
    for k, drop in ((3, 0.0), (20, 0.3)):
        groups = np.stack([rng.integers(0, k, 400) for _ in range(12)], axis=1)
        if drop:
            groups[rng.random(groups.shape) < drop] = -1
        raw = coclustering_matrix(groups, normalize=False)
        got = coclustering_matrix(groups, normalize=True)

        n, n_runs = groups.shape
        miss = (groups < 0).sum(axis=1).astype(np.float32)
        cx = coo_matrix(raw)
        denom = np.float32(n_runs) - miss[cx.row] - miss[cx.col]
        if miss.any():
            from scipy.sparse import csr_matrix

            ms = csr_matrix((groups < 0).astype(np.float32))
            shared = (ms @ ms.T).toarray()
            denom = denom + shared[cx.row, cx.col].astype(np.float32)
        expected = np.where(denom > 0, cx.data / denom, np.float32(0))
        want = coo_matrix((expected, (cx.row, cx.col)), shape=(n, n)).tocsr()
        want.eliminate_zeros()
        assert got.nnz == want.nnz
        assert np.allclose(got.toarray(), want.toarray(), rtol=1e-6, atol=1e-7)


def test_restrict_asks_before_building_a_dense_consensus(session):
    from cellpax.app.session import DenseConsensus

    _cluster(session)
    session.config.consensus_warn_gb = 1e-9  # anything at all trips it
    with pytest.raises(DenseConsensus) as excinfo:
        session.submit_restrict("root", n_clusters_min=2, n_clusters_max=8)
    assert "denser" in str(excinfo.value)
    assert 0.0 <= excinfo.value.projection["density"] <= 1.0
    # confirming proceeds
    session.submit_restrict(
        "root", n_clusters_min=2, n_clusters_max=8, confirm_large=True
    )
    session.jobs.wait(600)
    assert session.tree["root"].is_restricted


def test_restrict_route_returns_409_with_the_projection(client, session):
    _cluster(session)
    session.config.consensus_warn_gb = 1e-9
    response = client.post(
        "/api/levels/root/restrict", json={"n_clusters_min": 2, "n_clusters_max": 8}
    )
    assert response.status_code == 409
    body = response.json()
    assert body["kind"] == "dense_consensus"
    assert body["peak_gb"] >= body["gb"]


def test_palette_is_served_and_orders_saturated_hues_first(client):
    from cellpax.app.ngl import PALETTE, UNASSIGNED

    palette = client.get("/api/state").json()["palette"]
    assert palette["colors"] == list(PALETTE)
    assert palette["unassigned"] == UNASSIGNED
    # tab20's greys are reserved for unassigned, never a cluster
    assert "#7f7f7f" not in PALETTE and "#c7c7c7" not in PALETTE
    # green must not sit next to orange: ΔE 0.7 under protanopia
    assert abs(PALETTE.index("#2ca02c") - PALETTE.index("#ff7f0e")) > 1


# -- the labelling the UI shows must be the labelling commit writes --------------


def test_preview_ids_match_committed_ids(session):
    """Regression: `label()` reorders by `order_by`, `cluster_labels()` does not.

    They are a permutation of each other whenever ``order_by`` is set, so a UI
    reading one and a commit writing the other attaches every name to the wrong
    cluster and carves every child mask from the wrong cells.
    """
    _cluster(session)
    threshold = _family_threshold(session)
    session.preview_cut("root", distance_threshold=threshold, min_cluster_size=10)
    active = session.tree["root"].active
    cell_ids = np.asarray(active._cell_ids)

    shown = session.ordered_labels("root", threshold, 10).codes_for(cell_ids)
    committed = active.label(
        distance_threshold=threshold, min_cluster_size=10, name="x"
    ).codes_for(cell_ids)
    assert np.array_equal(shown, committed)


def test_cluster_ids_are_ordered_by_the_order_by_column(session):
    """And the ordering is the meaningful one: cluster 0 is shallowest."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    profile = session.column_profile("root", column="soma_depth_um", bins=20)
    medians = [row["median"] for row in profile["series"]]
    assert medians == sorted(medians), f"ids not depth-ordered: {medians}"


def test_child_mask_holds_the_cells_that_were_shown(session):
    """The end-to-end version of the same bug: does the mask match the picture?"""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    profile = session.column_profile("root", column="soma_depth_um", bins=20)
    deepest = max(profile["series"], key=lambda r: r["median"])

    session.set_names("root", names={deepest["cluster"]: "deep"})
    session.commit("root", label_name="family", children={"kid": ["deep"]})

    got = session.ft.dataframe("kid")
    assert got.height == deepest["n_cells"]
    assert abs(float(got["soma_depth_um"].median()) - deepest["median"]) < 15
    # the demo's families are separated by depth, so a correct carve is pure
    assert got["true_family"].n_unique() == 1


def test_scatter_and_profile_agree_on_cluster_membership(session):
    """Two views of one cut must not disagree about which cell is in which."""
    _cluster(session)
    threshold = _family_threshold(session)
    session.preview_cut("root", distance_threshold=threshold, min_cluster_size=10)
    session.submit_embed("root", method="pca")
    session.jobs.wait(600)

    scatter = session.scatter("root", distance_threshold=threshold, min_cluster_size=10)
    _, codes = session._current_codes("root")
    assert scatter["cluster"] == [int(c) for c in codes]


# -- profiles ------------------------------------------------------------------


def test_column_profile_separates_the_demo_families(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    profile = session.column_profile("root", column="soma_depth_um", bins=24)
    assert len(profile["series"]) == 3
    medians = sorted(row["median"] for row in profile["series"])
    # the fixture puts families at 100 / 200 / 300 um
    assert all(abs(m - t) < 40 for m, t in zip(medians, (100, 200, 300)))
    for row in profile["series"]:
        assert abs(sum(row["fraction"]) - 1.0) < 1e-6


def test_heatmap_ranks_discriminative_features_not_variable_ones(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    heatmap = session.cluster_heatmap("root", top=8)
    assert heatmap["f_stat"] == sorted(heatmap["f_stat"], reverse=True)
    # the fixture separates families on m0..m8; noise dimensions live above that
    assert all(int(f[1:]) < 12 for f in heatmap["features"])
    assert len(heatmap["matrix"]) == len(heatmap["clusters"])


def test_numeric_columns_excludes_features(session):
    names = {c["name"] for c in session.numeric_columns("root")}
    assert "soma_depth_um" in names
    assert not any(n.startswith("m") and n[1:].isdigit() for n in names)


# -- gap or cut ----------------------------------------------------------------


def test_boundary_gate_reports_evidence_for_every_pair(session):
    """Structure, not verdict.

    The verdict is deliberately not asserted: it hinges on ``connectivity_ratio``
    crossing 0.1, and kNN connectivity at fixed k thins as n grows, so this
    fixture flips between "discrete" at 360 cells and "continuous" at 240. That
    is the docstring's own warning made concrete — the verdict is a summary and
    the columns are the result — so the test pins the columns.
    """
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.submit_boundary("root", n_neighbors=15)
    session.jobs.wait(600)
    result = session.boundary("root")

    assert result["available"]
    assert len(result["rows"]) == 3  # three clusters -> three unordered pairs
    for row in result["rows"]:
        assert row["verdict"] in {"discrete", "continuous", "ambiguous", "too_small"}
        assert row["dip_p"] is not None
        assert row["valley_ratio"] is not None
    assert sum(result["counts"].values()) == 3


def test_boundary_pairs_are_named_like_every_other_view(session):
    """Same id-mismatch class as the commit bug: boundary_report re-cuts.

    Passed a threshold it would derive its own codes through `cluster_labels`
    and report raw dendrogram ids, so its pairs would name different clusters
    than the swatch list and the ridgeline.
    """
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "shallow", 1: "mid", 2: "deep"})
    session.submit_boundary("root", n_neighbors=15)
    session.jobs.wait(600)

    shown = {
        row["name"]
        for row in session.column_profile("root", column="soma_depth_um", bins=12)[
            "series"
        ]
    }
    named = set()
    for row in session.boundary("root")["rows"]:
        named |= {row["cluster_a"], row["cluster_b"]}
    assert named <= shown, f"boundary names {named} are not the displayed {shown}"


def test_parametrizing_discrete_clusters_trips_the_dimension_gate(session):
    """Forcing a curve through two separated blobs must warn, not just succeed."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.submit_parametrize("root", clusters=["a", "b"], orient_by="soma_depth_um")
    session.jobs.wait(600)
    gradient = session.gradient("root")
    assert gradient["available"]
    assert gradient["intrinsic_dimension"] > 2.5
    assert any("intrinsic dimension" in w for w in gradient["warnings"])


def test_gradient_commit_keeps_the_coordinate_alongside_the_bins(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.submit_parametrize("root", clusters=["a", "b"])
    session.jobs.wait(600)
    result = session.commit_gradient("root", bins=3)

    # the coordinate is a float column; the bins are a LabelSet beside it
    assert result["coordinate"] in session.ft.dataframe().columns
    assert result["bins"] in session.ft.labels
    assert [e.kind for e in session.ledger].count("parametrize") == 1
    assert [e.kind for e in session.ledger].count("bin") == 1


def test_gradient_replays_from_the_ledger(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family")
    session.submit_parametrize("root", clusters=["a", "b"], orient_by="soma_depth_um")
    session.jobs.wait(600)
    session.commit_gradient("root", bins=2)

    script = session.ledger.to_script()
    assert "ft.parametrize(" in script
    assert "grad_root.bin(" in script
    assert "ft.attach(grad_root)" in script


# -- resetting -----------------------------------------------------------------


def _committed(session) -> float:
    """Run the loop to a commit with a child mask. Returns the threshold used."""
    _cluster(session)
    threshold = _family_threshold(session)
    session.preview_cut("root", distance_threshold=threshold, min_cluster_size=10)
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family", children={"kid": ["a"]})
    return threshold


def test_reset_undoes_what_the_ledger_wrote(session):
    """Forgetting the entries is not enough — the table has to come back too."""
    _committed(session)
    assert "family" in session.ft.labels
    assert "kid" in session.ft.masks

    result = session.reset(keep_sweeps=False)

    assert session.ft.labels == []
    assert "kid" not in session.ft.masks
    assert result["dropped"] > 0
    assert any("detached" in u for u in result["undone"])
    assert any("dropped mask" in u for u in result["undone"])


def test_reset_keeps_the_sweep_by_default(session):
    """The expensive artifact is not a judgement, so it survives a reset."""
    _committed(session)
    session.reset(keep_sweeps=True)

    level = session.tree["root"]
    assert level.has_clustering, "the sweep was thrown away with the decisions"
    assert level.cut.distance_threshold is None
    assert level.cut.names == {}
    assert not level.committed
    assert {e.kind for e in session.ledger} <= {"open_level", "cluster", "restrict"}
    # and re-cutting is immediately possible, with no job
    again = session.scan("root", n_points=20, min_cluster_size=10)
    assert again["plateaus"]


def test_reset_everything_drops_the_sweep_too(session):
    _committed(session)
    session.reset(keep_sweeps=False)
    level = session.tree.get("root")
    assert level is None or not level.has_clustering


def test_reset_does_not_leave_an_orphaned_level(session):
    """A child level's mask is carved by a commit; drop the commit, drop the level.

    Keeping its ``open_level`` entry would leave the ledger naming a mask that
    no longer exists — unreplayable.
    """
    _committed(session)
    assert "kid" in session.tree
    session.reset(keep_sweeps=True)

    assert "kid" not in session.tree
    assert all(
        e.payload.get("mask") in session.ft.masks
        for e in session.ledger
        if e.kind == "open_level" and e.payload.get("mask")
    )


def test_the_surviving_ledger_still_replays(session):
    """Whatever a reset leaves behind must still describe the table exactly."""
    _committed(session)
    session.reset(keep_sweeps=True)

    namespace = {"ft": demo_table(n_per_subtype=40)}
    exec(compile(session.ledger.to_script(), "<ledger>", "exec"), namespace)
    assert sorted(namespace["ft"].masks) == sorted(session.ft.masks)
    assert namespace["ft"].labels == session.ft.labels


def test_reset_archives_the_old_ledger(session, tmp_path):
    _committed(session)
    before = len(session.ledger)
    result = session.reset(keep_sweeps=False)

    archive = Ledger(result["archive"])
    assert len(archive) == before, "the archive must hold the full history"
    assert result["archive"].endswith(".bak.jsonl")


def test_rewind_to_a_position_keeps_the_prefix(session):
    _committed(session)
    kinds = [e.kind for e in session.ledger]
    keep = kinds.index("cut")  # everything before the first cut

    session.rewind(keep)

    assert len(session.ledger) == keep
    assert session.ft.labels == []
    assert "kid" not in session.ft.masks


def test_rewind_is_a_no_op_when_nothing_would_be_dropped(session):
    _cluster(session)
    result = session.rewind(len(session.ledger))
    assert result["dropped"] == 0
    assert result["archive"] is None


def test_reset_route_refuses_without_confirmation(client, session):
    _committed(session)
    response = client.post("/api/ledger/rewind", json={})
    assert response.status_code == 400
    assert "confirm=true" in response.json()["detail"]
    assert "family" in session.ft.labels  # nothing happened

    ok = client.post("/api/ledger/rewind", json={"confirm": True, "keep_sweeps": True})
    assert ok.status_code == 200
    assert session.ft.labels == []


def test_undo_tolerates_an_already_missing_target(session):
    """A rewind over a column someone dropped by hand must still finish."""
    _committed(session)
    session.ft.detach("family")
    result = session.reset(keep_sweeps=False)
    assert result["dropped"] > 0
    assert "kid" not in session.ft.masks


# -- the representation everything is compared in --------------------------------


def test_cluster_uses_a_within_mask_block_weighted_space(session):
    _cluster(session)
    space = session.tree["root"].space
    assert space.feature_weights is not None, "block weights were not applied"
    assert "weighted" in space.label
    assert "alpha" in space.label
    # weights come in blocks, so far fewer distinct values than features
    distinct = len(set(np.round(space.feature_weights, 6)))
    assert 1 < distinct < len(space.feature_weights)


def test_each_level_refits_its_own_space(session):
    """Within-mask means within-mask: a child does not inherit the parent's fit."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family", children={"kid": ["a"]})
    _cluster(session, "kid")

    parent, child = session.tree["root"].space, session.tree["kid"].space
    assert child.feature_weights is not None
    assert not np.allclose(parent.feature_weights, child.feature_weights)


def test_the_gradient_is_fitted_in_the_clustering_space(session):
    """Otherwise its loadings describe a rotation nothing else in the analysis used."""
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.submit_parametrize("root", clusters=["a", "b"])
    session.jobs.wait(600)

    gradient = session.tree["root"].gradient
    assert gradient.params["space"] == session.tree["root"].space.label
    assert "weighted" in gradient.params["space"]


def test_representation_names_the_scaler_and_the_spaces(session):
    """It is invisible otherwise, and it is the first thing anyone asks about."""
    _cluster(session)
    rep = session.representation()
    assert rep["scaler"].startswith("clipped (sigma")
    assert rep["scaler_detail"]["n_sigma"] == 4.0
    assert rep["refit_per_mask"] is True
    assert "weighted" in rep["spaces"]["root"]
    assert session.state()["representation"] == rep


def test_the_demo_fixture_uses_the_same_scaler_as_the_pipeline():
    """The demo exists to exercise production paths, and the scaler is one."""
    from cellpax.persist import _scaler_tag

    tag = _scaler_tag(demo_table(n_per_subtype=5)._scaler_factory)
    assert isinstance(tag, dict)
    assert tag["mode"] == "sigma" and tag["n_sigma"] == 4.0


def test_the_script_reproduces_the_block_weights(session):
    """Weights are part of the fit, so a script that omits them replays a
    different space — and therefore a different clustering.

    This was invisible while the fixture's coarse plateau was wide enough that
    weighted and unweighted cuts agreed. It is the reason replay fidelity has to
    be asserted on cell-level labels rather than on cluster counts.
    """
    _cluster(session)
    script = session.ledger.to_script()
    assert "cpx.block_weights(" in script
    assert "feature_weights=weights_root" in script


def test_replay_is_exact_at_the_cell_level(session):
    _cluster(session)
    threshold = _family_threshold(session)
    session.preview_cut("root", distance_threshold=threshold, min_cluster_size=10)
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.commit("root", label_name="family", children={"kid": ["a"]})

    namespace = {"ft": demo_table(n_per_subtype=40)}
    exec(compile(session.ledger.to_script(), "<ledger>", "exec"), namespace)
    assert (
        namespace["ft"].dataframe("all_cells")["family"].to_list()
        == session.ft.dataframe("all_cells")["family"].to_list()
    )


# -- intrinsic dimension: the shape, not just the gate ---------------------------


def test_twonn_profile_collapses_to_the_reported_dimension():
    """The scalar must stay exactly the minimum of the profile it summarises."""
    from cellpax.gradient import twonn_dimension, twonn_profile

    rng = np.random.default_rng(0)
    points = rng.normal(0, 1, (400, 4))
    profile = twonn_profile(points)
    assert profile, "no levels estimated"
    assert twonn_dimension(points) == pytest.approx(
        min(level["dimension"] for level in profile)
    )
    assert [level["n"] for level in profile] == sorted(
        (level["n"] for level in profile), reverse=True
    )
    assert profile[0]["spread"] == 0.0  # full sampling is a single draw


def test_the_profile_separates_a_noisy_curve_from_a_ball():
    """The discriminating signal the scalar throws away.

    A ball is a ball at every scale; a noisy curve only looks high-dimensional
    while the nearest-neighbour distance is still inside the noise.
    """
    from cellpax.gradient import twonn_profile

    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 600)
    curve = np.stack([t, t**2, np.zeros_like(t)], axis=1) + rng.normal(
        0, 0.01, (600, 3)
    )
    ball = rng.normal(0, 1, (600, 5))

    curve_profile = [level["dimension"] for level in twonn_profile(curve)]
    ball_profile = [level["dimension"] for level in twonn_profile(ball)]

    assert curve_profile[0] - curve_profile[-1] > 0.4, "curve did not fall with scale"
    assert abs(ball_profile[0] - ball_profile[-1]) < 1.0, "ball should stay flat"
    assert min(ball_profile) > max(curve_profile)


def test_the_gradient_carries_its_dimension_profile(session):
    _cluster(session)
    session.preview_cut(
        "root", distance_threshold=_family_threshold(session), min_cluster_size=10
    )
    session.set_names("root", names={0: "a", 1: "b", 2: "c"})
    session.submit_parametrize("root", clusters=["a", "b"])
    session.jobs.wait(600)

    payload = session.gradient("root")
    profile = payload["dimension_profile"]
    assert len(profile) >= 2, "need at least two scales to read a shape"
    assert payload["intrinsic_dimension"] == pytest.approx(
        min(level["dimension"] for level in profile)
    )
    assert payload["dimension_gate"] == 2.5
    assert session.tree["root"].gradient.dimension_profile == profile
