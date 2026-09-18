"""The ``Clustering`` object: ft.cluster -> Clustering -> LabelSet -> ft.attach."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from cellpax.clustering import Clustering, SimilarityMatrix
from cellpax.featuretable import FeatureTable

_PARAMS = dict(n_neighbors=15, n_times=3, seed=0, n_jobs=1)


def _table(n: int = 80, dim: int = 20, sep: int = 3) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1.0, (n, dim))
    coords[n // 2 :, :sep] += 8.0
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
            "keep": [True, True, True, False] * (n // 4),
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(dim)])
    ft.add_mask("core", pl.col("keep"))
    return ft


# -- the chain -----------------------------------------------------------------


def test_cluster_returns_a_clustering_that_labels_itself() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS)

    assert isinstance(clus, Clustering)
    labels = clus.label(distance_threshold=0.5, name="subclass")
    ft.attach(labels)

    assert labels.mask == "core"
    assert len(labels) == int(ft.mask_series("core").sum())
    assert "subclass" in ft.dataframe("core").columns


def test_clustering_is_still_a_similarity_matrix() -> None:
    """Subclassing, so every existing SimilarityMatrix use keeps working."""
    ft = _table()
    clus = ft.cluster("core", **_PARAMS)

    assert isinstance(clus, SimilarityMatrix)
    assert clus.shape[0] == clus.shape[1]
    assert clus.linkage.shape[1] == 4
    assert clus.cluster_labels(0.5).shape[0] == clus.shape[0]
    x, counts = clus.cluster_count_curve(n_points=10)
    assert len(x) == len(counts) == 10


def test_clustering_records_its_provenance() -> None:
    ft = _table()
    ft.define_features("motion", columns=["m0", "m1"])

    default = ft.cluster("core", **_PARAMS)
    assert default.mask == "core"
    assert default.columns == tuple(ft.feature_columns)
    assert default.space == "pca(0.95)"
    np.testing.assert_array_equal(default.cell_ids, ft._cell_ids("core"))

    narrow = ft.cluster("core", columns="motion", pca=False, **_PARAMS)
    assert narrow.columns == ("m0", "m1")
    assert narrow.space == "scaled"

    assert "mask='core'" in repr(default) and "pca(0.95)" in repr(default)


def test_cell_ids_is_a_copy() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS)

    ids = clus.cell_ids
    ids[0] = -12345
    assert clus.cell_ids[0] != -12345


def test_clustering_rejects_mismatched_cell_ids() -> None:
    matrix = np.eye(4)
    with pytest.raises(ValueError, match="must describe the same cells"):
        Clustering(matrix, cell_ids=np.arange(3), mask="all")


# -- agreement with the old two-step form --------------------------------------


def test_label_matches_ft_label_with_an_explicit_mask() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")

    direct = clus.label(distance_threshold=0.5, name="x")
    via_table = ft.label("run", mask="core", distance_threshold=0.5, name="x")

    np.testing.assert_array_equal(direct.codes, via_table.codes)
    np.testing.assert_array_equal(direct.cell_ids, via_table.cell_ids)
    assert direct.mask == via_table.mask == "core"


def test_ft_label_takes_the_mask_from_a_clustering() -> None:
    ft = _table()
    ft.cluster("core", **_PARAMS, name="run")

    # no mask= given, yet it does not fall back to "all"
    labels = ft.label("run", distance_threshold=0.5)
    assert labels.mask == "core"
    assert len(labels) == int(ft.mask_series("core").sum())


def test_ft_label_refuses_a_mask_the_clustering_was_not_computed_on() -> None:
    ft = _table()
    ft.cluster("core", **_PARAMS, name="run")

    with pytest.raises(ValueError, match="was computed on mask 'core'"):
        ft.label("run", mask="all", distance_threshold=0.5)


def test_ft_label_still_accepts_a_bare_similarity_matrix() -> None:
    """The escape hatch for matrices computed outside the table."""
    ft = _table()
    from cellpax.clustering import fauxnograph_coclustering

    data = ft.features("core", scaled=True)
    bare = SimilarityMatrix(
        fauxnograph_coclustering(data, n_neighbors=15, n_times=3, seed=0, n_jobs=1),
        normalized=True,
    )
    labels = ft.label(bare, mask="core", distance_threshold=0.5)
    assert labels.mask == "core"


# -- persistence ---------------------------------------------------------------


def test_provenance_survives_save_load(tmp_path) -> None:
    from cellpax import load_feature_table, save_feature_table

    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")
    expected = clus.label(distance_threshold=0.5, name="subclass")

    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")
    back = load_feature_table(path, name="t")

    restored = back.clustering("run")
    assert isinstance(restored, Clustering)
    assert restored.mask == "core"
    assert restored.columns == tuple(ft.feature_columns)
    assert restored.space == "pca(0.95)"
    np.testing.assert_array_equal(
        restored.label(distance_threshold=0.5, name="subclass").codes, expected.codes
    )


def test_load_defaults_the_mask_when_an_old_manifest_lacks_provenance(tmp_path) -> None:
    """Analyses saved before provenance existed still load, as mask 'all'."""
    from datafolio import DataFolio

    from cellpax import load_feature_table, save_feature_table

    ft = _table()
    ft.cluster(**_PARAMS, name="run")  # the implicit "all" mask

    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")
    folio = DataFolio(path)
    manifest = folio.get("t/manifest")
    for meta in manifest["clusterings"].values():
        for key in ("mask", "columns", "space"):
            meta.pop(key, None)
    folio.add("t/manifest", manifest, overwrite=True)

    restored = load_feature_table(path, name="t").clustering("run")
    assert restored.mask == "all"
    assert restored.columns == ()
    assert restored.label(distance_threshold=0.5).mask == "all"


# -- ordering (order_by) -------------------------------------------------------


def _depth_table(n: int = 200, dim: int = 12) -> tuple[FeatureTable, np.ndarray]:
    """Four well-separated blobs whose depths are deliberately out of blob order."""
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 6, size=(4, dim))
    coords = np.vstack([c + rng.normal(0, 1.0, (n // 4, dim)) for c in centers])
    depth = np.concatenate(
        [np.full(n // 4, d) for d in (400.0, 100.0, 300.0, 200.0)]
    ) + rng.normal(0, 3, n)
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(n), dtype=pl.Int64),
            **{f"m{i}": coords[:, i] for i in range(dim)},
            "soma_depth_um": depth,
        }
    )
    ft = FeatureTable(df, features=[f"m{i}" for i in range(dim)])
    ft.add_mask("core", pl.col("cell_id") >= 0)
    return ft, depth


def _means_in_id_order(labels, values: np.ndarray) -> list[float]:
    return [float(np.nanmean(values[labels.codes == i])) for i in labels.ids]


def test_order_by_numbers_clusters_by_the_column() -> None:
    ft, depth = _depth_table()
    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um")

    assert clus.order_by == "soma_depth_um"
    means = _means_in_id_order(clus.label(distance_threshold=0.5), depth)
    assert means == sorted(means)


def test_order_by_descending_and_order_false() -> None:
    ft, depth = _depth_table()
    down = ft.cluster(
        "core", **_PARAMS, order_by="soma_depth_um", order_ascending=False
    )
    means = _means_in_id_order(down.label(distance_threshold=0.5), depth)
    assert means == sorted(means, reverse=True)

    raw = down.label(distance_threshold=0.5, order=False)
    assert _means_in_id_order(raw, depth) != means


def test_order_by_renames_the_default_id_names_to_match() -> None:
    """A fresh cut's names are just its ids, so they must track the renumbering."""
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um")
    labels = clus.label(distance_threshold=0.5)

    assert labels.names == [str(i) for i in labels.ids]


def test_order_by_survives_null_values() -> None:
    ft, depth = _depth_table()
    ft._df = ft._df.with_columns(
        pl.when(pl.col("cell_id") < 3)
        .then(None)
        .otherwise(pl.col("soma_depth_um"))
        .alias("soma_depth_um")
    )
    depth = depth.copy()
    depth[:3] = np.nan

    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um")
    means = _means_in_id_order(clus.label(distance_threshold=0.5), depth)
    assert not any(np.isnan(means))
    assert means == sorted(means)


def test_order_by_is_consistent_across_thresholds() -> None:
    """The point of ordering at cut time: two cuts number the same way."""
    ft, depth = _depth_table()
    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um")

    for threshold in (0.3, 0.5, 0.7):
        means = _means_in_id_order(clus.label(distance_threshold=threshold), depth)
        assert means == sorted(means)


def test_ft_label_applies_the_clusterings_ordering() -> None:
    ft, depth = _depth_table()
    ft.cluster("core", **_PARAMS, order_by="soma_depth_um", name="run")

    means = _means_in_id_order(ft.label("run", distance_threshold=0.5), depth)
    assert means == sorted(means)


def test_order_by_rejects_an_unknown_column() -> None:
    ft, _ = _depth_table()
    with pytest.raises(KeyError, match="order_by"):
        ft.cluster("core", **_PARAMS, order_by="nope")


def test_order_by_survives_save_load(tmp_path) -> None:
    from cellpax import load_feature_table, save_feature_table

    ft, depth = _depth_table()
    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um", name="run")
    expected = clus.label(distance_threshold=0.5)

    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")
    restored = load_feature_table(path, name="t").clustering("run")

    assert restored.order_by == "soma_depth_um"
    np.testing.assert_array_equal(
        restored.label(distance_threshold=0.5).codes, expected.codes
    )


# -- sorted_matrix -------------------------------------------------------------


def test_sorted_matrix_blocks_the_diagonal() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    sm = clus.sorted_matrix(distance_threshold=0.5)

    assert sm.matrix.shape == (200, 200)
    assert sm.boundaries[0] == 0 and sm.boundaries[-1] == 200
    assert sm.sizes.sum() == 200
    assert len(sm.names) == len(sm.sizes) == len(sm.boundaries) - 1

    means = sm.block_means()
    off = means[~np.eye(len(means), dtype=bool)]
    assert np.diag(means).min() > off.max()  # blocks are the structure


def test_sorted_matrix_orders_rows_within_a_block_by_leaf_order() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    sm = clus.sorted_matrix(distance_threshold=0.5)

    rank = np.empty(clus.shape[0], dtype=np.int64)
    rank[clus.leaf_order] = np.arange(clus.shape[0])
    for start, stop in zip(sm.boundaries[:-1], sm.boundaries[1:]):
        block = rank[sm.order[start:stop]]
        assert list(block) == sorted(block)


def test_sorted_matrix_follows_a_labelsets_own_order() -> None:
    ft, depth = _depth_table()
    clus = ft.cluster("core", **_PARAMS, order_by="soma_depth_um")
    labels = clus.label(distance_threshold=0.5)

    sm = clus.sorted_matrix(labels)
    assert sm.names == labels.names
    block_depth = [
        float(np.nanmean(depth[sm.order[a:b]]))
        for a, b in zip(sm.boundaries[:-1], sm.boundaries[1:])
    ]
    assert block_depth == sorted(block_depth)
    # cutting at a threshold agrees with cutting into a LabelSet first
    np.testing.assert_array_equal(
        clus.sorted_matrix(distance_threshold=0.5).order, sm.order
    )


def test_sorted_matrix_keeps_unassigned_as_a_trailing_block() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    sm = clus.sorted_matrix(distance_threshold=0.05, min_cluster_size=15)

    if (sm.codes == -1).any():  # only meaningful when the cut actually drops cells
        assert sm.names[-1] == "unassigned"
        assert (sm.codes[sm.boundaries[-2] :] == -1).all()

    dropped = clus.sorted_matrix(
        distance_threshold=0.05, min_cluster_size=15, include_unassigned=False
    )
    assert (dropped.codes >= 0).all()
    assert "unassigned" not in dropped.names


def test_sorted_matrix_carries_cell_ids_in_sorted_order() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    sm = clus.sorted_matrix(distance_threshold=0.5)

    np.testing.assert_array_equal(sm.cell_ids, clus.cell_ids[sm.order])


def test_sorted_matrix_guards_and_subsamples() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)

    with pytest.raises(ValueError, match="subsample"):
        clus.sorted_matrix(distance_threshold=0.5, max_cells=50)

    small = clus.sorted_matrix(distance_threshold=0.5, subsample=60)
    assert small.n_cells <= 70
    assert small.matrix.shape == (small.n_cells, small.n_cells)
    assert small.sizes.sum() == small.n_cells


def test_sorted_matrix_accepts_a_plain_code_array() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    codes = clus.cluster_labels(0.5) - 1

    sm = clus.sorted_matrix(codes)
    assert sm.sizes.sum() == clus.shape[0]

    with pytest.raises(ValueError, match="entries but the matrix"):
        clus.sorted_matrix(codes[:-1])


def test_sorted_matrix_needs_something_to_sort_by() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    with pytest.raises(ValueError, match="labels= or distance_threshold="):
        clus.sorted_matrix()


# -- threshold diagnostics -----------------------------------------------------


def test_threshold_scan_reports_cells_not_just_clusters() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    scan = clus.threshold_scan(n_points=8, min_cluster_size=10)

    assert scan.height == 8
    assert scan.columns == [
        "distance_threshold",
        "n_clusters",
        "n_assigned",
        "n_unassigned",
        "largest_cluster",
        "median_cluster_size",
    ]
    total = clus.shape[0]
    assert (scan["n_assigned"] + scan["n_unassigned"] == pl.Series([total] * 8)).all()
    # raising the threshold can only ever assign more cells, never fewer
    assigned = scan["n_assigned"].to_numpy()
    assert (np.diff(assigned) >= 0).all()


def test_threshold_scan_matches_an_actual_cut() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    row = clus.threshold_scan([0.5], min_cluster_size=10).row(0, named=True)
    labels = clus.label(distance_threshold=0.5, min_cluster_size=10)

    assert row["n_clusters"] == labels.n_clusters
    assert row["n_unassigned"] == labels.n_unassigned


def test_consensus_strength_finds_cells_that_can_never_join() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)
    strength = clus.consensus_strength()

    assert strength.shape == (clus.shape[0],)
    assert ((strength >= 0) & (strength <= 1)).all()
    # a cell that never co-clusters with anything is a permanent singleton
    lonely = np.flatnonzero(strength == 0)
    if lonely.size:
        labels = clus.cluster_labels(0.99, min_cluster_size=2)
        assert (labels[lonely] == -1).all()


def test_consensus_strength_excludes_the_diagonal_without_touching_it() -> None:
    """Each cell's best co-clustering with some *other* cell, matrix left intact."""
    ft, _ = _depth_table()
    clus = ft.cluster("core", **_PARAMS)

    matrix = clus.similarity_matrix
    before_diagonal = matrix.diagonal().copy()
    before_nnz = matrix.nnz

    strength = clus.consensus_strength()

    dense = matrix.toarray().astype(float)
    np.fill_diagonal(dense, 0.0)
    np.testing.assert_allclose(strength, dense.max(axis=1), atol=1e-6)

    assert clus.similarity_matrix is matrix
    assert matrix.nnz == before_nnz
    np.testing.assert_array_equal(matrix.diagonal(), before_diagonal)


# -- partitions: the runs behind the consensus ---------------------------------


def test_cluster_keeps_the_individual_leiden_runs() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster(
        "core", n_neighbors=15, n_times=3, seed=0, n_jobs=1, resolution=(0.2, 1.0)
    )

    parts = clus.partitions
    assert parts is not None
    assert parts.n_cells == clus.shape[0]
    assert parts.n_runs == 6  # 2 resolutions x 3 times
    assert sorted(set(parts.resolution.tolist())) == [0.2, 1.0]
    assert (parts.n_neighbors == 15).all()
    assert parts.cluster_counts().shape == (6,)


def test_partition_summary_and_by_setting() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster(
        "core", n_neighbors=15, n_times=3, seed=0, n_jobs=1, resolution=(0.2, 1.0)
    )

    summary = clus.partitions.summary()
    assert summary.height == 6
    assert summary["n_clusters"].min() >= 1

    by_setting = clus.partitions.by_setting()
    assert by_setting.height == 2
    assert by_setting["n_runs"].to_list() == [3, 3]
    # a higher resolution cannot produce a coarser partition
    coarse, fine = by_setting.sort("resolution")["median_clusters"].to_list()
    assert coarse <= fine


def test_partitions_filter_selects_runs() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster(
        "core", n_neighbors=15, n_times=3, seed=0, n_jobs=1, resolution=(0.2, 1.0)
    )
    parts = clus.partitions

    low = parts.filter(resolution_max=0.5)
    assert low.n_runs == 3
    assert (low.resolution == 0.2).all()
    assert low.labels.shape == (parts.n_cells, 3)

    with pytest.raises(ValueError, match="no runs match"):
        parts.filter(resolution_min=99.0)


def test_restrict_reconsenses_from_the_kept_runs() -> None:
    ft, _ = _depth_table()
    clus = ft.cluster(
        "core",
        n_neighbors=15,
        n_times=3,
        seed=0,
        n_jobs=1,
        resolution=(0.2, 1.0),
        order_by="soma_depth_um",
        name="run",
    )

    broad = clus.restrict(resolution_max=0.5)
    assert isinstance(broad, Clustering)
    assert broad.shape == clus.shape
    assert broad.partitions.n_runs == 3
    # provenance rides along, ordering included
    assert broad.mask == clus.mask
    assert broad.columns == clus.columns
    assert broad.space == clus.space
    assert broad.order_by == "soma_depth_um"
    np.testing.assert_array_equal(broad.cell_ids, clus.cell_ids)

    # equivalent to consensing the filtered runs by hand
    from cellpax.clustering import coclustering_matrix

    expected = coclustering_matrix(
        clus.partitions.filter(resolution_max=0.5).labels, normalize=True
    )
    np.testing.assert_allclose(
        broad.similarity_matrix.todense(), expected.todense(), atol=1e-6
    )


def test_partitions_survive_a_reload_so_restrict_still_works(tmp_path) -> None:
    """The runs persist, so a reloaded clustering can still be re-consensed.

    They used to be session-only, which left ``restrict`` dead after a reload and made
    the expensive part of clustering unrecoverable. Storing the runs also makes the saved
    form far smaller than the consensus matrix they imply.
    """
    from cellpax import load_feature_table, save_feature_table

    ft, _ = _depth_table()
    original = ft.cluster("core", **_PARAMS, name="run")
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    restored = load_feature_table(path, name="t").clustering("run")
    assert restored.partitions is not None
    assert restored.partitions.n_runs == original.partitions.n_runs
    np.testing.assert_array_equal(
        restored.partitions.labels, original.partitions.labels
    )
    # the consensus is derived from the runs on load, so it matches what was saved
    np.testing.assert_allclose(
        restored.similarity_matrix.toarray(),
        original.similarity_matrix.toarray(),
        atol=1e-6,
    )
    narrowed = restored.restrict(
        resolution_max=float(original.partitions.resolution.max())
    )
    assert narrowed.partitions is not None


def test_restrict_needs_partitions() -> None:
    """A Clustering built straight from a matrix has no runs to narrow."""
    from cellpax.clustering import Clustering

    matrix = np.eye(5)
    bare = Clustering(matrix, cell_ids=np.arange(1, 6), mask="all", normalized=True)
    assert bare.partitions is None
    with pytest.raises(ValueError, match="no partitions"):
        bare.restrict(resolution_max=0.5)


def test_coarse_runs_give_a_stronger_consensus_than_a_mixed_sweep() -> None:
    """Averaging grains dilutes the coarse structure — the whole point of restrict."""
    ft, _ = _depth_table()
    clus = ft.cluster(
        "core",
        n_neighbors=15,
        n_times=4,
        seed=0,
        n_jobs=1,
        resolution=(0.2, 1.5, 3.0),
    )
    broad = clus.restrict(resolution_max=0.5)

    # cells the coarse runs keep together are only partly together in the blend
    coarse = broad.similarity_matrix.todense()
    mixed = clus.similarity_matrix.todense()
    together = np.asarray(coarse) > 0.99
    np.fill_diagonal(together, False)
    if together.any():
        assert np.asarray(mixed)[together].mean() < 1.0


# -- stale embeddings ----------------------------------------------------------


def test_redefining_a_mask_drops_its_embeddings_with_a_warning() -> None:
    """A stale embedding is worse than none: it was computed on the old membership."""
    ft, _ = _depth_table()
    ft.add_mask("part", pl.col("cell_id") < 60)
    ft.embed("part", method="pca", name="e")
    assert ft.dataframe("part", embedding="e").height == 60

    with pytest.warns(UserWarning, match="'part' was redefined"):
        ft.add_mask("part", pl.col("cell_id") >= 0)  # redefined wider
    with pytest.raises(KeyError, match="No embedding 'e'"):
        ft.dataframe("part", embedding="e")

    ft.embed("part", method="pca", name="e")  # re-embedding fixes it
    assert ft.dataframe("part", embedding="e").height == 200


def test_a_covering_embedding_still_joins_cleanly() -> None:
    ft, _ = _depth_table()
    ft.embed("core", method="pca", name="e")
    frame = ft.dataframe("core", embedding="e")

    assert frame["e0"].null_count() == 0
    assert frame.height == int(ft.mask_series("core").sum())


# -- label/column name collisions in the dataframe view ------------------------


def _attached_subset(ft: FeatureTable, name: str, n_covered: int) -> None:
    """Attach a label covering only the first ``n_covered`` cells, under ``name``."""
    from cellpax.labels import LabelSet

    ids = ft._cell_ids("core")[:n_covered]
    ft.attach(LabelSet(ids, np.zeros(n_covered, dtype=int), name=name, mask="core"))


def test_passed_labels_win_over_an_attached_column_of_the_same_name() -> None:
    """The bug: polars suffixed the incoming column '_right' and readers of
    labels.name silently got the stale attached column instead."""
    ft, _ = _depth_table()
    _attached_subset(ft, "pheno", 40)
    clus = ft.cluster("core", **_PARAMS)
    labels = clus.label(distance_threshold=0.8, name="pheno")

    frame = ft.dataframe(labels=labels)

    assert not any(c.endswith("_right") for c in frame.columns)
    assert [c for c in frame.columns if "pheno" in c] == ["pheno", "pheno_id"]
    # the view shows what was passed, not the 40-cell attached column
    assert frame["pheno"].null_count() == 0
    assert frame.height == len(labels)
    assert sorted(frame["pheno"].unique().to_list()) == sorted(labels.names)


def test_shadowing_leaves_the_table_itself_untouched() -> None:
    ft, _ = _depth_table()
    _attached_subset(ft, "pheno", 40)
    clus = ft.cluster("core", **_PARAMS)

    ft.dataframe(labels=clus.label(distance_threshold=0.8, name="pheno"))

    assert ft.dataframe("core")["pheno"].is_not_null().sum() == 40
    with pytest.raises(ValueError, match="already in the table"):
        ft.attach(clus.label(distance_threshold=0.8, name="pheno"))


def test_dataframe_by_attached_column_name_still_round_trips() -> None:
    """labels='col' resolves from that column and joins back under the same name."""
    ft, _ = _depth_table()
    _attached_subset(ft, "pheno", 40)

    frame = ft.dataframe("core", labels="pheno")
    assert [c for c in frame.columns if "pheno" in c] == ["pheno", "pheno_id"]
    assert frame["pheno"].is_not_null().sum() == 40


def test_embedding_columns_also_win_a_name_clash() -> None:
    ft, _ = _depth_table()
    ft.embed("core", method="pca", name="e")
    ft.add_column(np.zeros(200), "e0", mask="core")  # a column that collides

    frame = ft.dataframe("core", embedding="e")
    assert not any(c.endswith("_right") for c in frame.columns)
    assert frame["e0"].null_count() == 0
    assert frame["e0"].abs().sum() > 0  # the embedding's values, not the zeros


# -- attach / detach name clashes ----------------------------------------------


def _simple_table(extra: pl.Series | None = None) -> FeatureTable:
    frame = pl.DataFrame(
        {"cell_id": pl.Series(range(20), dtype=pl.Int64), "m0": np.zeros(20)}
    )
    if extra is not None:
        frame = frame.with_columns(extra)
    return FeatureTable(frame, features=["m0"])


def _flat(name: str = "label", value: int = 0):
    from cellpax.labels import LabelSet

    return LabelSet(np.arange(20), np.full(20, value), names=[str(value)], name=name)


def test_attach_overwrite_replaces_an_attached_label() -> None:
    """The re-cut loop: LabelSet's default name is 'label', so this collides fast."""
    ft = _simple_table()
    ft.attach(_flat(value=0))
    ft.attach(_flat(value=1), overwrite=True)

    assert ft.dataframe()["label"].unique().to_list() == ["1"]
    assert ft.labelset("label").names == ["1"]
    assert [c for c in ft.dataframe().columns if "label" in c] == ["label", "label_id"]


def test_attach_without_overwrite_still_refuses() -> None:
    ft = _simple_table()
    ft.attach(_flat())
    with pytest.raises(ValueError, match="overwrite=True"):
        ft.attach(_flat())


def test_attach_does_not_suffix_on_a_clash() -> None:
    ft = _simple_table()
    ft.attach(_flat())
    with pytest.raises(ValueError):
        ft.attach(_flat())
    assert not any(c.endswith("_right") for c in ft.dataframe().columns)


def test_attach_warns_differently_about_a_foreign_column() -> None:
    """A lone clashing column isn't attach's leftovers, so don't call it a label."""
    ft = _simple_table(extra=pl.Series("label", ["x"] * 20))
    with pytest.raises(ValueError, match="may be your own data"):
        ft.attach(_flat())


def test_detach_removes_a_half_written_pair() -> None:
    """attach used to advise a detach that then refused, leaving no way forward."""
    ft = _simple_table(extra=pl.Series("label_id", [1] * 20))
    with pytest.raises(ValueError):
        ft.attach(_flat())

    ft.detach("label")  # previously KeyError -- a dead end
    ft.attach(_flat())
    assert [c for c in ft.dataframe().columns if "label" in c] == ["label", "label_id"]


def test_detach_still_raises_when_nothing_is_there() -> None:
    ft = _simple_table()
    with pytest.raises(KeyError, match="Unknown label"):
        ft.detach("nope")


# -- the swept axes reach the Clustering's provenance ---------------------------


def test_alpha_and_graph_type_are_recorded_in_the_space_string() -> None:
    ft = _table()
    plain = ft.cluster("core", **_PARAMS)
    assert plain.space == "pca(0.95)"

    whitened = ft.cluster("core", alpha=0.5, **_PARAMS)
    assert whitened.space == "pca(0.95, alpha=0.5)"

    floored = ft.cluster(
        "core", alpha=0.5, eigenvalue_floor=ft.space("core").noise_floor, **_PARAMS
    )
    assert "floor=" in floored.space


def test_graph_type_reaches_the_partitions_through_ft_cluster() -> None:
    ft = _table()
    clus = ft.cluster("core", graph_type=["knn", "umap_fuzzy"], **_PARAMS)
    assert sorted(set(clus.partitions.graph_type.tolist())) == ["knn", "umap_fuzzy"]
    assert clus.partitions.n_runs == 2 * _PARAMS["n_times"]


def test_restrict_isolates_one_graph_type() -> None:
    ft = _table()
    clus = ft.cluster("core", graph_type=["knn", "umap_fuzzy"], **_PARAMS)
    only_knn = clus.restrict(graph_type="knn")
    assert set(only_knn.partitions.graph_type.tolist()) == {"knn"}
    assert only_knn.space == clus.space
    assert only_knn.mask == clus.mask


def test_axis_stability_is_reachable_from_the_clustering() -> None:
    from cellpax.clustering import axis_stability

    ft = _table()
    clus = ft.cluster("core", graph_type=["knn", "umap_fuzzy"], **_PARAMS)
    labels = clus.cluster_labels(0.5)
    frame = axis_stability(clus.partitions, labels)
    assert frame.height == 2
    assert frame["graph_type"].to_list() == ["knn", "umap_fuzzy"]


def test_pca_false_rejects_a_whitening_strength() -> None:
    ft = _table()
    with pytest.raises(ValueError, match="needs pca to be on"):
        ft.cluster("core", pca=False, alpha=0.5, **_PARAMS)


# -- graph provenance ----------------------------------------------------------


def test_graph_provenance_lists_every_consumer() -> None:
    ft = _table()
    ft.cluster("core", graph_type=["knn", "umap_fuzzy"], **_PARAMS, name="run")
    ft.embed("core", method="pca", name="plain")

    frame = ft.graph_provenance()
    assert set(frame["consumer"]) == {"clustering", "embedding"}

    clustering_row = frame.filter(pl.col("name") == "run").row(0, named=True)
    assert clustering_row["space"] == "pca(0.95)"
    assert clustering_row["graph_type"] == "knn,umap_fuzzy"
    assert clustering_row["n_neighbors"] == "15"

    embedding_row = frame.filter(pl.col("name") == "plain").row(0, named=True)
    # an embedding builds its own neighbour structure; it is never a Leiden graph object
    assert embedding_row["graph_type"] == "internal:pca"
    assert embedding_row["space"] == "scaled"


def test_a_clustering_and_embedding_sharing_a_space_warns() -> None:
    """The tautology case: comparing their neighbourhoods would be circular."""
    ft = _table()
    ft.cluster("core", **_PARAMS, name="run")
    ft.embed("core", method="pca", name="same", space=ft.space("core"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ft.graph_provenance()
    assert any("close to circular" in str(w.message) for w in caught)


def test_separately_built_spaces_do_not_warn() -> None:
    ft = _table()
    ft.cluster("core", **_PARAMS, name="run")
    ft.embed("core", method="pca", name="plain")  # full scaled space, not pca(0.95)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ft.graph_provenance()
    assert not any("circular" in str(w.message) for w in caught)


# -- triage --------------------------------------------------------------------


def test_triage_labels_splits_cells_into_three_countable_groups() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS)
    labels = clus.label(distance_threshold=0.5)

    frame = ft.triage_labels(labels, mask="core", n_neighbors=10)
    assert frame.height == labels.cell_ids.shape[0]
    assert set(frame["verdict"]).issubset({"own", "other", "mixed", "unassigned"})
    assert set(frame.columns) >= {
        "own_fraction",
        "top_other_label",
        "top_other_fraction",
        "verdict",
        "label_name",
    }
    counts = frame.group_by("verdict").len()
    assert counts["len"].sum() == frame.height


def test_triage_calls_well_separated_cells_their_own() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS)
    labels = clus.label(distance_threshold=0.5)
    frame = ft.triage_labels(labels, mask="core", n_neighbors=5)
    assert (frame["verdict"] == "own").mean() > 0.8


# -- the hierarchy through the FeatureTable ------------------------------------


def test_nested_levels_convert_to_attachable_label_sets() -> None:
    from cellpax.labels import LabelSet

    ft = _table()
    clus = ft.cluster("core", **_PARAMS)
    hierarchy = clus.hierarchy()

    levels = [c for c in hierarchy.nested_labels.columns if c.startswith("level_")]
    for level in levels:
        labels = LabelSet.from_labels(
            hierarchy.nested_labels["cell_id"].to_numpy(),
            hierarchy.nested_labels[level].to_numpy(),
            unassigned=-1,
            name=level,
            mask="core",
        )
        ft.attach(labels)
    assert all(level in ft.dataframe().columns for level in levels)


def test_hierarchy_survives_a_save_and_reload(tmp_path) -> None:
    """It is derived from the runs, and the runs now persist."""
    from cellpax import load_feature_table, save_feature_table

    ft = _table()
    original = ft.cluster("core", **_PARAMS, name="run")
    path = tmp_path / "f.zarr"
    save_feature_table(ft, path, name="t")

    restored = load_feature_table(path, name="t").clustering("run")
    np.testing.assert_allclose(
        restored.hierarchy().nested_levels["n_clusters"].to_numpy(),
        original.hierarchy().nested_levels["n_clusters"].to_numpy(),
    )
    # and merge_support, which needs the runs, works again after a reload
    assert restored.merge_support(n_bands=2, max_merges=4).height == 4


# -- restrict preserves normalization, soft labels, propagation attach ----------


def test_restrict_preserves_an_unnormalized_consensus() -> None:
    ft = _table()
    clus = ft.cluster("core", **{**_PARAMS, "normalize": False}, name="raw")
    narrowed = clus.restrict(n_neighbors=_PARAMS["n_neighbors"])
    assert narrowed.normalized is False
    assert narrowed.max_value == clus.max_value


def test_soft_labels_read_high_for_own_cluster_members() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")
    labels = clus.label(distance_threshold=0.5, name="kind")
    soft = clus.soft_labels(labels)

    assert soft.height == clus.shape[0]
    p_columns = [c for c in soft.columns if c.startswith("p_")]
    assert len(p_columns) == labels.n_clusters
    codes = labels.codes
    for i, column in enumerate(p_columns):
        own = soft[column].to_numpy()[codes == i]
        other = soft[column].to_numpy()[(codes != i) & (codes != -1)]
        assert own.mean() > other.mean()


def test_soft_labels_match_a_brute_force_mean_over_members() -> None:
    """The definition, computed the slow obvious way, against the sparse one.

    Guards the indicator-matmul rewrite: entry (i, k) is the mean of row i over
    cluster k's members, with cell i itself left out of both sum and count.
    """
    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")
    labels = clus.label(distance_threshold=0.5, name="kind")
    soft = clus.soft_labels(labels)

    dense = clus.similarity_matrix.toarray().astype(float) / clus.max_value
    codes = labels.codes
    for k, column in enumerate(c for c in soft.columns if c.startswith("p_")):
        members = np.flatnonzero(codes == k)
        expected = np.array(
            [
                np.sum(dense[i, members[members != i]])
                / max(members[members != i].size, 1)
                for i in range(clus.shape[0])
            ]
        )
        np.testing.assert_allclose(soft[column].to_numpy(), expected, atol=1e-6)


def test_soft_labels_leaves_the_consensus_matrix_untouched() -> None:
    """It reads the matrix; it must never edit (or copy) the session's largest object."""
    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")
    labels = clus.label(distance_threshold=0.5, name="kind")

    matrix = clus.similarity_matrix
    before_diagonal = matrix.diagonal().copy()
    before_nnz, before_sum = matrix.nnz, float(matrix.sum())

    clus.soft_labels(labels)

    assert clus.similarity_matrix is matrix
    assert matrix.nnz == before_nnz
    assert float(matrix.sum()) == pytest.approx(before_sum)
    np.testing.assert_array_equal(matrix.diagonal(), before_diagonal)


def test_soft_labels_gives_no_column_to_unassigned_cells() -> None:
    ft = _table()
    clus = ft.cluster("core", **_PARAMS, name="run")
    codes = clus.cluster_labels(0.5)
    codes[:3] = -1

    soft = clus.soft_labels(codes)
    p_columns = [c for c in soft.columns if c.startswith("p_")]
    assert len(p_columns) == len({int(v) for v in codes if v != -1})
    assert "p_-1" not in soft.columns
    # Unassigned cells still get scored against every cluster.
    assert soft.height == clus.shape[0]
    assert soft.select(p_columns).head(3).to_numpy().shape == (3, len(p_columns))


def test_attaching_a_propagation_carries_confidence_alongside() -> None:
    ft = _table()
    ft.add_mask("labeled", pl.col("cell_id") <= 40, based_on="core")
    clus = ft.cluster("labeled", **_PARAMS, name="run")
    core = clus.label(distance_threshold=0.5, name="kind")
    result = ft.propagate_labels(core, to="core", n_neighbors=5)

    ft.attach(result)
    frame = ft.dataframe("core")
    assert "kind_nn" in frame.columns
    assert "kind_nn_confidence" in frame.columns
    assert frame["kind_nn_confidence"].drop_nulls().max() <= 1.0

    ft.detach("kind_nn")
    assert "kind_nn_confidence" not in ft.columns


# -- weighted and externally-supplied spaces -------------------------------------


def test_cluster_accepts_feature_weights_and_records_the_weighted_space() -> None:
    from cellpax.diagnostics import block_weights

    ft = _table()
    names = list(ft.feature_columns)
    weights = block_weights(ft.features("core", scaled=True), names)

    plain = ft.cluster("core", **_PARAMS, name="plain")
    weighted = ft.cluster("core", **_PARAMS, feature_weights=weights, name="weighted")

    assert "weighted" in weighted.space and "weighted" not in plain.space
    assert weighted.params["feature_weights"] is not None
    assert plain.params["feature_weights"] is None
    assert (weighted.similarity_matrix != plain.similarity_matrix).nnz > 0


def test_cluster_with_a_prebuilt_space_matches_feature_weights() -> None:
    from cellpax.diagnostics import block_weights

    ft = _table()
    weights = block_weights(ft.features("core", scaled=True), ft.feature_columns)
    space = ft.space("core", feature_weights=weights)

    by_weights = ft.cluster("core", **_PARAMS, feature_weights=weights, name="w")
    by_space = ft.cluster("core", **_PARAMS, space=space, name="w")

    assert (by_weights.similarity_matrix != by_space.similarity_matrix).nnz == 0
    assert by_space.params["pca"] is None  # the external-space marker


def test_a_passed_space_refuses_conflicting_representation_arguments() -> None:
    ft = _table()
    space = ft.space("core")
    with pytest.raises(ValueError, match="not both"):
        ft.cluster("core", **_PARAMS, space=space, feature_weights=np.ones(20))
    with pytest.raises(ValueError, match="drop pca="):
        ft.cluster("core", **_PARAMS, space=space, pca=False)
    with pytest.raises(ValueError, match="with_alpha"):
        ft.cluster("core", **_PARAMS, space=space, alpha=0.5)
    with pytest.raises(ValueError, match="feature_weights are frozen"):
        ft.cluster("core", **_PARAMS, pca=False, feature_weights=np.ones(20))


def test_boundary_report_refuses_the_wrong_geometry_for_weighted_runs() -> None:
    from cellpax.diagnostics import block_weights

    ft = _table()
    weights = block_weights(ft.features("core", scaled=True), ft.feature_columns)
    clus = ft.cluster("core", **_PARAMS, feature_weights=weights, name="run")
    lbl = clus.label(distance_threshold=0.5, name="kind")

    # in-session: the weighted space is cached, found by digest
    report = ft.boundary_report("run", labels=lbl)
    assert report.height >= 1

    # if the cache is gone, refusing beats reporting on the wrong geometry
    ft._space_cache.clear()
    with pytest.raises(ValueError, match="wrong geometry"):
        ft.boundary_report("run", labels=lbl)
    # ...and an explicit space= resolves it
    space = ft.space("core", feature_weights=weights)
    assert ft.boundary_report("run", labels=lbl, space=space).height >= 1


def test_boundary_report_demands_a_space_for_external_space_runs() -> None:
    ft = _table()
    space = ft.space("core")
    clus = ft.cluster("core", **_PARAMS, space=space, name="run")
    lbl = clus.label(distance_threshold=0.5, name="kind")
    with pytest.raises(ValueError, match="externally supplied space"):
        ft.boundary_report("run", labels=lbl)
    assert ft.boundary_report("run", labels=lbl, space=space).height >= 1
