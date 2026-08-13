"""The consensus as a tree: merge stability, nested levels, per-cell stability."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import adjusted_rand_score

from cellpax.clustering import Clustering, fauxnograph_coclustering

_PARAMS = dict(n_neighbors=[15, 30], n_times=3, seed=0, n_jobs=1)


def _two_scales(per: int = 70, dim: int = 6):
    """Two broad types, each split into two subtypes — real structure at two scales.

    The situation the resolution sweep is geomspaced for, and the one a single flat cut
    has to choose between resolving and missing.
    """
    rng = np.random.default_rng(0)
    centers = [0.0, 1.7, 9.0, 10.7]
    coords = np.vstack(
        [
            rng.normal(np.concatenate([[c], np.zeros(dim - 1)]), 0.5, (per, dim))
            for c in centers
        ]
    )
    broad = np.repeat([0, 0, 1, 1], per)
    fine = np.repeat([0, 1, 2, 3], per)
    return coords, broad, fine


def _clustering(coords: np.ndarray, **overrides):
    params = {**_PARAMS, **overrides}
    matrix, partitions = fauxnograph_coclustering(
        coords,
        resolution_parameter=list(np.geomspace(0.05, 3.0, 8)),
        return_partitions=True,
        **params,
    )
    return Clustering(
        matrix,
        cell_ids=np.arange(1, coords.shape[0] + 1),
        mask="all",
        normalized=True,
        partitions=partitions,
    )


# -- merge stability is a transform of the linkage, not a new measurement ---------


def test_merge_frequency_is_max_value_minus_height() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    table = clus.merge_table()
    np.testing.assert_allclose(
        table["coclustering_frequency"].to_numpy(),
        clus.max_value - table["height"].to_numpy(),
    )


def test_merge_table_covers_every_merge_with_consistent_sizes() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    table = clus.merge_table()
    assert table.height == coords.shape[0] - 1
    np.testing.assert_array_equal(
        table["n_cells"].to_numpy(),
        table["left_size"].to_numpy() + table["right_size"].to_numpy(),
    )
    assert table["n_cells"].max() == coords.shape[0]


# -- nested levels ---------------------------------------------------------------


def test_levels_run_coarsest_to_finest_and_nest() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    frame = clus.nested_labels()
    levels = [c for c in frame.columns if c.startswith("level_")]
    assert len(levels) >= 2

    counts = [frame[c].n_unique() for c in levels]
    assert counts == sorted(counts), f"levels are not coarse-to-fine: {counts}"

    # each finer cluster must sit inside exactly one coarser cluster
    for coarse, fine in zip(levels, levels[1:]):
        per_fine = frame.group_by(fine).agg(pl.col(coarse).n_unique().alias("parents"))
        assert per_fine["parents"].max() == 1


def test_levels_recover_structure_at_both_scales() -> None:
    """The point of keeping the ladder: one cut cannot hold both answers."""
    coords, broad, fine = _two_scales()
    clus = _clustering(coords)
    frame = clus.nested_labels()
    levels = [c for c in frame.columns if c.startswith("level_")]

    broad_best = max(adjusted_rand_score(broad, frame[c].to_numpy()) for c in levels)
    fine_best = max(adjusted_rand_score(fine, frame[c].to_numpy()) for c in levels)
    assert broad_best > 0.9
    assert fine_best > 0.7
    # and they are not the same level, which is why one flat vector cannot serve
    broad_level = max(
        levels, key=lambda c: adjusted_rand_score(broad, frame[c].to_numpy())
    )
    fine_level = max(
        levels, key=lambda c: adjusted_rand_score(fine, frame[c].to_numpy())
    )
    assert broad_level != fine_level


def test_nested_levels_describes_each_cut() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    levels = clus.nested_levels()
    assert set(levels.columns) == {
        "level",
        "height",
        "coclustering_frequency",
        "n_clusters",
        "n_unassigned",
    }
    heights = levels["height"].to_numpy()
    assert np.all(np.diff(heights) < 0), "level_0 must be the tallest (coarsest) cut"
    np.testing.assert_allclose(
        levels["coclustering_frequency"].to_numpy(), clus.max_value - heights
    )


def test_repeated_cluster_counts_are_dropped_and_logged(caplog) -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    heights = np.full(6, 0.5 * clus.max_value)  # every cut identical
    with caplog.at_level("INFO", logger="cellpax.clustering"):
        frame = clus.nested_labels(heights)
    levels = [c for c in frame.columns if c.startswith("level_")]
    assert len(levels) == 1
    assert "dropped 5" in caplog.text


def test_a_zero_height_cut_is_the_finest_available() -> None:
    """Not quite singletons: cells that co-clustered in every run merge at height 0."""
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    finest = clus.nested_labels([0.0])["level_0"].n_unique()
    default = clus.nested_labels()
    assert finest >= max(
        default[c].n_unique() for c in default.columns if c.startswith("level_")
    )
    assert finest < coords.shape[0]


def test_a_level_that_drops_every_cluster_is_not_emitted() -> None:
    """An all-unassigned column would read as a granularity rather than as nothing."""
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    with pytest.raises(ValueError, match="no cut height"):
        clus.nested_labels([0.0], min_cluster_size=coords.shape[0] + 1)


def test_min_cluster_size_warns_instead_of_emitting_broken_nesting() -> None:
    """Above 1, strict nesting genuinely cannot hold, so say so rather than pretend."""
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clus.nested_labels(n_levels=8, min_cluster_size=25)
    for record in caught:
        if "not strictly nested" in str(record.message):
            assert "min_cluster_size=25" in str(record.message)
            break


def test_cell_ids_are_carried_onto_the_frame() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    frame = clus.nested_labels()
    assert frame.columns[0] == "cell_id"
    np.testing.assert_array_equal(frame["cell_id"].to_numpy(), clus.cell_ids)


# -- per-cell stability ----------------------------------------------------------


def test_cell_stability_is_consensus_strength() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    np.testing.assert_array_equal(clus.cell_stability(), clus.consensus_strength())


def test_cell_stability_frame_pairs_ids_with_scores() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    frame = clus.cell_stability_frame()
    assert frame.columns == ["cell_id", "stability"]
    assert frame.height == coords.shape[0]
    assert frame["stability"].min() >= 0.0
    assert frame["stability"].max() <= 1.0


def test_boundary_cells_are_less_stable_than_core_cells() -> None:
    """The expectation the stability figure is meant to check, pinned on planted data.

    Two populations that genuinely interdigitate — separated by about two standard
    deviations along one axis, so there is a real boundary region rather than a gap.
    Cells in it should score lower, which is what makes a UMAP coloured by stability
    informative rather than uniform.
    """
    rng = np.random.default_rng(0)
    boundary_axis = np.concatenate(
        [rng.normal(-1.0, 1.0, 150), rng.normal(1.0, 1.0, 150)]
    )
    coords = np.column_stack([boundary_axis, rng.normal(0.0, 1.0, (300, 3))])
    clus = _clustering(coords, n_neighbors=[15])
    stability = clus.cell_stability()

    distance_to_boundary = np.abs(coords[:, 0])
    near = distance_to_boundary < np.quantile(distance_to_boundary, 0.25)
    far = distance_to_boundary > np.quantile(distance_to_boundary, 0.75)
    assert stability[near].mean() < stability[far].mean()


# -- the bundled object ----------------------------------------------------------


def test_hierarchy_bundles_the_pieces_consistently() -> None:
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    hierarchy = clus.hierarchy()

    assert hierarchy.n_cells == coords.shape[0]
    assert hierarchy.n_levels == hierarchy.nested_levels.height
    np.testing.assert_array_equal(hierarchy.linkage, clus.linkage)
    np.testing.assert_array_equal(hierarchy.cell_stability, clus.cell_stability())
    np.testing.assert_array_equal(hierarchy.leaf_order, clus.leaf_order)

    frame = hierarchy.cell_stability_frame()
    assert "stability" in frame.columns
    assert any(c.startswith("level_") for c in frame.columns)
    assert frame.height == coords.shape[0]


def test_dendrogram_frame_has_three_segments_per_merge() -> None:
    coords, _, _ = _two_scales()
    hierarchy = _clustering(coords).hierarchy()
    segments = hierarchy.dendrogram_frame()
    assert segments.height == 3 * (coords.shape[0] - 1)
    assert set(segments.columns) == {
        "merge",
        "x0",
        "y0",
        "x1",
        "y1",
        "height",
        "coclustering_frequency",
    }
    # leaf positions follow leaf_order, so the drawing lines up with sorted_matrix
    assert segments["x0"].min() >= 0
    assert segments["x1"].max() <= coords.shape[0] - 1


# -- which resolutions supported a merge -----------------------------------------


def test_merge_support_separates_types_from_subtypes() -> None:
    """Coarse runs keep the broad merges; only fine runs make the subtype splits.

    The distinction the pooled frequency cannot express, and the reason the runs are
    worth keeping.
    """
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    support = clus.merge_support(n_bands=3, max_merges=8)

    bands = [c for c in support.columns if c.startswith("support_band_")]
    assert len(bands) == 3
    # the broad merges (each joining ~half the cells) hold in the coarse band and not
    # in the fine one
    broad = support.filter(pl.col("n_cells") >= coords.shape[0] // 2)
    broad = broad.filter(pl.col("coclustering_frequency") > 0)
    assert not broad.is_empty()
    assert broad["support_band_0"].mean() > broad[bands[-1]].mean()


def test_merge_support_needs_the_runs() -> None:
    matrix = np.eye(6)
    bare = Clustering(matrix, cell_ids=np.arange(1, 7), mask="all", normalized=True)
    with pytest.raises(ValueError, match="needs the individual runs"):
        bare.merge_support()


def test_merge_support_logs_both_caps(caplog) -> None:
    """A bounded diagnostic that does not say what it bounded reads as complete."""
    coords, _, _ = _two_scales()
    clus = _clustering(coords)
    with caplog.at_level("INFO", logger="cellpax.clustering"):
        clus.merge_support(n_bands=2, max_merges=5, max_block=10)
    assert "annotating the top 5" in caplog.text
    assert "max_block=10" in caplog.text
