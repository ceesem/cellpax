"""Step 4: LabelSet — clear labels, relabeling verbs, and ft.attach."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cellpax.featuretable import FeatureTable
from cellpax.labels import LabelSet


def test_labelset_construction_and_frame() -> None:
    ls = LabelSet([10, 20, 30, 40], [0, 0, 1, -1], name="subclass")
    assert ls.ids == [0, 1]
    assert ls.names == ["0", "1"]
    assert ls.counts() == {"0": 2, "1": 1}
    assert ls.mask is None
    frame = ls.to_frame()
    assert frame.columns == ["cell_id", "subclass", "subclass_id"]
    assert frame["subclass"].to_list() == ["0", "0", "1", None]  # -1 -> null


def test_names_at_construction_and_rename_by_list() -> None:
    from cellpax.labels import Label

    # a list in cluster-id order, no Label objects needed
    ls = LabelSet([1, 2, 3, 4], [0, 0, 1, -1], names=["exc", "inh"], name="cls")
    assert ls.names == ["exc", "inh"]
    assert ls.to_names() == ["exc", "exc", "inh", None]

    # or a {id: name} mapping
    assert LabelSet([1, 2], [0, 1], names={0: "a", 1: "b"}).names == ["a", "b"]

    # names win over a name in meta, but meta's other fields survive
    both = LabelSet(
        [1, 2],
        [0, 1],
        meta={0: Label(id=0, name="stale", color="#f00")},
        names=["fresh", "b"],
    )
    assert both.names == ["fresh", "b"]
    assert both.cluster("fresh").color == "#f00"

    # rename takes the same two forms
    ls.rename(["E", "I"])
    assert ls.names == ["E", "I"]
    assert ls.rename({"E": "exc"}).names == ["exc", "I"]

    with pytest.raises(ValueError, match=r"expected 2 names for clusters \[0, 1\]"):
        LabelSet([1, 2], [0, 1], names=["only-one"])
    with pytest.raises(TypeError, match="not a string"):
        LabelSet([1, 2], [0, 1], names="exc")


def test_labelset_mask_provenance() -> None:
    ls = LabelSet([1, 2], [0, 0], name="cls", mask="left")
    assert ls.mask == "left"
    assert "mask='left'" in repr(ls)

    same_mask = ls.combine(LabelSet([3, 4], [0, 1], name="cls", mask="left"))
    assert same_mask.mask == "left"

    different_mask = ls.combine(LabelSet([5, 6], [0, 0], name="cls", mask="right"))
    assert different_mask.mask is None


def test_relabeling_verbs() -> None:
    ls = LabelSet([1, 2, 3, 4, 5], [0, 1, 2, 0, 1])
    ls.rename({0: "L2a", 1: "L2b", 2: "L3"})
    assert ls.names == ["L2a", "L2b", "L3"]
    ls.set_colors({"L2a": "#f00"})
    assert ls.cluster("L2a").color == "#f00"

    ls.merge(["L2a", "L2b"], into="L2")
    assert set(ls.names) == {"L2", "L3"}
    assert ls.counts()["L2"] == 4

    ls.reorder(["L3", "L2"])
    assert ls.names == ["L3", "L2"]  # renumbered 0,1 in that order
    with pytest.raises(ValueError, match="every cluster exactly once"):
        ls.reorder(["L3"])
    with pytest.raises(KeyError, match="Unknown label name"):
        ls.rename({"nope": "x"})
    with pytest.raises(ValueError, match="at least two"):
        ls.merge(["L2"], into="x")


def test_reorder_by_mapping() -> None:
    ls = LabelSet([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])
    # cluster 0 -> depth ~800, cluster 1 -> ~200, cluster 2 -> ~500
    depths = {1: 810, 2: 790, 3: 190, 4: 210, 5: 490, 6: 510}
    ls.reorder_by(depths)
    assert ls.names == ["1", "2", "0"]  # renumbered ascending by mean depth

    ls2 = LabelSet([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])
    ls2.reorder_by(depths, ascending=False)
    assert ls2.names == ["0", "2", "1"]


def test_reorder_by_array_aligned_to_cell_ids() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 0, 1, 1])
    # values given in a different cell order than the LabelSet's own — cluster 0
    # (cells 1,2) -> {5, 100}, cluster 1 (cells 3,4) -> {200, 300}
    ls.reorder_by([300, 200, 100, 5], cell_ids=[4, 3, 2, 1])
    assert ls.names == ["0", "1"]  # cluster 0's mean (52.5) < cluster 1's (250)

    ls_desc = LabelSet([1, 2, 3, 4], [0, 0, 1, 1])
    ls_desc.reorder_by([300, 200, 100, 5], cell_ids=[4, 3, 2, 1], ascending=False)
    assert ls_desc.names == ["1", "0"]

    ls2 = LabelSet([1, 2, 3, 4], [0, 0, 1, 1])
    with pytest.raises(KeyError):
        ls2.reorder_by([100, 5, 200], cell_ids=[4, 2, 1])  # missing cell 3

    with pytest.raises(ValueError, match="one entry per cell_id"):
        ls2.reorder_by([1, 2], cell_ids=[1, 2, 3])


def test_combine_disjoint_labelsets() -> None:
    a = LabelSet([1, 2], [0, 0], name="cls").rename({0: "exc"})
    b = LabelSet([3, 4], [0, 1], name="cls").rename({0: "inh_a", 1: "inh_b"})
    combined = a.combine(b)
    assert combined.to_frame().sort("cell_id")["cls"].to_list() == [
        "exc",
        "exc",
        "inh_a",
        "inh_b",
    ]
    with pytest.raises(ValueError, match="disjoint"):
        a.combine(LabelSet([2, 5], [0, 0]))


def test_from_labels_sorts_and_names_directly() -> None:
    ls = LabelSet.from_labels(
        [1, 2, 3, 4, 5], ["L5IT", "L23IT", "L5IT", None, "L23IT"], name="subclass"
    )
    # ids assigned in sorted order of the unique non-null values
    assert ls.names == ["L23IT", "L5IT"]
    frame = ls.to_frame().sort("cell_id")
    assert frame["subclass"].to_list() == ["L5IT", "L23IT", "L5IT", None, "L23IT"]


def test_from_labels_numeric_sentinel_and_ordering() -> None:
    ls = LabelSet.from_labels([1, 2, 3, 4], [3, 1, 1, -99], unassigned=-99)
    assert ls.names == ["1", "3"]  # sorted numerically, not string-wise
    assert ls.to_frame().sort("cell_id")["label"].to_list() == ["3", "1", "1", None]


def test_labelset_to_enum_and_filter() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 1, 0, 1], name="subclass")
    ls.rename({0: "L5IT", 1: "L23IT"})
    Labels = ls.to_enum("ITLabels")
    assert Labels.L5IT == 0 and Labels.L23IT == 1
    assert [m.name for m in Labels] == ["L5IT", "L23IT"]
    # the id column compares directly against enum members (IntEnum == int)
    frame = ls.to_frame()
    assert frame.filter(pl.col("subclass_id") == Labels.L5IT).height == 2
    # names needing sanitizing still yield valid members
    ls.rename({"L5IT": "L5 IT-a"})
    assert ls.to_enum().L5_IT_a == 0


def test_labelset_apply_enum() -> None:
    from enum import IntEnum

    class ITLabels(IntEnum):
        L5IT = 0
        L23IT = 1

    ls = LabelSet([1, 2, 3], [0, 1, 0])
    ls.apply_enum(ITLabels)
    assert ls.names == ["L5IT", "L23IT"]
    # round-trips back to an equivalent enum
    assert ls.to_enum("ITLabels").L23IT == 1
    # colliding names raise on to_enum
    ls.rename({"L23IT": "L5IT"})
    with pytest.raises(ValueError, match="collide"):
        ls.to_enum()


def test_per_cell_arrays_and_alignment() -> None:
    ls = LabelSet([10, 20, 30, 40], [0, 0, 1, -1], name="subclass")
    assert ls.codes.tolist() == [0, 0, 1, -1]
    assert ls.to_names() == ["0", "0", "1", None]
    assert len(ls) == 4
    assert ls.n_clusters == 2
    assert ls.n_unassigned == 1
    assert ls.assigned.tolist() == [True, True, True, False]

    # codes is a copy -- relabeling can't mutate it underneath the caller
    codes = ls.codes
    ls.merge([0, 1], into="M")
    assert codes.tolist() == [0, 0, 1, -1]
    ls.rename({0: "0"})

    # arbitrary row order, plus cells the label set doesn't cover
    assert ls.codes_for([40, 30, 10, 999]).tolist() == [-1, 0, 0, -1]
    assert ls.codes_for([999], missing=-7).tolist() == [-7]


def test_decode_and_with_codes_round_trip() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 0, 1, 1], name="subclass")
    ls.rename({0: "L23IT", 1: "L5IT"}).set_colors({"L5IT": "#f00"})

    assert ls.decode([1, 0, -1]) == ["L5IT", "L23IT", None]
    assert ls.decode(np.array([[0], [1]])) == ["L23IT", "L5IT"]

    # a classifier's predictions over other cells, named like the training set
    predicted = ls.with_codes([7, 8, 9], np.array([1, 0, 1]), name="subclass_pred")
    assert predicted.name == "subclass_pred"
    assert predicted.names == ["L23IT", "L5IT"]
    assert predicted.to_names() == ["L5IT", "L23IT", "L5IT"]
    assert predicted.cluster("L5IT").color == "#f00"
    assert predicted.mask is None  # different cells, so no inherited provenance

    with pytest.raises(ValueError, match=r"no cluster for ids \[2\]"):
        ls.with_codes([7, 8], [0, 2])
    with pytest.raises(ValueError, match="no cluster for ids"):
        ls.decode([9])


def test_copy_subset_and_drop_unassigned() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 0, 1, -1], name="subclass").rename({0: "A", 1: "B"})

    branch = ls.copy(name="branch")
    branch.merge(["A", "B"], into="AB")
    assert ls.names == ["A", "B"]  # original untouched
    assert branch.names == ["AB"] and branch.name == "branch"

    train = ls.subset([3, 1])
    assert train.cell_ids.tolist() == [3, 1]
    assert train.codes.tolist() == [1, 0]  # ids stay comparable with the parent
    assert train.cluster("A").id == 0
    with pytest.raises(KeyError, match="not in this label set"):
        ls.subset([1, 99])

    fittable = ls.drop_unassigned()
    assert fittable.cell_ids.tolist() == [1, 2, 3]
    assert fittable.n_unassigned == 0


def test_unassign_sends_a_cluster_back_to_unassigned() -> None:
    ls = LabelSet([1, 2, 3, 4, 5], [0, 0, 1, 2, 2], names=["A", "B", "C"])

    ls.unassign("B")
    assert ls.names == ["A", "C"]
    assert ls.ids == [0, 2]  # a gap, like merge leaves
    assert ls.codes.tolist() == [0, 0, -1, 2, 2]
    assert ls.n_unassigned == 1
    with pytest.raises(KeyError, match="Unknown label name"):
        ls.cluster("B")

    # several at once, by id or name
    ls.unassign([0, "C"])
    assert ls.ids == [] and ls.n_unassigned == 5


def test_compact_after_merge_gives_contiguous_codes() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 1, 2, 2])
    ls.merge([0, 1], into="A")
    assert ls.ids == [0, 2]  # merge leaves a gap
    ls.compact()
    assert ls.ids == [0, 1]
    assert ls.names == ["A", "2"]
    assert ls.codes.tolist() == [0, 0, 1, 1]


def test_catalog_and_color_map() -> None:
    ls = LabelSet([1, 2, 3, 4], [0, 0, 1, -1], name="subclass")
    ls.rename({0: "A", 1: "B"}).set_colors({"A": "#f00"})
    ls.set_descriptions({"B": "the other one"})

    catalog = ls.catalog()
    assert catalog.columns == ["id", "name", "color", "description", "n_cells"]
    assert catalog.to_dicts() == [
        {"id": 0, "name": "A", "color": "#f00", "description": None, "n_cells": 2},
        {
            "id": 1,
            "name": "B",
            "color": None,
            "description": "the other one",
            "n_cells": 1,
        },
    ]
    assert ls.color_map() == {"A": "#f00"}

    empty = LabelSet([1, 2], [-1, -1]).catalog()
    assert empty.height == 0 and empty.columns[0] == "id"


def test_duplicate_names_and_cell_ids_are_not_silent() -> None:
    ls = LabelSet([1, 2, 3], [0, 1, 2]).rename({0: "x", 1: "x"})
    assert ls.counts() == {"x": 2, "2": 1}  # summed, not one cluster dropped
    assert ls.catalog()["id"].to_list() == [0, 1, 2]  # kept apart here
    with pytest.raises(ValueError, match="ambiguous"):
        ls.rename({"x": "y"})

    with pytest.raises(ValueError, match="cell_ids must be unique"):
        LabelSet([1, 1], [0, 0])


def test_reorder_tolerates_metadata_for_absent_clusters() -> None:
    from cellpax.labels import Label

    ls = LabelSet([1, 2, 3], [0, 0, 1], meta={7: Label(id=7, name="ghost")})
    ls.reorder([1, 0])
    assert ls.names == ["1", "0"]
    assert ls.ids == [0, 1]  # the absent cluster's metadata is dropped

    # combine offsets past metadata-only ids instead of overwriting them
    exc = LabelSet([1, 2], [0, 0], meta={1: Label(id=1, name="reserved")}).rename(
        {0: "exc"}
    )
    inh = LabelSet([3, 4], [0, 1]).rename({0: "inh_a", 1: "inh_b"})
    combined = exc.combine(inh)
    assert combined.ids == [0, 2, 3]  # not [0, 1, 2], which would clobber id 1
    assert combined.counts() == {"exc": 2, "inh_a": 1, "inh_b": 1}
    assert combined.cluster(1).name == "reserved"


def _two_blobs(n: int = 60) -> FeatureTable:
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.3, (n // 2, 3)), rng.normal(8, 0.3, (n // 2, 3))]
    )
    df = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "m0": coords[:, 0],
            "m1": coords[:, 1],
            "m2": coords[:, 2],
        }
    )
    return FeatureTable(df, features=["m0", "m1", "m2"])


def test_ft_labelset_producers_record_mask() -> None:
    ft = _two_blobs(60)
    ft.add_mask("half", pl.col("cell_id") <= 30)

    default_mask = ft.overcluster(n_neighbors=10, resolution=1.0, seed=0)
    assert default_mask.mask == "all"

    explicit_mask = ft.overcluster(mask="half", n_neighbors=10, resolution=1.0, seed=0)
    assert explicit_mask.mask == "half"

    ft.cluster(mask="half", n_neighbors=10, n_times=2, seed=0, n_jobs=1, name="run")
    threshold_labels = ft.label("run", mask="half", distance_threshold=1.0)
    assert threshold_labels.mask == "half"

    choir_labels = ft.cluster_choir("run", mask="half", min_cluster_size=3, n_jobs=1)
    assert choir_labels.mask == "half"


def test_ft_dataframe_with_labels_infers_mask() -> None:
    ft = _two_blobs(60)
    ft.add_mask("half", pl.col("cell_id") <= 30)
    ft.embed("half", method="pca", n_components=2, name="pca")

    labels = ft.overcluster(mask="half", n_neighbors=10, resolution=1.0, seed=0)
    # no mask= passed at all -- inferred from labels.mask
    plot_df = ft.dataframe(embedding="pca", labels=labels)
    assert plot_df.height == 30
    assert {"pca0", "pca1", "leiden", "leiden_id"} <= set(plot_df.columns)
    assert set(plot_df["cell_id"].to_list()) == set(labels.cell_ids.tolist())


def test_ft_labels_property_tracks_attached_labelsets() -> None:
    ft = _two_blobs(60)
    assert ft.labels == []

    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    ft.attach(labels)
    assert ft.labels == ["subclass"]

    ft.attach(labels, name="subclass2")
    assert ft.labels == ["subclass", "subclass2"]

    # a metadata _id column with no matching base column isn't mistaken for one
    ft.add_column(np.arange(60), "root_id")
    assert ft.labels == ["subclass", "subclass2"]


def test_ft_label_and_attach_roundtrip() -> None:
    ft = _two_blobs(60)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    assert len(labels.ids) == 2
    labels.rename(["A", "B"])

    ft.attach(labels)
    df = ft.dataframe()
    assert "subclass" in df.columns
    assert set(df["subclass"].to_list()) == {"A", "B"}

    # the _id column comes along too, for IntEnum-style filtering
    assert "subclass_id" in df.columns
    L = labels.to_enum("Subclass")
    a_ids = df.filter(pl.col("subclass") == "A")["subclass_id"].to_list()
    assert set(a_ids) == {int(L.A)}

    # a partial label set leaves other cells null
    partial = LabelSet([1, 2, 3], [0, 0, 0], name="partial").rename({0: "X"})
    ft.attach(partial)
    assert ft.dataframe()["partial"].null_count() == 57
    assert ft.dataframe()["partial_id"].null_count() == 57


def test_classifier_round_trip_through_featuretable() -> None:
    """The xgboost-shaped loop: LabelSet -> y, fit, predictions -> LabelSet -> column."""
    from sklearn.ensemble import RandomForestClassifier

    ft = _two_blobs(60)
    ft.add_mask("train", pl.col("cell_id") <= 40)
    ft.cluster(mask="train", n_neighbors=10, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", mask="train", distance_threshold=0.5, name="subclass")
    names = [f"C{i}" for i in labels.ids]
    labels.rename(names).set_colors({names[0]: "#f00"})

    # scaled features are mask-relative, so fit and predict in one mask's space;
    # y is aligned to its rows by cell id rather than by assumed row order, and
    # cells the labels don't cover come back -1
    X = ft.features(scaled=True)
    cell_ids = ft.dataframe()[ft.id_column].to_numpy()
    y = labels.codes_for(cell_ids)
    fit = y != -1
    assert fit.sum() == len(labels)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X[fit], y[fit])
    assert sorted(model.classes_.tolist()) == labels.ids

    predicted = labels.with_codes(cell_ids, model.predict(X), name="subclass_pred")
    assert predicted.names == names
    assert predicted.cluster(names[0]).color == "#f00"  # identities carried over

    ft.attach(predicted)
    assert ft.dataframe()["subclass_pred"].null_count() == 0
    # the model reproduces the labels of the cells it was fit on
    joined = ft.dataframe().join(
        labels.to_frame(id_column=ft.id_column), on=ft.id_column, how="inner"
    )
    assert joined["subclass_pred"].to_list() == joined["subclass"].to_list()


def test_ft_detach_removes_label_columns() -> None:
    ft = _two_blobs(60)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    ft.attach(labels)
    assert ft.labels == ["subclass"]

    ft.detach("subclass")
    assert ft.labels == []
    assert "subclass" not in ft.dataframe().columns
    assert "subclass_id" not in ft.dataframe().columns

    with pytest.raises(KeyError, match="Unknown label"):
        ft.detach("subclass")


def test_ft_reorder_labels_by_column() -> None:
    ft = _two_blobs(60)
    ft.add_column(
        np.concatenate([np.full(30, 800.0), np.full(30, 200.0)]), "soma_depth_um"
    )
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    assert len(labels.ids) == 2

    ft.reorder_labels(labels, "soma_depth_um")
    depth_col = ft.dataframe()["soma_depth_um"].to_numpy()
    label_col = labels.to_frame(id_column="cell_id")["subclass_id"].to_numpy()
    means = [depth_col[label_col == i].mean() for i in labels.ids]
    assert means == sorted(means)


def test_ft_labelset_reconstructs_from_attached_column() -> None:
    ft = _two_blobs(60)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    labels.rename(["A", "B"])
    ft.attach(labels)

    rebuilt = ft.labelset("subclass")
    assert rebuilt.name == "subclass"
    assert rebuilt.mask == "all"
    assert sorted(rebuilt.names) == ["A", "B"]
    assert (
        rebuilt.to_frame()["subclass"].to_list() == ft.dataframe()["subclass"].to_list()
    )


def test_attach_labelset_round_trips_identities_and_mask() -> None:
    ft = _two_blobs(60)
    ft.add_mask("half", pl.col("cell_id") <= 30)
    labels = ft.overcluster(mask="half", n_neighbors=10, resolution=1.0, seed=0)
    labels.rename([f"C{i}" for i in labels.ids])
    labels.set_colors({"C0": "#1f77b4"}).set_descriptions({"C0": "the first one"})
    ft.attach(labels, name="subclass")  # renamed on the way in

    rebuilt = ft.labelset("subclass")
    assert rebuilt.names == labels.names
    assert rebuilt.color_map() == {"C0": "#1f77b4"}  # a column can't hold this
    assert rebuilt.cluster("C0").description == "the first one"
    assert rebuilt.mask == "half"  # the mask attach recorded, not the default
    assert len(rebuilt) == 30

    # an explicit mask still wins, over the whole table if asked
    everything = ft.labelset("subclass", mask="all")
    assert everything.mask == "all"
    assert len(everything) == 60
    assert everything.n_unassigned == 30  # off-mask cells are null -> -1
    assert everything.color_map() == {"C0": "#1f77b4"}

    # renaming in the table wins over the recorded name; color still follows the id
    renamed = [
        "renamed" if value == "C0" else value
        for value in ft.dataframe("half")["subclass"].to_list()
    ]
    ft.add_column(renamed, "subclass", mask="half")
    assert ft.labelset("subclass").cluster("renamed").color == "#1f77b4"

    # detach forgets the identities: re-attaching a colorless set of the same name
    # brings back no colors from the old record
    ft.detach("subclass")
    ft.attach(LabelSet([1, 2], [0, 0], names=["C0"], name="subclass"))
    assert ft.labelset("subclass").color_map() == {}


def test_attach_over_an_existing_label_column_raises() -> None:
    ft = _two_blobs(10)
    labels = LabelSet(np.arange(1, 11), [0] * 10, names=["X"], name="subclass")
    ft.attach(labels)
    with pytest.raises(ValueError, match="already in the table"):
        ft.attach(labels)
    ft.attach(labels, name="subclass2")  # a different name is fine
    assert ft.labels == ["subclass", "subclass2"]
    assert ft.labelset("subclass2").names == ["X"]


def test_ft_labelset_from_plain_string_column_without_id_companion() -> None:
    ft = _two_blobs(60)
    ft.add_column(
        pl.Series(["exc"] * 30 + ["inh"] * 30), "celltype"
    )  # never touched a LabelSet
    rebuilt = ft.labelset("celltype")
    assert sorted(rebuilt.names) == ["exc", "inh"]
    frame = rebuilt.to_frame().sort("cell_id")
    assert frame["celltype"].to_list() == ["exc"] * 30 + ["inh"] * 30


def test_ft_dataframe_compare_reorder_accept_column_name() -> None:
    ft = _two_blobs(60)
    ft.cluster(n_neighbors=15, n_times=3, seed=0, n_jobs=1, name="run")
    labels = ft.label("run", distance_threshold=0.5, name="subclass")
    ft.attach(labels)
    other = ft.label("run", distance_threshold=0.5, name="rerun")
    ft.attach(other)

    # dataframe(labels=...) works with a column name, no LabelSet needed
    plot_df = ft.dataframe(labels="subclass")
    assert "subclass" in plot_df.columns

    # compare() likewise
    cmp = ft.compare("subclass", "rerun")
    assert cmp.agreement()["ari"] == 1.0

    # reorder_labels() likewise -- resolves "subclass" back into a LabelSet
    ft.add_column(
        np.concatenate([np.full(30, 800.0), np.full(30, 200.0)]), "soma_depth_um"
    )
    reordered = ft.reorder_labels("subclass", "soma_depth_um")
    assert isinstance(reordered, LabelSet)
