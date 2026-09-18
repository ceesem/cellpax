"""Joining feature tables from separate datasets into one harmonized space."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from cellpax import (
    FeatureTable,
    LabelSet,
    clipped_scaler_factory,
    cross_dataset_classification,
    dataset_mixing,
    join_datasets,
    list_analyses,
    quantile_scaler_factory,
)
from cellpax.featuretable import FittedScaler
from cellpax.persist import _arrays_frame, _frame_arrays
from cellpax.space import _scaler_from_records, _scaler_records

FEATURES = ["f0", "f1", "f2", "f3"]
SUBCLASS_CENTERS = {"A": (0.0, 0.0), "B": (7.0, 0.0), "C": (0.0, 7.0)}
# a batch effect whose direction reverses between subclasses — the case a single
# per-feature map cannot remove (mid-layer spine deficit, deep-layer excess)
SUBCLASS_DISTORTION = {"A": (1.8, 2.5), "B": (0.5, -2.5), "C": (1.3, 1.5)}


def _dataset(
    n_per: int,
    *,
    distort: bool,
    seed: int,
    first_id: int = 1,
    subclasses: tuple[str, ...] = ("A", "B", "C"),
) -> FeatureTable:
    """Three subclasses, each with two subtypes separated along ``f2``.

    ``distort`` stretches and shifts ``f2`` differently per subclass and warps ``f3``
    nonlinearly, so the within-subclass organization is the same but the datasets
    disagree about where it sits.
    """
    rng = np.random.default_rng(seed)
    blocks, subclass, subtype = [], [], []
    for name in subclasses:
        cx, cy = SUBCLASS_CENTERS[name]
        for sub, offset in (("0", 2.5), ("1", -2.5)):
            x = rng.normal(size=(n_per, 4))
            x[:, 0] += cx
            x[:, 1] += cy
            x[:, 2] += offset
            if distort:
                scale, shift = SUBCLASS_DISTORTION[name]
                x[:, 2] = x[:, 2] * scale + shift
                x[:, 3] = np.exp(x[:, 3] / 2)
            blocks.append(x)
            subclass += [name] * n_per
            subtype += [f"{name}{sub}"] * n_per
    values = np.vstack(blocks)
    df = pl.DataFrame(
        {
            "root_id": pl.Series(
                range(first_id, first_id + len(values)), dtype=pl.Int64
            ),
            "subclass": subclass,
            "subtype": subtype,
            **{f: values[:, i] for i, f in enumerate(FEATURES)},
        }
    )
    return FeatureTable(df, FEATURES, id_column="root_id")


def _pair(n_per: int = 80) -> dict[str, FeatureTable]:
    return {
        "minnie": _dataset(n_per, distort=False, seed=0),
        "v1dd": _dataset(n_per, distort=True, seed=1),
    }


def _naive_concat(tables: dict[str, FeatureTable]) -> FeatureTable:
    frames = []
    for position, (name, table) in enumerate(tables.items()):
        frames.append(
            table.dataframe().with_columns(
                pl.lit(name).alias("dataset"),
                (pl.col("root_id") + (position + 1) * 1_000_000).alias("cell_id"),
            )
        )
    return FeatureTable(pl.concat(frames), FEATURES, id_column="cell_id")


# --------------------------------------------------------------------------- #
# scalers
# --------------------------------------------------------------------------- #


def test_quantile_scaler_factory_carries_its_configuration() -> None:
    factory = quantile_scaler_factory(clip=(2.0, 98.0), output_distribution="uniform")
    assert factory._cellpax_scaler_params == {
        "kind": "quantile",
        "clip": [2.0, 98.0],
        "output_distribution": "uniform",
        "n_quantiles": 1000,
        "subsample": 100_000,
        "random_state": 0,
    }
    steps = factory().named_steps
    assert list(steps) == ["clipper", "scaler"]
    assert type(quantile_scaler_factory(clip=None)()).__name__ == "QuantileTransformer"
    with pytest.raises(ValueError, match="clip"):
        quantile_scaler_factory(clip=(50.0, 10.0))


@pytest.mark.parametrize(
    "factory",
    [
        StandardScaler,
        RobustScaler,
        quantile_scaler_factory(clip=None),
        quantile_scaler_factory(),
        clipped_scaler_factory(),
        clipped_scaler_factory(mode="sigma"),
    ],
    ids=["standard", "robust", "quantile", "clip-quantile", "robust-pct", "robust-sig"],
)
def test_scaler_records_round_trip_and_invert(factory) -> None:
    rng = np.random.default_rng(0)
    raw = np.column_stack(
        [rng.normal(size=400), np.exp(rng.normal(size=400)), rng.gamma(2.0, size=400)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fitted = FittedScaler(factory(), ["ihs", "log", "sqrt"]).fit(raw)
    meta, arrays = _scaler_records(fitted)
    rebuilt = _scaler_from_records(meta, _frame_arrays(_arrays_frame(arrays)))
    assert np.allclose(rebuilt.transform(raw), fitted.transform(raw))
    # inside the clip bounds every one of these maps back to the raw values
    inner = raw[(np.abs(fitted.transform(raw)) < 1).all(axis=1)]
    assert np.allclose(
        fitted.inverse_transform(fitted.transform(inner)), inner, atol=0.05
    )


def test_inverse_transform_requires_an_invertible_scaler() -> None:
    class NoInverse:
        def fit(self, x):
            return self

        def transform(self, x):
            return x

    fitted = FittedScaler(NoInverse(), [None]).fit(np.ones((3, 1)))
    with pytest.raises(TypeError, match="inverse_transform"):
        fitted.inverse_transform(np.ones((3, 1)))


# --------------------------------------------------------------------------- #
# ids and structure
# --------------------------------------------------------------------------- #


def test_ids_are_minted_in_dataset_blocks_and_originals_kept() -> None:
    tables = _pair()
    ft = join_datasets(tables, scaler_factory=StandardScaler)
    frame = ft.dataframe()
    assert ft.id_column == "cell_id"
    assert frame.schema["cell_id"] == pl.Int64
    assert frame["cell_id"].n_unique() == ft.n_cells == 960
    assert frame.filter(pl.col("dataset") == "minnie")["cell_id"].min() == 1000
    assert frame.filter(pl.col("dataset") == "v1dd")["cell_id"].min() == 2000
    assert ft.source_id_column == "source_cell_id"
    for name, table in tables.items():
        part = frame.filter(pl.col("dataset") == name)
        assert (
            part["source_cell_id"].to_list() == table.dataframe()["root_id"].to_list()
        )
    assert ft.datasets == ["minnie", "v1dd"]
    assert ft.dataset_column == "dataset"
    assert ft.reference_dataset == "minnie"
    assert set(ft.masks) == {"all", "minnie", "v1dd"}


def test_id_lookups_round_trip_in_both_directions() -> None:
    ft = join_datasets(_pair(), scaler_factory=StandardScaler)
    joined = ft.cell_ids_for("v1dd", [5, 1, 300])
    assert joined.tolist() == [2004, 2000, 2299]
    back = ft.source_ids(joined)
    assert back["cell_id"].to_list() == joined.tolist()
    assert back["dataset"].to_list() == ["v1dd"] * 3
    assert back["source_cell_id"].to_list() == [5, 1, 300]
    assert ft.source_ids().height == ft.n_cells
    with pytest.raises(ValueError, match="not cells of dataset 'v1dd'"):
        ft.cell_ids_for("v1dd", [99_999])
    with pytest.raises(ValueError, match="not in the table"):
        ft.source_ids([1])
    with pytest.raises(KeyError, match="unknown dataset"):
        ft.cell_ids_for("nope", [1])


def test_labels_write_back_to_the_source_tables_by_original_id() -> None:
    tables = _pair()
    ft = join_datasets(tables, scaler_factory=StandardScaler)
    joint = LabelSet.from_labels(
        ft.dataframe()["cell_id"].to_numpy(),
        ft.dataframe()["subtype"].to_list(),
        name="joint",
    )
    parts = ft.labels_by_dataset(joint)
    assert list(parts) == ["minnie", "v1dd"]
    tables["v1dd"].attach(parts["v1dd"])
    written = tables["v1dd"].dataframe()
    assert written["joint"].to_list() == written["subtype"].to_list()
    assert parts["v1dd"].color_map() == joint.color_map()


def test_unjoined_tables_refuse_the_dataset_verbs() -> None:
    table = _dataset(20, distort=False, seed=0)
    assert table.datasets == []
    assert table.dataset_column is None
    with pytest.raises(ValueError, match="not built by join_datasets"):
        table.harmonize(table.dataframe(), "a")
    with pytest.raises(ValueError, match="not built by join_datasets"):
        table.source_ids()


def test_join_columns_are_protected() -> None:
    ft = join_datasets(_pair(), scaler_factory=StandardScaler, strata="subclass")
    for column in ("dataset", "source_cell_id", "stratum"):
        with pytest.raises(ValueError, match="where each cell came from"):
            ft.add_column(np.zeros(ft.n_cells), column)
    extra = ft.dataframe().select("cell_id", pl.lit(1.0).alias("new"))
    with pytest.warns(UserWarning, match="without harmonization"):
        ft.add_features(extra)


def test_describe_mentions_the_join() -> None:
    ft = join_datasets(
        _pair(), scaler_factory=quantile_scaler_factory(), strata="subclass"
    )
    text = ft.describe()
    assert "datasets (2)" in text
    assert "reference 'minnie'" in text
    assert "stratified by 'stratum'" in text
    assert "kind=quantile" in text


# --------------------------------------------------------------------------- #
# harmonization
# --------------------------------------------------------------------------- #


def test_reference_cells_are_unchanged_and_harmonize_reproduces_the_join() -> None:
    tables = _pair()
    ft = join_datasets(
        tables, scaler_factory=quantile_scaler_factory(), strata="subclass"
    )
    minnie = ft.dataframe("minnie").select(FEATURES).to_numpy()
    assert np.allclose(minnie, tables["minnie"].features())
    raw_v1dd = tables["v1dd"].dataframe()
    again = ft.harmonize(raw_v1dd, "v1dd")
    assert np.allclose(
        again.select(FEATURES).to_numpy(),
        ft.dataframe("v1dd").select(FEATURES).to_numpy(),
    )
    assert again["root_id"].to_list() == raw_v1dd["root_id"].to_list()
    array = ft.harmonize(
        raw_v1dd.select(FEATURES).to_numpy(), "v1dd", strata=raw_v1dd["subclass"]
    )
    assert np.allclose(array, again.select(FEATURES).to_numpy())
    with pytest.raises(ValueError, match="needs a stratum"):
        ft.harmonize(raw_v1dd.select(FEATURES), "v1dd")


def test_stratified_mapping_matches_reference_distribution_per_stratum() -> None:
    ft = join_datasets(
        _pair(200), scaler_factory=quantile_scaler_factory(clip=None), strata="subclass"
    )
    frame = ft.dataframe()
    for subclass in ("A", "B", "C"):
        part = frame.filter(pl.col("subclass") == subclass)
        ref = part.filter(pl.col("dataset") == "minnie")["f2"].to_numpy()
        mapped = part.filter(pl.col("dataset") == "v1dd")["f2"].to_numpy()
        assert abs(np.median(ref) - np.median(mapped)) < 0.3
        assert abs(np.std(ref) - np.std(mapped)) < 0.3


def test_fit_mask_changes_the_fit_but_every_cell_is_mapped() -> None:
    tables = _pair()
    for table in tables.values():
        table.add_mask("half", pl.col("root_id") % 2 == 0)
    full = join_datasets(tables, scaler_factory=StandardScaler)
    half = join_datasets(tables, scaler_factory=StandardScaler, fit_mask="half")
    assert half.n_cells == full.n_cells
    assert half.dataframe("v1dd")["f0"].null_count() == 0
    assert not np.allclose(
        half.dataset_scaler("v1dd").scaler.mean_,
        full.dataset_scaler("v1dd").scaler.mean_,
    )


def test_each_datasets_own_transforms_are_applied_and_disagreement_warns() -> None:
    tables = _pair()
    tables["v1dd"].preprocess(skew_screen=False, method="ihs", columns=["f3"])
    with pytest.warns(UserWarning, match="disagree on the preprocess transform"):
        ft = join_datasets(tables, scaler_factory=StandardScaler)
    assert ft.dataset_scaler("v1dd").transforms == [None, None, None, "ihs"]
    assert ft.dataset_scaler("minnie").transforms == [None, None, None, None]
    assert ft.transforms == {}


def test_strata_errors() -> None:
    tables = _pair()
    lonely = _dataset(80, distort=True, seed=1, subclasses=("A", "B"))
    lonely.add_column(["D"] * lonely.n_cells, "subclass_d")
    with pytest.raises(ValueError, match="do not exist in the reference"):
        join_datasets(
            {"minnie": tables["minnie"], "v1dd": lonely},
            scaler_factory=StandardScaler,
            strata={"minnie": "subclass", "v1dd": "subclass_d"},
        )
    with pytest.raises(ValueError, match="fewer than min_cells"):
        join_datasets(
            tables, scaler_factory=StandardScaler, strata="subclass", min_cells=500
        )
    holes = _dataset(80, distort=True, seed=1)
    holes.add_column(
        [None if i < 3 else s for i, s in enumerate(holes.dataframe()["subclass"])],
        "subclass",
        overwrite=True,
    )
    with pytest.raises(ValueError, match="have no 'subclass' value"):
        join_datasets(
            {"minnie": tables["minnie"], "v1dd": holes},
            scaler_factory=StandardScaler,
            strata="subclass",
        )


def test_reference_only_strata_pass_through() -> None:
    minnie = _dataset(80, distort=False, seed=0)
    v1dd = _dataset(80, distort=True, seed=1, subclasses=("A", "C"))
    ft = join_datasets(
        {"minnie": minnie, "v1dd": v1dd},
        scaler_factory=StandardScaler,
        strata="subclass",
    )
    only_ref = ft.dataframe().filter(pl.col("subclass") == "B")
    assert set(only_ref["dataset"]) == {"minnie"}
    with pytest.raises(KeyError):
        ft.dataset_scaler("minnie", "B")


def test_argument_errors() -> None:
    tables = _pair()
    with pytest.raises(ValueError, match="at least two"):
        join_datasets({"minnie": tables["minnie"]}, scaler_factory=StandardScaler)
    with pytest.raises(ValueError, match="invalid dataset name"):
        join_datasets(
            {"all": tables["minnie"], "b": tables["v1dd"]},
            scaler_factory=StandardScaler,
        )
    with pytest.raises(TypeError, match="not a StandardScaler instance|instance"):
        join_datasets(tables, scaler_factory=StandardScaler())
    with pytest.raises(TypeError, match="no inverse_transform"):
        join_datasets(tables, scaler_factory=lambda: _NoInverse())
    with pytest.raises(ValueError, match="not one of the datasets"):
        join_datasets(tables, scaler_factory=StandardScaler, reference="nope")
    fewer = _dataset(20, distort=False, seed=0)
    fewer.add_features(fewer.dataframe().select("root_id", pl.lit(0.0).alias("extra")))
    with pytest.raises(ValueError, match="different feature set"):
        join_datasets({"a": fewer, "b": tables["v1dd"]}, scaler_factory=StandardScaler)
    joined = join_datasets(
        {"a": fewer, "b": tables["v1dd"]},
        scaler_factory=StandardScaler,
        columns=FEATURES,
    )
    assert joined.feature_columns == FEATURES
    assert "extra" not in joined.columns
    clash = _dataset(20, distort=False, seed=0)
    clash.add_column(["x"] * clash.n_cells, "dataset")
    with pytest.raises(ValueError, match="reserves"):
        join_datasets({"a": clash, "b": tables["v1dd"]}, scaler_factory=StandardScaler)


class _NoInverse:
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x


# --------------------------------------------------------------------------- #
# carrying structure over
# --------------------------------------------------------------------------- #


def test_metadata_masks_collections_and_validity_are_carried() -> None:
    tables = _pair(40)
    minnie, v1dd = tables["minnie"], tables["v1dd"]
    minnie.add_column(np.arange(minnie.n_cells), "depth")
    for table in tables.values():
        table.add_mask("left", pl.col("f0") < 3)
        table.define_features("pair", columns=["f0", "f1"])
        table.set_validity(["f3"], where="left")
    minnie.add_mask("only_minnie", pl.col("f1") > 0)
    with pytest.warns(UserWarning, match="missing from some datasets"):
        ft = join_datasets(tables, scaler_factory=StandardScaler)
    frame = ft.dataframe()
    assert frame.filter(pl.col("dataset") == "v1dd")["depth"].null_count() == 240
    assert ft.mask_series("only_minnie").to_numpy()[240:].sum() == 0
    expected_left = np.concatenate(
        [minnie.mask_series("left").to_numpy(), v1dd.mask_series("left").to_numpy()]
    )
    assert np.array_equal(ft.mask_series("left").to_numpy(), expected_left)
    assert ft.collections["pair"].columns == ("f0", "f1")
    assert ft.validity_domains == {"f3": "left"}


def test_conflicting_structure_raises() -> None:
    tables = _pair(20)
    tables["minnie"].define_features("pair", columns=["f0", "f1"])
    tables["v1dd"].define_features("pair", columns=["f2", "f3"])
    with pytest.raises(ValueError, match="collection 'pair'"):
        join_datasets(tables, scaler_factory=StandardScaler)

    tables = _pair(20)
    tables["minnie"].add_mask("left", pl.col("f0") < 3)
    tables["minnie"].set_validity(["f3"], where="left")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="validity domains"):
        join_datasets(tables, scaler_factory=StandardScaler)

    tables = _pair(20)
    tables["minnie"].add_mask("v1dd", pl.col("f0") < 3)
    with pytest.raises(ValueError, match="share a name"):
        join_datasets(tables, scaler_factory=StandardScaler)


def test_feature_metadata_merges_and_conflicts_raise() -> None:
    tables = _pair(20)
    meta = pl.DataFrame({"feature_id": FEATURES, "family": ["a", "a", "b", "b"]})
    rebuilt = {
        name: FeatureTable(
            table.dataframe(), FEATURES, id_column="root_id", feature_metadata=meta
        )
        for name, table in tables.items()
    }
    ft = join_datasets(rebuilt, scaler_factory=StandardScaler)
    assert ft.var["family"].to_list() == ["a", "a", "b", "b"]
    rebuilt["v1dd"] = FeatureTable(
        tables["v1dd"].dataframe(),
        FEATURES,
        id_column="root_id",
        feature_metadata=meta.with_columns(pl.lit("z").alias("family")),
    )
    with pytest.raises(ValueError, match="annotate features differently"):
        join_datasets(rebuilt, scaler_factory=StandardScaler)


def test_attached_labels_are_carried_and_merged_by_name() -> None:
    tables = _pair(40)
    for name, table in tables.items():
        labels = table.labelset("subtype").copy(name="curated")
        if name == "minnie":
            labels.set_colors({"A0": "#ff0000"})
        table.attach(labels)
    minnie_labels = tables["minnie"].labelset("curated")
    tables["minnie"].attach(
        _propagation_like(minnie_labels, np.linspace(0, 1, len(minnie_labels))),
        name="nn",
    )
    ft = join_datasets(tables, scaler_factory=StandardScaler)
    carried = ft.labelset("curated")
    assert carried.n_clusters == 6
    assert carried.color_map()["A0"] == "#ff0000"
    frame = ft.dataframe()
    assert frame["curated"].to_list() == frame["subtype"].to_list()
    assert "nn_confidence" in frame.columns
    assert frame.filter(pl.col("dataset") == "v1dd")["nn"].null_count() == 240
    assert ft.labelset("nn").mask == "minnie"


def _propagation_like(labels: LabelSet, confidence: np.ndarray):
    class _Propagation:
        def __init__(self) -> None:
            self.labels = labels
            self.confidence = confidence

    return _Propagation()


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def test_dataset_mixing_reads_zero_when_mixed_and_large_when_separated() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(600, 3))
    datasets = np.array(["a", "b"] * 300)
    mixed = dataset_mixing(points, datasets, n_neighbors=20)
    assert abs(mixed["gap"][0]) < 0.05
    apart = points.copy()
    apart[datasets == "b"] += 20
    separated = dataset_mixing(apart, datasets, n_neighbors=20)
    assert separated["gap"][0] > 0.45
    assert separated["dataset"].to_list() == [None, "a", "b"]


def test_dataset_mixing_adjusts_for_composition_within_strata() -> None:
    rng = np.random.default_rng(0)
    # stratum x is 90% dataset a, stratum y 90% dataset b; within each, perfectly mixed
    x = rng.normal(size=(500, 2))
    y = rng.normal(size=(500, 2)) + 30
    points = np.vstack([x, y])
    datasets = np.array(["a"] * 450 + ["b"] * 50 + ["a"] * 50 + ["b"] * 450)
    rng.shuffle(datasets[:500])
    rng.shuffle(datasets[500:])
    strata = np.array(["x"] * 500 + ["y"] * 500)
    unadjusted = dataset_mixing(points, datasets, n_neighbors=20)
    adjusted = dataset_mixing(points, datasets, strata=strata, n_neighbors=20)
    assert unadjusted["gap"][0] > 0.25
    assert abs(adjusted["gap"][0]) < 0.05
    assert set(adjusted.drop_nulls("stratum")["stratum"]) == {"x", "y"}


def test_transfer_score_separates_unlearnable_labels() -> None:
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(size=(100, 2)), rng.normal(size=(100, 2)) + 10])
    features = np.vstack(
        [features, features + rng.normal(scale=0.1, size=features.shape)]
    )
    labels = np.array(
        ["p"] * 100 + ["q"] * 100 + ["p"] * 100 + ["q_new"] * 100, dtype=object
    )
    datasets = np.array(["a"] * 200 + ["b"] * 200)
    score = cross_dataset_classification(
        features, labels, datasets, train="a", test="b", seed=0
    )
    assert score.n_cells == 200
    assert score.accuracy == pytest.approx(0.5)
    assert score.accuracy_shared == pytest.approx(1.0)
    by_label = score.by_label()
    new = by_label.filter(pl.col("truth") == "q_new").row(0, named=True)
    assert new["in_train"] is False
    assert new["top_prediction"] == "q"
    confusion = score.confusion()
    assert confusion.group_by("truth").agg(pl.col("fraction").sum())[
        "fraction"
    ].to_list() == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="different datasets"):
        cross_dataset_classification(features, labels, datasets, train="a", test="a")


# --------------------------------------------------------------------------- #
# integration: the point of the whole thing
# --------------------------------------------------------------------------- #


def test_stratified_join_mixes_datasets_and_transfers_labels() -> None:
    tables = _pair(150)
    naive = _naive_concat(tables)
    naive_gap = naive.dataset_mixing(dataset_column="dataset", strata="subclass")
    naive_transfer = naive.cross_dataset_classification(
        "subtype", train="minnie", test="v1dd", dataset_column="dataset"
    )

    factory = quantile_scaler_factory()
    global_join = join_datasets(tables, scaler_factory=factory)
    global_gap = global_join.dataset_mixing(strata="subclass")
    stratified = join_datasets(tables, scaler_factory=factory, strata="subclass")
    stratified_gap = stratified.dataset_mixing()
    transfer = stratified.cross_dataset_classification(
        "subtype", train="minnie", test="v1dd"
    )

    assert naive_gap["gap"][0] > 0.15
    assert stratified_gap["gap"][0] < global_gap["gap"][0] < naive_gap["gap"][0]
    assert abs(stratified_gap["gap"][0]) < 0.05
    per_stratum = stratified_gap.drop_nulls("stratum")
    assert per_stratum["gap"].abs().max() < 0.1
    assert transfer.accuracy > 0.95 > naive_transfer.accuracy

    # subclasses stay separable in the reference's units, and within one subclass the
    # joint clustering recovers subtypes with both datasets in every cluster
    stratified.add_mask("sub_a", pl.col("subclass") == "A")
    frame = stratified.dataframe("sub_a").with_columns(
        pl.Series(
            "cut",
            KMeans(2, n_init=10, random_state=0).fit_predict(
                stratified.features("sub_a", scaled=True)
            ),
        )
    )
    assert adjusted_rand_score(frame["subtype"], frame["cut"]) > 0.9
    composition = frame.group_by("cut").agg(pl.col("dataset").n_unique())
    assert composition["dataset"].to_list() == [2, 2]


def test_new_cells_are_harmonized_then_labelled_through_the_joined_fits() -> None:
    tables = _pair(150)
    stratified = join_datasets(
        tables, scaler_factory=quantile_scaler_factory(), strata="subclass"
    )
    stratified.attach(stratified.labelset("subtype").copy(name="truth"))
    held_out = _dataset(30, distort=True, seed=7, first_id=90_000)
    rows = stratified.harmonize(held_out.dataframe(), "v1dd").rename(
        {"root_id": "cell_id"}
    )
    projected = stratified.project_labels(rows, "truth", name="truth")
    predicted = projected.labels.to_frame().join(
        rows.select("cell_id", "subtype"), on="cell_id"
    )
    assert (predicted["truth"] == predicted["subtype"]).mean() > 0.9


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory",
    [
        StandardScaler,
        clipped_scaler_factory(),
        clipped_scaler_factory(mode="sigma"),
        quantile_scaler_factory(),
        quantile_scaler_factory(clip=None, output_distribution="uniform"),
    ],
    ids=["standard", "robust-pct", "robust-sigma", "clip-quantile", "quantile"],
)
@pytest.mark.parametrize("strata", [None, "subclass"])
def test_joined_table_round_trips(tmp_path: Path, factory, strata) -> None:
    tables = _pair(60)
    ft = join_datasets(
        tables,
        scaler_factory=factory,
        strata=strata,
        output_scaler_factory=quantile_scaler_factory(),
    )
    ft.space()
    ft.save(tmp_path / "folio", "joined")
    back = FeatureTable.load(tmp_path / "folio", "joined")

    assert back.dataframe().equals(ft.dataframe())
    assert back.datasets == ft.datasets
    assert back.reference_dataset == ft.reference_dataset
    assert back.stratum_column == ft.stratum_column
    assert back.source_id_column == ft.source_id_column
    assert np.allclose(back.features(scaled=True), ft.features(scaled=True))
    held_out = _dataset(20, distort=True, seed=9, first_id=50_000).dataframe()
    assert np.allclose(
        back.harmonize(held_out, "v1dd").select(FEATURES).to_numpy(),
        ft.harmonize(held_out, "v1dd").select(FEATURES).to_numpy(),
    )
    assert (
        back.cell_ids_for("v1dd", [3]).tolist() == ft.cell_ids_for("v1dd", [3]).tolist()
    )
    assert list(back._space_cache) == list(ft._space_cache)


def test_unrecordable_scaler_warns_at_join_and_refuses_to_save(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="saving it will raise"):
        ft = join_datasets(_pair(20), scaler_factory=MinMaxScaler)
    with pytest.raises(TypeError, match="per-dataset scaler of dataset"):
        ft.save(tmp_path / "folio", "joined")
    assert list_analyses(tmp_path / "folio") == []


def test_unjoined_table_reloads_without_datasets(tmp_path: Path) -> None:
    table = _dataset(20, distort=False, seed=0)
    table.save(tmp_path / "folio", "plain")
    assert FeatureTable.load(tmp_path / "folio", "plain").datasets == []
