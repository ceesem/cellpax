"""What the feature matrix looks like before any clustering runs on it.

Six questions that decide whether a clustering result is even interpretable, each
answered as a frame rather than as advice:

:func:`feature_correlation`
    Which features are measuring the same thing. Redundant blocks are what make PCA
    truncation insufficient on its own — see :class:`~cellpax.space.FittedSpace` and its
    ``alpha`` — and they are obvious in an ordered correlation matrix and invisible in an
    unordered one.
:func:`tie_report` / :func:`duplicate_rows`
    Whether low-cardinality features have produced cells at identical coordinates. A tie
    block is an atom in the neighbour graph: every member has the same neighbours, and
    the weighting schemes that subtract a distance to the nearest neighbour have nothing
    to subtract.
:func:`clip_comparison`
    How many cells each clipping rule actually clips at this cohort's size. The percentile
    rule and the sigma rule agree at twenty thousand cells and disagree completely at
    five hundred, and which one is in use is usually inherited rather than chosen.
:func:`covariate_sensitivity`
    Which features track a nuisance covariate such as reconstruction completeness. A
    feature that rises and falls with how much of the cell sits inside the volume is
    measuring truncation, not biology — measurable but uninformative on peripheral
    cells — and is the raw material for a restricted validity domain.
:func:`stratum_shift`
    Which features shift distribution across strata — datasets, volumes, extraction
    versions. A feature that moves when the stratum changes is measuring the dataset
    rather than the cells, and does not travel: it wants a per-stratum validity domain
    or a seat outside the portable collection.

None of these need a clustering to have been run, and all of them are cheap. They are the
things worth reading *before* attributing a result to biology.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp, spearmanr

from cellpax.clustering import SortedMatrix

# Robust sigma per IQR, for a Gaussian.
_IQR_TO_SIGMA = 1.3489795003921634


def _feature_names(columns: Any, n_features: int) -> list[str]:
    """Validate ``columns`` as actual feature names, aligned to a matrix width.

    Rejects a bare string explicitly. These functions take an array rather than a
    ``FeatureTable``, so they cannot resolve a collection *name* — and ``list("analysis")``
    silently becomes eight one-character "names", which the width check only catches by
    luck. Fail on the type instead.
    """
    if isinstance(columns, str):
        raise TypeError(
            f"columns must be feature names, not the string {columns!r} — that would "
            f"iterate into {len(columns)} single characters. These functions take a "
            "feature matrix and cannot resolve a collection name; pass "
            f"ft.collections[{columns!r}].columns, or use the FeatureTable method, "
            "which resolves it for you."
        )
    names = [str(name) for name in columns]
    if len(names) != n_features:
        raise ValueError(
            f"features has {n_features} columns but {len(names)} names were given"
        )
    return names


def feature_correlation(
    features: np.ndarray,
    columns: Any,
    *,
    method: str = "pearson",
    absolute: bool = True,
    block_threshold: float = 0.5,
    linkage_method: str = "average",
) -> SortedMatrix:
    """Feature-by-feature correlation, ordered so redundant blocks are contiguous.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix. Scaled or raw — correlation is scale-invariant.
    columns : sequence of str
        Feature names, in the column order of ``features``.
    method : {'pearson', 'spearman'}, default 'pearson'
        Correlation type. ``'spearman'`` ranks first, so it catches monotone but
        non-linear redundancy.
    absolute : bool, default True
        Cluster on ``1 - |r|``, so strongly *anti*-correlated features group together —
        usually what you want, since either way they carry one measurement.
    block_threshold : float, default 0.5
        Where to cut the feature dendrogram, in the same ``1 - |r|`` units: the default
        groups features correlating above about 0.5. Raise it to merge more
        loosely-related features into one block, lower it to split.
    linkage_method : str, default 'average'
        Linkage for ordering the features.

    Returns
    -------
    SortedMatrix
        The same shape of object ``Clustering.sorted_matrix`` produces, so the plotting
        recipe in its docstring applies unchanged — with feature-level meanings:
        ``matrix`` is the reordered correlation matrix, ``cell_ids`` holds the feature
        names in sorted order (they are the row identifiers here), ``codes`` the block each
        feature fell into, and ``boundaries`` the block edges, i.e. the divider lines that
        make a redundant block visible as a square.

    >>> sm = feature_correlation(ft.features(mask, columns="analysis"),   # doctest: +SKIP
    ...                          ft.collections["analysis"].columns)
    >>> ax.imshow(sm.matrix, vmin=-1, vmax=1, cmap="RdBu_r")             # doctest: +SKIP
    >>> ax.set_xticks(range(sm.n_cells), sm.cell_ids, rotation=90)       # doctest: +SKIP
    >>> ax.hlines(sm.boundaries[1:-1] - 0.5, *ax.get_xlim())             # doctest: +SKIP

    What to look for: a block of *k* features correlating near 1.0 contributes *k* times
    its share to Euclidean distance while carrying one measurement's worth of
    information. PCA rotates that into one high-eigenvalue component but does not reweight
    it, which is the case ``alpha`` exists for. The largest block's size is a good guide
    to how much whitening is likely to matter.
    """
    features = np.asarray(features, dtype=float)
    names = _feature_names(columns, features.shape[1])
    if features.shape[1] < 2:
        raise ValueError("need at least two features to correlate")

    if method == "pearson":
        values = features
    elif method == "spearman":
        values = np.apply_along_axis(_rankdata, 0, features)
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    # A constant feature has zero variance, so numpy divides by zero and returns nan for
    # its whole row. That is a handled case — it correlates with nothing — so silence the
    # warning rather than letting a nan reach the linkage.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    distance = 1.0 - (np.abs(correlation) if absolute else correlation)
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    link = linkage(squareform(distance, checks=False), method=linkage_method)
    order = leaves_list(link)
    blocks = fcluster(link, t=block_threshold, criterion="distance")

    ordered_blocks = blocks[order]
    edges = np.flatnonzero(np.diff(ordered_blocks)) + 1
    boundaries = np.concatenate([[0], edges, [len(order)]])
    sizes = np.diff(boundaries)
    return SortedMatrix(
        matrix=correlation[np.ix_(order, order)],
        order=order,
        codes=ordered_blocks.astype(np.int64),
        names=[f"block_{i}" for i in range(len(sizes))],
        boundaries=boundaries,
        cell_ids=np.array(names, dtype=object)[order],
        sizes=sizes.astype(np.int64),
    )


def block_weights(
    features: np.ndarray,
    columns: Any,
    *,
    method: str = "mfa",
    block_threshold: float = 0.5,
    linkage_method: str = "average",
) -> np.ndarray:
    """Per-feature weights that stop a correlated block counting once per member.

    The other way to keep a redundant feature block from dominating Euclidean distance.
    Where ``FittedSpace``'s ``alpha`` flattens the whole eigenvalue spectrum — and pays
    for it by amplifying low-variance components whose *directions* a finite sample
    barely determines — this acts on the features directly, so the component directions
    are left alone. It is the better-targeted instrument when the diagnosis is
    "eleven columns measure one thing".

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` **scaled** matrix — the same one
        :meth:`~cellpax.space.FittedSpace.fit` takes.
    columns : sequence of str
        Feature names, in the column order of ``features``. Must be actual names: a bare
        string is rejected rather than iterated into characters.
    method : {'mfa', 'sqrt_k'}, default 'mfa'
        ``'mfa'`` divides each block by the standard deviation along its own first
        principal direction, after Escofier & Pagès' multiple factor analysis. That is an
        *adaptive* ``1/√k``: it collapses to it for a perfectly correlated block and
        barely touches a loosely correlated one, so a block of eight at r≈0.3 is not
        punished as hard as a block of eight at r≈0.99. ``'sqrt_k'`` is the plain
        size-based version.
    block_threshold : float, default 0.5
        Where the feature dendrogram is cut, in ``1 - |r|`` units — see
        :func:`feature_correlation`, which defines the blocks.
    linkage_method : str, default 'average'
        Linkage used to group the features.

    Returns
    -------
    numpy.ndarray
        One weight per feature, aligned to ``columns``. Pass to
        ``ft.space(..., feature_weights=w)`` so the weighting is frozen with the fit and
        applied by ``transform``; multiplying by hand works for a look but leaves the
        weights outside the space, where ``embed`` and ``project`` will skip them.

    Notes
    -----
    Singleton blocks get weight 1.0, so an uncorrelated feature is untouched. A block
    whose leading direction has zero variance (constant features) is also left at 1.0
    rather than dividing by zero.
    """
    features = np.asarray(features, dtype=float)
    names = _feature_names(columns, features.shape[1])
    if method not in {"mfa", "sqrt_k"}:
        raise ValueError(f"method must be 'mfa' or 'sqrt_k', got {method!r}")

    sorted_matrix = feature_correlation(
        features,
        names,
        block_threshold=block_threshold,
        linkage_method=linkage_method,
    )
    position = {name: index for index, name in enumerate(names)}
    weights = np.ones(len(names), dtype=float)
    for lo, hi in zip(sorted_matrix.boundaries[:-1], sorted_matrix.boundaries[1:]):
        block = [str(f) for f in sorted_matrix.cell_ids[lo:hi]]
        if len(block) == 1:
            continue
        index = [position[name] for name in block]
        if method == "sqrt_k":
            weights[index] = 1.0 / np.sqrt(len(index))
            continue
        centered = features[:, index] - features[:, index].mean(axis=0)
        sigma = np.linalg.svd(centered, compute_uv=False)[0] / np.sqrt(
            max(centered.shape[0] - 1, 1)
        )
        if sigma > 0:
            weights[index] = 1.0 / sigma
    return weights


def tie_report(features: np.ndarray, columns: Any) -> pl.DataFrame:
    """Per-feature cardinality and how many cells sit at each extreme.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix.
    columns : sequence of str
        Feature names, in the column order of ``features``.

    Returns
    -------
    pl.DataFrame
        One row per feature — ``feature``, ``n_distinct``, ``frac_distinct``,
        ``n_at_min``, ``n_at_max``, ``n_zero`` — sorted so the lowest-cardinality features
        come first.

    Notes
    -----
    Low ``frac_distinct`` marks the discrete features — integer branch and synapse counts,
    small-integer Strahler means — where many cells share a value. On its own that is
    harmless; it matters when enough such features are used together that whole cells
    coincide, which is what :func:`duplicate_rows` measures.

    ``n_at_min`` / ``n_at_max`` are worth reading after clipping rather than before: a
    percentile clip *creates* ties by pinning its tails to a bound, so a feature with
    twenty cells at exactly its maximum has usually been clipped rather than measured
    that way.
    """
    features = np.asarray(features)
    names = _feature_names(columns, features.shape[1])
    n_rows = features.shape[0]
    rows = []
    for index, name in enumerate(names):
        column = features[:, index]
        finite = column[np.isfinite(column)]
        distinct = int(np.unique(finite).size)
        rows.append(
            {
                "feature": name,
                "n_distinct": distinct,
                "frac_distinct": distinct / n_rows if n_rows else 0.0,
                "n_at_min": int((column == finite.min()).sum()) if finite.size else 0,
                "n_at_max": int((column == finite.max()).sum()) if finite.size else 0,
                "n_zero": int((column == 0).sum()),
            }
        )
    return pl.DataFrame(rows).sort("frac_distinct")


def duplicate_rows(features: np.ndarray) -> pl.DataFrame:
    """Cells sitting at exactly the same coordinates, as a one-row summary.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix. Pass the same columns a graph will be built on —
        a subset collides far more readily than the full set.

    Returns
    -------
    pl.DataFrame
        One row: ``n_rows``, ``n_unique_rows``, ``n_duplicate_rows`` (cells that are not
        the first member of their group), ``n_duplicate_groups``, ``max_tie_block``.

    Notes
    -----
    A tie block is an atom in the neighbour graph: its members are mutually at distance 0,
    so they have identical neighbourhoods and no weighting can separate them. It also
    breaks the two graph weightings that use a local distance scale — ``umap_fuzzy``
    subtracts each cell's distance to its nearest neighbour, and ``knn_distance`` divides
    by the mean neighbour distance, both of which are zero inside a block.
    :func:`~cellpax.clustering.kneighbor_graph` detects that case and warns, but knowing
    the count in advance is what tells you whether to expect it.

    Expect zero on a few dozen continuous morphometrics — the discrete features cannot
    force a collision across all the continuous ones. It becomes real on a small column
    subset, or on features that are all counts.
    """
    features = np.asarray(features)
    n_rows = features.shape[0]
    _, counts = np.unique(features, axis=0, return_counts=True)
    return pl.DataFrame(
        {
            "n_rows": [n_rows],
            "n_unique_rows": [int(counts.size)],
            "n_duplicate_rows": [int(n_rows - counts.size)],
            "n_duplicate_groups": [int((counts > 1).sum())],
            "max_tie_block": [int(counts.max()) if counts.size else 0],
        }
    )


def clip_comparison(
    features: np.ndarray,
    columns: Any,
    *,
    lower: float = 0.1,
    upper: float = 99.9,
    n_sigma: float = 5.0,
) -> pl.DataFrame:
    """How many cells each clipping rule would clip, per feature, at this cohort's size.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` **raw** matrix — robust scaling is applied internally,
        mirroring what :func:`~cellpax.clustering.make_clipped_scaler` does. Non-finite
        entries are left out of every statistic rather than poisoning their feature's
        whole row, with a warning naming the features that carry them.
    columns : sequence of str
        Feature names, in the column order of ``features``.
    lower, upper : float, default 0.1 and 99.9
        The percentile rule to evaluate.
    n_sigma : float, default 5.0
        The sigma rule to evaluate, in IQR units.

    Returns
    -------
    pl.DataFrame
        One row per feature, sorted by ``n_clipped_sigma`` descending:
        ``n_clipped_percentile`` and ``n_clipped_sigma`` (what each rule actually takes),
        ``expected_percentile_cells`` (``ceil(n × lower/100)`` per tail — what the
        percentile rule clips *by construction*), ``max_abs_sigma``, and
        ``n_sigma_gaussian_equivalent``.

    Notes
    -----
    The comparison is the point, in two directions. A percentile rule clips a fixed
    fraction whatever the data looks like, so ``n_clipped_percentile`` tracks ``n`` rather
    than tracking how extreme anything is. And at small ``n`` its bound is not robust: at
    n=500 with ``lower=0.1`` the bound is interpolated between the two most extreme
    observations, so an outlier partly sets the bound that is supposed to clip it and the
    clipping weakens as the outlier grows. The sigma rule has neither property — it clips
    what lies beyond ``±n_sigma`` and nothing otherwise, identically at n=500 and n=21000,
    from a median and IQR that a handful of extreme cells cannot move.

    ``max_abs_sigma`` is the useful companion column: it says how extreme the most extreme
    cell actually is, which is what decides whether any clipping is warranted at all.

    ``n_sigma`` is in IQR units, since that is what ``RobustScaler`` divides by. For a
    Gaussian, IQR ≈ 1.349σ, so ``5.0`` is about ±6.7 Gaussian σ; the
    ``n_sigma_gaussian_equivalent`` column states that conversion so the number is not
    misread. A useful sweep is nearer 3–5 than 5–10.
    """
    features = np.asarray(features, dtype=float)
    names = _feature_names(columns, features.shape[1])
    non_finite = ~np.isfinite(features)
    if non_finite.any():
        offenders = [name for name, hit in zip(names, non_finite.any(axis=0)) if hit]
        warnings.warn(
            "non-finite values are ignored in the clipping statistics; features "
            f"affected: {offenders}",
            RuntimeWarning,
            stacklevel=2,
        )
        features = np.where(non_finite, np.nan, features)
    n_rows = features.shape[0]
    median = np.nanmedian(features, axis=0)
    q75, q25 = np.nanpercentile(features, [75, 25], axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > 0, iqr, 1.0)
    scaled = (features - median) / scale

    low_bounds = np.nanpercentile(scaled, lower, axis=0)
    high_bounds = np.nanpercentile(scaled, upper, axis=0)
    percentile_clipped = ((scaled < low_bounds) | (scaled > high_bounds)).sum(axis=0)
    sigma_clipped = (np.abs(scaled) > n_sigma).sum(axis=0)

    return pl.DataFrame(
        {
            "feature": names,
            "n_rows": np.full(len(names), n_rows, dtype=np.int64),
            "n_clipped_percentile": percentile_clipped.astype(np.int64),
            "n_clipped_sigma": sigma_clipped.astype(np.int64),
            "expected_percentile_cells": np.full(
                len(names), int(np.ceil(n_rows * lower / 100.0)), dtype=np.int64
            ),
            "max_abs_sigma": np.nanmax(np.abs(scaled), axis=0),
            "n_sigma": np.full(len(names), float(n_sigma)),
            # RobustScaler divides by the IQR, and IQR = 1.349 sigma for a Gaussian, so a
            # bound of n_sigma IQR units sits at n_sigma * 1.349 Gaussian sigma.
            "n_sigma_gaussian_equivalent": np.full(
                len(names), float(n_sigma) * _IQR_TO_SIGMA
            ),
        }
    ).sort("n_clipped_sigma", descending=True)


def covariate_sensitivity(
    features: np.ndarray,
    covariate: np.ndarray,
    names: Any,
    *,
    min_finite: int = 10,
) -> pl.DataFrame:
    """Which features track a nuisance covariate, as one rank correlation per feature.

    The first half of building a validity domain from data. Take a
    reconstruction-completeness metric — the fraction of a cell's axon inside the
    volume, say — and ask which features rise and fall with it. A truncated
    reconstruction still yields an axon length; the number is measurable, just
    meaningless, and no downstream feature-importance method can recover that
    distinction from the matrix alone. This ranks the candidates for restriction.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix. Raw or scaled — rank correlation is
        invariant to any monotone rescaling.
    covariate : numpy.ndarray
        ``(n_cells,)`` numeric nuisance covariate, e.g. fraction of axon inside the
        volume. Non-finite entries are left out of every correlation, with a warning
        naming how many — a covariate that is itself partly unmeasured is worth
        knowing about before trusting what it exonerates.
    names : sequence of str
        Feature names, in the column order of ``features``.
    min_finite : int, default 10
        Features with fewer finite (feature, covariate) pairs than this get null
        statistics rather than a correlation estimated from almost nothing — and
        rather than being dropped silently, so the row still says the feature was too
        sparse to judge. ``n_finite`` is reported either way.

    Returns
    -------
    pl.DataFrame
        One row per feature, sorted by ``|spearman_rho|`` descending with nulls last:
        ``feature``, ``spearman_rho``, ``p_value``, ``n_finite``.

    Notes
    -----
    How to read it: a feature strongly correlated with completeness is measuring
    truncation, not biology, and belongs in a restricted validity domain —
    informative only on cells where the covariate says the measurement can be
    trusted. The converse does not hold: a near-zero rho is *necessary but not
    sufficient* evidence of safety. Rank correlation sees only monotone dependence,
    and truncation can corrupt a feature non-monotonically — a tortuosity that first
    rises and then falls as more of the arbor is cut away has a rho near zero and is
    still uninformative on truncated cells. A low rho earns a feature a closer look,
    not a clean bill.

    Spearman rather than Pearson because completeness metrics are bounded and their
    relationships to morphometrics are rarely linear; ranks catch any monotone
    dependence and are indifferent to the covariate's units. A feature (or covariate)
    that is constant on the finite rows gets null statistics — a rank correlation
    with nothing to rank is undefined, not zero.
    """
    features = np.asarray(features, dtype=float)
    feature_names = _feature_names(names, features.shape[1])
    covariate = np.asarray(covariate, dtype=float)
    if covariate.shape != (features.shape[0],):
        raise ValueError(
            f"covariate must have one value per cell: features has "
            f"{features.shape[0]} rows but covariate has shape {covariate.shape}"
        )
    covariate_finite = np.isfinite(covariate)
    n_bad = int((~covariate_finite).sum())
    if n_bad:
        warnings.warn(
            f"the covariate itself has {n_bad} non-finite values; those cells are "
            "left out of every correlation",
            RuntimeWarning,
            stacklevel=2,
        )
    rows = []
    for index, name in enumerate(feature_names):
        column = features[:, index]
        mask = covariate_finite & np.isfinite(column)
        n_finite = int(mask.sum())
        rho = p_value = None
        if n_finite >= max(min_finite, 2):
            paired_feature = column[mask]
            paired_covariate = covariate[mask]
            # A constant side leaves nothing to rank: undefined, not zero.
            if np.ptp(paired_feature) > 0 and np.ptp(paired_covariate) > 0:
                result = spearmanr(paired_feature, paired_covariate)
                rho = float(result.statistic)
                p_value = float(result.pvalue)
        rows.append(
            {
                "feature": name,
                "spearman_rho": rho,
                "p_value": p_value,
                "n_finite": n_finite,
            }
        )
    frame = pl.DataFrame(
        rows,
        schema_overrides={
            "spearman_rho": pl.Float64,
            "p_value": pl.Float64,
            "n_finite": pl.Int64,
        },
    )
    return frame.sort(pl.col("spearman_rho").abs(), descending=True, nulls_last=True)


def stratum_shift(
    features: np.ndarray,
    strata: np.ndarray,
    names: Any,
    *,
    reference: Any = None,
    min_cells: int = 20,
) -> pl.DataFrame:
    """Which features shift distribution across strata, per feature and stratum.

    The second half of building a validity domain from data. Strata are typically
    datasets or volumes; a feature whose distribution moves when the stratum changes
    is measuring the dataset — acquisition, segmentation, extraction version — rather
    than the cells, and either wants a per-stratum validity domain or a seat outside
    the portable collection altogether.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n_cells, n_features)`` matrix.
    strata : numpy.ndarray
        ``(n_cells,)`` stratum labels, strings or ints.
    names : sequence of str
        Feature names, in the column order of ``features``.
    reference : optional
        What each stratum is compared against. ``None`` (default) pools all *other*
        strata, so each stratum is asked whether it differs from everyone else. A
        stratum label compares every stratum against that one — the right shape when
        one dataset is the anchor the others are supposed to match. The reference
        stratum itself is kept in the output with null statistics rather than
        skipped, so every (feature, stratum) pair has a row and ``n_cells`` is still
        stated.
    min_cells : int, default 20
        Comparisons where either side has fewer finite values than this get null
        statistics — a KS statistic on a dozen cells is mostly noise — with
        ``n_cells`` still reported, so the small stratum is visible rather than
        silently absent.

    Returns
    -------
    pl.DataFrame
        Long form, one row per (feature, stratum), sorted by ``ks_statistic``
        descending with nulls last: ``feature``, ``stratum``, ``n_cells``,
        ``ks_statistic``, ``p_value``, ``median_shift_iqr``.

    Notes
    -----
    ``ks_statistic`` is the two-sample Kolmogorov–Smirnov statistic on the finite
    values: the largest gap between the two empirical CDFs, distribution-free, in
    [0, 1]. Distribution-free cuts both ways — it responds to *any* difference,
    including a pure location shift that per-dataset quantile normalization would
    remove entirely. That is why ``median_shift_iqr`` sits beside it:
    (stratum median − reference median) / reference IQR, a robust location shift in
    units that mean the same thing across features (null when the reference IQR is
    0). Read the pair together. Large KS *and* large ``|median_shift_iqr|`` is
    "shifted but possibly salvageable by rank- or quantile-normalization"; large KS
    with a small median shift is a shape-level difference — different spread,
    different tails, different modes — that no per-stratum recentering will fix, and
    the stronger argument for restricting the feature.
    """
    features = np.asarray(features, dtype=float)
    feature_names = _feature_names(names, features.shape[1])
    strata = np.asarray(strata)
    if strata.shape != (features.shape[0],):
        raise ValueError(
            f"strata must have one label per cell: features has "
            f"{features.shape[0]} rows but strata has shape {strata.shape}"
        )
    labels = np.unique(strata).tolist()
    if reference is not None and reference not in labels:
        raise ValueError(f"reference {reference!r} is not one of the strata: {labels}")
    rows = []
    for index, name in enumerate(feature_names):
        column = features[:, index]
        finite = np.isfinite(column)
        for label in labels:
            in_stratum = strata == label
            values = column[in_stratum & finite]
            row = {
                "feature": name,
                "stratum": label,
                "n_cells": int(values.size),
                "ks_statistic": None,
                "p_value": None,
                "median_shift_iqr": None,
            }
            if reference is not None and label == reference:
                rows.append(row)
                continue
            if reference is None:
                reference_values = column[~in_stratum & finite]
            else:
                reference_values = column[(strata == reference) & finite]
            if values.size >= min_cells and reference_values.size >= min_cells:
                result = ks_2samp(values, reference_values)
                row["ks_statistic"] = float(result.statistic)
                row["p_value"] = float(result.pvalue)
                q75, q25 = np.percentile(reference_values, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    row["median_shift_iqr"] = float(
                        (np.median(values) - np.median(reference_values)) / iqr
                    )
            rows.append(row)
    frame = pl.DataFrame(
        rows,
        schema_overrides={
            "n_cells": pl.Int64,
            "ks_statistic": pl.Float64,
            "p_value": pl.Float64,
            "median_shift_iqr": pl.Float64,
        },
    )
    return frame.sort("ks_statistic", descending=True, nulls_last=True)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — scipy.stats.rankdata without the scipy.stats import."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.shape[0], dtype=float)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks
