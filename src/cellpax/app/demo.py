"""A synthetic FeatureTable, so the app and its tests can drive the real loop.

Not a mock: this builds an actual ``FeatureTable`` and every code path the app
takes against real data runs against it unchanged. The structure is deliberately
hierarchical — three broad families that each split into two — so the descent has
something to descend *into* and the threshold scan has more than one plateau to
find.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cellpax.clustering import clipped_scaler_factory
from cellpax.featuretable import FeatureTable

#: Broad families, and how many subtypes each splits into.
_FAMILIES = {"alpha": 2, "beta": 2, "gamma": 2}


def demo_table(
    n_per_subtype: int = 60,
    *,
    n_features: int = 24,
    family_sep: float = 9.0,
    subtype_sep: float = 2.6,
    seed: int = 0,
) -> FeatureTable:
    """A two-level synthetic cell population with soma depth and a root id."""
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    family_names: list[str] = []
    subtype_names: list[str] = []

    for family_index, (family, n_subtypes) in enumerate(_FAMILIES.items()):
        family_shift = np.zeros(n_features)
        family_shift[family_index * 3 : family_index * 3 + 3] = family_sep
        for subtype_index in range(n_subtypes):
            subtype_shift = np.zeros(n_features)
            offset = 9 + family_index * 2 + subtype_index
            subtype_shift[offset % n_features] = subtype_sep * (subtype_index + 1)
            coords = (
                rng.normal(0.0, 1.0, (n_per_subtype, n_features))
                + family_shift
                + subtype_shift
            )
            blocks.append(coords)
            family_names.extend([family] * n_per_subtype)
            subtype_names.extend([f"{family}{subtype_index + 1}"] * n_per_subtype)

    coords = np.vstack(blocks)
    n = coords.shape[0]
    # A depth gradient correlated with family, so order_by has something to order
    # and the ids of two different cuts stay comparable.
    depth = np.array(
        [100.0 * (list(_FAMILIES).index(f) + 1) for f in family_names]
    ) + rng.normal(0.0, 25.0, n)

    frame = pl.DataFrame(
        {
            "cell_id": pl.Series(range(1, n + 1), dtype=pl.Int64),
            "root_id": pl.Series(
                [864691000000000000 + i for i in range(n)], dtype=pl.Int64
            ),
            **{f"m{i}": coords[:, i] for i in range(n_features)},
            "soma_depth_um": depth,
            "true_family": family_names,
            "true_subtype": subtype_names,
            "axon_frac_inside": rng.uniform(0.5, 1.0, n),
        }
    )
    table = FeatureTable(
        frame,
        features=[f"m{i}" for i in range(n_features)],
        # matches the real pipeline: a sigma-clipped robust scaler rather than
        # the plain StandardScaler default. The demo exists to exercise the same
        # code paths as production, and the scaler is one of them — a percentile
        # rule's breakpoint drops below one cell in the small cohorts a deep
        # descent produces, while a sigma bound is identical at n=300 and n=21000.
        scaler_factory=clipped_scaler_factory(mode="sigma", n_sigma=4.0),
        seed=seed,
    )
    table.define_features("analysis", columns=[f"m{i}" for i in range(n_features)])
    table.add_mask("all_cells", pl.col("cell_id") > 0)
    return table
