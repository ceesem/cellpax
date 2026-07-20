"""CHOIR-style statistically-validated cluster resolution.

An alternative to cutting a dendrogram at a single distance threshold — which can
never merge some branches while splitting others at the same level. Following
CHOIR (Sant et al., *Nature Genetics* 2025; www.choirclustering.com,
github.com/corceslab/CHOIR), each split in a hierarchical tree is kept only if
its two child clusters are distinguishable by a random-forest classifier beyond a
permutation null. Branches whose split is not significant are merged; branches
with real substructure keep splitting. No global threshold to tune.

The per-node decision is ported from CHOIR's ``ComparisonUtils`` logic:

- Balanced-accuracy of RFs trained to tell the two clusters apart, over repeated
  balanced two-fold splits (real labels vs. label-permuted null).
- **Split** iff all hold: ``mean_accuracy >= min_accuracy`` (0.5 floor); the
  observed mean accuracy is in the upper ``alpha`` tail of the permuted accuracies;
  and (when ``use_variance``) the observed accuracy *variance* is in the lower
  ``alpha`` tail of the bootstrapped permuted variance — i.e. the separation is
  not just high but stably high. The variance condition is what prevents
  over-splitting a single population from a data-derived cut.
- Fixed ``alpha`` per comparison (CHOIR applies no explicit multiple-comparison
  correction).

This prunes an existing hierarchy (e.g. a fauxnograph ``SimilarityMatrix``) over a
feature matrix; it is a faithful adaptation of CHOIR's test, not a port of the
full R package (which also builds its own tree and re-selects features per node).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import to_tree


def _distinguishable(
    xa: np.ndarray,
    xb: np.ndarray,
    *,
    alpha: float,
    n_iterations: int,
    n_estimators: int,
    sample_max: int,
    min_accuracy: float,
    use_variance: bool,
    rng: np.random.Generator,
    n_jobs: int,
) -> bool:
    """CHOIR's merge/split test: are two clusters RF-distinguishable beyond null?"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    iterations = max(1, n_iterations // 2)
    real: list[float] = []
    null: list[float] = []
    for _ in range(iterations):
        m = min(len(xa), len(xb), sample_max)
        if m < 2:
            return False
        a = xa[rng.choice(len(xa), size=m, replace=False)]
        b = xb[rng.choice(len(xb), size=m, replace=False)]
        x = np.vstack([a, b])
        y = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
        y_perm = rng.permutation(y)
        seed = int(rng.integers(0, 2**31))
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for labels, sink in ((y, real), (y_perm, null)):
            for train, test in splitter.split(x, labels):
                clf = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
                )
                clf.fit(x[train], labels[train])
                sink.append(balanced_accuracy_score(labels[test], clf.predict(x[test])))

    real_acc = np.array(real)
    null_acc = np.array(null)
    mean_acc = float(real_acc.mean())
    if mean_acc < min_accuracy:
        return False
    if (null_acc >= mean_acc).mean() >= alpha:  # not in the upper alpha tail
        return False
    if use_variance:
        var_acc = float(real_acc.var())
        boot = np.array(
            [
                np.var(rng.choice(null_acc, size=len(real_acc), replace=True))
                for _ in range(1000)
            ]
        )
        if (boot <= var_acc).mean() >= alpha:  # variance not stably low
            return False
    return True


def choir_labels(
    linkage: np.ndarray,
    features: np.ndarray,
    *,
    alpha: float = 0.05,
    min_cluster_size: int = 20,
    min_accuracy: float = 0.5,
    n_iterations: int = 100,
    n_estimators: int = 100,
    sample_max: int = 1000,
    use_variance: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Resolve a hierarchy into clusters by keeping only significant splits.

    Walks the dendrogram top-down: at each node whose children are both at least
    ``min_cluster_size``, the split is kept (recurse) iff the children pass
    CHOIR's distinguishability test; otherwise the node collapses to one cluster.
    Returns a 0-based label per row of ``features`` (rows must align with the
    linkage's observations).
    """
    root = to_tree(linkage)
    rng = np.random.default_rng(seed)

    def resolve(node) -> list[list[int]]:
        if node.is_leaf():
            return [[node.id]]
        left, right = node.left, node.right
        if left.count < min_cluster_size or right.count < min_cluster_size:
            return [node.pre_order()]
        a_idx = np.array(left.pre_order())
        b_idx = np.array(right.pre_order())
        if _distinguishable(
            features[a_idx],
            features[b_idx],
            alpha=alpha,
            n_iterations=n_iterations,
            n_estimators=n_estimators,
            sample_max=sample_max,
            min_accuracy=min_accuracy,
            use_variance=use_variance,
            rng=rng,
            n_jobs=n_jobs,
        ):
            return resolve(left) + resolve(right)
        return [node.pre_order()]

    labels = np.full(features.shape[0], -1, dtype=np.int64)
    for cluster_id, members in enumerate(resolve(root)):
        labels[np.array(members)] = cluster_id
    return labels
