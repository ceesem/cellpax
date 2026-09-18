"""The live analysis state: one FeatureTable, a tree of levels, a job queue.

Everything expensive in this application funnels through one fact. Building a
consensus linkage materialises a dense condensed distance vector of
``n(n-1)/2`` float64 — 4.8 GB at 34.6k cells — and scipy's ``nn_chain`` works on
its own copy, so the transient peak is about twice that. It cannot be avoided by
sparsity, because a missing entry in the consensus matrix means a pair that never
co-clustered, i.e. the *largest* distance rather than zero.

Two consequences run through this module:

* **Jobs are serialised.** One worker thread, one expensive operation at a time.
  Two concurrent linkage builds would not be slow, they would be fatal.
* **Cutting is free, so it is inline.** Once the linkage is cached, a threshold
  change is an ``fcluster`` call and ``soft_labels`` is a column sum. Those run
  in the request and return immediately, which is what makes scrubbing a
  threshold feel like scrubbing rather than like submitting.

The methods below are split along exactly that line: :meth:`Session.submit`
covers the four operations that build something (space+cluster, restrict, embed,
merge support), and everything else answers from cached state.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from cellpax.app import profiles
from cellpax.app.config import AppConfig
from cellpax.app.ledger import Ledger
from cellpax.app.levels import LevelTree
from cellpax.clustering import (
    _coverage_verdict,
    _plateau_frame,
    consensus_density,
)

#: Operations that build heavy state and therefore run on the worker.
JOB_KINDS = (
    "cluster",
    "restrict",
    "embed",
    "merge_support",
    "boundary",
    "parametrize",
    "replay",
)


@dataclass
class Job:
    """A unit of expensive work, and its progress as the client sees it."""

    id: int
    kind: str
    level: str
    label: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    message: str = ""
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started is not None:
            elapsed = (self.finished or time.time()) - self.started
        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "elapsed": elapsed,
            "error": self.error,
        }


class JobRunner:
    """A single worker thread. Deliberately not a pool.

    Concurrency here would mean two linkage builds at once, and the peak
    allocation of one is already the binding constraint on the machine.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Job, Callable[[Callable[[str], None]], Any]]] = (
            queue.Queue()
        )
        self._jobs: dict[int, Job] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cellpax-jobs"
        )
        self._thread.start()

    def submit(
        self,
        kind: str,
        level: str,
        label: str,
        work: Callable[[Callable[[str], None]], Any],
    ) -> Job:
        with self._lock:
            job = Job(id=self._next_id, kind=kind, level=level, label=label)
            self._next_id += 1
            self._jobs[job.id] = job
        self._queue.put((job, work))
        return job

    def _loop(self) -> None:
        while True:
            job, work = self._queue.get()
            with self._lock:
                job.status = "running"
                job.started = time.time()

            def progress(message: str, _job: Job = job) -> None:
                with self._lock:
                    _job.message = message

            try:
                result = work(progress)
            except Exception as exc:  # a failed job must not take the server with it
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.message = traceback.format_exc(limit=6)
                    job.finished = time.time()
            else:
                with self._lock:
                    job.status = "done"
                    job.result = result
                    job.message = ""
                    job.finished = time.time()
            finally:
                self._queue.task_done()

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)[:limit]
            return [job.to_dict() for job in jobs]

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(j.status in ("queued", "running") for j in self._jobs.values())

    def wait(self, timeout: float = 120.0) -> None:
        """Block until the queue drains. For tests and the headless replay path."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.busy:
                return
            time.sleep(0.02)
        raise TimeoutError("jobs did not finish in time")


class Session:
    """One open analysis: the table, the descent, the ledger, the worker."""

    def __init__(
        self,
        table: Any,
        config: AppConfig,
        *,
        ledger: Ledger | None = None,
    ) -> None:
        self.ft = table
        self.config = config
        self.ledger = (
            ledger if ledger is not None else Ledger(config.resolved_ledger_path())
        )
        self.tree = LevelTree()
        self.jobs = JobRunner()
        self._scan_cache: dict[tuple, pl.DataFrame] = {}
        self._support_cache: dict[str, pl.DataFrame] = {}
        self._boundary_cache: dict[str, pl.DataFrame] = {}
        self._label_cache: dict[tuple, Any] = {}

    # -- levels -----------------------------------------------------------------

    def open_level(
        self,
        name: str,
        *,
        mask: str | None,
        parent: str | None = None,
        columns: str | None = None,
        order_by: str | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        """Start a level on a mask. The root call has ``parent=None``."""
        if mask is not None and mask not in self.ft.masks:
            raise ValueError(
                f"mask {mask!r} is not defined; masks are {sorted(self.ft.masks)}"
            )
        level = self.tree.add(
            name,
            mask=mask,
            parent=parent,
            columns=columns if columns is not None else self.config.columns,
            order_by=order_by if order_by is not None else self.config.order_by,
        )
        if record:
            self.ledger.append(
                "open_level",
                name,
                mask=mask,
                parent=parent,
                columns=level.columns,
                order_by=level.order_by,
            )
        return level.to_dict()

    def level_cell_count(self, mask: str | None) -> int:
        return int(self.ft.mask_series(mask).sum())

    # -- jobs -------------------------------------------------------------------

    def submit_cluster(
        self,
        level_name: str,
        *,
        n_neighbors: list[int],
        graph_type: list[str],
        resolution: list[float],
        n_times: int,
        alpha: float = 0.25,
        use_block_weights: bool = True,
        confirm_large: bool = False,
    ) -> Job:
        """Fit the space and run the consensus sweep. The big one."""
        level = self.tree[level_name]
        n_cells = self.level_cell_count(level.mask)
        if n_cells > self.config.linkage_warn_cells and not confirm_large:
            raise LargeLinkage(n_cells, self.config.linkage_warn_cells)

        params = {
            "mask": level.mask,
            "n_neighbors": list(n_neighbors),
            "graph_type": list(graph_type),
            "resolution": [float(r) for r in resolution],
            "n_times": int(n_times),
            "alpha": float(alpha),
            "columns": level.columns,
            "order_by": level.order_by,
            "name": f"{level_name}_consensus",
            "use_block_weights": bool(use_block_weights),
        }

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            from cellpax.diagnostics import block_weights

            weights = None
            if use_block_weights:
                progress("computing block weights (MFA)")
                scaled = self.ft.features(
                    level.mask, scaled=True, columns=level.columns
                )
                names = (
                    list(self.ft.collections[level.columns].columns)
                    if level.columns
                    else None
                )
                if names is not None:
                    weights = block_weights(scaled, names)

            progress("fitting the space")
            space = self.ft.space(
                level.mask,
                columns=level.columns,
                alpha=alpha,
                feature_weights=weights,
            )

            total = len(graph_type) * len(n_neighbors)
            progress(
                f"consensus sweep: {total} graph builds, "
                f"{len(resolution) * n_times} Leiden runs each"
            )
            clustering = self.ft.cluster(
                level.mask,
                space=space,
                n_neighbors=list(n_neighbors),
                graph_type=list(graph_type),
                resolution=list(resolution),
                n_times=int(n_times),
                order_by=level.order_by,
                name=params["name"],
            )
            progress("building the linkage")
            _ = clustering.linkage  # pay the 4.8 GB here, inside the job

            level.space = space
            level.clustering = clustering
            level.restricted = None
            level.cluster_params = params
            self.tree.touch(level_name)
            self._invalidate(level_name)
            return {"n_runs": int(clustering.partitions.n_runs)}

        self.ledger.append("cluster", level_name, **params)
        return self.jobs.submit(
            "cluster", level_name, f"cluster {level_name} ({n_cells:,} cells)", work
        )

    def submit_restrict(
        self,
        level_name: str,
        *,
        n_clusters_min: int | None = None,
        n_clusters_max: int | None = None,
        resolution_min: float | None = None,
        resolution_max: float | None = None,
        graph_type: list[str] | None = None,
        n_neighbors: list[int] | None = None,
        confirm_large: bool = False,
    ) -> Job:
        """Narrow the pooled runs. Rebuilds a linkage, hence a job."""
        level = self.tree[level_name]
        if level.clustering is None:
            raise ValueError(f"level {level_name!r} has no clustering to restrict")
        window = {
            "n_clusters_min": n_clusters_min,
            "n_clusters_max": n_clusters_max,
            "resolution_min": resolution_min,
            "resolution_max": resolution_max,
            "graph_type": graph_type,
            "n_neighbors": n_neighbors,
        }
        kwargs = {k: v for k, v in window.items() if v is not None}

        # Price the restricted consensus before building it. Filtering the runs
        # is column selection on the partition array — cheap — so the density of
        # the matrix it implies can be predicted exactly here, and this is the
        # operation that most needs predicting: narrowing to coarse grain picks
        # the runs with the biggest clusters, and every cluster adds a size**2
        # block, so the restricted matrix is routinely far denser than the pool.
        if level.clustering.partitions is not None:
            subset = level.clustering.partitions.filter(**kwargs)
            projection = consensus_density(subset.labels)
            if (
                projection["peak_gb"] > self.config.consensus_warn_gb
                and not confirm_large
            ):
                raise DenseConsensus(projection, self.config.consensus_warn_gb)

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            progress("re-consensing over the narrowed runs")
            restricted = level.clustering.restrict(**kwargs)
            progress("building the linkage")
            _ = restricted.linkage
            level.restricted = restricted
            level.restrict_params = kwargs
            self.tree.touch(level_name)
            self._invalidate(level_name)
            return {"n_runs": int(restricted.partitions.n_runs)}

        self.ledger.append("restrict", level_name, mask=level.mask, **window)
        return self.jobs.submit("restrict", level_name, f"restrict {level_name}", work)

    def submit_embed(self, level_name: str, *, method: str = "umap") -> Job:
        level = self.tree[level_name]

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            progress(f"embedding with {method}")
            name = f"{level_name}_{method}"
            self.ft.embed(
                level.mask,
                method=method,
                name=name,
                space=level.space,
                columns=level.columns,
            )
            return {"embedding": name}

        return self.jobs.submit(
            "embed", level_name, f"embed {level_name} ({method})", work
        )

    def submit_merge_support(self, level_name: str, *, n_bands: int = 4) -> Job:
        """Per-band merge support. Not free: it walks the tree and every run."""
        level = self.tree[level_name]
        active = self._require_active(level_name)

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            progress("walking merges and scoring per-band support")
            frame = active.merge_support(n_bands=n_bands)
            self._support_cache[level_name] = frame
            return {"n_merges": int(frame.height)}

        return self.jobs.submit(
            "merge_support", level_name, f"merge support {level_name}", work
        )

    # -- cheap reads ------------------------------------------------------------

    def _require_active(self, level_name: str) -> Any:
        level = self.tree[level_name]
        active = level.active
        if active is None:
            raise ValueError(
                f"level {level_name!r} has no clustering yet — run the sweep first"
            )
        self.tree.touch(level_name)
        return active

    def _invalidate(self, level_name: str) -> None:
        self._scan_cache = {
            k: v for k, v in self._scan_cache.items() if k[0] != level_name
        }
        self._support_cache.pop(level_name, None)
        self._boundary_cache.pop(level_name, None)
        self._label_cache = {}

    def coverage(self, level_name: str) -> dict[str, Any]:
        """Does the grain window actually contain runs, and from where?

        The check to run before trusting a restriction. Seeds at one setting
        reproduce each other, so realised cluster counts arrive in knots; a
        window can easily fall between two of them, or catch only one graph
        type — in which case the pool is no longer marginalising over graph
        construction, whatever the ensemble was swept over.
        """
        level = self.tree[level_name]
        if level.clustering is None or level.clustering.partitions is None:
            raise ValueError(f"level {level_name!r} has no runs to summarise")
        summary = level.clustering.partitions.summary()
        window = level.restrict_params
        lo = window.get("n_clusters_min")
        hi = window.get("n_clusters_max")
        inside = summary
        if lo is not None:
            inside = inside.filter(pl.col("n_clusters") >= lo)
        if hi is not None:
            inside = inside.filter(pl.col("n_clusters") <= hi)
        by_axis = (
            inside.group_by("graph_type", "n_neighbors")
            .len()
            .sort("graph_type", "n_neighbors")
        )
        counts = (
            summary.group_by("n_clusters")
            .len()
            .sort("n_clusters")
            .rename({"len": "n_runs"})
        )
        n_axes_total = summary.select(
            pl.struct("graph_type", "n_neighbors").n_unique()
        ).item()
        n_axes_inside = (
            inside.select(pl.struct("graph_type", "n_neighbors").n_unique()).item()
            if inside.height
            else 0
        )
        return {
            "n_runs_total": int(summary.height),
            "n_runs_in_window": int(inside.height),
            "n_axes_total": int(n_axes_total),
            "n_axes_in_window": int(n_axes_inside),
            "realised_counts": _records(counts),
            "by_axis": _records(by_axis),
            "window": {k: v for k, v in window.items() if v is not None},
            # The honest verdict, since this is the check people skip.
            "verdict": _coverage_verdict(
                int(inside.height), int(n_axes_inside), int(n_axes_total)
            ),
        }

    def axis_stability(
        self, level_name: str, *, by: str = "graph_type"
    ) -> dict[str, Any]:
        """ARI of each ensemble slice against a reference cut of the active pool.

        Scored over *all* runs, deliberately, even when a restriction is active:
        the slice that disagrees may be exactly the one the restriction dropped,
        and that is the thing worth seeing.
        """
        from cellpax.clustering import axis_stability as _axis_stability

        level = self.tree[level_name]
        if level.clustering is None:
            raise ValueError(f"level {level_name!r} has no clustering")
        active = self._require_active(level_name)
        max_value = float(active.max_value)
        rows = []
        # A single probe threshold would be a hidden choice, so score at three
        # and report whether the ranking survives.
        for probe in (0.4, 0.5, 0.6):
            reference = active.cluster_labels(probe * max_value)
            frame = _axis_stability(level.clustering.partitions, reference, by=(by,))
            rows.append((probe, frame))
        orderings = [
            tuple(frame.sort("mean_ari", descending=True)[by].to_list())
            for _, frame in rows
        ]
        return {
            "by": by,
            "probes": [
                {"threshold": probe, "rows": _records(frame)} for probe, frame in rows
            ],
            "ranking_stable": len(set(orderings)) == 1,
            "ranking": list(orderings[0]),
        }

    def scan(
        self,
        level_name: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
        n_points: int = 40,
        min_cluster_size: int = 1,
    ) -> dict[str, Any]:
        """``threshold_scan`` over a range, plus the plateaus read off it.

        Cheap: each point is one ``fcluster``. This is the scrubbing surface.
        """
        active = self._require_active(level_name)
        top = float(active.linkage[:, 2].max())
        lo = 0.0 if lo is None else float(lo)
        hi = top if hi is None else float(hi)
        key = (level_name, lo, hi, n_points, min_cluster_size)
        if key not in self._scan_cache:
            grid = np.linspace(lo, hi, int(n_points))
            self._scan_cache[key] = active.threshold_scan(
                grid, min_cluster_size=int(min_cluster_size)
            )
        frame = self._scan_cache[key]
        return {
            "max_height": top,
            "max_value": float(active.max_value),
            "min_cluster_size": int(min_cluster_size),
            "rows": _records(frame),
            "plateaus": _records(_plateau_frame(frame)),
        }

    def merge_support(self, level_name: str) -> dict[str, Any]:
        """The cached per-band support table, plus the ceiling it implies."""
        frame = self._support_cache.get(level_name)
        if frame is None:
            return {"available": False, "rows": [], "ceiling": None}
        bands = [c for c in frame.columns if c.startswith("support_band_")]
        annotated = (
            frame.with_columns(
                pl.when(pl.max_horizontal(bands) <= 0.5)
                .then(pl.lit("unsupported"))
                .when(pl.min_horizontal(bands) > 0.5)
                .then(pl.lit("all-band"))
                .otherwise(pl.lit("fine-only"))
                .alias("verdict")
            )
            if bands
            else frame.with_columns(pl.lit("unknown").alias("verdict"))
        )
        unsupported = annotated.filter(pl.col("verdict") == "unsupported")
        ceiling = (
            float(unsupported.select(pl.col("height").min()).item())
            if unsupported.height
            else None
        )
        return {
            "available": True,
            "bands": bands,
            "ceiling": ceiling,
            "rows": _records(annotated.sort("height", descending=True).head(200)),
        }

    # -- what the clusters are made of ------------------------------------------

    def ordered_labels(
        self,
        level_name: str,
        threshold: float,
        min_cluster_size: int,
        *,
        label_name: str | None = None,
    ) -> Any:
        """The LabelSet a commit at this cut would produce. The only labelling.

        Everything the user sees must come from here, because ``label()`` and
        ``cluster_labels()`` do not agree. With ``order_by`` set, ``label()``
        renumbers clusters by that column's per-cluster aggregate — cluster 0 is
        the shallowest — while ``cluster_labels()`` returns raw dendrogram order.
        The two are a permutation of each other, so showing one and committing
        the other silently attaches every name to the wrong cluster and carves
        every child mask from the wrong cells.

        Cached per (threshold, size) so scrubbing does not rebuild it.
        """
        level = self.tree[level_name]
        active = self._require_active(level_name)
        key = (level_name, float(threshold), int(min_cluster_size), label_name)
        hit = None if label_name is not None else self._label_cache.get(key)
        if hit is None:
            hit = active.label(
                distance_threshold=float(threshold),
                min_cluster_size=int(min_cluster_size),
                name=label_name or f"{level_name}_preview",
            )
            # names and colours ride along, so every surface that shows a
            # cluster — the swatch list, the ridgeline, the boundary table —
            # calls it whatever you called it
            if level.cut.names:
                hit = hit.rename(dict(level.cut.names))
            if level.cut.colors:
                hit = hit.set_colors(dict(level.cut.colors))
            if label_name is None:
                self._label_cache = {key: hit}  # only the current cut is kept
        return hit

    def _current_codes(self, level_name: str) -> tuple[Any, np.ndarray]:
        """The active clustering and its per-row codes at the current cut."""
        level = self.tree[level_name]
        active = self._require_active(level_name)
        if not level.cut.is_set():
            raise ValueError("choose a threshold first — these views describe a cut")
        labels = self.ordered_labels(
            level_name,
            float(level.cut.distance_threshold),
            int(level.cut.min_cluster_size),
        )
        return active, labels.codes_for(np.asarray(active._cell_ids))

    def numeric_columns(self, level_name: str) -> list[dict[str, Any]]:
        level = self.tree[level_name]
        return profiles.numeric_columns(self.ft, level.mask)

    def column_profile(
        self, level_name: str, *, column: str, bins: int = 40
    ) -> dict[str, Any]:
        """Depth (or any metadata column) distributed per cluster."""
        level = self.tree[level_name]
        active, codes = self._current_codes(level_name)
        return profiles.column_profile(
            self.ft,
            level.mask,
            np.asarray(active._cell_ids),
            codes,
            column,
            bins=bins,
            names=level.cut.names,
        )

    def cluster_heatmap(self, level_name: str, *, top: int = 40) -> dict[str, Any]:
        level = self.tree[level_name]
        active, codes = self._current_codes(level_name)
        return profiles.cluster_heatmap(
            self.ft,
            level.mask,
            np.asarray(active._cell_ids),
            codes,
            columns=level.columns,
            top=top,
            names=level.cut.names,
        )

    def feature_scatter(
        self, level_name: str, *, feature: str, against: str, scaled: bool = False
    ) -> dict[str, Any]:
        level = self.tree[level_name]
        active, codes = self._current_codes(level_name)
        return profiles.feature_scatter(
            self.ft,
            level.mask,
            np.asarray(active._cell_ids),
            codes,
            feature=feature,
            against=against,
            scaled=scaled,
        )

    def feature_names(self, level_name: str) -> list[str]:
        level = self.tree[level_name]
        return profiles._feature_names(self.ft, level.columns)

    # -- gap or cut -------------------------------------------------------------

    def submit_boundary(self, level_name: str, *, n_neighbors: int = 15) -> Job:
        """Ask whether each cluster boundary is a gap or a cut. A job.

        Not free: it rebuilds the space the clustering was computed in, builds a
        kNN graph, runs a dip test per pair and estimates a density valley.

        This is the step the app was missing, and its absence begged the
        question: a tool that only offers "cut into k discrete clusters" will
        always answer that the data is discrete. The verdict here is what
        licenses a label — or sends you to :meth:`submit_parametrize` instead.
        """
        level = self.tree[level_name]
        active, _ = self._current_codes(level_name)

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            progress("dip test, connectivity, density valley per pair")
            # `labels=`, not `distance_threshold=`: passing a threshold makes
            # boundary_report re-cut with `cluster_labels`, whose ids are raw
            # dendrogram order. Its `cluster_a`/`cluster_b` would then name
            # different clusters than everything else on screen.
            frame = self.ft.boundary_report(
                active,
                labels=self.ordered_labels(
                    level_name,
                    float(level.cut.distance_threshold),
                    int(level.cut.min_cluster_size),
                ),
                min_cluster_size=int(level.cut.min_cluster_size),
                n_neighbors=int(n_neighbors),
                space=level.space,
            )
            self._boundary_cache[level_name] = frame
            return {"n_pairs": int(frame.height)}

        self.ledger.append(
            "boundary",
            level_name,
            mask=level.mask,
            n_neighbors=int(n_neighbors),
            distance_threshold=level.cut.distance_threshold,
            min_cluster_size=level.cut.min_cluster_size,
        )
        return self.jobs.submit(
            "boundary", level_name, f"gap-or-cut {level_name}", work
        )

    def boundary(self, level_name: str) -> dict[str, Any]:
        """The cached per-pair verdicts, worst-separated first."""
        frame = self._boundary_cache.get(level_name)
        if frame is None:
            return {"available": False, "rows": [], "counts": {}}
        counts = (
            frame.group_by("verdict").len().sort("len", descending=True)
            if "verdict" in frame.columns
            else frame.clear()
        )
        return {
            "available": True,
            "rows": _records(frame),
            "counts": {r["verdict"]: r["len"] for r in _records(counts)},
            "columns": list(frame.columns),
        }

    def submit_parametrize(
        self,
        level_name: str,
        *,
        clusters: list[str],
        orient_by: str | None = None,
        nuisance: list[str] | None = None,
        name: str | None = None,
    ) -> Job:
        """Fit a 1-D coordinate through clusters a boundary called continuous.

        The alternative to a label, not a decoration on one. Two warnings ride
        along and are surfaced rather than logged away: the **dimension gate**
        (a curve through genuinely higher-dimensional structure compresses it
        into an artifact) and the **nuisance tripwire** (truncation manufactures
        gradients, and a coordinate that tracks reconstruction quality is not
        biology acquiring a name).
        """
        level = self.tree[level_name]
        if level.labels is None and not level.cut.is_set():
            raise ValueError("parametrize needs a cut to select clusters from")
        gradient_name = name or f"{level_name}_axis"

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            import warnings as _warnings

            progress("estimating intrinsic dimension, fitting principal curve")
            labels = level.labels
            if labels is None:
                active = level.active
                labels = active.label(
                    distance_threshold=float(level.cut.distance_threshold),
                    min_cluster_size=int(level.cut.min_cluster_size),
                    name=f"{level_name}_tmp",
                )
                if level.cut.names:
                    labels = labels.rename(dict(level.cut.names))
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                gradient = self.ft.parametrize(
                    level.mask,
                    labels=labels,
                    clusters=list(clusters),
                    # the same block-weighted space the clusters were found in;
                    # without it the curve is fit in a plain PCA and its
                    # loadings describe a rotation nothing else used
                    space=level.space,
                    orient_by=orient_by,
                    nuisance=list(nuisance) if nuisance else None,
                    name=gradient_name,
                )
            level.gradient = gradient
            level.gradient_warnings = [str(w.message) for w in caught]
            return {
                "name": gradient_name,
                "n_cells": len(gradient),
                "warnings": level.gradient_warnings,
            }

        self.ledger.append(
            "parametrize",
            level_name,
            mask=level.mask,
            clusters=list(clusters),
            orient_by=orient_by,
            nuisance=list(nuisance) if nuisance else None,
            name=gradient_name,
            label_name=level.labels.name if level.labels is not None else None,
        )
        return self.jobs.submit(
            "parametrize", level_name, f"parametrize {level_name}", work
        )

    def gradient(self, level_name: str) -> dict[str, Any]:
        """The fitted coordinate, its loadings, and the warnings it carries."""
        level = self.tree[level_name]
        gradient = getattr(level, "gradient", None)
        if gradient is None:
            return {"available": False}
        loadings = gradient.loadings()
        return {
            "available": True,
            "name": gradient.name,
            "n_cells": len(gradient),
            "intrinsic_dimension": (
                None
                if gradient.intrinsic_dimension is None
                else float(gradient.intrinsic_dimension)
            ),
            "dimension_profile": list(getattr(gradient, "dimension_profile", [])),
            "dimension_gate": 2.5,
            "coordinate": [float(v) for v in gradient.coordinate],
            "cell_id": [int(v) for v in gradient.cell_ids],
            # cellpax already sorts these by |spearman_rho| descending, so the
            # head is the answer to "what varies along this axis"
            "loadings": _records(loadings.head(30)),
            "warnings": getattr(level, "gradient_warnings", []),
        }

    def commit_gradient(
        self,
        level_name: str,
        *,
        bins: int | None = None,
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Attach the coordinate, and optionally declared cuts of it.

        Attaching the coordinate is the honest default — it keeps the continuum
        a continuum. ``bins`` cuts it into named intervals *alongside* the
        coordinate, which stays attached: the labels stay honest because the
        thing they were cut from is still there and the cut points are declared
        numbers anyone can revise.
        """
        level = self.tree[level_name]
        gradient = getattr(level, "gradient", None)
        if gradient is None:
            raise ValueError("nothing to commit: no gradient fitted")
        self.ft.attach(gradient)
        result: dict[str, Any] = {"coordinate": gradient.name, "bins": None}
        if bins:
            binned = gradient.bin(
                int(bins), names=names, name=f"{gradient.name}_binned"
            )
            self.ft.attach(binned, overwrite=True)
            result["bins"] = binned.name
        self.ledger.append(
            "bin",
            level_name,
            mask=level.mask,
            gradient=gradient.name,
            bins=int(bins) if bins else None,
            names=list(names) if names else None,
        )
        return result

    def preview_cut(
        self, level_name: str, *, distance_threshold: float, min_cluster_size: int = 1
    ) -> dict[str, Any]:
        """What this threshold gives, without committing to it."""
        active = self._require_active(level_name)
        level = self.tree[level_name]
        level.cut.distance_threshold = float(distance_threshold)
        level.cut.min_cluster_size = int(min_cluster_size)
        # the ids shown here are the ids a commit will write — see ordered_labels
        labels = self.ordered_labels(
            level_name, float(distance_threshold), int(min_cluster_size)
        )
        codes = labels.codes_for(np.asarray(active._cell_ids))
        assigned = codes[codes >= 0]
        ids, counts = np.unique(assigned, return_counts=True)
        support = self.merge_support(level_name)
        ceiling = support.get("ceiling")
        return {
            "distance_threshold": float(distance_threshold),
            "min_cluster_size": int(min_cluster_size),
            "coclustering_frequency": float(active.max_value)
            - float(distance_threshold),
            "n_clusters": int(ids.size),
            "n_assigned": int(assigned.size),
            "n_unassigned": int(codes.size - assigned.size),
            "clusters": [
                {"id": int(i), "n_cells": int(c), "name": level.cut.names.get(int(i))}
                for i, c in zip(ids, counts)
            ],
            "largest_cluster": int(counts.max()) if counts.size else 0,
            "median_cluster_size": float(np.median(counts)) if counts.size else 0.0,
            # The one cross-check worth surfacing at the moment of cutting.
            "above_support_ceiling": (
                None if ceiling is None else bool(float(distance_threshold) > ceiling)
            ),
            "support_ceiling": ceiling,
        }

    def flow(
        self,
        level_name: str,
        *,
        heights: list[float] | None = None,
        n_levels: int = 5,
        min_cluster_size: int = 1,
    ) -> dict[str, Any]:
        """Cell flow between adjacent cuts of the ladder — the alluvial.

        Two things make this not simply a dendrogram redrawn. ``min_cluster_size``
        sends cells to ``-1`` at one level and not another, so an "unassigned"
        band exists and drains as the cut coarsens (only ever in that direction:
        coarsening merges, so a group that clears the size floor keeps clearing
        it). And flows are matched by membership, never by cluster id, because
        ``cluster_labels`` renumbers to a contiguous ``0..k-1`` after dropping —
        ids are not stable across levels even when the membership is.
        """
        active = self._require_active(level_name)
        top = float(active.linkage[:, 2].max())
        if heights is None:
            heights = list(np.linspace(0.1 * top, 0.9 * top, int(n_levels)))
        heights = sorted((float(h) for h in heights), reverse=True)  # coarse -> fine

        # ordered per height for the same reason as everywhere else: a cluster
        # coloured blue on the map must be the blue ribbon here
        cell_ids = np.asarray(active._cell_ids)
        codes = [
            active.label(
                distance_threshold=float(h),
                min_cluster_size=int(min_cluster_size),
                name="_flow",
            ).codes_for(cell_ids)
            for h in heights
        ]
        nodes: list[dict[str, Any]] = []
        for index, (height, column) in enumerate(zip(heights, codes)):
            ids, counts = np.unique(column, return_counts=True)
            for cluster_id, count in zip(ids, counts):
                nodes.append(
                    {
                        "level": index,
                        "height": height,
                        "cluster": int(cluster_id),
                        "n_cells": int(count),
                        "unassigned": bool(int(cluster_id) < 0),
                    }
                )
        edges: list[dict[str, Any]] = []
        for index in range(len(codes) - 1):
            left, right = codes[index], codes[index + 1]
            pairs, counts = np.unique(
                np.stack([left, right], axis=1), axis=0, return_counts=True
            )
            for (source, target), count in zip(pairs, counts):
                edges.append(
                    {
                        "level": index,
                        "source": int(source),
                        "target": int(target),
                        "value": int(count),
                    }
                )
        return {
            "heights": heights,
            "max_value": float(active.max_value),
            "min_cluster_size": int(min_cluster_size),
            "nodes": nodes,
            "edges": edges,
        }

    def scatter(
        self,
        level_name: str,
        *,
        embedding: str | None = None,
        distance_threshold: float | None = None,
        min_cluster_size: int = 1,
        color_by: str = "cluster",
    ) -> dict[str, Any]:
        """Embedding coordinates plus a per-cell colour code.

        ``color_by='stability'`` returns the per-cell consensus strength instead
        of the cut — the companion picture, since low scorers should concentrate
        on interdigitated boundaries. Scattered uniformly instead, the
        instability is not about boundaries and the cut is not the thing to fix.
        """
        level = self.tree[level_name]
        active = self._require_active(level_name)
        name = embedding or self._find_embedding(level)
        if name is None:
            return {
                "available": False,
                "reason": "no embedding computed for this level",
            }
        frame = self.ft.embedding(level.mask, name=name)
        coord_cols = [c for c in frame.columns if c.startswith(name)][:2]
        if len(coord_cols) < 2:
            return {"available": False, "reason": f"embedding {name!r} is not 2-D"}
        xs = frame[coord_cols[0]].to_numpy()
        ys = frame[coord_cols[1]].to_numpy()

        if color_by == "stability":
            values = active.cell_stability()
            return {
                "available": True,
                "embedding": name,
                "color_by": "stability",
                "x": [float(v) for v in xs],
                "y": [float(v) for v in ys],
                "value": [float(v) for v in values],
                "cell_id": [int(v) for v in active._cell_ids],
            }
        threshold = (
            level.cut.distance_threshold
            if distance_threshold is None
            else float(distance_threshold)
        )
        if threshold is None:
            codes = np.zeros(xs.shape[0], dtype=np.int64)
        else:
            codes = self.ordered_labels(
                level_name, float(threshold), int(min_cluster_size)
            ).codes_for(np.asarray(active._cell_ids))
        return {
            "available": True,
            "embedding": name,
            "color_by": "cluster",
            "x": [float(v) for v in xs],
            "y": [float(v) for v in ys],
            "cluster": [int(v) for v in codes],
            "cell_id": [int(v) for v in active._cell_ids],
        }

    def _find_embedding(self, level: Any) -> str | None:
        for mask, name in self.ft.embeddings:
            if mask == (level.mask or "__all__") and name.startswith(level.name):
                return name
        for mask, name in self.ft.embeddings:
            if mask == (level.mask or "__all__"):
                return name
        return None

    def soft_labels(self, level_name: str, *, limit: int = 500) -> dict[str, Any]:
        """Per-cell mean co-clustering with each cluster — the gray zone.

        An honest ensemble frequency, not a calibrated probability: rows need not
        sum to one, and a cell far from everything can still score high against
        its nearest cluster.
        """
        level = self.tree[level_name]
        active = self._require_active(level_name)
        if not level.cut.is_set():
            raise ValueError("choose a threshold before reading soft labels")
        frame = active.soft_labels(
            self.ordered_labels(
                level_name,
                float(level.cut.distance_threshold),
                int(level.cut.min_cluster_size),
            )
        )
        prob_cols = [c for c in frame.columns if c.startswith("p_")]
        if not prob_cols:
            return {"rows": [], "columns": []}
        values = frame.select(prob_cols).to_numpy()
        ordered = np.sort(values, axis=1)
        margin = (
            ordered[:, -1] - ordered[:, -2] if values.shape[1] > 1 else ordered[:, -1]
        )
        frame = frame.with_columns(pl.Series("margin", margin)).sort("margin")
        return {
            "columns": prob_cols,
            "rows": _records(frame.head(limit)),
            "n_cells": int(frame.height),
        }

    # -- committing -------------------------------------------------------------

    def set_names(
        self,
        level_name: str,
        *,
        names: dict[int, str] | None = None,
        colors: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        level = self.tree[level_name]
        if names:
            level.cut.names.update({int(k): str(v) for k, v in names.items()})
        if colors:
            level.cut.colors.update({int(k): str(v) for k, v in colors.items()})
        self._label_cache = {}
        self.ledger.append(
            "rename",
            level_name,
            names={str(k): v for k, v in level.cut.names.items()},
            colors={str(k): v for k, v in level.cut.colors.items()},
        )
        return level.to_dict()

    def commit(
        self,
        level_name: str,
        *,
        label_name: str | None = None,
        children: dict[str, list[str]] | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        """Attach the cut, carve the child masks, and open the next levels.

        This is the only method that mutates the ``FeatureTable``. It records the
        threshold *and* the restriction and size floor it was chosen under,
        because a threshold read without those is not reproducible.
        """
        level = self.tree[level_name]
        active = self._require_active(level_name)
        if not level.cut.is_set():
            raise ValueError("nothing to commit: no threshold chosen")
        name = label_name or f"{level_name}_label"

        # The cut is recorded here rather than on every scrub. Previewing a
        # threshold is a look, not a decision, and a ledger that logged each
        # slider position would bury the choices in the motion that led to them.
        if record:
            self.ledger.append(
                "cut",
                level_name,
                mask=level.mask,
                name=name,
                distance_threshold=float(level.cut.distance_threshold),
                min_cluster_size=int(level.cut.min_cluster_size),
                restricted=level.is_restricted,
                restrict_params=level.restrict_params,
                # The names as they stand at commit, not as they were typed.
                # `rename` entries are the audit trail of the naming; this is
                # the state that has to be reproduced, and it is what the
                # generated script renders.
                names={str(k): v for k, v in level.cut.names.items()},
                colors={str(k): v for k, v in level.cut.colors.items()},
            )

        # the same construction every view used, under its final name
        labels = self.ordered_labels(
            level_name,
            float(level.cut.distance_threshold),
            int(level.cut.min_cluster_size),
            label_name=name,
        )
        self.ft.attach(labels, overwrite=True)
        level.labels = labels
        level.committed = True

        # Recorded before the children are opened, so replaying the ledger
        # attaches the parent's labels before a child mask tries to select on
        # the column they create.
        if record:
            self.ledger.append(
                "commit",
                level_name,
                mask=level.mask,
                name=name,
                distance_threshold=level.cut.distance_threshold,
                min_cluster_size=level.cut.min_cluster_size,
                restricted=level.is_restricted,
                restrict_params=level.restrict_params,
                children={k: list(v) for k, v in (children or {}).items()},
            )

        opened: list[str] = []
        for child, members in (children or {}).items():
            self.ft.add_mask(
                child, pl.col(name).is_in(list(members)), based_on=level.mask
            )
            self.open_level(
                child,
                mask=child,
                parent=level_name,
                columns=level.columns,
                order_by=level.order_by,
                record=record,
            )
            opened.append(child)

        return {"label": name, "children": opened, "level": level.to_dict()}

    def note(self, level_name: str, text: str) -> dict[str, Any]:
        entry = self.ledger.append("note", level_name, text=text)
        return {"at": entry.at, "text": text}

    # -- rewinding --------------------------------------------------------------

    #: Entries that record work rather than judgement. A "reset decisions"
    #: keeps these, because re-running a sweep at 34.6k cells costs a linkage
    #: build and rewinding a cut does not invalidate the tree it was cut from.
    _WORK_KINDS = ("open_level", "cluster", "restrict")

    def rewind(self, keep: int, *, archive: bool = True) -> dict[str, Any]:
        """Drop the trailing entries, undoing what they wrote to the table."""
        return self._rewind_to(
            list(range(min(max(0, int(keep)), len(self.ledger)))), archive=archive
        )

    def reset(
        self, *, keep_sweeps: bool = True, archive: bool = True
    ) -> dict[str, Any]:
        """Clear the descent. Two scopes, because they cost very different amounts.

        ``keep_sweeps=True`` (the default) throws away every judgement — cuts,
        names, commits, boundaries, gradients — and keeps the consensus sweeps
        and grain windows that produced the trees they were judgements about.
        That is almost always what "let me start over" means in practice: the
        expensive thing is the linkage, and rewinding a cut does not invalidate
        it.

        ``keep_sweeps=False`` is the true clean slate, and it means re-running
        every sweep.

        Either way the ledger stays honest: what survives is exactly what still
        describes the table, so replaying it reproduces the state you are left
        in.
        """
        keep = (
            [i for i, e in enumerate(self.ledger) if e.kind in self._WORK_KINDS]
            if keep_sweeps
            else []
        )
        result = self._rewind_to(keep, archive=archive)
        root = self.config.root_mask
        if root is not None and root in self.ft.masks and root not in self.tree:
            self.open_level(root, mask=root)
            result["remaining"] = len(self.ledger)
        return result

    def _rewind_to(self, keep: list[int], *, archive: bool = True) -> dict[str, Any]:
        """Keep these entry positions; reverse the effects of the rest.

        Truncating the ledger on its own would be a lie: the entries being
        forgotten also attached label columns and carved masks, and a record
        that no longer describes the table is the exact failure the ledger
        exists to prevent. Reversing them is tractable only because the app's
        entire write surface is four calls — ``attach`` for labels, ``add_mask``
        for children, ``attach`` for a coordinate and its bins. Nothing else in
        the app touches the table.

        The old ledger is copied aside first. Losing a morning of decisions to a
        misclick is worse than an extra file.
        """
        wanted = set(keep)
        dropped = [e for i, e in enumerate(self.ledger) if i not in wanted]
        if not dropped:
            return {
                "dropped": 0,
                "undone": [],
                "archive": None,
                "remaining": len(self.ledger),
            }

        undone: list[str] = []
        for entry in reversed(dropped):
            undone.extend(self._undo(entry))

        # A level whose mask the undo just removed cannot survive, and neither
        # can its entries. Without this a scoped reset keeps the `open_level`
        # for a child carved by a commit it also dropped — an orphan whose mask
        # no longer exists, which would make the ledger unreplayable.
        entries = self.ledger.entries
        alive = {
            entry.level
            for index, entry in enumerate(entries)
            if index in wanted
            and entry.kind == "open_level"
            and (
                entry.payload.get("mask") is None
                or entry.payload["mask"] in self.ft.masks
            )
        }
        orphans = {entries[i].level for i in wanted} - alive
        if orphans:
            wanted = {i for i in wanted if entries[i].level in alive}
            undone.append(f"dropped orphaned level(s) {sorted(orphans)}")

        path = self._archive_ledger() if archive else None
        self.ledger.keep(sorted(wanted))
        self._rebuild_tree()
        self._scan_cache, self._support_cache = {}, {}
        self._boundary_cache, self._label_cache = {}, {}
        return {
            "dropped": len(dropped),
            "undone": undone,
            "archive": str(path) if path else None,
            "remaining": len(self.ledger),
        }

    def _undo(self, entry: Any) -> list[str]:
        """Reverse one entry's effect on the table. Missing targets are fine.

        Tolerant by design: a rewind after a partial failure, or over an entry
        whose column someone already dropped by hand, should still clean up
        everything else rather than stopping halfway.
        """
        done: list[str] = []
        payload = entry.payload
        if entry.kind == "commit":
            for child in payload.get("children") or {}:
                try:
                    self.ft.drop_mask(child)
                    done.append(f"dropped mask {child!r}")
                except (KeyError, ValueError):
                    pass
            name = payload.get("name")
            if name:
                try:
                    self.ft.detach(name)
                    done.append(f"detached {name!r}")
                except KeyError:
                    pass
        elif entry.kind == "bin":
            gradient = payload.get("gradient")
            for column in (f"{gradient}_binned", gradient):
                if not column:
                    continue
                try:
                    self.ft.detach(column)
                    done.append(f"detached {column!r}")
                except KeyError:
                    pass
        return done

    def _archive_ledger(self) -> Any:
        from pathlib import Path
        from shutil import copyfile

        path = self.ledger.path
        if path is None or not Path(path).exists():
            return None
        stamp = int(time.time())
        target = Path(path).with_suffix(f".{stamp}.bak.jsonl")
        copyfile(path, target)
        return target

    def _rebuild_tree(self) -> None:
        """Bring the level tree back in line with the surviving entries.

        A level survives if its ``open_level`` entry did; within a survivor,
        each piece of state survives only if the entry that created it did. The
        clustering is the exception and is kept regardless — see :meth:`rewind`.
        """
        from cellpax.app.levels import CutState

        surviving = {e.level for e in self.ledger if e.kind == "open_level"}
        for name in [level.name for level in self.tree]:
            if name not in surviving:
                self.tree.remove(name)
        for level in self.tree:
            kinds = {e.kind for e in self.ledger.for_level(level.name)}
            if "restrict" not in kinds:
                level.restricted = None
                level.restrict_params = {}
            if "cut" not in kinds:
                level.cut = CutState()
                level.labels = None
                level.committed = False
            if "parametrize" not in kinds:
                level.gradient = None
                level.gradient_warnings = []
            if "rename" not in kinds:
                level.cut.names, level.cut.colors = {}, {}
            level.children = [c for c in level.children if c in surviving]

    # -- state ------------------------------------------------------------------

    def representation(self) -> dict[str, Any]:
        """What the cells are actually being compared in, named plainly.

        Surfaced because it is invisible otherwise. The scaler is the table's,
        set once at construction and refit per mask; the space is per level and
        carries the block weights and whitening. Neither shows up anywhere on
        screen unless it is put there, and "which scaler is this using" is not a
        question anyone should have to read source to answer.
        """
        from cellpax.persist import _scaler_tag

        try:
            tag = _scaler_tag(self.ft._scaler_factory)
        except Exception:
            tag = "custom"
        if isinstance(tag, dict):
            mode = tag.get("mode", "percentile")
            detail = (
                f"±{tag.get('n_sigma')}σ"
                if mode == "sigma"
                else f"{tag.get('lower')}–{tag.get('upper')} pct"
            )
            scaler = f"clipped ({mode}, {detail})"
        else:
            scaler = str(tag)
        spaces = {
            level.name: level.space.label
            for level in self.tree
            if level.space is not None
        }
        return {
            "scaler": scaler,
            "scaler_detail": tag,
            "refit_per_mask": True,
            "spaces": spaces,
        }

    def state(self) -> dict[str, Any]:
        return {
            "levels": self.tree.to_list(),
            "jobs": self.jobs.snapshot(),
            "busy": self.jobs.busy,
            "masks": sorted(self.ft.masks),
            "labels": sorted(self.ft.labels),
            "embeddings": [list(e) for e in self.ft.embeddings],
            "collections": sorted(self.ft._collection_names),
            "n_cells": int(self.ft.n_cells),
            "representation": self.representation(),
            "ledger_size": len(self.ledger),
            "demo": self.config.is_demo,
            "linkage_warn_cells": self.config.linkage_warn_cells,
        }


class DenseConsensus(Exception):
    """Raised when a restriction would build a consensus matrix past the limit.

    The counterpart to :class:`LargeLinkage`, and the one people are not
    expecting: the intuition that restricting *narrows* the analysis is right
    about the runs and backwards about the memory. Fewer, coarser runs means
    bigger clusters, and ``A @ A.T`` gains a ``size**2`` block per cluster — so
    the matrix gets denser as the window gets coarser, and a pool that was
    comfortably sparse can come back near-solid.
    """

    def __init__(self, projection: dict[str, float], limit_gb: float) -> None:
        self.projection = projection
        self.limit_gb = limit_gb
        super().__init__(
            f"this window projects a {projection['density']:.0%}-dense consensus over "
            f"{int(projection['n_cells']):,} cells — about "
            f"{projection['gb']:.1f} GB for the matrix and {projection['peak_gb']:.1f} GB "
            f"at peak, against a {limit_gb:.0f} GB limit. Coarser windows are denser, "
            f"not sparser. Re-send with confirm=true to proceed."
        )


class LargeLinkage(Exception):
    """Raised when a sweep would build a linkage past the configured limit.

    Not a refusal — a confirmation. The condensed distance vector is
    ``n(n-1)/2`` float64 and scipy works on a copy of it, so the caller deserves
    to see the number before the allocation happens rather than after.
    """

    def __init__(self, n_cells: int, limit: int) -> None:
        self.n_cells = n_cells
        self.limit = limit
        self.gb = n_cells * (n_cells - 1) / 2 * 8 / 1e9
        super().__init__(
            f"{n_cells:,} cells needs a ~{self.gb:.1f} GB condensed distance matrix "
            f"(about {2 * self.gb:.1f} GB peak, since scipy works on a copy); "
            f"limit is {limit:,}. Re-send with confirm=true to proceed."
        )


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Frame to JSON-safe records, with numpy scalars unwrapped."""
    return [
        {k: (None if v is None else _plain(v)) for k, v in row.items()}
        for row in frame.iter_rows(named=True)
    ]


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, float) and (value != value):  # NaN is not JSON
        return None
    return value
