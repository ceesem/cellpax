"""The HTTP surface. Thin on purpose — the thinking lives in :mod:`session`.

Routes divide along the cost line that shapes the whole app. ``/jobs/*`` returns
a job id immediately and the client watches it; everything else answers from
cached state inside the request. A route that could take thirty seconds and one
that takes three milliseconds should not look the same to the client, because
the UI has to render them differently to stay honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cellpax.app.config import AppConfig
from cellpax.app.ledger import Ledger
from cellpax.app.ngl import PALETTE, UNASSIGNED, NeuroglancerLinks, palette_for
from cellpax.app.session import DenseConsensus, LargeLinkage, Session

STATIC = Path(__file__).parent / "static"


# Request bodies live at module scope, not inside ``create_app``. With
# ``from __future__ import annotations`` every annotation is a string, and
# FastAPI resolves them against the module globals — a closure-local model is
# invisible there and silently degrades to a query parameter.


class OpenLevel(BaseModel):
    name: str
    mask: str | None = None
    parent: str | None = None
    columns: str | None = None
    order_by: str | None = None


class ClusterRequest(BaseModel):
    n_neighbors: list[int] = [15, 30, 60]
    graph_type: list[str] = ["knn", "snn_jaccard", "umap_fuzzy"]
    resolution: list[float] | None = None
    n_times: int = 3
    alpha: float = 0.25
    use_block_weights: bool = True
    confirm: bool = False


class RestrictRequest(BaseModel):
    confirm: bool = False
    n_clusters_min: int | None = None
    n_clusters_max: int | None = None
    resolution_min: float | None = None
    resolution_max: float | None = None
    graph_type: list[str] | None = None
    n_neighbors: list[int] | None = None


class EmbedRequest(BaseModel):
    method: str = "umap"


class NamesRequest(BaseModel):
    names: dict[str, str] | None = None
    colors: dict[str, str] | None = None


class CommitRequest(BaseModel):
    label_name: str | None = None
    children: dict[str, list[str]] | None = None


class NoteRequest(BaseModel):
    text: str


class RewindRequest(BaseModel):
    # Explicit and required: this undoes attached labels and carved masks, so it
    # must not be reachable by a stray POST.
    confirm: bool = False
    #: How many leading entries to keep. Omit for a scoped reset instead.
    keep: int | None = None
    #: On a reset (keep=None): hold on to the sweeps and grain windows and drop
    #: only the judgements. False re-runs every linkage.
    keep_sweeps: bool = True
    archive: bool = True


class BoundaryRequest(BaseModel):
    n_neighbors: int = 15


class ParametrizeRequest(BaseModel):
    clusters: list[str]
    orient_by: str | None = None
    nuisance: list[str] | None = None
    name: str | None = None


class GradientCommitRequest(BaseModel):
    bins: int | None = None
    names: list[str] | None = None


def build_session(config: AppConfig) -> Session:
    """Load the table the app will drive, or the demo fixture."""
    if config.folio is None:
        from cellpax.app.demo import demo_table

        table = demo_table()
        config = config
        if config.columns is None:
            config.columns = "analysis"
        if config.order_by is None:
            config.order_by = "soma_depth_um"
        if config.root_mask is None:
            config.root_mask = "all_cells"
    else:
        import datafolio

        from cellpax.featuretable import FeatureTable

        folio = datafolio.DataFolio(str(config.folio))
        if config.table_name is None:
            raise ValueError("config sets 'folio' but not 'table_name'")
        table = FeatureTable.load(folio, config.table_name)

    session = Session(table, config, ledger=Ledger(config.resolved_ledger_path()))
    root = config.root_mask
    if root is not None and root in table.masks:
        session.open_level(root, mask=root)
    return session


def create_app(
    config: AppConfig | None = None, *, session: Session | None = None
) -> Any:
    """Build the FastAPI app. ``session`` is injected by the tests."""
    config = config or AppConfig()
    state = session if session is not None else build_session(config)
    links = NeuroglancerLinks(state.config.neuroglancer)

    app = FastAPI(title="cellpax", docs_url="/api/docs")
    app.state.session = state

    _build_cache: dict[str, Any] = {}

    def _client_build() -> str:
        """A short digest of the client files, so staleness is visible.

        This app is edited while it is running and a browser tab outlives many
        of those edits. A tab silently executing a build from an hour ago looks
        exactly like a bug in the current code — buttons that never enable,
        jobs that never register — and costs far more to diagnose than it does
        to prevent.

        Rehashed only when an mtime moves, because this rides on the job poll
        and would otherwise re-read the client on a timer forever.
        """
        import hashlib

        names = ("app.js", "app.css", "index.html")
        stamp = tuple(
            (STATIC / n).stat().st_mtime_ns if (STATIC / n).exists() else 0
            for n in names
        )
        if _build_cache.get("stamp") != stamp:
            digest = hashlib.blake2b(digest_size=4)
            for name in names:
                path = STATIC / name
                if path.exists():
                    digest.update(path.read_bytes())
            _build_cache["stamp"] = stamp
            _build_cache["value"] = digest.hexdigest()
        return str(_build_cache["value"])

    @app.middleware("http")
    async def _no_store(request: Any, call_next: Any) -> Any:
        # Never cache the client. These are localhost files; revalidation costs
        # nothing, and a stale tab costs an afternoon.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    def _fail(exc: Exception) -> None:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- state ------------------------------------------------------------------

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        payload = state.state()
        payload["neuroglancer"] = {"available": links.available}
        # one palette, served once, so swatches / scatter / neuroglancer agree
        payload["palette"] = {"colors": list(PALETTE), "unassigned": UNASSIGNED}
        payload["build"] = _client_build()
        payload["sweep_defaults"] = {
            **state.config.sweep,
            "resolution": state.config.resolution_grid(),
        }
        return payload

    @app.get("/api/jobs")
    def get_jobs() -> dict[str, Any]:
        # `build` rides along here rather than only on /api/state: this is the
        # one endpoint polled on a timer, so it is the only place a tab finds
        # out the client changed under it without being told to look.
        return {
            "jobs": state.jobs.snapshot(),
            "busy": state.jobs.busy,
            "build": _client_build(),
        }

    # -- levels -----------------------------------------------------------------

    @app.post("/api/levels")
    def open_level(body: OpenLevel) -> dict[str, Any]:
        try:
            return state.open_level(
                body.name,
                mask=body.mask if body.mask is not None else body.name,
                parent=body.parent,
                columns=body.columns,
                order_by=body.order_by,
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.post("/api/levels/{name}/cluster")
    def run_cluster(name: str, body: ClusterRequest) -> dict[str, Any]:
        resolution = body.resolution or state.config.resolution_grid()
        try:
            job = state.submit_cluster(
                name,
                n_neighbors=body.n_neighbors,
                graph_type=body.graph_type,
                resolution=resolution,
                n_times=body.n_times,
                alpha=body.alpha,
                use_block_weights=body.use_block_weights,
                confirm_large=body.confirm,
            )
        except LargeLinkage as exc:
            # 409, not 400: the request is well-formed and will be honoured on
            # re-send. The client shows the number and asks.
            return JSONResponse(
                status_code=409,
                content={
                    "needs_confirmation": True,
                    "n_cells": exc.n_cells,
                    "gb": round(exc.gb, 1),
                    "peak_gb": round(2 * exc.gb, 1),
                    "detail": str(exc),
                },
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    @app.post("/api/levels/{name}/restrict")
    def run_restrict(name: str, body: RestrictRequest) -> dict[str, Any]:
        window = body.model_dump()
        confirm = window.pop("confirm", False)
        try:
            job = state.submit_restrict(name, confirm_large=confirm, **window)
        except DenseConsensus as exc:
            # Same 409-and-ask shape as an oversized linkage. Worth showing
            # rather than just running: the intuition that restricting narrows
            # the problem is exactly backwards for this allocation.
            return JSONResponse(
                status_code=409,
                content={
                    "needs_confirmation": True,
                    "kind": "dense_consensus",
                    "density": round(exc.projection["density"], 4),
                    "gb": round(exc.projection["gb"], 1),
                    "peak_gb": round(exc.projection["peak_gb"], 1),
                    "n_cells": int(exc.projection["n_cells"]),
                    "detail": str(exc),
                },
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    @app.post("/api/levels/{name}/embed")
    def run_embed(name: str, body: EmbedRequest) -> dict[str, Any]:
        try:
            job = state.submit_embed(name, method=body.method)
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    @app.post("/api/levels/{name}/merge-support")
    def run_merge_support(name: str, n_bands: int = 4) -> dict[str, Any]:
        try:
            job = state.submit_merge_support(name, n_bands=n_bands)
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    # -- cheap reads ------------------------------------------------------------

    @app.get("/api/levels/{name}/coverage")
    def get_coverage(name: str) -> dict[str, Any]:
        try:
            return state.coverage(name)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/axis-stability")
    def get_axis_stability(name: str, by: str = "graph_type") -> dict[str, Any]:
        try:
            return state.axis_stability(name, by=by)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/scan")
    def get_scan(
        name: str,
        lo: float | None = None,
        hi: float | None = None,
        n_points: int = 40,
        min_cluster_size: int = 1,
    ) -> dict[str, Any]:
        try:
            return state.scan(
                name, lo=lo, hi=hi, n_points=n_points, min_cluster_size=min_cluster_size
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/merge-support")
    def get_merge_support(name: str) -> dict[str, Any]:
        try:
            return state.merge_support(name)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/preview")
    def get_preview(
        name: str, distance_threshold: float, min_cluster_size: int = 1
    ) -> dict[str, Any]:
        try:
            return state.preview_cut(
                name,
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/flow")
    def get_flow(
        name: str, n_levels: int = 5, min_cluster_size: int = 1
    ) -> dict[str, Any]:
        try:
            return state.flow(
                name, n_levels=n_levels, min_cluster_size=min_cluster_size
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/scatter")
    def get_scatter(
        name: str,
        embedding: str | None = None,
        distance_threshold: float | None = None,
        min_cluster_size: int = 1,
        color_by: str = "cluster",
    ) -> dict[str, Any]:
        try:
            return state.scatter(
                name,
                embedding=embedding,
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                color_by=color_by,
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/soft-labels")
    def get_soft_labels(name: str, limit: int = 300) -> dict[str, Any]:
        try:
            return state.soft_labels(name, limit=limit)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    # -- what the clusters are made of ------------------------------------------

    @app.get("/api/levels/{name}/columns")
    def get_columns(name: str) -> dict[str, Any]:
        try:
            return {
                "numeric": state.numeric_columns(name),
                "features": state.feature_names(name),
            }
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/profile")
    def get_profile(name: str, column: str, bins: int = 40) -> dict[str, Any]:
        try:
            return state.column_profile(name, column=column, bins=bins)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/heatmap")
    def get_heatmap(name: str, top: int = 40) -> dict[str, Any]:
        try:
            return state.cluster_heatmap(name, top=top)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/levels/{name}/feature-scatter")
    def get_feature_scatter(
        name: str, feature: str, against: str, scaled: bool = False
    ) -> dict[str, Any]:
        try:
            return state.feature_scatter(
                name, feature=feature, against=against, scaled=scaled
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    # -- gap or cut -------------------------------------------------------------

    @app.post("/api/levels/{name}/boundary")
    def run_boundary(name: str, body: BoundaryRequest) -> dict[str, Any]:
        try:
            job = state.submit_boundary(name, n_neighbors=body.n_neighbors)
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    @app.get("/api/levels/{name}/boundary")
    def get_boundary(name: str) -> dict[str, Any]:
        try:
            return state.boundary(name)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.post("/api/levels/{name}/parametrize")
    def run_parametrize(name: str, body: ParametrizeRequest) -> dict[str, Any]:
        try:
            job = state.submit_parametrize(
                name,
                clusters=body.clusters,
                orient_by=body.orient_by,
                nuisance=body.nuisance,
                name=body.name,
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)
        return {"job": job.to_dict()}

    @app.get("/api/levels/{name}/gradient")
    def get_gradient(name: str) -> dict[str, Any]:
        try:
            return state.gradient(name)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.post("/api/levels/{name}/gradient/commit")
    def commit_gradient(name: str, body: GradientCommitRequest) -> dict[str, Any]:
        try:
            return state.commit_gradient(name, bins=body.bins, names=body.names)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    # -- neuroglancer links -----------------------------------------------------

    def _cluster_cells(name: str, cluster: int) -> list[int]:
        level = state.tree[name]
        active = level.active
        if active is None or not level.cut.is_set():
            raise ValueError("choose a threshold before inspecting a cluster")
        import numpy as np

        # the same labelling the UI displayed and a commit would write
        cell_ids = np.asarray(active._cell_ids)
        codes = state.ordered_labels(
            name,
            float(level.cut.distance_threshold),
            int(level.cut.min_cluster_size),
        ).codes_for(cell_ids)
        return [int(c) for c in cell_ids[codes == cluster]]

    @app.get("/api/levels/{name}/clusters/{cluster}/neuroglancer")
    def get_ngl(name: str, cluster: int) -> dict[str, Any]:
        try:
            cells = _cluster_cells(name, cluster)
            level = state.tree[name]
            colors = palette_for([cluster], level.cut.colors)
            return links.link_for_cluster(
                state.ft,
                cells,
                color=colors.get(cluster),
                title=level.cut.names.get(cluster, f"cluster {cluster}"),
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    # -- committing -------------------------------------------------------------

    @app.post("/api/levels/{name}/names")
    def set_names(name: str, body: NamesRequest) -> dict[str, Any]:
        try:
            return state.set_names(
                name,
                names={int(k): v for k, v in (body.names or {}).items()},
                colors={int(k): v for k, v in (body.colors or {}).items()},
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.post("/api/levels/{name}/commit")
    def commit(name: str, body: CommitRequest) -> dict[str, Any]:
        try:
            return state.commit(
                name, label_name=body.label_name, children=body.children
            )
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.post("/api/levels/{name}/note")
    def add_note(name: str, body: NoteRequest) -> dict[str, Any]:
        return state.note(name, body.text)

    # -- ledger -----------------------------------------------------------------

    @app.get("/api/ledger")
    def get_ledger() -> dict[str, Any]:
        frame = state.ledger.to_frame()
        return {"entries": frame.to_dicts(), "n": len(state.ledger)}

    @app.post("/api/ledger/rewind")
    def rewind_ledger(body: RewindRequest) -> dict[str, Any]:
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="rewinding detaches label columns and drops child masks; "
                "re-send with confirm=true",
            )
        try:
            if body.keep is None:
                return state.reset(keep_sweeps=body.keep_sweeps, archive=body.archive)
            return state.rewind(body.keep, archive=body.archive)
        except (ValueError, KeyError) as exc:
            _fail(exc)

    @app.get("/api/ledger/script", response_class=PlainTextResponse)
    def get_script() -> str:
        return state.ledger.to_script()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "levels": len(state.tree),
            "busy": state.jobs.busy,
            "build": _client_build(),
        }

    # -- client -----------------------------------------------------------------

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

        @app.get("/")
        def index() -> Any:
            return FileResponse(str(STATIC / "index.html"))

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cellpax-app", description="Local labeling app for a cellpax descent."
    )
    parser.add_argument(
        "config", nargs="?", default=None, help="Path to an app config TOML."
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    config = AppConfig.from_toml(args.config) if args.config else AppConfig()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    import uvicorn

    app = create_app(config)
    banner = (
        "demo fixture" if config.is_demo else f"{config.folio} :: {config.table_name}"
    )
    print(f"cellpax app  ->  http://{config.host}:{config.port}   ({banner})")
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
