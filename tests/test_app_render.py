"""Run the client's chart renderers headlessly, against real API payloads.

The app has no build step and no JS test runner, which is right for a local
tool — but it left the drawing code exercised only by a person looking at it,
and rendering bugs shipped that way twice. This captures fixtures from the
actual endpoints, then executes every renderer under jsdom and fails on any
thrown error.

It proves the code *runs* and emits marks. It cannot prove the result looks
right; that still needs eyes. Skipped when node or jsdom are absent, so the
Python suite stays runnable without a JS toolchain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cellpax.app.config import AppConfig
from cellpax.app.demo import demo_table
from cellpax.app.ledger import Ledger
from cellpax.app.session import Session

pytest.importorskip("fastapi", reason="app extra not installed")

HARNESS = Path(__file__).parent / "js" / "render_smoke.mjs"


def _node() -> str | None:
    return shutil.which("node")


def _has_jsdom(node: str) -> bool:
    probe = subprocess.run(
        [node, "-e", "require.resolve('jsdom')"],
        capture_output=True,
        cwd=HARNESS.parent,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    _node() is None or not _has_jsdom(_node() or "node"),
    reason="needs node with jsdom: run `npm install jsdom` in tests/js",
)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> Path:
    """Drive the real session once and dump what the client would receive."""
    tmp = tmp_path_factory.mktemp("render")
    config = AppConfig(
        columns="analysis", order_by="soma_depth_um", root_mask="all_cells"
    )
    session = Session(
        demo_table(n_per_subtype=40), config, ledger=Ledger(tmp / "l.jsonl")
    )
    session.open_level("root", mask="all_cells")
    session.submit_cluster(
        "root",
        graph_type=["knn"],
        n_neighbors=[15],
        resolution=list(np.geomspace(0.05, 2.0, 12)),
        n_times=3,
    )
    session.jobs.wait(600)

    scan = session.scan("root", n_points=40, min_cluster_size=10)
    threshold = scan["plateaus"][0]["midpoint"]
    preview = session.preview_cut(
        "root", distance_threshold=threshold, min_cluster_size=10
    )
    session.set_names("root", names={0: "shallow", 1: "mid", 2: "deep"})
    session.submit_embed("root", method="pca")
    session.jobs.wait(600)
    session.submit_merge_support("root", n_bands=4)
    session.jobs.wait(600)
    session.submit_boundary("root", n_neighbors=15)
    session.jobs.wait(600)
    session.submit_parametrize("root", clusters=["shallow", "mid"])
    session.jobs.wait(600)

    payload = {
        "level": "root",
        "state": session.state(),
        "scan": scan,
        "preview": preview,
        "profile": session.column_profile("root", column="soma_depth_um", bins=24),
        "heatmap": session.cluster_heatmap("root", top=10),
        "flow": session.flow("root", n_levels=4, min_cluster_size=10),
        "scatter": session.scatter(
            "root", distance_threshold=threshold, min_cluster_size=10
        ),
        "stability": session.scatter("root", color_by="stability"),
        "featureScatter": session.feature_scatter(
            "root", feature="m0", against="soma_depth_um"
        ),
        "columns": {
            "numeric": session.numeric_columns("root"),
            "features": session.feature_names("root"),
        },
        "mergeSupport": session.merge_support("root"),
        "boundary": session.boundary("root"),
        "gradient": session.gradient("root"),
        "coverage": session.coverage("root"),
        "axisStability": session.axis_stability("root"),
        "softLabels": session.soft_labels("root", limit=20),
        "jobs": {"jobs": session.jobs.snapshot(), "busy": False},
        "ledger": {
            "entries": session.ledger.to_frame().to_dicts(),
            "n": len(session.ledger),
        },
    }
    path = tmp / "fixtures.json"
    path.write_text(json.dumps(payload))
    return path


def test_every_renderer_runs_without_throwing(fixtures):
    result = subprocess.run(
        [_node(), str(HARNESS), str(fixtures)],
        capture_output=True,
        text=True,
        cwd=HARNESS.parent,
    )
    print(result.stdout)
    assert result.returncode == 0, (
        f"renderer smoke failed\n{result.stdout}\n{result.stderr}"
    )


def test_the_harness_would_catch_a_chained_append():
    """Guard the guard: the bug that shipped must actually fail this harness.

    ``Node.append()`` returns undefined, so ``svg.append(x).append?.(y)`` throws
    on the property access before the optional call can short-circuit. A smoke
    test that would not have caught that is not worth having.
    """
    node = _node()
    # beside the harness: ESM resolves imports from the script's own directory,
    # not the working directory, so a probe in tmp_path cannot find jsdom
    script = HARNESS.parent / "_probe.mjs"
    script.write_text(
        "import { JSDOM } from 'jsdom';\n"
        "const { window } = new JSDOM('<svg id=s></svg>');\n"
        "const svg = window.document.querySelector('#s');\n"
        "const r = window.document.createElement('rect');\n"
        "try { svg.append(r).append?.(null); console.log('NO THROW'); }\n"
        "catch (e) { console.log('THREW:', e.constructor.name); }\n"
    )
    try:
        result = subprocess.run(
            [node, str(script)], capture_output=True, text=True, cwd=HARNESS.parent
        )
    finally:
        script.unlink(missing_ok=True)
    assert "THREW" in result.stdout, result.stdout + result.stderr
