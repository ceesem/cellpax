"""The loky resource-tracker wire-format shim.

CPython 3.13.10 changed the tracker protocol out from under loky's server. The
regression these guard is not cosmetic: unparsed messages never reach the
registry, so refcounts never drop and shared temp folders never get unlinked.

The integration tests run in subprocesses because the tracker server is a real
child process launched once per interpreter -- there is no way to exercise the
launch path in-process, and no way to un-launch it afterwards.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap

import cellpax_loky_compat
from cellpax_loky_compat import _to_legacy


def _run(body: str, **env_extra: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter, capturing the tracker's stderr too."""
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _json_msg(cmd: str, name: str, rtype: str) -> bytes:
    b64 = base64.urlsafe_b64encode(name.encode()).decode("ascii")
    payload = {"cmd": cmd, "rtype": rtype, "base64_name": b64}
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode()


class TestTranslation:
    def test_json_becomes_legacy(self):
        assert _to_legacy(_json_msg("REGISTER", "/tmp/pool", "folder")) == (
            b"REGISTER:/tmp/pool:folder\n"
        )

    def test_name_containing_colon_survives(self):
        """loky's parser rejoins the middle fields, so ':' in a path is safe."""
        line = _to_legacy(_json_msg("MAYBE_UNLINK", "/tmp/a:b/c", "file"))
        assert line == b"MAYBE_UNLINK:/tmp/a:b/c:file\n"
        splitted = line.strip().decode().split(":")
        cmd, name, rtype = splitted[0], ":".join(splitted[1:-1]), splitted[-1]
        assert (cmd, name, rtype) == ("MAYBE_UNLINK", "/tmp/a:b/c", "file")

    def test_probe_without_name(self):
        """The stdlib probe carries no base64_name; loky reads it as an empty name."""
        probe = b'{"cmd":"PROBE","rtype":"noop"}\n'
        assert _to_legacy(probe) == b"PROBE::noop\n"

    def test_legacy_passes_through(self):
        """loky's own probe already speaks the old format and must not be touched."""
        assert _to_legacy(b"PROBE:0:noop\n") == b"PROBE:0:noop\n"

    def test_unknown_shape_passes_through(self):
        """Garbage is handed to loky verbatim, so it reports what it always would."""
        assert _to_legacy(b'{"unexpected": 1}\n') == b'{"unexpected": 1}\n'
        assert _to_legacy(b"{not json\n") == b"{not json\n"


class TestApply:
    def test_idempotent_and_reports_state(self):
        from joblib.externals.loky.backend import resource_tracker as rt

        first = cellpax_loky_compat.apply()
        after_first = rt.main
        assert cellpax_loky_compat.apply() is first
        assert rt.main is after_first

    def test_opt_out_env(self):
        assert (
            _run(
                """
                import cellpax_loky_compat
                print(cellpax_loky_compat.apply())
                """,
                CELLPAX_NO_LOKY_COMPAT="1",
            ).stdout.strip()
            == "False"
        )

    def test_stands_down_when_loky_speaks_json(self, monkeypatch):
        """The shim must disappear on its own once joblib ships a fix."""
        monkeypatch.setattr(cellpax_loky_compat, "_loky_speaks_json", lambda: True)
        assert cellpax_loky_compat._needs_patch() is False

    def test_stands_down_when_loky_owns_the_client_side(self, monkeypatch):
        """If loky overrides _send, the format is upstream's business, not ours."""
        monkeypatch.setattr(
            cellpax_loky_compat._loky_rt.ResourceTracker,
            "_send",
            lambda self, cmd, name, rtype: None,
        )
        assert cellpax_loky_compat._needs_patch() is False

    def test_warns_if_the_tracker_is_already_running(self):
        """A late import cannot fix a server that is already up -- say so."""
        out = _run(
            """
            import warnings
            from joblib.externals.loky.backend import resource_tracker as rt
            rt.ensure_running()          # server up, running loky's own loop
            import cellpax_loky_compat
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cellpax_loky_compat.apply()
            print(any("already running" in str(w.message) for w in caught))
            """
        )
        assert out.stdout.strip() == "True", out.stderr

    def test_importing_cellpax_applies_it(self):
        out = _run(
            """
            import cellpax
            from joblib.externals.loky.backend import resource_tracker as rt
            import cellpax_loky_compat
            print(rt.main is cellpax_loky_compat.main)
            """
        )
        assert out.stdout.strip() == "True", out.stderr


_REFCOUNT_PROBE = """
    import os, shutil, tempfile, time
    import cellpax_loky_compat
    patched = cellpax_loky_compat.apply()
    from joblib.externals.loky.backend import resource_tracker as rt

    folder = tempfile.mkdtemp(prefix="cellpax_tracker_test_")
    rt.register(folder, "folder")
    rt.register(folder, "folder")
    rt.maybe_unlink(folder, "folder")
    time.sleep(1.0)
    held = os.path.exists(folder)
    rt.maybe_unlink(folder, "folder")
    time.sleep(1.0)
    gone = not os.path.exists(folder)
    shutil.rmtree(folder, ignore_errors=True)
    print(f"{patched} {held} {gone}")
"""


def test_refcounted_folder_cleanup():
    """The whole point: refcounts survive translation, so folders get unlinked.

    Two registrations and one release must leave the folder alone; the second
    release must delete it.
    """
    out = _run(_REFCOUNT_PROBE)
    applied, held_at_one, gone_at_zero = out.stdout.strip().split()
    assert applied == "True", out.stderr
    assert held_at_one == "True", out.stderr
    assert gone_at_zero == "True", out.stderr


def test_unpatched_leaks_on_an_affected_interpreter():
    """Pins down what the shim is actually buying, so it cannot be quietly dropped.

    On an interpreter/joblib pair that needs the shim, opting out must leak the
    folder. Where the shim is unnecessary, cleanup must work without it.
    """
    out = _run(_REFCOUNT_PROBE, CELLPAX_NO_LOKY_COMPAT="1")
    applied, _held, gone_at_zero = out.stdout.strip().split()
    assert applied == "False"

    if cellpax_loky_compat._needs_patch():
        assert gone_at_zero == "False", (
            "expected the unpatched tracker to leak the folder; if this now "
            "passes, joblib may have fixed the format and the shim can go"
        )
        assert "unknown resource type" in out.stderr
    else:
        assert gone_at_zero == "True", out.stderr


def test_memmapped_parallel_run_is_quiet():
    """A memmapped Parallel run must not print tracker tracebacks."""
    out = _run(
        """
        import cellpax  # applies the shim
        import numpy as np
        from joblib import Parallel, delayed

        def total(x):
            return float(x.sum())

        # max_nbytes=1 forces memmapping, which is what drives tracker traffic.
        print(Parallel(n_jobs=2, max_nbytes=1)(
            delayed(total)(np.ones(10_000)) for _ in range(4)
        ))
        """
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[10000.0, 10000.0, 10000.0, 10000.0]"
    assert "unknown resource type" not in out.stderr
    assert "resource_tracker" not in out.stderr, out.stderr
