r"""Bridge loky's resource tracker across CPython's tracker wire-format change.

``joblib.externals.loky.backend.resource_tracker.ResourceTracker`` subclasses the
stdlib ``multiprocessing.resource_tracker.ResourceTracker`` and inherits its
``_send``, but it ships *its own* tracker server (``main``) with its own parser.
CPython 3.13.10 rewrote the wire format from ``CMD:name:rtype\n`` to a JSON
payload carrying a base64-encoded name, so every message joblib sends now lands
in a parser that cannot read it::

    ValueError: Cannot register "REGISTER","rtype":"folder","base64_name" for
    automatic cleanup: unknown resource type ("L3Zhci9mb2xkZXJz...")

The server catches that per line and hands it to ``sys.excepthook``, so a
memmapped ``Parallel`` run prints one traceback per registered resource -- enough
to bury a notebook -- and, since nothing ever reaches the registry, the
refcounting that lets loky unlink shared temp folders never happens either.

Every process that touches a shared resource *writes* to the tracker (worker
processes register their own semaphores and memmaps), but exactly one process
*reads*: the tracker server. So this patches the reader.
``ResourceTracker._launch`` builds the server's command line as
``f"from {main.__module__} import main; main({r}, {VERBOSE})"``, resolving the
module-level ``main`` at launch time -- rebinding that name is therefore enough
to bring the server up here instead, translate the stream, and feed it to loky's
real loop. Nothing about loky's parsing, refcounting, or cleanup is reimplemented.

This is a separate top-level package on purpose: the tracker server imports it,
and importing ``cellpax`` there would drag numpy/scipy/sklearn into a daemon that
should stay tiny.

``apply()`` is conditional on all three legs of the mismatch still being true, so
it disables itself the moment joblib ships a fix. Set ``CELLPAX_NO_LOKY_COMPAT=1``
to opt out by hand.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import threading
import warnings
from multiprocessing.resource_tracker import ResourceTracker as _StdResourceTracker

from joblib.externals.loky.backend import resource_tracker as _loky_rt

__all__ = ["apply", "main"]

# Bound at import time, before apply() can rebind `_loky_rt.main`. The tracker
# server imports this module, so this has to still be loky's own loop there.
_LOKY_MAIN = _loky_rt.main

_OPT_OUT_ENV = "CELLPAX_NO_LOKY_COMPAT"


def _stdlib_speaks_json() -> bool:
    """Does this CPython's tracker client emit the JSON wire format?

    ``_make_probe_message`` is stateless, so calling it unbound is a direct read
    of the format rather than a version guess.
    """
    make_probe = getattr(_StdResourceTracker, "_make_probe_message", None)
    if make_probe is None:
        return False
    try:
        return make_probe(None).lstrip().startswith(b"{")
    except Exception:
        return False


def _loky_speaks_json() -> bool:
    """Has loky's tracker server learned the JSON format (i.e. is this fixed)?"""
    try:
        return "base64_name" in inspect.getsource(_LOKY_MAIN)
    except (OSError, TypeError):
        # No source to read. Assume not: translating a stream that needs no
        # translation is a no-op, so guessing wrong this way is harmless.
        return False


def _needs_patch() -> bool:
    if not _stdlib_speaks_json():
        return False
    if _loky_rt.ResourceTracker._send is not _StdResourceTracker._send:
        # loky overrides the client side, so upstream owns the format on both
        # ends and we should keep our hands off it.
        return False
    return not _loky_speaks_json()


def _to_legacy(line: bytes) -> bytes:
    r"""Rewrite one JSON tracker message as loky's ``CMD:name:rtype\n`` form.

    Non-JSON lines pass through untouched -- loky's own probe (``PROBE:0:noop``)
    already speaks the legacy format. A name containing ``:`` survives the round
    trip: loky's parser takes the first and last fields and rejoins the middle.
    """
    stripped = line.strip()
    if not stripped.startswith(b"{"):
        return line
    try:
        payload = json.loads(stripped)
        cmd = payload["cmd"].encode("ascii")
        rtype = payload["rtype"].encode("ascii")
        name = base64.urlsafe_b64decode(payload.get("base64_name", ""))
    except Exception:
        # Not a shape we know. Hand it over as-is and let loky's parser report
        # it exactly as it would have without us in the way.
        return line
    return cmd + b":" + name + b":" + rtype + b"\n"


def _pump(src_fd: int, dst_fd: int) -> None:
    try:
        with open(src_fd, "rb") as src:
            for line in src:
                os.write(dst_fd, _to_legacy(line))
    except BaseException:
        try:
            sys.excepthook(*sys.exc_info())
        except BaseException:
            pass
    finally:
        # EOF has to reach loky's loop or the server blocks forever and never
        # runs its shutdown cleanup.
        try:
            os.close(dst_fd)
        except OSError:
            pass


def main(fd: int, verbose: int = 0) -> None:
    """Run loky's tracker server behind a wire-format translator.

    Signature matches ``loky...resource_tracker.main``; this is what ``_launch``
    spawns once :func:`apply` has run. Signal handling stays on the main thread
    where loky installs it; only the translation runs on the helper thread.
    """
    if sys.platform == "win32":
        import msvcrt

        # loky hands the server an inheritable handle; take it down to an fd to
        # read, then hand loky's loop a handle for our pipe so its own
        # conversion still applies. Untested on Windows.
        fd = msvcrt.open_osfhandle(fd, os.O_RDONLY)

    read_fd, write_fd = os.pipe()
    threading.Thread(
        target=_pump,
        args=(fd, write_fd),
        daemon=True,
        name="cellpax-loky-tracker-translate",
    ).start()

    if sys.platform == "win32":
        import msvcrt

        read_fd = msvcrt.get_osfhandle(read_fd)

    _LOKY_MAIN(read_fd, verbose)


def apply() -> bool:
    """Route loky's resource tracker through the translating server.

    Idempotent, and safe to call before or after joblib has been used. Returns
    whether the patch is in place; ``False`` means it was skipped because it is
    not needed on this interpreter/joblib pair, or because ``CELLPAX_NO_LOKY_COMPAT``
    is set.
    """
    if os.environ.get(_OPT_OUT_ENV):
        return False
    if _loky_rt.main is main:
        return True
    if not _needs_patch():
        return False
    _loky_rt.main = main

    if _loky_rt._resource_tracker._fd is not None:
        # The server is already up, running loky's unpatched loop; only a
        # relaunch picks this up. Worth saying out loud, because the symptom
        # (a wall of tracker tracebacks) looks identical to the patch failing.
        warnings.warn(
            "loky's resource tracker was already running when the CellPax "
            "wire-format shim was applied, so it is still running the "
            "unpatched server. Restart the kernel/process, and import cellpax "
            "before the first joblib.Parallel call.",
            RuntimeWarning,
            stacklevel=2,
        )
    return True
