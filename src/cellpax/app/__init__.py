"""The local labeling app — a stateful driver for the recursive descent.

Not a plotting layer for cellpax and not a notebook. The library stays headless
and emits frames; this consumes them and owns the two things a notebook cannot:

* **A tree of levels.** One level is "mask -> space -> Clustering -> threshold ->
  child masks", and the descent applies it again to each child. In a notebook
  that recursion becomes copy-paste, because a cell is not a level. Here the
  level is the unit you navigate.
* **A decision ledger.** Append-only, replayable, and deliberately outside
  cellpax — ``DESIGN_PROPOSAL.md`` drops ledgers from the library on purpose.

Two facts about cost shape everything (see :mod:`cellpax.app.session`):

* The consensus linkage costs ~4.8 GB to build at 34.6k cells (~9.6 GB peak,
  since scipy copies the condensed vector) and then lives as a ~1 MB array. So
  the server is one long-lived process that builds it once, and jobs that build
  one are serialized — two at a time would be fatal on any normal machine.
* Once it is cached, cutting is free: ``cluster_labels`` is an ``fcluster`` call
  and ``soft_labels`` is a column sum. Threshold scrubbing is therefore an
  inline read, while ``cluster`` / ``restrict`` / ``embed`` are background jobs.

The UI keeps that distinction visible rather than making everything feel equally
slow.

Run it with ``poe app`` (demo fixture) or ``poe app my-config.toml``.
"""

from cellpax.app.config import AppConfig
from cellpax.app.ledger import Entry, Ledger
from cellpax.app.levels import Level, LevelTree
from cellpax.app.session import Session

__all__ = ["AppConfig", "Entry", "Ledger", "Level", "LevelTree", "Session"]
