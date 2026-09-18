# cellpax

Cell feature universe analysis for connectomics

## Team Tools & Domain Context

<!-- See team reference doc for CAVE ecosystem, caveclient, neuroglancer,
     task queue patterns, and connectomics-specific design decisions:
     [link TBD] -->

## Development Environment

```bash
# Install dependencies (creates virtual environment)
uv sync

# Add a new dependency
uv add <package>

# Linting / formatting
uv run ruff check src/
uv run ruff format src/
```

### Key Commands

| Command | Description |
|---------|-------------|
| `poe lab` | Launch Jupyter Lab |
| `poe profile` | Profile CPU with pyinstrument (HTML report) |
| `poe profile-all` | Profile CPU + memory with scalene |
| `poe scratch-lab` | Jupyter Lab in the scratch/ directory |
| `poe test` | Run pytest with coverage |
| `poe doc-preview` | Preview documentation locally |
| `poe drybump patch/minor/major` | Dry-run version bump |
| `poe bump patch/minor/major` | Bump version and create tag |

## Library Development

### Testing Strategy

Implement both integration tests and unit tests:

- **Integration tests**: Real-world workflows, end-to-end functionality
- **Unit tests**: Individual methods, edge cases, error conditions
- **Coverage target**: >90% line coverage, >85% branch coverage

```bash
poe test                                                       # full suite with coverage
uv run pytest tests/test_foo.py -v                             # single file
uv run pytest --cov=cellpax --cov-report=html tests/  # HTML report
```

### Release Process

```bash
poe drybump patch   # preview what will change
poe bump patch      # bump version, commit, tag (also: minor, major)
```

This updates `pyproject.toml`, `src/cellpax/__init__.py`, commits, and tags.

### Scratch Dir

`poe scratch-lab` opens Jupyter Lab in `scratch/`, isolated from the main package environment.

---

## Project Overview

<!-- Describe what this project does and why it exists -->

## Architecture & Key Files

<!-- List key files and how they fit together -->

## Data & External Dependencies

<!-- Document datasets, services, APIs, and file paths this project depends on -->

## Conventions & Patterns

<!-- Describe coding conventions, naming patterns, and design decisions -->

### The consensus matrix is ~50% dense — never change its format

`Clustering.similarity_matrix` is CSR only for the storage, not because it is sparse:
measured density on a real descent is **50%**, because coarse runs in the sweep put
almost every pair together at least once. At 34.6k cells that is ~6e8 nonzeros, 4.8 GB,
and the largest object in the session.

So anything that reformats or densifies it is a kernel kill, not a slowdown. In
particular `tolil()` costs ~80 bytes per nonzero against CSR's 8 — a 10x blow-up, ~48 GB
at descent scale — and it is tempting because `setdiag` is the natural way to drop the
diagonal. Don't: `setdiag(0)` on a CSR *copy* only clears entries, so it is
structure-preserving and warning-free. Better still, don't copy — subtract
`matrix.diagonal()` from the result afterwards.

Likewise, per-cluster `matrix[:, members]` column slicing is O(nnz) *per cluster*.
Right-multiply by a sparse one-hot cluster indicator instead: one O(nnz) pass for all
clusters at once, no copy, no dtype upcast. `soft_labels` is the worked example;
`sorted_matrix` takes the other route and refuses outright above `max_cells`.

## Notes

<!-- Miscellaneous notes, gotchas, TODOs -->

### The loky resource-tracker traceback flood (keep this fixed)

**Symptom.** Every `joblib.Parallel` run that memmaps anything prints one traceback
per shared resource, usually enough to bury a notebook:

```
ValueError: Cannot register "REGISTER","rtype":"folder","base64_name" for automatic
cleanup: unknown resource type ("L3Zhci9mb2xkZXJz...")
```

**Cause.** `loky.backend.resource_tracker.ResourceTracker` subclasses the stdlib
tracker and inherits its `_send`, but ships its own tracker *server* with its own
parser. CPython 3.13.10 changed the wire format from `CMD:name:rtype` to JSON with a
base64 name, so loky's parser cannot read what joblib now sends. It is **not** purely
cosmetic: nothing reaches the registry, so refcounts never drop and shared temp
folders leak (`tests/test_loky_compat.py` pins that down).

**Fix.** `src/cellpax_loky_compat/` — a translating tracker server. `cellpax/__init__.py`
calls `apply()`, which rebinds `resource_tracker.main`; `_launch` builds the server's
command line from `main.__module__`, so the server comes up in our module, translates
the stream, and hands it to loky's real loop. The reader is patched rather than the
writers because every worker process writes to the tracker but only one process reads.

**Why it keeps coming back, and why it should not now.** Previous fixes were edits to
the installed loky in a venv, which `uv sync` reverts (same trap as `poe fix-omp` in
v1dd_feature_pax). This one lives in CellPax source, so it survives syncs and travels
to every project that imports cellpax.

**If you touch this:**

- `apply()` is conditional on all three legs of the mismatch and stands down on its own
  once joblib ships a fix — do not make it unconditional. As of 2026-08, joblib 1.5.3
  and loky 3.5.6 are current and still affected; when a fixed release lands, bump and
  delete this shim plus its test.
- Keep `cellpax_loky_compat` a **separate top-level package**, listed in
  `[tool.uv.build-backend] module-name`. The tracker server imports it, and importing
  `cellpax` there would pull numpy/scipy/sklearn into a daemon that should stay tiny.
- The shim must be applied before the first `Parallel` call. A late `import cellpax`
  warns instead of silently doing nothing.
- Escape hatch: `CELLPAX_NO_LOKY_COMPAT=1`.

<!-- BACKLOG.MD MCP GUIDELINES START -->
<!-- backlog.md-instructions-version: 1.50.1 -->

<CRITICAL_INSTRUCTION>

## BACKLOG WORKFLOW INSTRUCTIONS

This project uses Backlog.md MCP for all task and project management activities.

**CRITICAL GUIDANCE**

- If your client supports MCP resources, read `backlog://workflow/overview` to understand when and how to use Backlog for this project.
- If your client only supports tools or the above request fails, call `backlog.get_backlog_instructions()` to load the tool-oriented overview. Use the `instruction` selector when you need `task-creation`, `task-execution`, or `task-finalization`.

- **First time working here?** Read the overview resource IMMEDIATELY to learn the workflow
- **Already familiar?** You should have the overview cached ("## Backlog.md Overview (MCP)")
- **When to read it**: BEFORE creating tasks, or when you're unsure whether to track work

These guides cover:
- Decision framework for when to create tasks
- Search-first workflow to avoid duplicates
- Links to detailed guides for task creation, execution, and finalization
- MCP tools reference

You MUST read the overview resource to understand the complete workflow. The information is NOT summarized here.

</CRITICAL_INSTRUCTION>

<!-- BACKLOG.MD MCP GUIDELINES END -->
