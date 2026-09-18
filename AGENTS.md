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

## Notes

<!-- Miscellaneous notes, gotchas, TODOs -->

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
