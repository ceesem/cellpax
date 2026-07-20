# CellPax

CellPax turns a table of per-cell features into a **reviewed, versioned, and
shareable cell-type annotation** — with every step kept as durable, reproducible
provenance.

If you cluster cells, look at the result, decide "candidate 4 is really two
types" or "these cells are artifacts, exclude them," and then need to hand a
clean, defensible annotation to a collaborator or a downstream tool — that
back-and-forth is exactly what CellPax records. Nothing is lost to a notebook
cell you overwrote.

## What you get

- **One immutable study.** Everything — the cell universe, features, clustering,
  human decisions, taxonomy, and final labels — lives in a single content-addressed
  store you can reopen in a fresh process and re-validate.
- **Preview, then keep.** Expensive computations are *previewed* freely and only
  promoted to a named **revision** when you want a durable checkpoint. Revisions
  form a history you can walk and compare.
- **A review ledger, not overwritten labels.** Merges, splits, exclusions, manual
  corrections, and ambiguity are appended as **decisions** with a rationale and
  evidence — never silent edits.
- **Versioned taxonomy, independent assignments.** Correcting a label creates a
  new **assignment set**, not a new taxonomy version, so your vocabulary stays
  stable while the annotation evolves.
- **Self-contained releases.** Publish an **annotation release** that bundles the
  taxonomy, labels, decision history, quality summary, a resolved recipe, and a
  generated enum binding — consumable without importing any clustering machinery.

## A 30-second look

```python
from cellpax import Study

study = Study.open("my-study", read_only=True)
release = study.get_annotation_release("cells-v1")
bundle = study.load_annotation_release(release)

bundle.assignments          # every cell's taxon, status, source, confidence
bundle.taxonomy             # the exact vocabulary version this release used
Taxon = bundle.taxonomy_enum()
Taxon.EXCITATORY.color      # rich metadata, generated from the taxonomy
```

## Install

CellPax uses [uv](https://docs.astral.sh/uv/) for its environment:

```bash
uv sync
```

During development the [DataFolio](https://github.com/) 2.0 storage substrate is
resolved from the sibling `../datafolio` checkout.

## Where to go next

- **[Tutorial](tutorial.md)** — build a complete study from raw features to a
  published release, end to end. Start here if you're new.
- **[User Guide](guide.md)** — task-oriented "how do I…" reference organized by
  concept: studies, feature spaces, clustering, review, taxonomy, releases, and
  the trust model.
- **[Function Reference](reference/api.md)** — the full API, generated from the
  source.
- **[Core contracts (ADR 0001)](decisions/0001-core-contracts-v1.md)** — the
  architectural decisions behind the immutable substrate.
