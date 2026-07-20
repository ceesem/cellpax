# Tutorial: from features to a published annotation

This walkthrough builds a complete CellPax study from scratch: you'll register a
cell universe, cluster it, review the clusters into named cell types, and publish
a self-contained release a collaborator can consume. Every snippet below is part
of one runnable script — paste them in order.

By the end you will have used the whole pipeline:

**universe → features → scope → feature space → representation → clustering →
candidates → review → taxonomy → assignment set → views → release**.

## 0. Set up

We'll use a small synthetic dataset of 40 cells that fall into two obvious
groups, so the clustering has something clean to find.

```python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from cellpax import (
    CandidateCutConfig,
    ClusteringConfig,
    FeatureDefinition,
    FeatureSpaceConfig,
    RepresentationConfig,
    Study,
    TaxonDefinition,
    taxonomy_table,
)

rng = np.random.default_rng(0)
blob_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.4, size=(20, 3))
blob_b = rng.normal(loc=[6.0, 6.0, 6.0], scale=0.4, size=(20, 3))
coords = np.vstack([blob_a, blob_b])
cell_ids = list(range(1, 41))

path = Path(tempfile.mkdtemp()) / "connectome-study"
```

## 1. Create the study and register the universe

A study is one immutable, content-addressed store. The **universe** is its single
authoritative list of cells; everything else must reference a subset of it. You
can hand it a plain list of ids.

```python
study = Study.create(path, created_by="tutorial")
study.register_universe(cell_ids)
```

Cell ids are coerced to a non-null, unique `Int64` column. The universe is set
once and cannot be replaced — this is what lets every later artifact trust its
cell ids.

## 2. Register features and their catalog

Feature values live in a **feature block**, paired with a **catalog** that
documents what each column means. Describe the columns with `FeatureDefinition`
objects and CellPax builds the strict catalog for you — the catalog travels with
the features, so a release is self-describing.

```python
values = pl.DataFrame(
    {
        "cell_id": pl.Series(cell_ids, dtype=pl.Int64),
        "morph_0": coords[:, 0],
        "morph_1": coords[:, 1],
        "morph_2": coords[:, 2],
    }
)
block = study.register_feature_block(
    values,
    [
        FeatureDefinition("morph_0", modality="morphology", family="shape"),
        FeatureDefinition("morph_1", modality="morphology", family="shape"),
        FeatureDefinition("morph_2", modality="morphology", family="shape"),
    ],
)
```

Registration is content-addressed: register the identical block twice and you get
the same id back, not a duplicate.

## 3. Build a revision with the fluent builder

Everything downstream — scope, feature space, representation, clustering — is
*previewed* (materialized and given a stable id) and then **kept** as a named
**revision** when you want a durable checkpoint. `study.build()` gives you a
builder that holds each artifact so you don't re-pass them at every step, and
chains kept revisions into a history automatically.

Start with the cells to work on and the features to use, then keep your first
checkpoint:

```python
build = study.build()
build.scope(cell_ids).select(block)   # all cells, all features in the block
inputs = build.keep("inputs")
```

`scope(cell_ids)` selects cells; `select(block)` selects every feature in the
block (pass a list of feature ids to narrow it). Neither needs a description —
provide one with `derivation_text=...` when you want to record intent.

## 4. Feature space and representation

A **feature space** applies a transform to the selected values; a
**representation** produces coordinates for clustering or visualization. Use the
typed config factories — no magic strings, and your editor autocompletes the
parameters:

```python
build.feature_space(FeatureSpaceConfig.standard_scaler())
build.representation(RepresentationConfig.pca(n_components=2))
represented = build.keep("scaled + pca")
```

Each transform learns from a `fit_scope`, which defaults to the current scope —
pass `fit_scope=...` explicitly only when you want to fit on a trusted core and
apply to a wider set.

## 5. Cluster once, cut cheaply

Clustering is split into two steps on purpose. The expensive consensus work is a
**clustering run**, stored once. Turning it into a flat partition is a cheap
**candidate set** — you can take many different cuts of one run without paying for
it again.

```python
build.clustering(ClusteringConfig.fauxnograph(n_neighbors=(10,)))
build.candidates(CandidateCutConfig.distance(threshold=0.5))
revision = build.keep("candidates")

definitions = study.candidate_definitions(revision.candidate_set_id)
print(definitions["candidate_id"].to_list(), definitions["n_cells"].to_list())
# -> [0, 1] [20, 20]
```

!!! note "fauxnograph runs in parallel"
    The built-in `fauxnograph` backend uses all cores by default. Pass
    `ClusteringConfig.fauxnograph(n_neighbors=(10,), n_jobs=1)` for a
    deterministic single-threaded run, or to silence joblib worker warnings in
    some environments.

!!! tip "Prefer explicit calls?"
    The builder is optional sugar. Every step maps to a `study.preview_*` call
    plus a final `study.keep(...)`; the [User Guide](guide.md) shows that form.

The two blobs came back as candidates `0` and `1`, twenty cells each. Candidates
are just *numbered groups* — they carry no meaning until you review them.

## 6. Define a taxonomy

The **taxonomy** is your controlled vocabulary of cell types. Build it from
concise `TaxonDefinition` objects; input order becomes display order, and omitted
display/lifecycle fields get sensible defaults. It is versioned and immutable:
once `1.0.0` is registered, its rows never change.

```python
study.register_taxonomy(
    taxonomy_table(
        "cells",
        "1.0.0",
        [
            TaxonDefinition(1, "excitatory", label="Exc", color="#d62728"),
            TaxonDefinition(2, "inhibitory", label="Inh", color="#1f77b4"),
        ],
    )
)
```

For a hierarchy, add `parent_id=...`; use the longer keyword fields only when a
taxon needs to override its label-derived names.

## 7. Review: assign candidates to types

Review is an **append-only ledger**. Each `append_decision` records an action, its
targets, a taxon, and — required — a rationale. Decisions form a linear branch;
here we assign each candidate to a type on the `main` branch.

```python
ids = definitions["candidate_id"].to_list()

study.append_decision(
    revision=revision, review_branch="main", action="assign",
    target_kind="candidates", target_ids=[ids[0]], taxon_id=1,
    rationale="Compact, well-separated cluster; excitatory morphology.",
)
head = study.append_decision(
    revision=revision, review_branch="main", action="assign",
    target_kind="candidates", target_ids=[ids[1]], taxon_id=2,
    rationale="Second cluster is inhibitory.",
)
```

Beyond `assign`, the ledger supports `merge`, `split`, `exclude`,
`mark_ambiguous`, `assign_parent_only`, and manual per-cell corrections — see the
[User Guide](guide.md#reviewing-decisions).

## 8. Reduce decisions into an assignment set

An **assignment set** is the immutable result of reducing a decision head into
one row per cell. Correcting labels later produces a *new* assignment set — the
taxonomy version stays fixed.

```python
assignment_set = study.create_assignment_set_from_decisions(
    taxonomy_name="cells", taxonomy_version="1.0.0",
    review_branch="main", decision_head=head,
)
study.assignments(assignment_set)  # one row per cell: taxon, status, source, ...
```

## 9. Look at the result with views

**Views** are read-only, fixed-schema projections. `study.view(name, ...)` joins
scope, candidates, assignments, and taxonomy for you:

```python
study.view("cells", revision, assignment_set=assignment_set)      # per-cell table
study.view("embedding", revision, assignment_set=assignment_set)  # x/y + colors
study.view("taxonomy", assignment_set=assignment_set)             # counts per taxon
study.view("release_summary", revision, assignment_set=assignment_set)
```

An embedding view drops straight into a plot or a serialized report component:

```python
from cellpax.plotting import embedding_scatter
from cellpax.views import serialize_view

frame = study.view("embedding", revision, assignment_set=assignment_set)
embedding_scatter(frame)              # a matplotlib figure, colored by taxon
component = serialize_view("embedding", frame)   # deterministic JSON bytes
```

## 10. Publish a release

A **release** freezes a source revision + assignment set into a self-documenting
bundle: taxonomy, assignments, decision lineage, quality summary, a resolved
recipe, a replay script, and a generated enum binding.

```python
release = study.create_annotation_release(
    "cells-v1", source_revision=revision, assignment_set=assignment_set
)

bundle = study.load_annotation_release(release)
Taxon = bundle.taxonomy_enum()
print(Taxon.EXCITATORY, Taxon.EXCITATORY.color)   # -> Taxon.EXCITATORY #d62728

study.validate_annotation_release(release)         # re-derives and checksums everything
```

## 11. Consume it downstream

A consumer needs only the release bundle — no clustering or review code. The
Trajan adapter (or the generic `decorate_cells`) joins released labels onto any
cell table:

```python
from cellpax.adapters.trajan import decorate_cells

consumer = pl.DataFrame(
    {"cell_id": pl.Series(cell_ids, dtype=pl.Int64), "extra": range(40)}
)
decorated = decorate_cells(consumer, bundle)
# adds taxon_id, assignment_status, taxon_short_name, taxon_color, ...
```

## 12. Reopen and validate

Everything you built survives a fresh process. Reopening and validating proves
the study is internally consistent and content-addressed as claimed.

```python
study.validate()

reopened = Study.open(path, read_only=True)
reopened.validate()
reopened.load_annotation_release("cells-v1")
```

## Where to go next

You've touched every stage of the pipeline. The **[User Guide](guide.md)** goes
deeper on each: the preview/keep model and the builder, fit-vs-application scopes,
the full set of review actions, propagating labels to non-core cells, coverage and
provenance on assignments, and the trust model behind `validate()`.
