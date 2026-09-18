---
id: TASK-1
title: Standardize public API typing and NumPy docstrings
status: Done
assignee:
  - codex
created_date: '2026-09-08 15:59'
updated_date: '2026-09-08 16:21'
labels: []
dependencies: []
modified_files:
  - src/cellpax/assign.py
  - src/cellpax/clustering.py
  - src/cellpax/compare.py
  - src/cellpax/featuretable.py
  - src/cellpax/gradient.py
  - src/cellpax/labels.py
  - src/cellpax/persist.py
  - src/cellpax/propagate.py
  - src/cellpax/space.py
  - src/cellpax/validate.py
  - docs/reference/api.md
  - mkdocs.yml
  - tests/test_public_api_docs.py
priority: high
type: docs
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the complete user-facing CellPax API to a consistent, readable documentation and typing standard. Public functions, classes, methods, and properties should expose useful domain types and clear NumPy-style documentation without forcing excessive dtype-level precision or verbose boilerplate on trivial accessors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every public API member exported through cellpax has an appropriate docstring, with concise summaries permitted for trivial properties and accessors
- [x] #2 Every non-trivial public function and method documents its accepted inputs and returned value using NumPy-style sections
- [x] #3 Public class constructors and result containers document their user-relevant parameters or attributes, including shapes, ordering, optionality, and mutation or copy behavior where relevant
- [x] #4 Public signatures use concrete CellPax domain types where known and retain flexible typing only at intentionally open third-party or scalar-value boundaries
- [x] #5 Docstrings agree with current signatures, including forwarded keyword arguments and recently added parameters
- [x] #6 Important user-actionable errors and optional dependency failures are documented without requiring exhaustive Raises sections for every validation branch
- [x] #7 The generated API reference displays useful signature annotations without duplicate public API entries
- [x] #8 The full test suite, lint checks, and strict documentation build pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish a lightweight public-docstring convention: summary on every public member; NumPy Parameters/Returns for callable inputs and outputs; Attributes for public result containers; selective Raises for actionable failures; concise property docs allowed.
2. Correct the public type surface by adding the five missing annotations and replacing bare Any returns/arguments with known CellPax domain types through TYPE_CHECKING-safe forward references and small reusable aliases where they improve clarity.
3. Update exported classes and functions module by module, preserving strong existing prose while restructuring it into scannable NumPy sections and reconciling every documented name with its signature.
4. Update API-generation configuration so annotations are visible and the reference page does not render the same objects twice.
5. Add a regression test that statically checks the agreed public documentation/type invariants, then run focused tests, the full suite with coverage, Ruff, and the strict MkDocs build. Record any intentional exceptions in the test or documentation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete pending final coverage run: standardized public API type annotations and NumPy-style docstrings across exported functions/classes and FeatureTable methods; corrected flatten_labels' conditional tuple return; enabled rendered signature annotations; removed duplicate umbrella API rendering and added all missing explicit exports; added tests/test_public_api_docs.py to enforce documentation, annotations, signature/doc agreement, and one-entry-per-export API coverage. Verification so far: 675 tests passed without coverage, Ruff passed, strict MkDocs build passed.

Final verification: `uv run poe test` passed all 675 tests with coverage instrumentation (8 expected warnings); `uv run ruff check src/ tests/test_public_api_docs.py` passed; `uv run mkdocs build --strict` passed; `git diff --check` passed. The API contract test also confirms every exported callable/member has documentation and annotations, signature parameters are represented in NumPy Parameters sections, exported classes describe Parameters or Attributes, and the reference page has exactly one directive per export.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Standardized CellPax's user-facing API documentation and typing across exported functions, result classes, and FeatureTable workflows. Added concrete domain return/argument types where known, complete NumPy-style Parameters/Returns/Attributes sections, and selective actionable Raises guidance. Corrected the conditional `flatten_labels` return type. Enabled signature annotations in generated docs, removed duplicate umbrella rendering, added missing explicit API entries, and introduced a regression test enforcing the public API documentation contract. Verification: 675 tests pass with coverage instrumentation, Ruff passes, strict MkDocs build passes, and the diff is whitespace-clean.
<!-- SECTION:FINAL_SUMMARY:END -->
