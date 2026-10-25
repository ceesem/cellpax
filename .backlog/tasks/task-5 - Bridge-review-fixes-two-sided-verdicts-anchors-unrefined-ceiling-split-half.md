---
id: TASK-5
title: >-
  Bridge review fixes: two-sided verdicts, anchors, unrefined ceiling,
  split-half
status: Done
assignee: []
created_date: '2026-10-22 03:04'
labels:
  - clustering
  - datasets
dependencies: []
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Acting on the independent review (2026-10-21). Casey's decisions: fix one-sided reproduction (1), unrefined LOO ceiling (6), headline over all clusters / unverifiable never shared (3), split-half option (4); keep the Voronoi null (anchors name shared regions, not boundaries) (2); note seed-only SEs (5); defer area/coverage handling (7). Goal clarified: aligned names in the two morphological spaces as proposed anchors for comparison analyses; clustering stays driven by statistics plus biological priors.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-sided verdicts add nested/split; BridgeScore.anchors(); reproduce_share/anchored_share/anchored_cells; GrainCorrespondence.matched() and matched curve/band by default; ceiling never refined; transfer_bridge(mask=) holds out the rest (held_out); bridging.spatial_halves, anchor-based suggest_names, anchor_rows; notebook regenerated (refine off by default, anchor table). Not done, by decision: correspondence-shuffle/gradient nulls, bootstrap SEs, has_z-primary analysis.
<!-- SECTION:NOTES:END -->
