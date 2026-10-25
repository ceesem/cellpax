---
id: TASK-3
title: 'Transfer bridge phase 2: consensus-level scoring, 1-SE band, A-refinement'
status: Done
assignee: []
created_date: '2026-10-22 00:18'
updated_date: '2026-10-22 01:38'
labels:
  - clustering
  - datasets
dependencies: []
priority: medium
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
score_labels on nested levels of shortlisted restricts, suggest_band with 1-SE rule, optional diffusion of the Z prior over target A graph (applied identically to null and ceiling). Gated on phase-1 real-data evidence. Plan: /Users/caseysm/.claude-work/plans/i-am-considering-a-quizzical-crown.md
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done (2026-10-21): masks-based ft.transfer_bridge(a, b) + strata=; strata_ami/strata_excess; VerdictRule + per-cluster verdicts (BridgeScore.clusters/verdicts(), verdict counts per run pair with null counterpart); soft-purity path shared by score and score_runs; GrainCorrespondence.curve() (log-binned, pooled SE) + suggest_band() 1-SE rule with at_edge warning. Real data: inh reproduces_excess peaks grain ~25-30 (turns over past 44); strata_excess peaks ~18-20. Exc (8k/dataset subsample): reproduces_excess peaks ~7-8, strata_excess near 0 until k>=23.
Remaining: A-refinement of the Z vote over target A graph (same estimator for null and ceiling); consensus-level scoring of restrict(resolution_max=band) cuts end to end.

Done (2026-10-21): refine_columns/refine_alpha (label spreading over the receiving population's own A graph; identical for cross, ceiling, null; never extends coverage); score_consensus(band=...) via shared _score_partitions core. Real data: refinement raises strata ceiling (inh 0.65->0.83 at a=0.8) and strata_excess at every grain; a=0.8 over-smooths fine grain (unverifiable 26->39 at k28, band peak 28->9), a=0.5 keeps fine-grain verdicts ~unchanged -> default 0.5. Consensus at band: inh level pairs k~10-19 reproduces_excess ~7, strata_excess 0.11-0.15; exc consensus fine levels strata_excess ~0 (fine exc structure not verifiable through Z).
<!-- SECTION:NOTES:END -->
