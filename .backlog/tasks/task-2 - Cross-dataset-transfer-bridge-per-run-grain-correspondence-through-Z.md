---
id: TASK-2
title: 'Cross-dataset transfer bridge: per-run grain correspondence through Z'
status: Done
assignee: []
created_date: '2026-10-22 00:18'
updated_date: '2026-10-22 01:22'
labels:
  - clustering
  - datasets
dependencies: []
priority: high
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tier 1 phase 1 of semi-aligned clustering: cluster each dataset in its own A space, score cross-dataset reproducibility of every stored Leiden run by carrying labels through the aligned embedding Z. See plan /Users/caseysm/.claude-work/plans/i-am-considering-a-quizzical-crown.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TransferBridge holds sparse cross and leave-one-out within-dataset Z vote matrices (optional mutual-only)
- [x] #2 carry() is one sparse one-hot matmul; -1 source cells do not vote; empty rows abstain
- [x] #3 score() reports AMI, coverage, per-cluster recall, within-dataset Z ceiling and efficiency in both directions
- [x] #4 score_runs() scores matched-grain run pairs without any consensus rebuild, with a k-means-in-A null and excess
- [x] #5 GrainCorrespondence frame + ridge(); ft.transfer_bridge entry point; tests in tests/test_bridge.py
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented src/cellpax/bridge.py + ft.transfer_bridge; tests/test_bridge.py (16 tests); guide 'Semi-aligned clustering', api.md, changelog.

Deviations from plan, found while calibrating the fixture:
- Default null is random Voronoi tessellations in each dataset's own features (independent seeds), not k-means: k-means competes with real structure (finds true discrete types too, zeroing excess where the clustering is right). null='kmeans' kept as an explicit competitor baseline.
- excess = ami - null_ami, NOT efficiency - null_efficiency: tessellations cut through populations, so their ceiling is low and the efficiency ratio inflates (null efficiency 0.6-0.77 vs null AMI 0.17-0.42 on the fixture). Efficiency is the blur diagnostic, not the reproducibility score.
- Added worst_efficiency (per-cluster minimum): global AMI is carried by shared coarse structure; a refuted fine split left global efficiency at 0.86 while its per-cluster efficiency was ~0.6.
- Per-cluster reading: high ceiling + low efficiency = refuted; low ceiling = unverifiable through this bridge (efficiency ~1 by construction, not support).

Remaining for this task: the real-data evidence run (does efficiency/excess separate blur from irreproducibility on the actual A/Z?).

Real-data evidence run done 2026-10-21 (joint_embedding_ad data, inh + exc subsample): Z aligns (mixing gap 0.12/0.18 vs 0.5 in A); subclass control efficiency 0.83; inh A partitions reproduce beyond null within subclasses; Z-space clustering gets many more clusters refuted.
<!-- SECTION:NOTES:END -->
