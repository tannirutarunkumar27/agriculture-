# Outer study planning implementation

Date: 2026-10-13. Task R04 is IN_PROGRESS, not fully complete.

Implemented separate planner `../restart/outer_protocol.py`; existing frozen
search/training/aggregation files remain unchanged. The planner verifies raw
and prepared assets/runtime/source identity, writer-locks completed search,
checks all 1120 completed receipts and cached metrics, recomputes selections
and requires an exact match to saved selections before producing the plan.
No NPZ test arrays are loaded and no model inference or training occurs.

Frozen output: `../results/restart_outer_protocol_v1/` containing config,
plan, selections, planner snapshot and hash manifest. 320 planned runs:
eight geography holdouts x five seeds x eight methods (four neural and four
conventional). Seeds: 123, 2024, 3407, 9999, 2718. Deterministic references
repeated over seeds are not independent observations; regional folds share
training data and seeds do not generate new regional datasets.

Neural LR and algorithm-specific rounds are the existing inner selections.
Final refit uses the last fixed round, never outer-test checkpoint selection.
Negative predictions retained, undefined R2 explicit; equal-group RMSE and
pooled errors distinguished. Support panels must be training-derived. Budget
edge limitations and earlier seed123 diagnostic exposure remain disclosed.

Actual commands from repository root:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_outer_protocol -v
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_protocol --output paper2/results/restart_outer_protocol_v1
```

Actual outcomes: five tests passed, zero failures/errors; freeze returned
FROZEN_PLAN_ONLY_NO_OUTER_INFERENCE, planned_runs=320. Tests cover counts,
unique IDs, selected budget preservation, bad seeds, missing folds and invalid
rounds. They do not certify a runner that does not yet exist.

Next implement separate runner code, bind that code prospectively before any
study execution, test exact interrupted resume, malformed input rejection,
writer exclusion, failure journaling, cache integrity and ambiguous inference
receipts. Run synthetic integration smoke before real outer evaluation.
Do not treat this plan-only package as authorization to skip runner verification.
KG, ranking, privacy and remote deployment remain gated separately.
