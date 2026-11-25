# Outer batch coordinator implementation

2026-10-13. R04 remains IN_PROGRESS pending independent edge-case audit and
persistent verification evidence before real study execution.

Added restart/outer_runner.py and test_outer_runner.py. Separate execution
package results/restart_outer_execution_v1 binds planner, training core,
coordinator, regression client, metric/hash utilities and aggregation source
snapshots to the existing frozen 320-run plan and runtime. No prior frozen code
or raw data changed. Source edits now require a new prospective package version.

Coordinator defaults to dry-run and one-run cap. Actual execution requires
--execute; existing outputs require --resume. All arrays/partitions and source
hashes checked; train/test IDs must match, validation arrays forbidden. Study
writer lock excludes concurrent execution. Failures save traceback/attempt and
stop; retry requires explicit --retry-failed and cannot override ambiguous test
inference. Completed cached predictions verified without model inference.
Support context retained and checked against partition metadata on continuation.

Actual commands:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_outer_protocol paper2.restart.test_outer_training paper2.restart.test_outer_runner -v
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_runner --freeze --output paper2/results/restart_outer_execution_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_runner --output paper2/results/restart_outer_study_v1 --max-runs 8
```

Combined suite: 19 passed, zero failures/errors after correcting a test fixture
path. First attempt had one error because the fixture was outside the results
root; production guard correctly rejected it. Nonfatal joblib NumPy deprecation
continues. Freeze returned FROZEN_EXECUTION_CODE_NO_OUTER_INFERENCE. No real
training/inference performed. Synthetic coordinator tests cover cap/cache resume,
failure stop/retry journaling, changed assets/context, ambiguous inference,
writer exclusion and no-write dry-run.

Next: independent code/package binding and receipt edge-case review; persist
verification logs; implement independent cached-output audit and aggregate
metrics including supported panels and undefined R2. Only then execute a bounded
real starter batch. No test-score-based tuning or automatic budget amendment.
