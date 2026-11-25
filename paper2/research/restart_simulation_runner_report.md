# Matched simulation runner, recovery and stability amendment

2026-10-14. R08 remains IN_PROGRESS. The execution/ledger stage is implemented
and starter runs audited. This is public-data simulation, with no application
privacy certification or secure aggregation.

## Runner and accounting

Added equal-slot control averaging, whole-vector clipped controls and the
simulated Gaussian variant. Every training round atomically checkpoints the
global model and its round-tagged event list. Exact recovery restores both;
distinct deterministic streams are derived per version/run/round. These are
simulation seeds, not private randomness. Local shuffling is separate.

The shared ledger is reconstructed from verified checkpoints, keyed by unique
event IDs; it rejects altered/removed historical events. It composes events
across overlapping folds, seeds and noisy variants for each public client slot.
Unnoised controls have null privacy budgets; they are never labeled epsilon=0.
The runner checks frozen sources/runtime/data, caps new runs, requires explicit
resume and OS writer lock, journals failures and rejects ambiguous inference.
Completed predictions are checked from cache. Final fit receipts bind model and
accounting hashes, while independent audit checks final checkpoint equality,
events, shared budgets and cached IDs/targets/errors without model reinference.

## Observed failure and documented fix

The v1 synthetic suite passed 23 tests. Its first real batch completed three
Angers/seed123 runs (equal average, clipped average and sigma2). The sigma4
run committed round47 and then stopped on non-finite local regression loss.
Failure traceback and checkpoints are retained under restart_privacy_simulation_v1.
Independent audit PASS_PARTIAL verified the three completed runs and all 125
noisy events (78 sigma2 + 47 sigma4). No sigma8 run was started in v1.

This is local optimization instability after large simulated noise perturbs
the global state. The separately versioned v2 local client clips the gradient's
global L2 norm to 1 at each SGD step. The same stabilization applies to all five
variants. LR, round budgets, model capacity, clipping C=1 and noise multipliers
are unchanged. No outcomes were removed or silently substituted.

stable_regression_training.py and stable_simulation_training.py implement the
amendment. stable_privacy_simulation_runner.py freezes new code separately and
uses new random streams. amend_privacy_stability.py preserves the failure hash
and prior event ledger in restart_privacy_study_protocol_v2. The shared v2
ledger includes v1 events with version-prefixed IDs, preventing both omission
and collisions when version2 reuses the same public fold/seed identifiers.

## Actual acceptance and starter outcomes

- V2 suite: 24 tests passed, zero failures/errors. Includes exact recovery for
  all variants, capped integration, duplicate/tampered event rejection, completed
  model reuse, and a regression test showing bounded local steps from a large
  starting model. Logs: restart_stable_simulation_tests_v1.
- V2 first five runs completed successfully. An explicit cap1 resume verified
  the completed entries and trained the next equal-average seed2024 control.
  Final status: 6/200 complete, 194 pending, zero v2 failure files.
- Audit PASS_PARTIAL at restart_privacy_verified_v3: all six model/receipt/cache
  records verified without reinference. Ledger: 359 events = 125 prior-version
  events + 234 new noisy events (three variants x 78 rounds).

For Angers/seed123, v2 RMSE is about 4.16 t/ha for both controls, 203.27 for
sigma2, 522.91 for sigma4 and 4650.03 for sigma8. Numerical stabilization
prevents the observed loss overflow, but does not establish useful accuracy.
These unfavorable results remain recorded. Full cross-region comparisons are
incomplete; no positive privacy-utility finding or superiority is claimed.
Existing outcomes and the failure-informed amendment make this development
work, not new independent confirmation. No settings were tuned to test scores.

## Actual commands

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_simulation_runner --output paper2/results/restart_simulation_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.privacy_simulation_runner --freeze --output paper2/results/restart_privacy_execution_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.privacy_simulation_runner --execute --cap 5 --output paper2/results/restart_privacy_simulation_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_privacy_simulation --output paper2/results/restart_privacy_verified_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_stable_simulation --output paper2/results/restart_stable_simulation_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.amend_privacy_stability --output paper2/results/restart_privacy_study_protocol_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.stable_privacy_simulation_runner --freeze --output paper2/results/restart_privacy_execution_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.stable_privacy_simulation_runner --execute --cap 5 --output paper2/results/restart_privacy_simulation_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_stable_privacy_simulation --output paper2/results/restart_privacy_verified_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.stable_privacy_simulation_runner --execute --cap 1 --resume --output paper2/results/restart_privacy_simulation_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_stable_privacy_simulation --output paper2/results/restart_privacy_verified_v3
```

Next continue the fixed v2 matrix in capped batches with audits, then summarize
matched errors and simulated budgets. Keep v1 closed with its recorded failure;
do not retry its unstable trajectory or alter frozen code. No new dependency,
raw-data change, baseline overwrite, Paper 1 or manuscript modification occurred.
