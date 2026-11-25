# R12 second bounded control-search batch

2026-10-14. R12b bounded execution/audit subtask VERIFIED_COMPLETE. Overall
R12 remains IN_PROGRESS:24/224 candidates complete,200 pending. Active task
is conditional crop-system harvested seed dry-matter regression. The current
study is development evidence, not independent confirmation or crop ranking.

## Acceptance and dependencies

Dependencies: unchanged restart_control_search_protocol_v1,
restart_control_search_execution_v1, restart_public_data_v1 and frozen
candidate engine. Read the six project ledgers and user autonomous instructions,
inspected actual runner/auditor and Git state, and checked for Python writers.
None was active before starting; the runner also acquires its OS writer lock.
Code, source snapshots, runtime, plan/config and public-data identities passed
verification before execution and again during the independent audit.

Acceptance: fresh relevant tests pass; explicit capped resume reuses completed
runs; all completed candidates have valid receipts/checkpoints and independently
reproducible final/best inner metrics; failures/attempts are inspected; settings
are unchanged; incomplete search cannot select final settings. All passed for
this batch. Frozen sources/packages were not edited or regenerated.

## Actual execution and verification

- Fresh scoped suite:17 tests passed, zero failures/errors. Persistent evidence:
  results/restart_control_search_tests_v3. The separate26 dependency tests from
  the prior phase were not rerun; their evidence is retained, not recounted as
  new tests in this session. No claim that the entire repository suite was run.
- Executed20 additional candidates using the existing output with explicit
  --execute --resume --cap20. Total24 completed,200 pending.
- Independent audit PASS_PARTIAL_INNER_SEARCH verifies all24 receipt asset
  hashes, full histories, first-tie minima, budget reports, final/best validation
  checkpoints and equal-client training metrics. Evidence:
  results/restart_control_search_verified_v2/verification.json.
- Exactly24 attempt records, all COMPLETE, one per candidate; zero FAILED or
  active STARTED attempts after completion. The four original completion receipt
  hashes match the preceding audit, confirming they were not replaced/retrained.
- No selection.json exists. No outer-model inference or noisy training occurred.
  No active Python writer remained after the batch.

Coverage:four configurations each on outer_0_inner_1 through outer_0_inner_6,
holding Angers out and validating respectively on Auzeville,
Copenhagen+Taastrup, Grignon, Jyndevad, Kassel and Reading. This is not yet
complete even for the first outer group's seven inner folds. It is not a
repeated-seed or full eight-group comparison. Ten of24 candidates select the
240-round cap; fixed-budget minima do not establish convergence. Target scales,
learning rates, seed and budget remain as originally frozen. No interim
hyperparameters or pooled/equal-group superiority claims are reported.

## Commands executed

From the repository root with the existing isolated environment:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_search_runner --output paper2/results/restart_control_search_tests_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --resume --cap 20
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v2
```

Read-only PowerShell checks additionally inspected attempt statuses, fold
coverage, budget-cap counts, selection absence and original completion hashes.
No errors needed repair in this phase. No packages installed, raw data changed,
previous results overwritten, Paper1/open LaTeX edited or automation duplicated.
Authored changes are the six progress ledgers and this report; generated changes
are20 new run folders, updated study progress, new test logs and a new audit.

## Next action and gates

Continue the remaining200 frozen candidates in bounded batches; use a new audit
directory each time. Example next commands, not executed in this phase:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --resume --cap 20
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v3
```

Only complete verified224-run selection can supply equal-inner-group means,
frozen-order ties and median-round prospective refits. Do not tune from exposed
outer scores or change the grid/budget silently. Any further refit/noise study
needs a separately frozen protocol. Prior baseline/guard/DP simulation outcomes
remain exposed development results. Independent agronomic guidance, crop ranking,
release-certified privacy/SecAgg and remote deployment gates remain unchanged.
