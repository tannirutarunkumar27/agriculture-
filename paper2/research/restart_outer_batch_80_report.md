# Outer-study continuation to 80 verified runs

2026-10-13. Executed 72 additional frozen runs, retaining the first eight.
Total 80/320 complete, 240 pending, zero failure files. Completed Angers and
Auzeville outer groups across all five seeds and eight methods.

Actual commands:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_runner --execute --resume --max-runs 72 --output paper2/results/restart_outer_study_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_outer_study --output paper2/results/restart_outer_verified_v2
```

Runner: PARTIAL, completed=80, pending=240, executed_this_invocation=72.
Independent audit: PASS_PARTIAL, verified=80, model_reinference=false. All
frozen code/runtime/data/settings checks passed during execution and audit.
No new tests were run this phase; earlier scoped suite remains 22 passed.
No model/code/data edits, installations, guidance or privacy experiments.

Next: continue remaining six groups in bounded resume batches, audit completed
caches and then implement complete matched-group/seed summaries. Exposed outer
scores must not be used to adjust frozen settings. Two-group completion is not
final cross-region evidence; unresolved acceptance tests remain in R04 ledger.
