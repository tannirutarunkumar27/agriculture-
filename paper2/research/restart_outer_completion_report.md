# Frozen outer-study execution completed

Date: 2026-10-13. All remaining 240 runs completed successfully. Final total:
**320/320 complete, zero pending, zero failure files**. The attempt journals
contain exactly 320 records, all COMPLETE, one attempt per run.

Coverage: eight geographic outer groups x five seeds x eight methods. Groups:
Angers, Auzeville, Copenhagen+Taastrup, Grignon, Jyndevad, Kassel, Reading and
Sanmarcoargentano. Methods: centralized MLP, FedAvg, FedProx, FedNova, random
forest, ridge, training mean and XGBoost. Seeds: 123, 2024, 3407, 9999, 2718.

Actual commands from repository root:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_runner --execute --resume --max-runs 240 --output paper2/results/restart_outer_study_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_outer_study --output paper2/results/restart_outer_verified_v3
```

Runner returned COMPLETE, completed=320, planned=320, pending=0 and
executed_this_invocation=240. Independent audit returned PASS_COMPLETE,
verified=320, planned=320, pending=0 and model_reinference=false. Audit checks
frozen source/runtime/data bindings, model and receipt hashes, cached targets
and IDs, support context and independently calculated metrics. Per-run R2 and
training-supported species/composition metrics are included in verification.json.
No model was loaded for repeated inference during this audit.

Final evidence:

- ../results/restart_outer_study_v1/progress.json
- ../results/restart_outer_study_v1/runs/ (models, predictions, receipts, journals)
- ../results/restart_outer_verified_v3/verification.json
- ../results/restart_outer_execution_v1/ (unchanged frozen execution package)

No source changes, hyperparameter changes, dependency installations or test
reruns were needed during this execution phase. The previously saved scoped
suite has 22 passing tests (restart_outer_tests_v2). Existing Paper 1, raw data
and manuscript were preserved.

Execution is verified complete. Matched-group/seed aggregation, comparative
interpretation and figures remain separate work. Report equal-group versus
pooled errors separately, account for correlated folds and deterministic
reference repetitions, and retain support panels. Seed repeats are not new
independent regions. Fixed-budget convergence limitations remain documented;
do not tune from the now-exposed outer results. Knowledge guidance, ranking,
privacy and field deployment are not established by these baseline runs.
