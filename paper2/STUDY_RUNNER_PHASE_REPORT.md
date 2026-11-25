# Step 2 implementation and verification report

Date: 2026-10-12. Scope: bounded/resumable frozen-plan runner, not the complete
235-run study or privacy/knowledge-guided implementation.

## Files created

- `study_runner.py`: execution controls, kernel writer lock, frozen ownership,
  training receipts, checkpoint recovery, evaluation journal, cached outputs,
  completed-run verification and fail-closed error records.
- `scripts/run_baseline_study.py`: dry-run-by-default CLI, explicit execution,
  run-ID selection, per-invocation budget, resume and inspected failure retry.
- `tests/test_study_runner.py`: fourteen orchestration/recovery tests.
- `research/study_runner_guide.md`: usage, output layout and recovery rules.
- Saved outputs: `results/tests_v6/` and `results/baseline_study_v1/`.

## Verified behavior

Fresh full suite: **70 passed, zero failed**. Tests exercise unfinished-round
resume, completed-round recovery without retraining, cached-bundle recovery
without repeated inference, ambiguous test interruption blocking, corruption
rejection, explicit retry and second-writer exclusion. Reduced-budget test
fixtures stub only frozen-package verification to permit short toy budgets;
actual training/evaluation/recovery code runs on real source rows. This is
separate from the unchanged REAL frozen-plan smoke run below.

The installed joblib/NumPy combination emits a nonfatal shape-setting
DeprecationWarning in the conventional-model test. It is preserved in the log;
no dependency upgrade or warning suppression was used to conceal a failure.
Models/metrics verified successfully under frozen versions. Any later
dependency amendment must be separately tested/versioned.

The existing trainer cannot resume when saved completed rounds equal target
rounds. Rather than modify the frozen trainer, the wrapper handles
post-training finalization from its checkpoint. Errors are neither hidden nor
silently skipped. Injected failures are clearly synthetic TEST fixtures, not
claimed real production incidents.

## Actual bounded frozen-plan execution

Four entries were chosen before execution and ran with unchanged frozen
settings, not reduced smoke budgets:

1. `canonical_seed123_iid_full_fedavg`: 80 rounds.
2. `canonical_seed123_class_skew_moderate_fedprox`: 80 rounds.
3. `maharashtra_seed123_reference_xgboost`: 200 boosting rounds at fixed selected settings.

Terminology correction verified during step 3: with multiclass `multi:softprob`,
200 boosting rounds produce 3,200 class-specific trees for Maharashtra's 16
classes (and 4,400 for the canonical 22 classes), not 200 total trees. The actual
frozen `n_estimators=200` setting was never changed.
4. `maharashtra_seed123_district_full_fednova`: 80 rounds.

All four completed. The repeat invocation with `--resume` verified completed
receipts/cached metrics and executed **zero** new runs, without model test
inference. Frozen-package verification remained PASS after real execution.

Study progress: **4/235 completed, 231 pending, zero failed runs**. These are
development-holdout outcomes, not a complete repeated-seed result or independent
confirmation. No statistical superiority/convergence/field benefit is claimed.
The full study has NOT been launched.

## Executed commands

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v6
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --run-id canonical_seed123_iid_full_fedavg --run-id canonical_seed123_class_skew_moderate_fedprox --run-id maharashtra_seed123_district_full_fednova --run-id maharashtra_seed123_reference_xgboost --max-runs 4
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --run-id canonical_seed123_iid_full_fedavg --run-id canonical_seed123_class_skew_moderate_fedprox --run-id maharashtra_seed123_district_full_fednova --run-id maharashtra_seed123_reference_xgboost --max-runs 4 --execute
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --run-id canonical_seed123_iid_full_fedavg --run-id canonical_seed123_class_skew_moderate_fedprox --run-id maharashtra_seed123_district_full_fednova --run-id maharashtra_seed123_reference_xgboost --max-runs 4 --execute --resume
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

## Limitations and next step

At-most-one automatic evaluation attempt is conservative: uncommitted test
inference after a crash is ambiguous and blocks automatic retries. Locally
generated joblib files are trusted only after hash checks; untrusted pickles
must never be loaded. Integrity checks/locking are not cryptographic
authentication, SecAgg or DP. Simulation uses public global train-only scaling.
Source provenance limitations and earlier development test exposure persist.

Next: execute remaining frozen runs in bounded batches, checking failures
and artifacts between batches. Aggregate only matching complete seed/scenario
groups; do not call four single-seed smoke entries a full comparative study.
Knowledge and formal privacy stay disabled pending their separate gates.
