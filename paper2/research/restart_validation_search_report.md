# Restart: frozen regression validation-search implementation

Historical starter-batch report. All 1,120 candidates subsequently completed;
see `restart_validation_completion_report.md` for current verified status.

Date: 2026-10-13. The bounded/resumable search is implemented, tested and frozen.
**Execution is partial: 20/1,120 candidates complete, 1,100 pending, zero observed
failures. No hyperparameter selection or outer-test evaluation is complete.**
Existing classification work, raw downloads, prepared splits, Paper 1 and the
open manuscript were preserved. No dependencies were installed or upgraded.

## Fixed scope and budget

This stage tunes baselines using only the new dataset's 56 INNER folds. The
eight OUTER test groups are not inferred on. Inner files are required to contain
train/validation only; unexpected/test arrays reject before training. Source,
preparation, fold and implementation identities are checked before execution.

Each inner fold has 20 candidates:

- Twelve neural: centralized MLP, FedAvg, sample-weighted FedProx and restricted
  plain-SGD FedNova, each at LR .0003/.001/.003; hidden32, batch32, 80 rounds,
  one local epoch, full participation, CPU one thread, MSE. FedProx mu=.01.
- Eight conventional: training mean; ridge alpha .1/1/10; Random Forest 100
  trees with minimum leaf1/3; XGBoost 100 boosting rounds, depth2/3, LR=.05.
- Tuning RNG seed42; future study seeds are separately specified. Existing
  seed123 smoke exposure is disclosed: this remains development-data evidence.

Totals: 672 neural + 448 conventional = **1,120 validation candidates**.
No outer test scores are used to pick a learning rate, checkpoint, model, grid
or budget. Fixed 80-round caps do not certify convergence of every case.
Retain validation curves and document budget limitations; amendments need a
new version, not silent changes to this frozen plan.

## Selection and future outer refit

Neural checkpoints use minimum inner-validation RMSE, first-round tie wins;
every configured round is still executed. At search completion, each outer
fold selects ONE common LR by averaging validation RMSE equally across its
seven inner geographic folds and the four neural algorithms. This shared
configuration is an explicit comparison choice, not separately optimized
performance for each method.

For each algorithm, outer-refit duration is the median selected round across
the seven inner folds at that common LR. Reference settings are chosen
independently by their equal-inner-group mean RMSE. Exact configuration ties
select the first candidate in frozen order. The routine refuses incomplete
results: no favorable subset or premature per-outer selection is produced.

These rules prepare a later outer-refit/evaluation stage. This implementation
does NOT yet execute that stage, evaluate rankings, enable knowledge, or provide
privacy guarantees. Final-study orchestration needs independent study seeds,
model/test receipts, outer prediction caches and resource accounting.

## Reliability and scientific boundaries

Default dry-run creates no execution output. Training requires `--execute` and
has a per-invocation cap (default1). Existing output requires `--resume` with
the same frozen identity. OS writer locking excludes concurrent writers.
Neural round checkpoints atomically record last and selected state/history.
Round/client RNG is deterministic; interrupted/resumed weights matched exactly
in a synthetic test. A final-round checkpoint can finalize without retraining.

Completed results are checked from cached validation predictions and model/file
hashes on resume, rather than trained again. Failures stop the batch, retain
attempt-specific errors/journal entries and require inspection plus explicit
`--retry-failed`. Never delete a failure or change settings to improve a score.
Validation re-inference is permitted for verification; outer-test inference is
not. Hashes are integrity bindings, not cryptographic authentication of clients.

The public-data simulator still uses centralized training-only preprocessing.
Logical research-site clients are not isolated farms. FedNova supports only
constant shared LR plain SGD; the original checked aggregation is unchanged.
Raw crop treatments are retrospective conditional descriptors, not certified
pre-sowing-known values. No best-crop labels, physiological priors or economic
benefits are inferred from this search.

## Executed checks and first real batch

- Initial 13 search tests passed; final expanded suite **14 passed, zero
  failures/errors**, including real frozen-plan tamper rejection without training.
  Evidence: `results/restart_search_tests_v1` and `restart_search_tests_v2`.
- Tests cover metric units/nonfinite inputs, unsafe config, exact checkpoint
  recovery, cached tampering, no-retraining finalization, outer-array rejection,
  no-write dry-run, bounded resume, overwrite rejection, explicit failure retry,
  writer exclusion, incomplete selection and tie/refit rules. Test observations
  are synthetic; they are not agricultural experiment scores.
- Frozen real plan created at `results/restart_validation_protocol_v1/` with
  source snapshots, settings, plan, prepared-data identity and runtime versions.
- Real dry-run confirmed 1,120 planned, 20-run cap, no training/output creation.
- Executed **20 real candidates**: all configurations on fixed
  `outer_0_inner_1` (178 training systems, 354 validation systems at Auzeville).
  The outer Angers group remains uninferred. All finished, zero failures.
- Independent verification reproduced all 20 selected local models' validation
  metrics and checked neural checkpoint-selection histories: **PASS_PARTIAL**.
  Metric tolerance rtol=atol=1e-7. No outer-test inference during verification.
- No `selected_configurations.json` should exist while 1,100 cases remain.
  Do not interpret one completed inner fold as the full search or superiority.

## Files and commands

New config: `configs/restart_validation_search_v1.json`.
New modules/entry points: `restart/validation_search.py`,
`restart/run_validation_search.py`, `restart/verify_validation_search.py`,
`restart/test_validation_search.py`. Existing regression client and aggregation
were reused unchanged and included in frozen source identities.

Executed from repo root:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/test_validation_search.py --output paper2/results/restart_search_tests_v1
& '.\.venv\Scripts\python.exe' -B paper2/restart/test_validation_search.py --output paper2/results/restart_search_tests_v2
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_validation_search.py --freeze
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_validation_search.py --output paper2/results/restart_validation_search_v1 --max-runs 20
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_validation_search.py --output paper2/results/restart_validation_search_v1 --max-runs 20 --execute
& '.\.venv\Scripts\python.exe' -B paper2/restart/verify_validation_search.py --output paper2/results/restart_validation_verified_v1
```

All commands exited successfully. `progress.json` under the execution folder
is the current completion ledger. Frozen package describes settings, not a
claim that all search training is done. Verification output refuses overwrite.

## Next concrete action

Continue the remaining frozen candidates in bounded batches, checking failures
after each successful invocation:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_validation_search.py --output paper2/results/restart_validation_search_v1 --max-runs 40 --execute --resume
```

This continuation command has NOT yet been executed in this report. Do not
alter frozen sources, protocol, preparation artifacts or dependency versions.
After all 1,120 complete, verify against selected models into a NEW verification
directory and review curves/selected configurations before implementing outer
refit and test evaluation. KG, ranking, privacy and final manuscript remain later.
