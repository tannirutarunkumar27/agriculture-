# New-dataset preparation and first training diagnostic

Date: 2026-10-13. Actual splits and preprocessing CREATED and VERIFIED; new-data
validation-only smoke EXECUTED. No final outer-test inference or research study.
Prior 235 classification runs are historical and not used as new-study scores.

## Completed evidence

- 607 system-level regression rows rebuilt from audited raw harvest outcomes.
- Nine original site labels become eight geographic groups: Copenhagen and
  Taastrup merged by the declared 1-km coordinate policy (about 0.37 km apart).
- Eight outer refit partitions, 56 inner training/validation partitions.
- All 64 independently verified: source/code/array hashes, row coverage,
  site/experiment/geographic disjointness, client exclusion, training-only
  numerical/category fits, support flags and reconstructed arrays.
- Each outer test row occurs exactly once across the outer refit partitions.
  Inner NPZ files do not contain outer test arrays.
- Ten data-preparation tests passed, zero failures/errors. Five new regression
  client tests passed, zero failures/errors. These 15 tests are separately
  scoped, not a claim that all historical 96 tests were rerun.
- Eight validation-only baseline smoke configurations completed on the fixed
  `outer_0_inner_1` fold: training mean, ridge, Random Forest regressor,
  XGBoost regressor, centralized MLP, FedAvg, FedProx and restricted FedNova.
  Training: 178 systems on six groups. Validation: 354 Auzeville systems.
  Angers outer holdout: 75 systems, NOT inferred on.
- Neural smoke: three rounds, one local epoch, hidden 32, batch 32, LR .001,
  plain SGD, FedProx mu .01, full participation, seed 123. These settings were
  recorded before execution and are diagnostic, not validated regression settings.
  Selected neural weights/histories and validation metrics saved.
- Source data and old frozen baseline package unchanged; old verification PASS.
  No rule approval, formal privacy or best-crop labels created.

## Files and saved artifacts

New modules under `paper2/restart/`: data_pipeline.py, prepare_regression.py,
verify_regression_data.py, test_data_pipeline.py, regression_training.py,
run_regression_smoke.py and test_regression_training.py. Established checked
aggregation is reused unchanged; classification loss/training was not reused.

`results/restart_regression_data_v1/`: row table, coordinate-distance review,
split summary, each fold's exact IDs/client map/preprocessor/NPZ and manifest.
`results/restart_regression_verified_v1/verification.json`: 64-fold PASS.
`results/restart_preparation_tests_v1/`: ten-test execution log/status.
`results/restart_regression_client_tests_v1/`: five-test execution log/status.
`results/restart_regression_smoke_v1/`: pre-execution manifest, validation-only
diagnostics and selected neural weights (not resumable study checkpoints).

The source metadata sand/silt descriptions appear swapped. Those fields are
excluded pending clarification; raw files are not edited. No climate workbook
or source PDF was rewritten. All numerical unit assumptions and exclusions are
documented in `restart_implementation_plan.md`.

## Commands actually executed

```powershell
& 'C:/Users/tanne/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B paper2/restart/prepare_regression.py --output paper2/results/restart_regression_data_v1
& 'C:/Users/tanne/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B paper2/restart/test_data_pipeline.py --output paper2/results/restart_preparation_tests_v1
& 'C:/Users/tanne/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B paper2/restart/verify_regression_data.py --data paper2/results/restart_regression_data_v1 --output paper2/results/restart_regression_verified_v1
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_regression_smoke.py --data paper2/results/restart_regression_data_v1 --verification paper2/results/restart_regression_verified_v1/verification.json --output paper2/results/restart_regression_smoke_v1
& '.\.venv\Scripts\python.exe' -B paper2/restart/test_regression_training.py --output paper2/results/restart_regression_client_tests_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

All commands exited successfully. Output creation refuses overwrite. Bundled
Python was used for source extraction/preparation; the existing repo environment
was used for Torch/sklearn/XGBoost smoke. Dependencies were not installed or
upgraded. No failures were observed in these executions.

## Interpretation and remaining work

The smoke is a wiring/finite-loss/aggregation diagnostic, NOT converged or
repeated-seed research evidence. The selected validation group contains three
species absent from its training groups; strong interpretation must distinguish
support. Existing default regressors and three-round MLPs do not establish
superiority. Negative predictions are retained; no metrics are silently clipped.

Data preparation does not finalize physiological rules, comparability, utilities,
privacy or production deployment. The initial task draft remains historical;
the new implementation plan explicitly amends its nine-site scheme to eight
geographic groups. Full-study training remains disabled until a new bounded
validation-search/refit/budget protocol and safe experiment runner are frozen.
The old `training_enabled=false` draft is not a statement that this separately
authorized engineering smoke never executed.

Next: implement and freeze a bounded validation-only regression convergence
search and resumable study coordinator using THESE saved splits, then execute
new baseline experiments. Keep KG/ranking/privacy off until their separate gates
are supported. Preserve the old classification study rather than deleting it.
