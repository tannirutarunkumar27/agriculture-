# R13 bounded runner and audited starter

2026-10-14. Execution-software subtask VERIFIED_COMPLETE; overall R13 remains
IN_PROGRESS. Five of40 frozen unnoised refit runs completed,35 pending.
Active objective is retrospective conditional harvested seed dry matter,t/ha.

## Inspection and implementation

Inspected current ledgers, actual frozen protocol/training and prior audit/test
evidence, Git state and active Python processes. No prior writer was active.
Earlier224 inner results remain complete; parent protocol verification includes
strict provenance and unchanged source/data/runtime checks. No frozen source
or archived result was edited or retrained.

New modules:public_refit_runner.py, public_refit_inference.py,
verify_public_refit.py and test_public_refit_runner.py. Separate execution
snapshot/identity:results/restart_public_refit_execution_v1, binding the existing
restart_public_refit_protocol_v1.40 planned runs/seeds/budgets are unchanged.

Runner default is non-mutating dry-run. Execution requires positive cap and
explicit existing-output resume, exclusive OS writer lock, separate paths and
validated bindings/inventory. Training accesses only frozen training tensors.
Test arrays are loaded for a new run only after fixed-budget fit finalization.
Raw model prediction multiplies by the declared target scale exactly once;
no prediction clipping or outer-test checkpoint selection occurs.

Intent is committed before model loading/test prediction. Completion receipts
bind fit, model, round checkpoint, support context, intent, raw prediction cache
and result. Completed caches verify without model forward calls/retraining.
Unknown test-inference intent without completion is fail-closed, even if retry
is requested. Interrupted training can recover exact checkpoints; failed
training requires explicit retry after inspection. Failure traces and attempt
identities/status/receipt hashes are retained, including valid completion-cache
recovery after a postcommit failure. Missing journals/orphaned assets reject.

Added explicit tensor shape/dtype/finite-state checks before partial checkpoint
recovery and final inference. PyTorch loading/comparison must not silently cast
altered checkpoint states; a float64-corrupted fixture is rejected before fit.
This is wrapper hardening, not a change to the frozen training method.

Independent cached audit recomputes RMSE/MAE/R2/negative predictions in float64,
verifies raw targets/row IDs and both seen/unseen species/composition panels,
including undefined-R2 and empty-panel handling. It performs no model forward
calls. Final complete-study equal-group versus pooled aggregation remains a
separate next analysis step using the existing tested aggregation.

## Actual tests, errors and execution

Final scoped suites61 passed, zero final errors/failures:
17 new runner/inference/cache tests at restart_public_refit_runner_tests_v3;
18 preparation/core tests at restart_public_refit_tests_v5;
26 dependency tests at restart_public_refit_dependency_tests_v2. Not a full
repository suite and not agronomic/privacy certification.

Initial runner_tests_v1 ran15 tests and retained one failure: an altered support
fixture was correctly rejected by the stronger fit signature before the later
context guard, but the assertion expected only the word binding. Corrected
the assertion to accept signature/binding, without weakening production guards.
v2's16 and final v3's17 tests passed; checkpoint schema and pre-fit dtype tests
were added. No actual training/inference failures occurred.

Freeze and dry-run passed. Explicit --execute --cap5 completed five new runs,
one COMPLETE attempt each. Audit PASS_PARTIAL_REFIT saved at
results/restart_public_refit_verified_v1/verification.json. All5 caches, receipts
and support panels pass without repeated model test inference. No writer remains.

All starter cases hold Angers out (75 test systems), fixed scale1/LR .01 and
133 refit rounds. These are development results on one region, not final
multi-region comparative evidence or five independent regional replications.

| Seed | RMSE,t/ha | MAE,t/ha | R2 |
| ---: | ---: | ---: | ---: |
| 123 | 1.646856 | 1.336460 | 0.095110 |
| 2024 | 1.662858 | 1.362851 | 0.077440 |
| 3407 | 1.704643 | 1.378887 | 0.030492 |
| 9999 | 1.684336 | 1.364983 | 0.053454 |
| 2718 | 1.587332 | 1.296413 | 0.159340 |

These observed scores are retained, not used to alter learning rate, budget or
checkpoint selection. No full-study superiority, privacy or field-benefit claim.

## Commands actually executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit_runner --output paper2/results/restart_public_refit_runner_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit_runner --output paper2/results/restart_public_refit_runner_tests_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit --output paper2/results/restart_public_refit_tests_v5
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_work_scaled_simulation --output paper2/results/restart_public_refit_dependency_tests_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit_runner --output paper2/results/restart_public_refit_runner_tests_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --freeze --output paper2/results/restart_public_refit_execution_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --output paper2/results/restart_public_refit_study_v1 --cap 5
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --output paper2/results/restart_public_refit_study_v1 --execute --cap 5
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_public_refit --output paper2/results/restart_public_refit_verified_v1
```

Authored changes:four new modules/tests, six project ledgers and this report.
Generated:new test evidence, execution freeze, five run folders/progress and
audit. No global/environment dependency installation, cloud cost, credentials,
network service, automation change, Paper1/raw-data/open LaTeX modification or
historical-result replacement. The new frozen execution sources must now remain
unchanged; future changes need a new prospective package.

## Next action and gates

Resume remaining35 runs with explicit bounded --execute --resume --cap N;
inspect active writers/failures first, and audit each batch to a new output.
Example next commands (not executed in this phase):

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --output paper2/results/restart_public_refit_study_v1 --execute --resume --cap 10
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_public_refit --output paper2/results/restart_public_refit_verified_v2
```

After all40 verify, aggregate all eight geographic groups/five seeds and retain
support panels/equal-group versus pooled distinctions. Existing outer outcomes
were already exposed; follow-up remains development. Grid-edge/cap/underfit
findings are not erased by passing software tests. Matched noise mechanisms,
release accounting and privacy claims require a separate prospective freeze;
this unnoised package provides none. Agronomic rules/ranking comparability and
remote/certified privacy/SecAgg gates remain as documented.
