# Training-support safeguard and sensitivity experiment

Date: 2026-10-14 (Asia/Calcutta). Task R11 VERIFIED_COMPLETE for the software
fix and specified post-hoc sensitivity experiment. Scientific confirmation
on independent data remains open.

## Defect and implemented policy

In the Auzeville outer fold, standardized soil_awc and P_amount are zero for
all training rows but reach 263 and 66 respectively on held-out rows. A neural
input connection receiving only zeros has no supervised gradient. Saved weight
diagnostics show these incoming weights remain essentially at initialization.
Unseen categories/species can activate the same type of unsupported connection.

support_guard.py fits a Boolean mask exclusively from x_train. Every column
identically zero in training is forced to zero for inference. This includes
numeric values, missing/category indicators and species flags. No threshold is
chosen from outcomes, and varying supported features are left unchanged.
Nonzero constant training columns are retained. This is a computational support
policy, not an agronomic interpretation or evidence that unseen species are
predictable. Metadata/support panels still record unseen status.

For all eight folds, guarded training matrices equal the original matrices
exactly. The sensitivity experiment therefore reuses the hash-verified original
model weights, retaining their complete training configuration and history.
There is no retraining, LR search, budget change or output clipping. A separate
frozen package, source snapshots, gate hashes, model provenance, inference
intent and cached completion receipts distinguish the new experiment.

## Verified execution

- Seven safeguard tests passed, with logs in restart_guard_tests_v2.
  Checks include unchanged training input, unsupported-value suppression,
  zero-gradient mechanism, invalid arrays and frozen gate/snapshot tampering.
- Eight starter evaluations succeeded, then all remaining 312 succeeded.
  Total 320/320 complete, zero failure files.
- Cached-output analysis verified parent model hashes, new receipts, cached
  target/ID equality, support context, model provenance and complete eight-fold
  coverage. It performs no further inference.
- A final explicit resume executed zero new evaluations and verified all 320
  completed entries. Both baseline and sensitivity results remain available.

## Comparison (RMSE in t/ha)

| Method | Original equal-group RMSE | Guarded equal-group RMSE +/- seed SD | Guarded pooled RMSE |
| --- | ---: | ---: | ---: |
| Centralized MLP | 6.0307 | 1.2130 +/- 0.0196 | 1.4527 |
| FedAvg | 5.8373 | 1.3438 +/- 0.0295 | 1.3697 |
| FedProx | 5.8373 | 1.3438 +/- 0.0295 | 1.3697 |
| FedNova | 5.8134 | 1.1820 +/- 0.0204 | 1.3632 |
| Random Forest | 1.3181 | 1.3181 +/- 0.0091 | 1.9958 |
| Ridge | 1.4694 | 1.4694 +/- 0.0000 | 2.1511 |
| Training mean | 1.4509 | 1.4509 +/- 0.0000 | 1.9379 |
| XGBoost | 1.2461 | 1.2461 +/- 0.0000 | 1.7335 |

FedAvg seed123 on Auzeville changes from RMSE 36.7482 to 0.9960, with MAE
0.8358 and R2 0.3405. This quantifies the combined support-gate effect; it does
not isolate the causal contribution of each masked feature. Seen/unseen-species
panels, per-region metrics and per-seed pooled MAE/R2 are saved in analysis.json.
All 265 unseen-species cases belong to Auzeville, so this panel supplies only
one unseen-species region. Seed SD is initialization variability, not a regional
confidence interval. The gate also removes unlearned species inputs.

The policy was motivated by already-exposed outer errors. Consequently these
are POST_HOC_SENSITIVITY results, not a new blind confirmation or evidence of
method novelty, knowledge guidance or privacy. Retain the original table and
disclose this amendment when writing the paper. Independent validation needs
new data or a separately justified design; reusing these folds cannot erase
the prior exposure.

## Files and commands

Implementation: restart/support_guard.py, restart/guard_sensitivity.py,
restart/test_support_guard.py. Frozen package: results/restart_guard_protocol_v1.
Evaluations: results/restart_guard_study_v1. Audit/summary:
results/restart_guard_analysis_v1/analysis.json. Tests:
results/restart_guard_tests_v1 and restart_guard_tests_v2.

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity tests --output paper2/results/restart_guard_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity freeze --output paper2/results/restart_guard_protocol_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity execute --output paper2/results/restart_guard_study_v1 --cap 8 --execute
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity execute --output paper2/results/restart_guard_study_v1 --cap 312 --execute --resume
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity tests --output paper2/results/restart_guard_tests_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity analyze --output paper2/results/restart_guard_analysis_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.guard_sensitivity execute --output paper2/results/restart_guard_study_v1 --cap 1 --execute --resume
```

Subsequent next step completed: remaining baseline runner mutation/recovery
acceptance checks, with 29 passing tests saved in restart_outer_acceptance_v1.
Checks reject changed execution snapshots, plan/config/selection mutations,
missing explicit resume, changed bindings and unbound recovery. Added
restart/test_outer_acceptance.py; frozen implementation remained unchanged.

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_outer_acceptance --output paper2/results/restart_outer_acceptance_v1
```

Next ready work: specify the privacy threat model and literature-supported method.
KG and ranking remain gated by source validity/comparability. No dataset,
Paper 1, frozen baseline source/package or open manuscript was modified.
