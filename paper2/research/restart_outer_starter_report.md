# First frozen outer-study batch

2026-10-13. Added independent cached-output verifier verify_outer_study.py.
It verifies package/source/runtime/data identities, model hashes (no model
loading), receipt identities/assets, inference intent, partition support,
cached target/ID equality with prepared test arrays and independently computed
RMSE/MAE/negative counts. Adds R2 (null if undefined) and supported-species/
composition panels. Analysis uses outcomes only after fixed inference; no tuning.

Actual commands:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_outer_study --tests --output paper2/results/restart_outer_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.outer_runner --execute --max-runs 8 --output paper2/results/restart_outer_study_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_outer_study --output paper2/results/restart_outer_verified_v1
```

19 tests passed before training. Real batch: eight complete, 312 pending, no
observed failure. Covers centralized MLP, FedAvg, FedProx, FedNova, RF, ridge,
training mean and XGBoost for Angers held out, seed123. Independent audit:
PASS_PARTIAL, eight verified, no model reinference. Existing frozen package,
settings, raw sources and Paper 1 unchanged. No new dependencies installed.

Outer-test scores have now been exposed; earlier untouched-test statements are
historical. Never modify LR/budget/architecture based on these scores. This
single-group/seed subset does not establish comparative generalization.

Fresh metric-unit tests added for known values, undefined R2 and bad arrays.
Next complete the remaining frozen runs in bounded resume batches, independently
audit completed results and implement complete matched-group/seed aggregation.
Additional package mutation/receipt edge-case tests remain useful acceptance
work; full project completion is not claimed. KG/ranking/privacy remain gated.
