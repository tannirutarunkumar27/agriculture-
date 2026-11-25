# Outer training and inference core

2026-10-13. R04 remains IN_PROGRESS. Implemented restart/outer_training.py
and seven synthetic tests in restart/test_outer_training.py. Real outer data
was not loaded, trained on or inferred on.

The refit primitive takes training arrays only, uses fixed previously selected
rounds, retains the last round and never selects on outer outcomes. Reuses the
unchanged regression client and established aggregation. Atomic checkpoints
persist round/history/state; completed fit receipts verify model identity/hash
and avoid retraining. Four reference regressor implementations are included.

Inference verifies the fitted model, writes persistent intent before execution,
then commits predictions, target/ID cache, metrics and hashed completion receipt.
Completed calls recompute cached metrics without model inference. If intent
exists without committed completion, execution stops rather than silently
repeating test inference. This conservatively requires manual diagnosis after
an ambiguous interruption. Caller MUST hold a study-wide writer lock.

Actual command:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_outer_training -v
```

Actual result: 7 tests passed, zero failures/errors. Test cases include exact
interrupted/resumed weights for centralized/FedAvg/FedProx/FedNova; completed
fit no-retraining; once-only cached evaluation and tamper rejection; ambiguous
inference refusal; changed identity rejection; mean/ridge/RF/XGBoost smokes;
invalid dtype/client membership rejection. All data is synthetic. A nonfatal
joblib NumPy 2.5 shape-deprecation warning appeared and was not suppressed.

Still required before any real outer execution: coordinator verifies frozen
plan/runtime/data/code binding, checks arrays against partition/preprocessing,
enforces caps/explicit resume/writer lock, journals failures, retains support
panels, saves persistent test logs and independent study audits. Freeze that
runner's exact code prospectively in a separate package. Do not directly call
these primitives on real data without those protections.

No privacy, KG, ranking, external deployment, raw-data or Paper 1 changes.
