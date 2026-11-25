# Step 2: bounded frozen-study runner

Implemented 2026-10-12. Entry point: `scripts/run_baseline_study.py`.
The original five baseline source files and frozen package remain unchanged.

## Execution controls

Default invocation is read-only dry-run: checks the frozen protocol, source
data, baseline code and dependency versions and lists selected IDs. Training
requires `--execute`. Default maximum new/resumed runs is one; set `--max-runs`
explicitly for a bounded batch. No CLI hyperparameter, seed, round-budget or
partition overrides. `--run-id` can be repeated for IDs already in the plan.

The same study output requires `--resume`. Existing output is never silently
replaced. A kernel-held file lock prevents two writers; the operating system
releases it after process exit, so no guessed stale-PID deletion is needed.
The persistent `runner.lock` file's existence alone does NOT mean a runner is
active. Runner code and protocol-verifier hashes bind the study output to the
executed implementation; changing them requires an amended version.

Example read-only plan inspection:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 1
```

Example NEXT bounded batch (not executed as part of step 2):

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 5 --execute --resume
```

It verifies previously completed runs before selecting the next unfinished
entries in frozen order. Completed centralized references are fit once per
dataset/seed, not repeated for every scenario. If a failed run is encountered,
the batch stops; it is not silently skipped to improve the result table.

## Training and recovery

All baseline engines train in validation-only mode. The wrapper checks exact
configuration, source/code/partition identity, complete round coverage, selected
validation NLL checkpoint, class ordering, model hashes and training-only
preprocessing before test evaluation.

Training attempts and logs are preserved. Neural interruptions resume the
saved last complete round at the fixed target budget. A complete final-round
checkpoint can finish validation/export without retraining even if the baseline
summary was never written. Conventional estimators have no mid-fit optimizer
checkpoint: an inspected failure with no usable saved model requires explicit
`--retry-failed` to start a fresh attempt, retaining the earlier attempt.

Failures save full tracebacks in per-run `errors/` and root `study_errors/`.
After inspection/fix, use `--resume --retry-failed` for pre-evaluation failures.
Incompatible source/settings/code/environment changes are rejected rather than
being silently treated as a compatible resume. Frozen budgets are never extended.

## Preventing automatic repeated test evaluation

Before the model touches test inputs, an atomic journal records `started` and
binds the chosen model's training receipt. One test-set prediction pass is
written to an atomic NPZ bundle with probabilities, public row IDs/labels and
identity/metric metadata. Summary exports and completion receipts are then
derived from this cached bundle. Subsequent verification recomputes metrics
from cached probabilities; it does not run the model on test inputs again.

| Saved state | Behavior |
|---|---|
| Incomplete neural training checkpoint | Resume remaining rounds before any test evaluation |
| Complete training checkpoint but no final summary | Recover validation/export, no repeated training |
| Committed prediction bundle but interrupted summary export | Rebuild summary from cached predictions, no new test inference |
| Valid completion receipt | Verify hashes and cached metrics, skip execution |
| Evaluation journal started but no committed bundle | Stop as ambiguous; even retry cannot automatically re-evaluate |
| Corrupted completed artifacts | Fail closed and record integrity error, no retraining/replacement |

This is **at most one automatic evaluation attempt**, not a claim of an
infallible distributed exactly-once transaction. A crash between inference and
durable bundle commit cannot prove whether inference happened. The conservative
response is to stop and require reviewed recovery/protocol amendment, not reset
or delete the journal. Atomic local writes/kernel locks are not cryptographic
security or formal privacy. Hardware/power-loss guarantees are not established.

## Outputs

Study root: binding, exact runner source snapshots, `study_progress.json` and
`completed_runs.csv`. Each run: frozen binding, training attempts/log, training
receipt/round records, evaluation journal, cached test bundle, selected metrics
and completion hashes. Per-class/confusion-matrix outputs are saved in summaries.
For Maharashtra, regional test metrics use cached probabilities; global-schema
Macro-F1 includes absent classes as zero, with class support reported.

NLL in final summaries uses the shared strict probability-metric evaluator for
all models, including normalization only within accepted float32 row-sum
tolerance. Neural checkpoint selection still uses original validation logit NLL.
Other frozen selection, class schema, settings and metrics are unchanged.

Time/step counts are measured local simulation values; communication fields
are tensor payload sizes, not network traffic or end-to-end deployment latency.
Conventional methods have no neural SGD step count, recorded as null rather
than fabricated zero updates. Logs omit raw training feature records; cached
test labels/probabilities are derived public-data evaluation artifacts.

The frozen package's `training_started=false` fields describe the STEP 1 freeze
operation, not current execution. Current study progress lives in the separate
study output. No raw source data, Paper 1 file or frozen artifact is rewritten.
