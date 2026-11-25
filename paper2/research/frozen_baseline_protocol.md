# Step 1: frozen repeated-seed baseline development protocol

Frozen 2026-10-12. Executable template: `configs/baseline_study_v1.json`.
Materialized package: `results/frozen_baseline_protocol_v1/`.

## Scope and evaluation status

Canonical benchmark is the primary controlled dataset; Maharashtra is an
additional district-labeled development dataset with unresolved measurement
provenance. Manipur remains a separate synthetic supplementary dataset and is
not included in this baseline phase. No cross-dataset merging or regional
fabrication. Independent field validation has NOT been established.

Study seeds: **123, 2024, 3407, 9999, 2718**. Tuning seed 42 is excluded from
this study, but new seeds do not create independent data. Previous Paper 1 and
Paper 2 development use of the datasets/holdouts remains disclosed. This is a
prospectively frozen DEVELOPMENT comparison, not external confirmation.

## Fixed model and training settings

Neural models: same 64-hidden-unit MLP, shared label schema and initialization
per dataset/seed; SGD LR=.2, momentum=0, weight decay=0, batch=32, local epochs=1,
rounds=80, CPU threads=1. FedProx mu=.01; restricted FedNova constant-LR plain
SGD normalization. No optimizer or capacity changes during the study.

Centralized reference uses 80 full-training-set passes. Federated local passes
match aggregate data exposure at full participation, not identical optimizer
paths or server updates. Partial participation intentionally reduces exposure;
actual selected clients, optimizer steps, time and tensor bytes are reported.
No adaptive local epochs, LR schedule or early stopping added after freezing.

Conventional models use the validation-selected settings:

| Model | Canonical | Maharashtra |
|---|---|---|
| Logistic regression | C=10, max_iter=2000 | Same |
| Random forest | 200 trees, min_samples_leaf=1 | Same |
| XGBoost | 200 boosting rounds, depth=3, LR=.1 | 200 boosting rounds, depth=6, LR=.1 |

The multiclass XGBoost setting is 200 boosting rounds, not 200 total trees.
Locally verified saved boosters contain 4,400 trees for 22 canonical classes
and 3,200 trees for 16 Maharashtra classes. This corrects documentation only;
the frozen `n_estimators=200` configurations and budgets are unchanged.

All transformations fit training rows only. Global mean/std and soil-color
vocabulary are shared public-simulation references, NOT private distributed
preprocessing. Hyperparameters come from independently checked validation
pilot selection; their provenance is copied into the frozen package.

## Client scenarios and run budget

Five clients per simulated scenario. For each dataset/seed:

- Random equal-group IID, participation=1.0.
- Label-wise Dirichlet alpha=.5, participation=1.0.
- Label-wise Dirichlet alpha=.1, participation=1.0.
- Unequal-size global Dirichlet alpha=.5, participation=1.0.
- Same IID partition with participation=.6 (three of five clients).
- Same IID partition with participation=.2 (one of five clients).

Maharashtra additionally uses its five observed district identifiers with full
participation. Partial participation is uniform without replacement, not an
asynchronous timeout or client failure experiment. No claim about combined
severe-skew/low-participation or sensor failures until separately tested.

Whole duplicate input profiles stay together; split grouping and holdout row
IDs are identical across scenarios within each dataset/seed. Minimum local
training size=5, bounded Dirichlet draw retries=200, actual distributions saved.
No test rows enter training clients. Fixed 80 rounds are a study budget, not a
guarantee of convergence. Per-round curves retain evidence of unfinished learning.

**235 planned runs:** 195 federated runs plus 40 centralized references. Each
centralized reference is fit once per dataset/seed and reused for comparisons
against scenarios sharing the exact same train/validation/test rows. It is not
repeated per scenario or chosen opportunistically. There are **45** unique
partition files; partial IID participation reuses the IID partition.

## Selection, metrics and analysis

Select the neural checkpoint with minimum global validation NLL; first round
wins exact ties. Test is evaluated once after checkpoint selection for each
completed run. No test-driven choice of hyperparameters, rounds, splits or
algorithms. No re-training on train+validation without a new protocol version.

Primary metric: Macro-F1 with the global class schema. Secondary: accuracy,
macro precision/recall, weighted F1, NLL, per-class report and confusion matrix.
Resources: actual local SGD steps, round/aggregation seconds, tensor payload
upload/download bytes. Payload bytes exclude actual network/crypto overhead;
do not call them measured network traffic. Do not invent unavailable memory
or end-to-end application latency measurements.

Analysis: descriptive mean and sample SD across five seeds, paired per-seed
algorithm differences on matching dataset/scenario/splits. Seed variability is
conditional on these datasets; overlapping splits are not independent new field
cohorts. No confirmatory p-values/statistical superiority under this initial
development protocol. Any later hypothesis-testing/confirmatory protocol must
be explicit and recorded before inspecting its outcomes.

## Reproducibility and error handling

Materialized protocol, 235-run plan, 45 row-ID/client partitions, source dataset
hashes, Python/dependency versions, tuning evidence and exact baseline code
snapshots are locally hash-checked. Model/settings/source/environment changes
require a newly frozen amended version. Existing artifacts are not overwritten.
Local hashes provide consistency checks, not signed/immutable security.

Fail visibly on incompatible schemas, stale/non-finite updates, malformed
messages or changed source files. Never silently skip a failed client or substitute
results. The next-stage runner must save run failures/checkpoints and resume
without mixing versions or evaluating the test repeatedly.

Knowledge guidance=false; formal privacy=false. This is a single-process
public-data simulation, not a multi-region deployment. At participation .2 only
one client trains per round, which cannot support a useful multi-client secure
aggregation privacy claim. SecAgg/DP will need a separate threat model/protocol.

## Executed checks

Fresh full suite: **56 passed, zero failed** (`results/tests_v5`). Includes
unapproved hyperparameter change, tuning-seed reuse, unsafe scenario names,
fabricated districts, protocol/partition tampering and false privacy flags.
Frozen package verification: PASS, 235 planned runs / 45 partitions,
`training_started=false`, `test_evaluated=false`.

Commands:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v5
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --config paper2/configs/baseline_study_v1.json --output paper2/results/frozen_baseline_protocol_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

Next step: implement and smoke-test the bounded, resumable frozen-plan runner.
No repeated-seed study training was started in this step.
