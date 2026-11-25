# Fixed-transform data and matched simulation plan

2026-10-14. Completed the next R08 preparation stage; no real model training
performed this stage. Privacy-utility runner and release ledger remain pending.

## Data preparation and verification

prepare_public_data.py reads the previously audited 607-system row table and
creates a new version with the fixed, data-independent 54-feature transform.
It retains all original targets, row IDs, clients, eight geography holdouts,
56 inner folds and training-derived support metadata. Support metadata is for
public-benchmark interpretation only; it is not fitted into the new transform.
Inner NPZ files contain train/validation only and omit outer test arrays.
Neither numerical statistics nor categorical vocabularies are fitted.

All 64 folds passed a fresh verifier: parent raw/data/source/runtime hashes,
exact unchanged targets/IDs, exact partition equality, fixed schema identity,
recomputed arrays, omitted test arrays in inner folds and once-only outer row
coverage. Public tensors: results/restart_public_data_v1. Evidence:
results/restart_public_data_verified_v1/verification.json.

## Prospective matched development plan

privacy_study_protocol.py froze 200 runs = eight outer groups x five seeds x
five variants: equal-slot averaging; clipped equal-slot averaging with C=1;
and clipped noisy averaging at std/sensitivity multipliers 2, 4 and 8.
Only the original FedAvg's selected per-fold LR and rounds are carried forward.
All variants within a fold/seed must share those settings, a fresh seeded model,
54-feature representation, local batching/epoch work and full participation.
The historical sample-weighted models are not directly substituted as controls.
No new tuning, target transformation, output clipping or early stopping is planned.

The existing outcomes have already been exposed. This is an additional public-
data development simulation, not independent confirmation or an application DP
release. Seeds for simulated noise must use a stream separate from data ordering.
Before execution, the coordinator must implement controls, verify resume identity,
persist accountant events atomically with each round's checkpoint, prevent
duplicated events/releases and freeze its exact execution code.

## Composition and claim boundary

The frozen plan computes the hypothetical noisy-only transcript budget for each
regional slot across every overlapping fold, seed and noisy variant. The slot
is fixed/public within each outer fold (seven training clients). At delta=1e-5,
the conservative full-matrix epsilon bounds range approximately 515.91--571.49.
These very large bounds do not support a useful strong-privacy claim for releasing
the entire noisy transcript. They illustrate why per-run epsilon must not be
presented as the privacy budget of all released experiments together.

The unnoised controls are excluded from that hypothetical budget; if they were
released on the same private data, the joint transcript would have no established
finite DP guarantee. All planning budgets include application_privacy_certified
false. Actual public-data research remains a simulator benchmark. Private
deployment needs an explicit limited release policy, reviewed randomness and
numerics, protected evaluation outputs and independently public roster/schema.

## Actual commands and tests

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_public_study paper2.restart.test_private_primitives -v
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.prepare_public_data prepare --output paper2/results/restart_public_data_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.prepare_public_data verify --output paper2/results/restart_public_data_verified_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.privacy_study_protocol --output paper2/results/restart_privacy_study_protocol_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_study --output paper2/results/restart_public_study_tests_v1
```

Initial 12 tests passed; final scoped suite including accounting has 20 passing
tests, zero failures/errors, with persistent logs. Four new planning tests verify
matched settings, duplicate rejection, valid clipping/noise and round budgets.
The real preparation verifier passed all 64 folds; plan freeze returned
FROZEN_PLAN_ONLY_NO_TRAINING, planned=200. Frozen source snapshots and hashes are
in restart_privacy_study_protocol_v1. Existing source/results and raw files are
preserved; no dependency installation or manuscript change.

Next implement matched control aggregators and a resumable simulator runner
with round/event accounting and cached test-inference receipts; test it on
synthetic inputs before freezing code and executing a bounded starter batch.
