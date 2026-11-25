# Validation convergence implementation report

Date: 2026-10-12. This report extends `PHASE_REPORT.md`, retaining its historical
results. Source rules remain unapproved. No private or deployed system claimed.

## Completed work

1. Added `evaluate_test=False` / CLI `--validation-only` to neural training,
   with test policy included in strict checkpoint resume identity. Tuning saves
   validation results only and never produces test predictions.
2. Implemented matched centralized LogisticRegression, RandomForest and XGBoost
   references with the same permitted training IDs/preprocessing and class
   schema as neural federation. Model configuration, hashes and versions saved.
3. Implemented a fixed, bounded validation search and ONE common neural LR per
   dataset; architecture, initialization, local epochs, batch size and other
   settings stay matched across algorithms.
4. Added probability validation, invalid-client dtype/label rejection and
   malformed non-tensor update rejection. Preserve fail-fast semantics.
5. Executed 38 candidate runs, recomputed all saved candidate validation
   metrics, verified best-round selection and preserved exact source versions.

## Actual error and repair

First expanded suite: 40 tests with one error, saved in `results/tests_v3`.
XGBoost uses NaN as its default missing-value sentinel in estimator parameters.
Strict manifest JSON writing failed in `centralized.py` via `train.write_json`.
`serializable_parameters` now records that library default explicitly as a
tagged value. Predictions, losses and metrics still reject NaN; serialization
was not globally relaxed. An added regression test checks standards-compliant
JSON. A separate float32 row-sum warning is fixed by normalization only after
probabilities pass finite/range/sum checks. This is not calibration.

Final fresh suite: **42 passed, zero failed** (`results/tests_v4`). It includes
an evaluation-call-count check proving neural tuning makes no test evaluation,
and validation-only execution tests for all three conventional references.

## Executed pilot

Protocol saved before training: `configs/convergence_pilot.json`, copied into
`results/convergence_pilot_v1/protocol.json`. Canonical IID and Maharashtra
district partitions; seed 42; 64-hidden-unit MLP; 80 rounds; local epochs=1;
batch=32; full participation; proximal mu=.01; LR candidates=.05/.1/.2.

24 neural candidates plus 14 conventional candidates = **38**. The common
selected neural LR is **.2** on both datasets, minimizing mean best validation
NLL across the four neural algorithms. Conventional models are individually
selected by validation NLL. No test values were used during this search.

All scores below are **single-seed validation Macro-F1**, not final test results:

| Model | Canonical IID | Maharashtra district clients |
|---|---:|---:|
| Centralized MLP | 98.48% | 98.07% |
| FedAvg | 97.22% | 86.53% |
| FedProx | 97.22% | 86.53% |
| FedNova (plain SGD) | 97.22% | 90.53% |
| Logistic regression | 97.26% | 85.72% |
| Random forest | 99.70% | 98.84% |
| XGBoost | 99.39% | 99.35% |

These are measured from the selected NLL checkpoint, not cherry-picked maximum
F1 rounds. FedAvg/FedNova's equal-work IID match is expected. The Maharashtra
difference is descriptive, not statistical superiority. Strong centralized
tree baselines remain important comparators; no KG-FCR result exists yet.

Best selected rounds near 80 and the upper LR boundary mean the pilot does not
prove convergence or optimal hyperparameters. The earlier three-round scores
were undertrained diagnostics; longer validation curves are now retained.

## Saved artifacts and verification

- `results/convergence_pilot_v1/validation_results.csv`: all 38 candidates.
- `results/convergence_pilot_v1/selected_configs.json`: common neural configs
  and selected conventional configurations, selected by validation only.
- Per-candidate directories: full manifests, summaries and neural round logs/
  checkpoints or conventional locally generated model files.
- `results/convergence_verified_v1/selected_validation_metrics.csv`: metrics
  independently recomputed from the selected saved models.
- `results/convergence_verified_v1/regional_validation_metrics.csv`: regional
  Maharashtra validation metrics, sample and class support. Global-schema
  Macro-F1 includes absent classes as zero; compare with this scope in mind.
- `results/convergence_verified_v1/verification.json`: **38/38** verified,
  no test evaluations or test prediction files accepted.
- `results/convergence_pilot_v1/source_snapshot`: five exact manifested Python
  source files, hashes matched before and after copying. Never load untrusted
  joblib checkpoints. These files preserve the original executed code version.
- Commands and original failed/successful test logs preserved in `STATUS.md`.

## Important limitations and next actions

This is a single-seed PUBLIC simulation pilot. Train-only global preprocessing
is not privately distributed. Epoch/data-exposure budgets are matched but
optimizer paths and exact update counts differ and are recorded. District
validation remains same-district, not unseen-region validation. Maharashtra
measurement provenance/units remain unverified; Manipur remains synthetic and
was not tuned in this bounded phase.

Earlier smoke runs already evaluated these held-out test partitions. Therefore
do not call them pristine external confirmation; no further test metrics were
inspected during tuning. Freeze an independent confirmatory protocol or obtain
new external measured data before publication claims. Add prospectively specified
convergence/heterogeneity and repeated-seed comparisons with uncertainty estimates.

Crop-specific agronomic records/units remain the gate for knowledge guidance.
Formal privacy, poisoning defenses, distributed deployment and Linux validation
remain separate future work. No existing Paper 1 source or raw dataset changed,
and no dependency was installed in this phase.
