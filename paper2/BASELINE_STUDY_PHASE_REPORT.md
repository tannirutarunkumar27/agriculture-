# Step 3: completed frozen baseline development study

Executed: 2026-10-12 local session. Paper 1, original raw datasets, the open
LaTeX document, frozen baseline implementation and frozen study settings were
not modified. No dependencies were installed.

## Completion and evidence

- All **235 planned runs completed**, zero failures, zero pending: 195
  federated and 40 centralized references; canonical 110, Maharashtra 125.
- Five seeds: 123, 2024, 3407, 9999, 2718; 45 frozen partition files.
- The four previously completed starter runs were retained and verified.
  This step executed the remaining 231 runs in five 40-run batches and a
  final 31-run batch. Completion totals: 44, 84, 124, 164, 204, 235.
- All 235 completions were independently rechecked through the analysis
  wrapper: source/config/partition/model/receipt identities and predictive
  metrics recomputed from cached probabilities. No model test inference was
  performed during analysis. Identical actual splits were checked for pairing.
- **47 complete five-seed groups**, **182 descriptive paired comparisons**,
  **910 individual paired-seed differences**. No incomplete groups excluded.
- Full final software suite: **80 tests passed, zero failed**, exit code 0.
  Evidence: `results/tests_v9/unittest.log` and `verification.json`.
  Earlier 76-test and 78-test runs remain in `tests_v7` and `tests_v8`.
  The existing nonfatal joblib/NumPy deprecation is recorded, not suppressed.
- Frozen protocol SHA-256:
  `01964d81236d1f97924bb20b840520939f77d89064667e9b19a03ac1f82cad62`.
  Verification PASS. The freeze manifest's `training_started=false` and
  `test_evaluated=false` describe the historical freeze, not current execution;
  current completion is in `results/baseline_study_v1/study_progress.json`.

## Saved outputs

Execution: `results/baseline_study_v1/` contains per-run selected models,
validation histories, training receipts, test journals, cached probability
bundles, metrics and completion records. Completed cases were not retrained
or re-evaluated during batch resume.

Analysis: `results/baseline_analysis_v1/`:

- `per_run_metrics.csv`: all 235 verified results and resource measurements.
- `aggregate_metrics.csv`: complete five-seed means and sample SDs.
- `paired_seed_differences.csv`: all 910 paired differences.
- `paired_difference_summary.csv`: 182 paired comparison means/sample SDs.
- `regional_metrics.csv`, `per_class_metrics.csv`: district and crop detail.
- `verification.json`: audit and source receipt hashes, analysis identity,
  completeness, metric units and interpretation constraints.

Predictive CSV metrics are fractions; multiply mean and SD by 100 for percent.
Paired Macro-F1 differences are already in percentage points. NLL is
dimensionless. Sample SD across seeds is not a field-generalization confidence
interval. Regional Macro-F1 uses the full crop schema with absent classes zero;
support and present-class counts accompany regional scores.

## Selected descriptive findings

Macro-F1 percentage mean +/- sample SD, five seeds. This is a small selection;
all scenarios and models remain in the machine-readable outputs.

| Dataset / condition | Model | Macro-F1 (%) |
| --- | --- | --- |
| Canonical / centralized reference | Random Forest | 99.27 +/- 0.63 |
| Canonical / centralized reference | XGBoost | 98.78 +/- 0.21 |
| Canonical / IID full participation | FedAvg | 97.30 +/- 1.12 |
| Canonical / severe class skew | FedAvg | 96.25 +/- 1.53 |
| Canonical / severe class skew | FedNova | 95.95 +/- 2.57 |
| Maharashtra / centralized reference | XGBoost | 98.38 +/- 0.67 |
| Maharashtra / centralized reference | centralized MLP | 96.19 +/- 0.99 |
| Maharashtra / IID full participation | FedAvg | 90.00 +/- 1.64 |
| Maharashtra / severe class skew | FedAvg | 75.15 +/- 7.91 |
| Maharashtra / severe class skew | FedNova | 84.87 +/- 4.81 |
| Maharashtra / district clients | FedAvg | 86.05 +/- 2.69 |
| Maharashtra / district clients | FedNova | 88.52 +/- 1.89 |
| Maharashtra / unequal client sizes | FedAvg | 94.23 +/- 1.65 |
| Maharashtra / unequal client sizes | FedNova | 92.73 +/- 2.05 |

FedNova-minus-FedAvg under Maharashtra severe skew is +9.73 +/- 6.15
percentage points across paired seeds. This benefit is condition-dependent:
FedNova's mean is lower under Maharashtra unequal sizes and canonical severe
skew. The fixed sample-weighted FedProx configuration does not demonstrate
consistent improvement. Strong centralized references remain necessary.
Identical rounded scores do not imply copied runs: restricted plain-SGD
FedNova reduces algebraically to FedAvg when selected local step counts are
equal; this property is tested. Small FedProx differences can also leave
hard predictions unchanged. Probability and per-run outputs are preserved.

## Implementation and verification commands

New analysis implementation: `study_analysis.py`,
`scripts/summarize_baseline_study.py`, ten analysis tests in
`tests/test_study_analysis.py`, and `research/baseline_study_analysis_guide.md`.
Analysis rejects incomplete studies by default. Explicit partial mode excludes
every incomplete seed group, including groups with zero observed seeds.
Existing output is refused, and a study writer lock protects verification.
Synthetic unit-test fixtures are not experimental measurements.

```powershell
# Run five successive batches, waiting for each successful exit:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 40 --execute --resume
# Final batch:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 31 --execute --resume
# Final full software checks:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v9
# Complete analysis (already executed; refuses an existing output directory):
& '.\.venv\Scripts\python.exe' -B paper2/scripts/summarize_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --study paper2/results/baseline_study_v1 --output paper2/results/baseline_analysis_v1
```

Documentation-only correction: XGBoost `n_estimators=200` means 200 boosting
rounds, not 200 total multiclass trees. Locally generated saved models contain
4,400 trees for canonical (22 classes) and 3,200 for Maharashtra (16 classes).
No configuration, saved result or frozen source was altered by that correction.

## Limits and next gated step

These are DEVELOPMENT-holdout results from reused datasets and overlapping
seed splits, not independent confirmation after earlier Paper 1/smoke use.
There are no confirmatory p-values, knowledge-guided results, formal privacy
guarantees or agricultural outcome/field-generalization claims. Datasets have
different features and crop taxonomies and must not be pooled as equivalent
tasks. Maharashtra measurement units/provenance remain unresolved; Manipur
synthetic data was excluded from this study.

The neural budgets and settings remained frozen. Equal local epochs do not
equate all optimizer trajectories; partial participation reduces data exposure.
The public training-only preprocessing in this local simulator is not private
preprocessing. Tensor payload counts are not network traffic and local CPU
timings are not deployment latency. Fixed 80-round budgets do not certify
convergence of every scenario. FedProx is the documented sample-weighted
variant; FedNova is restricted constant-LR plain SGD.

Next: resolve crop-specific agronomic source provenance and feature-unit
compatibility, define and test class-aligned knowledge guidance, then freeze
its ablations before training. Unverified crop rules must remain disabled.
Privacy/accounting, secure aggregation, distributed deployment and independent
validation are later separate phases. The completed baseline study alone does
not complete Paper 2.
