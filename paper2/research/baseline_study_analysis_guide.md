# Step 3: execution and descriptive analysis

This stage executes the existing 235-entry frozen development plan in bounded
batches. Baseline hyperparameters, training budgets, data sources, partitions,
runner and frozen identities must not change while the study runs. Completion
is determined by verified receipts, not by directory count or a printed score.

## Continue execution

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 40 --execute --resume
```

Repeat only after each batch exits successfully and `study_progress.json` shows
no failures. Completed cases are verified, not trained/evaluated again. Any
failure stops execution for inspection; do not silently remove it or change
settings to improve the output table. A batch limit is an invocation cap, not
an amendment to the fixed per-model 80-round budget.

## Complete-study analysis

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/summarize_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --study paper2/results/baseline_study_v1 --output paper2/results/baseline_analysis_v1
```

By default, this refuses an incomplete study before writing outputs. Analysis
locks the study against concurrent writers, verifies its identity and all
completed model/prediction/receipt artifacts, and derives scores from cached
probabilities without model test inference. Pairing checks actual shared
train/validation/test row identities, not just matching seed numbers.

For explicitly labeled interim development analysis, use `--allow-partial`
and a separate new version directory. Every seed-incomplete metric group is
excluded from aggregate comparisons, with its missing seeds recorded. Partial
outputs are never labeled the completed 235-run study. Existing output
directories are refused rather than overwritten.

## Outputs and units

- `per_run_metrics.csv`: verified per-run predictive/resource values.
- `aggregate_metrics.csv`: mean and sample SD only for complete five-seed
  dataset/scenario/model groups. Predictive fractions are in [0,1]; NLL is
  dimensionless. Multiply fraction mean/SD by 100 for percentage units.
- `paired_seed_differences.csv`: model-minus-comparator Macro-F1 differences
  in percentage points, with the five actual paired seeds retained.
- `paired_difference_summary.csv`: mean/sample SD of paired differences, not
  unpaired differences of pooled means. No confirmatory p-values.
- `regional_metrics.csv`: Maharashtra regional cached-test metrics and crop
  support; global-schema Macro-F1 counts absent crop classes as zero.
- `per_class_metrics.csv`: crop-specific precision/recall/F1 and support.
- `verification.json`: completeness, excluded groups, receipt hashes, analysis
  code version, frozen verification and scientific interpretation limits.

Full expected coverage: 235 runs, 47 complete metric groups and 182 descriptive
paired comparisons (910 individual paired-seed differences). Centralized
references are reused as comparators on the identical per-seed splits; their
scores are not treated as extra independent observations for every scenario.

Never mix null conventional-model SGD step counts with zeros or drop missing
seeds to obtain a favorable average. Means/SD across seeds are conditional on
these reused datasets and overlapping splits, not confidence intervals for
new agricultural regions. Single-process public preprocessing and tensor-byte
measurements do not imply formal privacy or actual network performance.

## Interpretation limits

All results remain DEVELOPMENT-holdout evidence. Earlier smoke and Paper 1 use
of the data is disclosed. No independent field confirmation, crop yield gain,
knowledge-guided improvement, formal privacy, statistical superiority or
production deployment follows from a high benchmark score. Fixed-budget
80-round runs may still be underconverged; preserve full validation curves.

Tests explicitly cover sample SD, paired percentage-point units, duplicate or
unexpected seeds, exclusion of incomplete groups, null resource measurements,
and refusal to generate an unqualified incomplete-study summary. Synthetic
unit-test fixtures are not experimental observations.
