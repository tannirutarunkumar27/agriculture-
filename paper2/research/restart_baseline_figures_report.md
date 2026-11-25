# R10 baseline publication figures

Date: 2026-10-14

## Outcome

The missing baseline-results figures are complete. Three figures were generated
from fresh cached audits of the complete original outer study and the separate
post-hoc zero-support sensitivity. No model was trained and no inference was
repeated.

Accepted output: `results/restart_baseline_figures_v2/`

- `original_baseline_rmse`: all eight original methods, equal-group and pooled
  RMSE, with all five seed values and descriptive seed SD. The log axis is
  explicit and is used only to retain visibility across the large error range.
- `original_xgboost_geographic_rmse`: the original XGBoost result by each of
  eight held-out groups, including unequal group sample counts.
- `posthoc_guard_sensitivity`: original versus guarded neural results, labelled
  prominently as post-hoc sensitivity after outer-test failures were exposed.

Each figure is available as PNG, PDF and SVG. Exact plotted values, captions,
source-analysis hashes, code hash and asset hashes are saved with the figures.
All three accepted PNGs were inspected at original resolution. Version 1 was
retained as development evidence; version 2 fixes crowded log tick labels and
a legend overlap found during visual review.

## Verification

Command:

```text
.venv\Scripts\python.exe -m paper2.restart.test_baseline_publication_figures --output paper2/results/restart_baseline_figure_tests_v2
```

Final rerun: `results/restart_baseline_figure_tests_v3/`; four tests passed,
zero failures/errors. The five existing R13/R14 figure tests also passed at
`results/restart_publication_figure_tests_v4/`, for nine final figure tests.
The baseline tests require the exact
eight-algorithm inventory, 320 original and 320 guarded evaluations, five seeds,
607 systems and eight groups; independently recompute plotted means/SDs; reject
partial, relabelled or inconsistent analyses; keep prospective and post-hoc
records separate; and render three formats without leaked figures.

Generation command:

```text
.venv\Scripts\python.exe -m paper2.restart.baseline_publication_figures --output paper2/results/restart_baseline_figures_v2
```

The generator first ran a fresh audit-backed original summary and a fresh
guard-cache/provenance analysis. Both were complete. The manifest records
`model_training=false`, `model_reinference=false`, and
`posthoc_guard_separated=true`.

## Interpretation boundaries

The original prospectively frozen outer results remain primary. The support
guard was designed after outer-test failures were visible, reused fixed models,
and is not independent confirmation or a replacement for the original result.
Seed SD describes initialization variability and is not a regional confidence
interval. Small groups with four to six systems remain visible. No field,
ranking, agronomic-utility or privacy claim follows from these figures.

## Next action

R10 figure generation is now complete across the original baseline, R13
unnoised refit and R14 matched-noise studies. The next ready R10 task is a
Paper-2-only evidence-to-claim matrix and manuscript skeleton using the accepted
figures and verified results. Paper 1 manuscript files remain untouched.
