# Restart: validation search completed

Date: 2026-10-13. Active dataset: new cereal-legume trials v2.0.0,
607 crop systems, eight geographic groups. Historical classification results
are not reused. All 1,120 frozen inner-validation candidates completed with
zero failures and zero pending runs. No outer-test inference occurred.

## Execution and verification

The previous 20 completions were retained. This phase executed two batches of
40, twelve batches of 80 and a final batch of 60, without changing frozen
settings, sources, data or runtime. All 1,120 attempt journals contain a single
attempt; no failed model attempt or silent replacement occurred. A tool-host
connection interruption was resolved by checking saved progress and the OS
writer lock before continuation; it did not require retraining completed runs.

Executed from repository root using the existing environment:

```powershell
# Bounded continuation; earlier batches used 40 and the final batch used 60.
& '.\.venv\Scripts\python.exe' -B paper2/restart/run_validation_search.py --output paper2/results/restart_validation_search_v1 --max-runs 80 --execute --resume
& '.\.venv\Scripts\python.exe' -B paper2/restart/verify_validation_search.py --output paper2/results/restart_validation_verified_v2
& '.\.venv\Scripts\python.exe' -B paper2/restart/summarize_validation_selection.py --output paper2/results/restart_validation_selection_v1
& '.\.venv\Scripts\python.exe' -B paper2/restart/test_validation_search.py --output paper2/results/restart_search_tests_v3
```

- Independent model/cache/source audit: PASS_COMPLETE, 1,120 verified runs.
  Selected model validation metrics reproduced within rtol/atol 1e-7; neural
  80-round histories and minimum-RMSE/first-tie checkpoint choices checked.
- Independent selection audit: PASS_COMPLETE_SELECTION_AUDIT. Recomputed all
  eight outer selections from completed inner results and matched saved settings.
- Fresh scoped search suite: 14 passed, zero failures/errors. Synthetic tests
  verify safeguards, not agronomic validity or field generalization.
- No frozen search/training/aggregation code was changed. The new selection
  summarizer is separate from the frozen training implementation.

Evidence: `../results/restart_validation_search_v1/progress.json`,
`../results/restart_validation_search_v1/selected_configurations.json`,
`../results/restart_validation_verified_v2/verification.json`,
`../results/restart_validation_selection_v1/selection_audit.json` and
`../results/restart_search_tests_v3/verification.json`.

## Selected neural outer-refit settings

Common LR is selected by equal weighting of seven inner groups and four neural
algorithms. Each algorithm's refit rounds are the median of its seven selected
inner checkpoint rounds. Conventional settings are selected separately using
equal-inner-group mean RMSE; their exact settings remain in the selection JSON.

| Outer group | Common LR | Centralized rounds | FedAvg | FedProx | FedNova |
| --- | ---: | ---: | ---: | ---: | ---: |
| Angers | .003 | 9 | 78 | 78 | 17 |
| Auzeville | .003 | 12 | 22 | 22 | 28 |
| Copenhagen + Taastrup | .001 | 23 | 80 | 80 | 57 |
| Grignon | .001 | 26 | 80 | 80 | 60 |
| Jyndevad | .003 | 10 | 60 | 60 | 20 |
| Kassel | .001 | 21 | 80 | 80 | 47 |
| Reading | .003 | 12 | 67 | 67 | 20 |
| Sanmarcoargentano | .001 | 21 | 80 | 80 | 47 |

## Interpretation and next boundary

286 of 672 neural candidates (42.6%) selected the final allowed round; four
of eight outer folds selected the highest LR in the frozen grid. These are
verified fixed-budget choices, not evidence of universal convergence or globally
optimal settings. Do not silently increase budgets or tune from outer scores.
Any convergence-sensitivity amendment must be prospectively versioned before
outer-test evaluation. Validation scores are selection evidence, not final
held-out research results; the earlier diagnostic exposure remains disclosed.

Next: specify/freeze study seeds, these refit settings, metric aggregation,
support panels and one-time outer-inference policy; implement/test a bounded
resumable outer runner before execution. Preserve negative predictions and
explicit unavailable R2. Report equal-group and pooled RMSE/MAE separately.
Knowledge guidance, ranking, formal privacy and deployment remain disabled or
unimplemented. No raw data, Paper 1 or open LaTeX source was changed.
