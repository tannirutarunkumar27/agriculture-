# Complete baseline comparison

All 320 runs re-audited: eight geographic groups, 607 systems per seed, five
seeds, eight methods. RMSE units are t/ha seed dry matter. Equal-group RMSE
averages the eight group errors per seed; pooled RMSE combines all 607 rows.

| Method | Equal-group RMSE mean +/- seed SD | Pooled RMSE mean +/- seed SD |
| --- | ---: | ---: |
| XGBoost | 1.2461 +/- 0.0000 | 1.7335 +/- 0.0000 |
| Random Forest | 1.3181 +/- 0.0091 | 1.9958 +/- 0.0360 |
| Training mean | 1.4509 +/- 0.0000 | 1.9379 +/- 0.0000 |
| Ridge | 1.4694 +/- 0.0000 | 2.1511 +/- 0.0000 |
| FedNova | 5.8134 +/- 1.5956 | 29.2405 +/- 9.8813 |
| FedProx | 5.8373 +/- 1.6226 | 28.2904 +/- 9.8478 |
| FedAvg | 5.8373 +/- 1.6226 | 28.2909 +/- 9.8478 |
| Centralized MLP | 6.0307 +/- 1.6250 | 30.4896 +/- 10.0605 |

XGBoost has the lowest observed error under both summaries. All methods have
negative pooled R2, including XGBoost (-0.1807). Neural errors are especially
large on Auzeville: FedAvg seed123 RMSE is 36.75 there versus 0.54--1.76 in its
other groups. Diagnose input support/scaling/extrapolation separately; the cause
is not established here. Preserve frozen results and do not tune from test scores.

There are 342 training-seen species/composition systems and 265 unseen systems.
All unseen cases occur in Auzeville, limiting cross-region interpretation.
Panel equal-group errors average only nonempty groups; counts are saved.
Seed SD represents initialization variability, not geographic confidence.
Deterministic repeated references can have zero SD. Folds share training data.
The 28 paired method comparisons are descriptive, with no significance claims.

Outputs: results/restart_outer_analysis_v1/analysis.json (40 method/panel
summaries, 200 method/seed/panel entries, regional metrics, 28 paired comparisons)
and audit/verification.json (fresh PASS_COMPLETE). No models retrained or
reinferred. Frozen code and settings unchanged.

Executed commands:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_outer_summary -v
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.summarize_outer --output paper2/results/restart_outer_analysis_v1
```

Three tests passed: unequal-group weighting, overlapping holdout rejection and
empty-panel handling. Complete real analysis verified eight disjoint geographic
groups per method/seed and all cached completion hashes and assets.
