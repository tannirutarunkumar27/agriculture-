# R14 complete execution, independent audit and descriptive analysis

2026-10-14. R14 VERIFIED_COMPLETE: all160 planned new runs completed with
160 COMPLETE attempts, zero failed attempts and zero pending runs. The40 R13
unnoised controls were reused from bound caches, without retraining. All eight
geographic holdouts and five declared seeds are present for each variant.

Independent audit: `results/restart_matched_noise_verified_v2/verification.json`,
PASS_COMPLETE_MATCHED_STUDY. Complete analysis:
`results/restart_matched_noise_analysis_v1/analysis.json`,
VERIFIED_COMPLETE_MATCHED_ANALYSIS. Analysis includes its own fresh complete
cached audit. No model test inference was repeated. Frozen protocol, source,
runtime, data, parent selections, journals, model/checkpoint and prediction
receipts passed their bindings. No settings were changed from exposed test scores.

The original session69296 finished and was no longer available to poll. Saved
batch progress reports156 newly executed runs after the four-run starter, total160.
Actual completions and all160 journals were inspected before the independent
audit. No new training process or duplicate writer was launched this session.

## Complete matched findings

Errors below are t/ha of harvested seed dry matter, mean +/- descriptive SD
across five seeds. Equal-group RMSE is primary; pooled RMSE weights systems.
These are repeated-seed development results, not independent geographic CIs.

| Variant | Equal-group RMSE | Pooled RMSE | Mean pooled R2 | Saved fits worse than matching training constant |
| --- | ---: | ---: | ---: | ---: |
| Cached unnoised reference | 1.067407 +/-0.021053 | 1.201189 +/-0.038476 | 0.432669 | 0/40 |
| Clipped control | 1.067407 +/-0.021053 | 1.201189 +/-0.038476 | 0.432669 | 0/40 |
| Noise multiplier2 | 1.182853 +/-0.115717 | 1.368078 +/-0.120487 | 0.260112 | 0/40 |
| Noise multiplier4 | 2.218085 +/-0.491871 | 2.770127 +/-0.852787 | -2.243366 | 29/40 |
| Noise multiplier8 | 5.628243 +/-0.500009 | 7.238383 +/-1.108712 | -19.970877 | 40/40 |

All40 clipped-control prediction arrays exactly match their unnoised reference
arrays. This establishes no detectable prediction difference in these pairs;
the study does not log enough clipping diagnostics to assert that clipping was
never numerically active. The public bound C=LR*4 matches the four gradient-
clipped local steps. Aggregate error increases with noise multiplier in this
fixed experiment. This is not proof of monotonic behavior for every group/seed.

Mean paired equal-group RMSE differences from the cached reference are
0.000000, +0.115446, +1.150678 and +4.560836 for clipped,2,4,8 respectively.
No inferential significance or universal superiority is claimed. High-noise
negative R2 and weak training fits are retained adverse findings, not execution
failures. Do not change noise/work/settings now to make those results favorable.

Original baselines and post-hoc support-guard studies used different development
protocols/preprocessing. Do not combine their numerical rankings with this
matched study as if every training condition were identical.

## Accounting and validity

23174 recorded events exactly match the frozen prospective full-study ledger:
8564 historical events retained unchanged plus14610 new noisy rounds. The audit
independently recomputed per-slot scalar rho/epsilon as well as cached metrics.
There is no sampling amplification. Conditional cumulative epsilon bounds at
delta1e-5 remain1297.067440--1377.757121 across slots, far too large to support
a strong-privacy claim. They apply only to an idealized noisy-only transcript.
Nonprivate inner selection, controls, diagnostics, test-metric releases, public
deterministic RNG and uncertified finite precision are not covered. No
application DP, secure aggregation or production privacy is established.

The complete analysis retains five support panels, regional errors and paired
seed differences using the original `summarize_outer.aggregate` unchanged.
There are607 distinct systems per seed, not3035 independent observations.
Seen panels cover342 systems/eight groups; unseen species/composition panels
cover265 Auzeville systems/one group. Geographic sizes are highly unequal,
including groups with only4 or6 systems. Prior outer exposure, grid-edge inner
choices,80/224 cap hits and11/56 weak selected inner fits remain limitations.

## Figures, tests and commands

R10b matched figures are accepted at `results/restart_matched_noise_figures_v1`:
three plots (RMSE overview, geographic RMSE, support-panel RMSE), each in
PNG/PDF/SVG with exact plot values, captions and source/asset hashes. All three
PNGs visually inspected: labels/legends readable, markers/error bars inside
limits, shared comparison scales, all five variants and support counts retained.
PDF/SVG use the same plotted figures; separate PDF rendering not performed.

Prior R14 preparation passed52 scoped tests. This continuation added five final
figure tests, all PASS, zero errors/failures. The targeted visibility regression
fixes the earlier R13 v1 seed-marker clipping; accepted R13 plots are v2.
No frozen training code or numerical experimental results were edited.

Successful commands using the existing `.venv/Scripts/python.exe`:

```text
-m paper2.restart.verify_matched_noise --output paper2/results/restart_matched_noise_verified_v2
-m paper2.restart.summarize_matched_noise --output paper2/results/restart_matched_noise_analysis_v1
-m paper2.restart.publication_figures --study matched --output paper2/results/restart_matched_noise_figures_v1
```

Also checked all40 clipped/reference cached prediction pairs using
numpy.array_equal:40/40 exact matches. No model forward, retraining, package
installation, cloud costs, network ports, raw-data changes, Paper1 edits or
open-LaTeX edits occurred. New local recovery/audit guide:
`research/restart_reproducibility_guide.md`; it explicitly does not claim Linux
validation or portable absolute-path manifests.

Next ready work: R10 baseline-study figures and evidence-based manuscript
synthesis, plus reproducibility packaging checks. R06/R07 agronomic validity and
ranking comparability remain blocked; R09 remote infrastructure absent; R08
certified privacy/SecAgg remains a separate unfinished requirement. Completed
R14 training must not be rerun. Hourly continuation remains active for open work.

Baseline-figure interfaces inspected for the next task: `summarize_outer.py`
provides the original320-run cached analysis; `guard_sensitivity.py:analyze`
provides the separate post-hoc320-run sensitivity analysis. A read-only lookup
of the nonexistent `summarize_guard.py` failed; file inventory identified the
correct existing module. No experiment/source change resulted. Keep original
baseline, post-hoc sensitivity and fixed-public matched studies explicitly
separate when presenting the next figures.
