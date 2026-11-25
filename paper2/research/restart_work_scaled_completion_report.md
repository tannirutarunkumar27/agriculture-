# Calibrated simulation completed and audited

2026-10-14. All 200 v3 runs complete, zero pending, zero failure files.
Complete independent audit and matched group/seed summary:
results/restart_privacy_analysis_v3. Earlier versions/results remain unchanged.

## Implemented fix

The old C=1 bound was poorly aligned with client deltas (mean norm .0088162
in the first public Angers round). Multiplier2 produced RMS noise norm24.1965
across1793 parameters. The new fixed four-step optimizer, gradient clip1 and
C=LR*4 align the update bound with public optimizer work. For LR .003, C=.012.
Noise and sensitivity scale together, retaining the accounted noise ratio.
This is not a bound fitted to private client norms. All five variants share
the changed work policy; streams are independent across version/run identities.

26 tests passed before execution: work independence from client size, update
bounds, accounting ratio, exact recovery, event integrity and matched controls.
Five starter runs and 195 subsequent runs completed. Audit PASS_COMPLETE checks
all200 models/final checkpoints, accounting receipts, IDs/targets and cached
errors without held-out reinference. Ledger:8564 events (359 prior +8205 v3).

## Full results

RMSE in t/ha seed dry matter. Five seeds, eight regions, 607 systems per seed.

| Variant | Equal-region RMSE mean +/- seed SD | Pooled RMSE mean |
| --- | ---: | ---: |
| Equal average | 3.2915 +/- 0.1399 | 3.4367 |
| Clipped average | 3.2915 +/- 0.1399 | 3.4367 |
| Noise multiplier2 | 3.2703 +/- 0.1384 | 3.4374 |
| Noise multiplier4 | 3.2246 +/- 0.1709 | 3.4360 |
| Noise multiplier8 | 3.1124 +/- 0.1950 | 3.3159 |

The extreme noisy errors are removed. However all variants have negative
pooled R2 and the controls fit poorly. This does not establish good predictive
utility or that noise improves generalization. Seed SD is not regional
confidence; correlated folds and development amendments require disclosure.
Seen/unseen species/composition and per-region/seed metrics remain saved.

## Remaining learning issue

Training-only diagnostics of all40 equal-average controls show higher training
error than a constant reference. The initial pooled comparison was extended
to the optimizer's matching equal-client MSE and its optimal equal-client
constant: all40 are worse there too. Two tests passed, including unequal-client
data where the pooled mean would incorrectly imply underfit. Initial diagnostic
output v1 is retained; v2 records both objectives. No held-out reinference.

Next R12: study optimization/target units using training and inner-validation,
then prospectively freeze any further amendment. Do not alter the v3 matrix
from its outer scores or silently replace adverse outcomes.

## Commands and evidence

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.work_scaled_privacy_simulation_runner --execute --cap 195 --resume --output paper2/results/restart_privacy_simulation_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.summarize_work_scaled_simulation --output paper2/results/restart_privacy_analysis_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.diagnose_public_control_fit --output paper2/results/restart_control_fit_diagnosis_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_matched_control_diagnosis -v
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.diagnose_public_control_fit --output paper2/results/restart_control_fit_diagnosis_v2
```

Supporting fix/test/freeze commands: restart_work_calibration_report.md.
Actual checks:26 pre-execution tests and two diagnostic tests passed.
No actual application privacy certification: controls are unnoised, seeds are
public simulation seeds, and sampler/release/SecAgg certification remains open.
No raw data, baseline, earlier frozen source, Paper1 or manuscript overwritten.
