# Public-work calibration of update clipping and simulated noise

2026-10-14. Follow-up development experiment, version3. Existing versions,
including adverse results and the v1 failure, are retained unchanged.

## Diagnosis

Recomputed the seven clients' first local update at the public Angers/seed123
initialization. Mean update norm: 0.0088162; maximum: 0.0347547. The model has
1793 parameters. With the v2 bound C=1 and seven slots, multiplier2 adds noise
of coordinate std 4/7, with RMS L2 norm sqrt(1793)*4/7 = 24.1965. Noise is
thus orders of magnitude larger than actual update signals. The previous
sensitivity/accounting formula was internally consistent; its clipping scale
was poorly aligned with the optimizer's parameter-update units.

Those norm diagnostics describe the public benchmark only. They are not used
to fit the replacement bound. A private implementation could not publish such
client diagnostics without additional protection.

## Fixed public-work design

V3 takes exactly four SGD minibatch steps per client per round, with gradient
L2 norm clipped to 1. A seeded permutation supplies batches; small clients
reshuffle/cycle as needed. All five variants use the same four-step policy.
No momentum, weight decay or changed LR is introduced. The public mathematical
bound on a local parameter delta is four times its shared learning rate.
Global update clipping uses C=LR*4, and Gaussian std remains
multiplier*(2C/7). Global clipping enforces the bound on the computed vector,
including possible floating-point differences from the analytic step bound.

This policy no longer calibrates against private client-size-dependent local
work. For LR .003, C=.012 instead of1; multiplier2 RMS noise becomes about
0.2904, preserving std/sensitivity=2. The Gaussian accounting ratio is unchanged
under matched scaling of sensitivity and noise. This does not eliminate the
small-client-population/dimensionality trade-off or certify private releases.

Model topology, targets, public preprocessing, seeds, per-fold LR/rounds,
variant names and noise multipliers remain as in the recorded development
design. Local work and the update clip bound are explicit amendments applied
to controls and noisy variants alike. New noise streams use the work-v3
namespace. The current shared ledger includes all 359 previously committed
v1/v2 noisy events, with unique historical prefixes.

## Actual testing and starter evidence

26 tests passed, zero failures/errors. New checks verify four-step work for
clients of sizes2 and200, the LR*4 delta bound, and unchanged accounting cost
when sensitivity/noise are scaled together. Recovery tests verify identical
weights/events across interruptions for every variant. Existing event
tampering, control averaging and cached-completion tests remain included.

V3 starter: five configurations completed and independently audited without
reinference. Angers/seed123 RMSE in t/ha:

| Variant | V2 starter | V3 starter |
| --- | ---: | ---: |
| Equal average | 4.16 | 3.82 |
| Clipped average | 4.16 | 3.82 |
| Noise multiplier2 | 203.27 | 4.03 |
| Noise multiplier4 | 522.91 | 3.88 |
| Noise multiplier8 | 4650.03 | 3.31 |

The extreme noise-driven predictions are absent in this starter. These
comparisons combine changed local work, calibrated clipping and independent
noise streams; they do not isolate a single causal contribution. A lower
single-seed noisy error does not prove that noise improves prediction.
The complete v3 matrix subsequently finished:200/200, zero failures.
Audit/comparison:restart_privacy_analysis_v3. Completion report:
restart_work_scaled_completion_report.md.

## Commands executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_work_scaled_simulation --output paper2/results/restart_work_scaled_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.amend_privacy_work_scale --output paper2/results/restart_privacy_study_protocol_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.work_scaled_privacy_simulation_runner --freeze --output paper2/results/restart_privacy_execution_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.work_scaled_privacy_simulation_runner --execute --cap 5 --output paper2/results/restart_privacy_simulation_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_work_scaled_privacy_simulation --output paper2/results/restart_privacy_verified_v4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.work_scaled_privacy_simulation_runner --execute --cap 195 --resume --output paper2/results/restart_privacy_simulation_v3
```

Files: work_scaled_regression_training.py, work_scaled_simulation_training.py,
work_scaled_privacy_simulation_runner.py, amend_privacy_work_scale.py,
test_work_scaled_simulation.py and verify_work_scaled_privacy_simulation.py.
Snapshot/hash packages version3 were created before its first training run.

All outcomes were already exposed before this diagnostic amendment. This is
a development sensitivity experiment, not independent validation or a method
novelty claim. Unnoised controls and public deterministic simulation seeds
preclude an application privacy guarantee. No raw dataset, original baseline,
earlier frozen source, Paper1 or open manuscript was modified.
