# R14 matched clipping/noise preparation

2026-10-14. R14a protocol and R14b core/coordinator acceptance VERIFIED_COMPLETE.
The actual160-run execution and comparative analysis remain separate tasks.

Fresh cached audits verified all40 R13 controls and200 v3 simulations, including
the8564-event prior ledger. R13 also binds the strict224-candidate selection audit.
No parent model reinference or retraining occurred. Frozen protocol:
`results/restart_matched_noise_protocol_v1`; execution:
`results/restart_matched_noise_execution_v1`. Both frozen before real R14 training.

Reuse40 existing controls via hashes of fit, prediction cache, completion and
attempt journals. Plan160 NEW runs:40 clipped controls +120 noisy variants with
multipliers2/4/8. All use fresh matched initialization (no trained warm start),
the same five seeds/eight holdouts, target scale1, LR.01, hidden32, batch32,
four local SGD steps, gradient clip1 and equal seven-client aggregation.
Outer0..7 budgets:133,129,114,149,106,109,106,128. These inner-selected budgets
are fixed; no outer-outcome tuning. C=.04, replacement sensitivity2C/7,
coordinate noise SD=multiplier*2C/7. New stream namespace:`r14-matched-v1:`.

Prior8564 event rows/IDs are preserved exactly. Planned14610 new events would
total23174. The unchanged Gaussian zCDP accountant follows the
[Bun--Steinke bound](https://arxiv.org/html/1605.02065). Hypothetical cumulative
per-slot epsilon bounds at delta1e-5 range1297.067440--1377.757121. These large
conditional bounds do not support strong privacy. They exclude nonprivate tuning,
unnoised controls, training diagnostics and test-metric releases. Public NumPy
seeds and finite precision are not certified release machinery. The actual
pipeline has NO application DP/SecAgg guarantee; production mode is rejected.
Unnoised-control budget is null, not zero. All conditional costs compose across
overlapping folds/seeds/configurations for each protected slot separately.

New implementation: `restart/matched_noise_protocol.py`,
`matched_noise_training.py`, `matched_noise_inference.py`,
`matched_noise_runner.py`, `verify_matched_noise.py`, and three test modules.
Features: strict tensor/roster/input checks; atomic state-plus-event checkpoints;
exact resume; once-only test intent before forward; fail-closed ambiguity;
bounded execution; explicit retry; OS writer exclusion; monotonic event ledger;
independent cached metrics/support and scalar-accounting audit.

41 relevant tests PASS, zero errors/failures:8 protocol,24 core/dependency
(8 new core,16 existing primitives/accountant),9 coordinator tests. Includes
all four variants,81-round synthetic smoke beyond legacy80 cap, exact interrupted
recovery, no cached model forward, historical ledger tampering, source/runtime
mutations, missing journals, active writers and ambiguous inference rejection.
Earlier protocol v1 also passed8; final protocol v2 adds a post-audit ledger hash
guard. No real results are inferred from these software tests.

Commands (existing `.venv/Scripts/python.exe`):

```text
-m paper2.restart.test_matched_noise_protocol --output paper2/results/restart_matched_noise_protocol_tests_v1
-m paper2.restart.test_matched_noise_protocol --output paper2/results/restart_matched_noise_protocol_tests_v2
-m paper2.restart.matched_noise_protocol freeze
-m paper2.restart.matched_noise_protocol verify
-m paper2.restart.test_matched_noise_training --output paper2/results/restart_matched_noise_training_tests_v1
-m paper2.restart.test_matched_noise_runner --output paper2/results/restart_matched_noise_runner_tests_v1
-m paper2.restart.matched_noise_runner --freeze --output paper2/results/restart_matched_noise_execution_v1
```

Protocol PASS_FROZEN_MATCHED_PROTOCOL; execution FROZEN_MATCHED_EXECUTION_NO_TRAINING.
Raw data, Paper1, open LaTeX, frozen older sources/results remain unchanged;
no installs. Process command-line discovery via Win32_Process was denied;
Get-Process and per-study OS locks remain available. One documentation patch
failed its out-of-order context check and changed no files; corrected patch
applied. No experimental failure during preparation.

Next: verified execution dry-run, bounded starter, independent audit, unchanged
remaining batches and full comparative analysis. Reuse `summarize_outer.aggregate`
unchanged with40 cached references, equal-group/pooled metrics and support panels.
Retain prior development exposure, unequal/tiny groups, five seeds not independent
regions, Auzeville-only265 unseen systems,80/224 capped inner candidates, grid-edge
choices and11/56 selected inner fits losing to the matching training constant.
R06/R07 require valid independent agronomic units/scope/comparability; certified
privacy/SecAgg, remote deployment and final evidence-based manuscript remain open.
