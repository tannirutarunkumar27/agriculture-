# R12 full fixed-public control-search completion

2026-10-14. R12 bounded inner-search task VERIFIED_COMPLETE. User-authorized
remaining200 candidates all completed;224/224 total, zero pending/failures.
This does not establish convergence, independent field confirmation, ranking
or privacy. Active target remains conditional harvested seed dry matter,t/ha.

## Acceptance and actual verification

Dependencies: unchanged restart_control_search_protocol_v1,
restart_control_search_execution_v1, restart_public_data_v1 and frozen engine.
Project ledgers, Git state and actual prior execution/audit evidence inspected;
no Python writer before starting. Exclusive OS lock acquired by the runner.
Config/plan/source snapshots/runtime/data bindings verified before execution
and during the complete audit. No training-budget or grid changes occurred.

Fresh scoped suite17 tests passed, zero errors/failures. Evidence:
results/restart_control_search_tests_v4. Previously26 dependency tests remain
retained, not recounted as newly rerun. No claim of a full-repository test run.

The resumed cap200 command executed exactly200 new candidates. All224 attempts
are COMPLETE, one per planned candidate. The previous24 completion hashes
match restart_control_search_verified_v2; their results were not replaced.
A subsequent completed-study resume returned executed0, confirming cache reuse.
Its progress.json describes that last verification invocation (executed0),
not the preceding200-run execution count. Attempt journals retain every run.

Full audit PASS_COMPLETE_INNER_SEARCH at
results/restart_control_search_verified_v3/verification.json independently
reproduces final/best inner validation predictions and final equal-client
training metrics, checks full histories/budget snapshots/first-tie minima,
receipt/model hashes and all eight complete-search choices. Selection means
and median refit rounds were independently recomputed. No outer-model
inference, noisy training, new dependency or remote/public service occurred.
No active Python writer remains after completion.

## Verified inner selections

Each outer group's choice uses seven equally weighted inner-validation groups,
not pooled sample counts. The frozen order resolves exact configuration ties;
checkpoint ties use the earliest round. Each candidate completes240 rounds;
refit budget is the median best round of the seven chosen-configuration curves.
All groups select public target scale1 and LR .01. Mean RMSE is a selected
development inner score, not an outer-test generalization estimate.

| Excluded outer group | Equal-inner-group mean best RMSE,t/ha | Median refit rounds | Chosen-configuration cap hits /7 |
| --- | ---: | ---: | ---: |
| Angers | 0.994603 | 133 | 2 |
| Auzeville | 1.049884 | 129 | 1 |
| Copenhagen+Taastrup | 1.042190 | 114 | 2 |
| Grignon | 1.024567 | 149 | 0 |
| Jyndevad | 1.052776 | 106 | 0 |
| Kassel | 1.104487 | 109 | 1 |
| Reading | 1.106085 | 106 | 1 |
| Sanmarcoargentano | 1.057746 | 128 | 0 |

Machine-readable full grid/selection evidence:
results/restart_control_search_v1/selection.json. All configurations and
cap/grid-edge flags remain available; no adverse candidate was removed.

## Limits and retained adverse findings

80/224 candidates selected the240-round cap;7/56 curves at the chosen
configurations also hit the cap. All selections lie at both the lower target-
scale and upper LR grid edges. Therefore this finite search does not establish
optimal hyperparameters or convergence. It must not be silently expanded on
the basis of exposed outer scores.

At the chosen inner checkpoints,11/56 training fits remain worse than their
matched equal-client optimal constant, computed using the saved training
diagnostic at the selected round. This is a descriptive fit diagnostic, not an
execution error. Do not claim the earlier public-transform underfit concern is
universally resolved or infer noisy-model performance from unnoised controls.

The full expanded search includes the four pilot configurations, executed
under its own frozen identity; these overlapping runs are not independent
replications. Earlier starter documentation referring to two overlaps applied
only to that first inner fold. Original baseline/guard/privacy-study outer
outcomes were already exposed; all follow-up work remains development.
Nested folds share data and are not56 independent regions. No statistical
superiority, deployment utility or application privacy guarantee is established.

## Commands actually executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_search_runner --output paper2/results/restart_control_search_tests_v4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --resume --cap 200
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --resume --cap 1
```

Read-only monitoring inspected completion counts and failure journals. Final
PowerShell checks inspected all224 attempt statuses, prior receipt hashes,
80 cap hits,56 selected checkpoint training-fit diagnostics and selections.
No errors required repair. Authored changes:the six progress ledgers and this
report. Generated artifacts:200 additional run folders, complete selection,
study progress, new test evidence and a full audit. Paper1, raw data, existing
frozen sources/packages, historical results and open LaTeX were preserved.

## Next task, not executed here

R13:prospectively freeze a separate repeated-seed fixed-public refit/matched
follow-up using the verified inner choices, with explicit prior-exposure,
support-panel, equal-group versus pooled metrics and once-only outer inference
receipts. Any new clipped/noisy comparison must define matched work and update
its mechanism/accounting before execution. Retain original privacy matrices.
Agronomic-rule, ranking, certified privacy/SecAgg and remote deployment gates
remain unresolved. This request completes the remaining R12 search only; no
new outer or privacy experiment was launched.
