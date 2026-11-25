# R12 expanded inner-control search: coordinator and starter

2026-10-14. R12a coordinator subtask VERIFIED_COMPLETE; R12 full search
IN_PROGRESS. Active objective remains retrospective conditional harvested seed
dry-matter regression, not best-crop classification. All prior outer outcomes
were exposed; this is development, not independent confirmation.

## Implementation and acceptance

New files:restart/control_search_runner.py, verify_control_search.py and
test_control_search_runner.py. Frozen candidate/training/aggregation sources
are reused unchanged. Execution freeze:results/restart_control_search_execution_v1,
binding original restart_control_search_protocol_v1, data, runtime, source
snapshots and coordinator/auditor code. No new packages installed.

Acceptance criteria passed: default non-mutating dry-run; positive execution
caps; explicit resume; exclusive OS writer lock; separate safe output paths;
protocol/source/data identity checks; complete cache hash/semantic verification;
identity-bound exact checkpoint recovery; completed-round finalization without
retraining; full failed-attempt tracebacks and explicit retry; last/best partial
checkpoint model metrics reproduced before recovery; incomplete selection
rejected; first exact ties; equal inner-group weighting rather than pooled
samples; median selected refit budget; independent partial/complete audits.

Dependencies: audited public transform/data, original frozen candidate engine
and 224-run plan. All training uses four gradient-clipped local SGD minibatch
steps per client, hidden32, batch32, seed42 and equal client aggregation.
Public target scales1/10 are applied for optimization and undone for scoring.
Frozen grid LR .003/.01 and240-round cap were not altered.

Tests actually executed:15 scoped tests first passed at
restart_control_search_tests_v1; then17 scoped tests passed at
restart_control_search_tests_v2 after adding partial-model integrity and
independent audit tests. A separate26-test dependency suite passed at
restart_control_search_dependencies_tests_v1. Final scoped/dependency total43,
zero errors/failures. This is not a claim that the entire repository suite was
rerun, nor that software tests establish agronomic validity or privacy.

## Actual bounded execution

Dry-run passed. Four expanded-plan candidates executed, each with one COMPLETE
attempt. results/restart_control_search_v1/progress.json records4 complete,
220 pending, no selection available. Independent audit PASS_PARTIAL_INNER_SEARCH
at restart_control_search_verified_v1 verifies receipt hashes, every saved
round, minimum/first-tie checkpoint, budget-prefix summaries and final/best
inner model metrics. Training metrics are independently computed with equal
client weighting. No outer-model inference or noisy training occurred.

All four use outer_0_inner_1: Angers excluded from training/selection and
Auzeville the inner validation group. RMSE units:t/ha; training MSE:t²/ha².

| Target scale | LR | Best validation RMSE | Best round | Final equal-client train MSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | .003 | 1.642139 | 240 | 2.135815 |
| 1 | .01 | 0.894949 | 239 | 0.788008 |
| 10 | .003 | 1.374479 | 240 | 1.339032 |
| 10 | .01 | 1.248485 | 215 | 0.904796 |

These are not final hyperparameter choices or multi-region superiority
evidence. Scale1/LR .01 performs best on this one inner fold; scale10 is not
uniformly preferable. Two minima hit the budget cap, and LR .01 is a grid
edge. Two .003 configurations overlap the earlier pilot, separately executed
under the expanded protocol identity; do not count these as independent
replications or pool duplicate findings into scientific sample sizes.

Final selection is allowed only after all224 candidates verify. For each of
eight outer groups: compare the four configurations using the unweighted mean
of seven inner-group best validation RMSE values; first configuration in the
frozen order wins exact ties; prospective refit rounds are the median of the
seven selected checkpoint rounds. Store every configuration and budget/grid
edge flag. No interim selection file exists.

## Commands actually executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_search_runner --output paper2/results/restart_control_search_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_search_runner --output paper2/results/restart_control_search_tests_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --freeze --output paper2/results/restart_control_search_execution_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --cap 4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_work_scaled_simulation --output paper2/results/restart_control_search_dependencies_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --cap 4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v1
```

Inspection corrected an initial wrong module location (paper2/ instead of
paper2/restart/); no writes occurred from that command. CIM process enumeration
was denied; Get-Process showed no Python processes before training, and the
runner's actual OS writer lock enforces exclusion. No training/test failures
occurred. Synthetic failure/tamper fixtures passed and are not real-study
failures. All test and experiment evidence remains saved.

## Next action and gates

Resume unchanged in bounded batches, checking existing writer state. Example
next command (not executed in this phase):

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_runner --output paper2/results/restart_control_search_v1 --execute --resume --cap 20
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v2
```

Do not rerun the completed original1120 candidates or closed privacy matrices.
Do not edit this new execution's frozen sources. Failures require inspecting
attempts/checkpoints before explicit --retry-failed. Repeated-seed refit and
any new matched privacy simulation require separate prospectively frozen
protocols after full selection; no changes based on exposed outer scores.
R06 agronomic-rule validity, R07 ranking comparability, R09 remote deployment
and release-certified privacy/SecAgg remain blocked/pending as documented.
Paper1, raw data, historical results/packages and open LaTeX were untouched.
