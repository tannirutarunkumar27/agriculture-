# Control conditioning and convergence pilot

2026-10-14. R12 is IN_PROGRESS. Four new inner-only diagnostic candidates
completed and independently verified. No outer-model inference or noisy training
was performed. This work follows the training-only underfit diagnosis; existing
baseline and v3 results remain unchanged.

## Prospective pilot

The first two inner folds in fixed manifest order were declared before training:
outer_0_inner_1 (validation Auzeville) and outer_0_inner_2 (validation
Copenhagen+Taastrup). Each compares fixed target scales1 and10 at LR .003,
seed42, hidden32, batch32 and four local gradient-clipped SGD steps. Training
runs to240 rounds and reports both80 and240-round prefixes. Targets are divided
by the public scale for MSE optimization and predictions multiplied by the same
scale for raw t/ha scoring. No private target statistics are fitted.

Checkpoint selection is minimum inner-validation RMSE, first exact tie wins.
All rounds are executed. This target transform changes optimization conditioning
and effective raw initialization; it does not change the raw target definition.
The trial is a development diagnostic, not independent confirmation or a final
hyperparameter selection. All previous exposure remains disclosed.

## Actual results

| Inner validation group | Target scale | RMSE at round80 | Best RMSE through240 | Best round | Final train equal-client MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auzeville | 1 | 2.6937 | 1.6421 | 240 | 2.1358 |
| Auzeville | 10 | 2.1966 | 1.3745 | 240 | 1.3390 |
| Copenhagen+Taastrup | 1 | 2.8432 | 1.0856 | 188 | 1.5326 |
| Copenhagen+Taastrup | 10 | 1.0752 | 1.0674 | 100 | 1.2707 |

RMSE units:t/ha. Constant-reference training MSE is1.6981 for the first fold
and1.8142 for the second. At240 rounds, scale10 beats the matching constant
on training data in both folds, whereas scale1 remains underfit in the first.
Both Auzeville candidates select the round cap, so convergence is not certified.
The scale10 Copenhagen candidate's best checkpoint is at100 despite improved
training fit at240, reinforcing validation checkpoint selection over longer-
training assumptions. Two folds cannot establish universally preferable settings.

## Verification

Three tests passed, zero failures/errors: correct target-unit restoration and
invalid scale rejection, explicit test-array rejection, and exact resumed
normalized-control weights/history. Logs:restart_control_pilot_tests_v1.

Independent audit PASS_COMPLETE_INNER_PILOT verifies all four source/data/
checkpoint identities, every saved round, budget-prefix reports and minimum/
first-tie selections. Final and selected inner checkpoints reproduce raw
validation metrics; final training MSE is independently recomputed with equal
client weighting. Only training/inner-validation models are inferred in audit.
Evidence:restart_control_pilot_verified_v1.

## Expanded search now frozen

224 planned new candidates:56 inner folds x2 target scales x2 learning rates
(.003/.01),240-round cap, seed42, reporting80/240 prefixes. This is a new
fixed-public-transform/control study, not rerunning the earlier1120-candidate
search. The expanded plan must select configurations using equal-inner-group
mean validation RMSE per outer group, with first-configuration ties and median
selected round over seven inner folds for prospective refit. No incomplete
search selection is allowed. No final settings were selected in this phase.

Package:results/restart_control_search_protocol_v1. Status:
FROZEN_EXPANDED_SEARCH_NO_TRAINING. A separate capped coordinator with receipt,
recovery and selection tests is required before execution. Freeze that execution
code separately; do not edit the existing pilot/frozen dependencies.

## Commands

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_convergence --output paper2/results/restart_control_pilot_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_convergence freeze --output paper2/results/restart_control_pilot_protocol_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_convergence execute --output paper2/results/restart_control_pilot_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_convergence --output paper2/results/restart_control_pilot_verified_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.freeze_control_search --output paper2/results/restart_control_search_protocol_v1
```

New sources:control_convergence.py,test_control_convergence.py,
verify_control_convergence.py,freeze_control_search.py; config:
configs/restart_control_search_v1.json. No dependencies installed. Paper1,
raw data, previous results/frozen sources and open manuscript preserved.
