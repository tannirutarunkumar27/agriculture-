# Reviewer-response execution record, 2026-10-15

Input review: user-provided simulated referee report. It is an assessment,
not independent evidence. Original 1120 inner candidates, 320 primary outer
runs, 320 post-hoc guard runs, 224 fixed-public search candidates, 40 refits,
160 matched-noise evaluations, and privacy simulations were not rerun or edited.

## RR01: novelty positioning — VERIFIED_COMPLETE

Reviewed 20 primary papers/records in `reviewer_response_literature_v1.md`
(eight agriculture-specific closest works, three foundational FL methods,
three tabular-model comparisons, three heterogeneity methods, and three
geographic-validation studies). Outcome: existing research already covers
federated yield prediction, client heterogeneity, drift remedies, and tabular
neural/tree comparisons. No new-method claim is currently supported. The
testable contribution is a matched diagnostic analysis, pending execution.

## RR02: descriptive geographic sensitivity — VERIFIED_COMPLETE

Implemented `restart/reviewer_sensitivity.py` and four relevant tests,
including reproduction of the archived XGBoost aggregate. The script consumes
only `results/restart_outer_analysis_v1/analysis.json` per-region metrics and
writes `results/reviewer_sensitivity_v1/analysis.json` once, with source SHA-256.
It performs no training or test inference. Six new tests (RR02+RR03) pass.

| Scenario | Weighting | Systems | XGBoost RMSE | FedAvg RMSE | Best mean-RMSE method |
|---|---|---:|---:|---:|---|
| All 8 groups | Equal group | 607 | 1.2461 | 5.8373 | XGBoost |
| All 8 groups | Pooled | 607 | 1.7335 | 28.2909 | XGBoost |
| Exclude groups with <=6 systems | Equal group | 593 | 1.5855 | 8.7473 | XGBoost |
| Exclude groups with <=6 systems | Pooled | 593 | 1.7505 | 28.6223 | XGBoost |
| Leave out Auzeville | Equal group | 253 | 1.1585 | 1.3834 | XGBoost |
| Leave out Auzeville | Pooled | 253 | 1.5402 | 1.6972 | FedNova (1.5298) |

The output also has paired XGBoost-versus-method per-group RMSE differences
for each of five computational seeds, and all eight leave-one-group scenarios
under three weighting rules. The near-tie and ranking change without Auzeville
show site composition affects interpretation. Removing groups is not
independent validation, and seed SD is not location uncertainty. Pooled RMSE
is recomputed from group sample counts and squared RMSE before square root.

## RR03: dependence and hash audit — PARTIALLY VERIFIED

Implemented `restart/reviewer_data_audit.py` and two tests. Input is the
versioned 607-row public regression CSV; output is aggregate-only
`results/reviewer_data_audit_v1/analysis.json` with source SHA-256. There are
37 experiment IDs; all 37 have multiple system rows, and one contributes 64.
Experiments remain within geographic groups. Kassel and Reading each have
one experiment (four systems), Sanmarcoargentano two experiments (six systems).
Thus 607 is not a count of independent field experiments. No effective sample
size or uncertainty bound is inferred without a defensible dependence model.

Fixed-public hashing uses eight SHA-256 buckets per categorical field.
`previous_crop` has 13 distinct observed nonmissing values mapped into seven
buckets: 10 values occupy colliding buckets, forming nine distinct colliding
pairs; 221 rows are missing this field. `crop_season` and `pest_control` have
two observed categories each and no observed distinct-category collisions.
This describes the current dataset, not collision behavior on future sites.

Prediction-time availability and cultivar/planting-density harmonization are
not resolved by these counts. In particular, actual availability of
`pest_control`, applied fertilizer and irrigation at a *prospective* prediction
time must be specified before any operational forecast claim. All current
results remain retrospective conditional regression.

## Commands and verification

`python` resolves to an inaccessible WindowsApps alias in this shell; this
did not indicate a code failure. Switched to the existing project `.venv`.
Executed `& '.\.venv\Scripts\python.exe' -B -m unittest
paper2.restart.test_reviewer_sensitivity paper2.restart.test_reviewer_data_audit
-v`: six passed, zero failed after a targeted fix to one test's nested JSON
lookup. Ran each analysis module once with explicit input/output paths above.
No new model training, external download, package installation or cloud cost.

## Remaining actions / blockers

RR04: inspect existing convergence traces and FedAvg/FedProx update semantics.
RR05: freeze a genuinely matched ablation matrix with inner-only selection and
compute/resource budgets; RR06: implement a separate resumable runner and
smoke-test it before expensive training. RR07 outer follow-up is downstream
and will remain test-exposed development evidence. RR08 support-guard
confirmation and new-site generalization require an untouched comparable
dataset. RR09 formal privacy/SecAgg or live deployment requires a separate
threat model, validated mechanism and authorized infrastructure. RR10 portable
clean-room reproduction and final citations remain pending. These cannot be
closed by optimizing scores on the already exposed outer groups.
