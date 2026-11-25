# V. Results

All reported results were obtained from completed cached evaluations that passed
independent provenance and metric audits. Unless stated otherwise, values are
means plus or minus sample standard deviation across the five fixed seeds.
Standard deviations describe computational variability; they are not confidence
intervals over independent geographic trials.

## A. Original eight-method outer evaluation

The original prospectively frozen study completed 320 evaluations: eight held-
out geographic groups, five seeds, and eight methods. Table 1 summarizes the
all-system errors. Equal-group RMSE was the primary measure, while pooled RMSE
weighted each of the 607 systems equally.

**Table 1. Original grouped outer-evaluation results (RMSE, t/ha).**

| Method | Equal-group RMSE | Pooled RMSE |
| --- | ---: | ---: |
| XGBoost | 1.2461 +/- 0.0000 | 1.7335 +/- 0.0000 |
| Random forest | 1.3181 +/- 0.0091 | 1.9958 +/- 0.0360 |
| Training mean | 1.4509 +/- 0.0000 | 1.9379 +/- 0.0000 |
| Ridge | 1.4694 +/- 0.0000 | 2.1511 +/- 0.0000 |
| FedNova | 5.8134 +/- 1.5956 | 29.2405 +/- 9.8813 |
| FedProx | 5.8373 +/- 1.6226 | 28.2904 +/- 9.8478 |
| FedAvg | 5.8373 +/- 1.6226 | 28.2909 +/- 9.8478 |
| Centralized MLP | 6.0307 +/- 1.6250 | 30.4896 +/- 10.0605 |

XGBoost had the lowest observed equal-group and pooled RMSE within this frozen
benchmark. This was a descriptive result, not an inferential superiority claim.
All eight methods had negative mean pooled R2; XGBoost's value was -0.1807.
Thus, its lower RMSE did not establish strong explained variance across the
combined held-out systems.

The gap between equal-group and pooled errors was especially large for the four
neural methods. Auzeville contained 354 of 607 systems and therefore dominated
the pooled summary. In one documented example, FedAvg at seed 123 had an
Auzeville RMSE of 36.7482 t/ha, compared with group RMSEs between approximately
0.54 and 1.76 t/ha elsewhere. The failure was retained in the original result;
neural hyperparameters and budgets were not changed after observing it.

The geographic XGBoost errors also varied despite its deterministic zero seed
SD: RMSE ranged from 0.5452 t/ha at Reading to 1.8594 t/ha at Auzeville.
Reading and Kassel contained only four systems each, so their apparently low
errors should not be interpreted as stable regional estimates. The accepted
figures `original_baseline_rmse` and `original_xgboost_geographic_rmse` show the
complete method and group patterns, including all seed values and group counts.

## B. Support-conditioned and post-hoc sensitivity analysis

Across the original splits, 342 systems had species and exact compositions seen
in the corresponding training sets. The remaining 265 systems had unseen
species/compositions, and all 265 belonged to Auzeville. Consequently, the
unseen panel represented one region rather than general cross-region
extrapolation.

Inspection after the outer failures showed that some encoded columns were zero
for every training row but nonzero in held-out data. For example, standardized
soil available water and phosphorus columns in the Auzeville holdout reached
values of approximately 263 and 66 even though both columns were identically
zero during training. A training-only zero-support guard was therefore evaluated
as a separate fixed-model sensitivity analysis. It set only all-training-zero
input columns to zero at inference and did not retrain models, change budgets,
or use outcomes to choose columns.

**Table 2. Post-hoc zero-support sensitivity (equal-group RMSE, t/ha).**

| Neural method | Original | Guarded post-hoc sensitivity |
| --- | ---: | ---: |
| Centralized MLP | 6.0307 +/- 1.6250 | 1.2130 +/- 0.0196 |
| FedAvg | 5.8373 +/- 1.6226 | 1.3438 +/- 0.0295 |
| FedProx | 5.8373 +/- 1.6226 | 1.3438 +/- 0.0295 |
| FedNova | 5.8134 +/- 1.5956 | 1.1820 +/- 0.0204 |

For FedAvg at seed 123, Auzeville RMSE changed from 36.7482 to 0.9960 t/ha.
This quantifies the behavior of the specified guard but does not isolate the
effect of an individual feature or establish performance on independently
collected data. The guard was designed after outer-test failures were exposed;
therefore Table 2 and `posthoc_guard_sensitivity` are explicitly post-hoc and
do not replace the original prospective results.

## C. Fixed-public unnoised refit

The separately frozen fixed-public stage used the 54-dimensional transform and
its own inner selection. All 40 outer refits completed and passed a cached-output
audit. The primary equal-group RMSE was 1.067407 +/- 0.021053 t/ha; pooled RMSE
was 1.201189 +/- 0.038476 t/ha. Equal-group MAE was 0.881704 +/- 0.022902,
pooled MAE was 0.948698 +/- 0.043564, and pooled R2 was
0.432669 +/- 0.036004.

Results differed across geographic groups, as shown in Table 3 and the accepted
`geographic_rmse` figure.

**Table 3. Fixed-public unnoised geographic results.**

| Held-out group | Systems | RMSE mean +/- seed SD | Mean R2 |
| --- | ---: | ---: | ---: |
| Angers | 75 | 1.657205 +/- 0.044743 | 0.083167 |
| Auzeville | 354 | 0.989938 +/- 0.059142 | 0.346560 |
| Copenhagen+Taastrup | 52 | 1.170048 +/- 0.032327 | -0.077797 |
| Grignon | 64 | 1.707368 +/- 0.061959 | -0.360441 |
| Jyndevad | 48 | 1.092247 +/- 0.053302 | 0.254387 |
| Kassel | 4 | 0.562621 +/- 0.011994 | -0.313808 |
| Reading | 4 | 0.433949 +/- 0.005893 | 0.662469 |
| Sanmarcoargentano | 6 | 0.925882 +/- 0.022788 | -0.109800 |

Four groups had negative mean R2 even though overall pooled R2 was positive.
The result demonstrates why pooled and equal-group summaries were retained
together and why small-group RMSE alone was insufficient for interpretation.
All 40 final training fits outperformed their matched equal-client constant, but
this did not guarantee generalization to every outer group.

The training-seen species panel contained 342 systems from all eight groups and
had equal-group RMSE 1.070628 +/- 0.024447 t/ha. The unseen panel contained 265
Auzeville systems and had RMSE 0.979646 +/- 0.065301 t/ha. The lower unseen-panel
error must not be interpreted as universal unseen-species capability because
the seen and unseen panels represented different systems and geographic
distributions. Exact-composition panels had the same membership in these splits.

The fixed-public result is a later development stage with a different feature
transform, weighting rule, selection search, and refit budget from the original
eight-method study. Its numerical values are therefore not presented as a
matched correction or superiority comparison against Table 1.

## D. Matched clipping and aggregation-noise utility

The matched follow-up added 160 new evaluations and reused 40 bound unnoised
reference caches, giving eight groups and five seeds for each of five variants.
All 40 clipped-control prediction arrays exactly matched the corresponding
unnoised reference arrays. This showed no detected prediction difference in
those pairs, but per-round diagnostics were insufficient to claim that clipping
was never active.

**Table 4. Matched aggregation-noise utility results (RMSE, t/ha).**

| Variant | Equal-group RMSE | Pooled RMSE | Mean pooled R2 | Fits worse than matched constant |
| --- | ---: | ---: | ---: | ---: |
| Cached unnoised reference | 1.067407 +/- 0.021053 | 1.201189 +/- 0.038476 | 0.432669 | 0/40 |
| Clipped control | 1.067407 +/- 0.021053 | 1.201189 +/- 0.038476 | 0.432669 | 0/40 |
| Noise multiplier 2 | 1.182853 +/- 0.115717 | 1.368078 +/- 0.120487 | 0.260112 | 0/40 |
| Noise multiplier 4 | 2.218085 +/- 0.491871 | 2.770127 +/- 0.852787 | -2.243366 | 29/40 |
| Noise multiplier 8 | 5.628243 +/- 0.500009 | 7.238383 +/- 1.108712 | -19.970877 | 40/40 |

Mean paired increases in equal-group RMSE relative to the cached reference were
0.000000, 0.115446, 1.150678, and 4.560836 t/ha for the clipped, multiplier-2,
multiplier-4, and multiplier-8 variants, respectively. Error increased with the
noise multiplier in this fixed experiment. The large negative R2 and weak
training fits at multipliers 4 and 8 were retained as adverse findings rather
than treated as execution failures.

Support-conditioned behavior also degraded with noise. At multiplier 2,
equal-group RMSE was 1.178797 +/- 0.113484 t/ha for 342 seen systems and
1.220762 +/- 0.216734 t/ha for the 265 Auzeville-only unseen systems. At
multiplier 8, the corresponding values were 5.591927 +/- 0.626727 and
7.376316 +/- 2.435713 t/ha. The accepted matched `rmse_overview`,
`geographic_rmse`, and `support_rmse` figures retain all variants, group counts,
seed values, and support counts.

The complete ledger contained 23,174 events: 8,564 retained historical events
and 14,610 new noisy-round events. Conditional noisy-only cumulative epsilon
bounds at delta \(10^{-5}\) ranged from 1297.067440 to 1377.757121 across
client slots. These values were too large to support a strong privacy claim and
did not cover nonprivate selection, controls, evaluation releases, deterministic
research randomness, or secure aggregation. The results therefore characterize
utility under controlled aggregation noise only; they do not establish
application differential privacy or production privacy.

## E. Summary of empirical findings

Three patterns were consistent with the evidence. First, the original frozen
baseline comparison favored tree ensembles in observed RMSE, while the neural
methods exhibited severe held-out support failures. Second, a training-only
support guard greatly changed those fixed-model neural predictions, but its
post-hoc design prevents it from serving as independent confirmation. Third,
the later fixed-public matched study produced substantially better unnoised
neural errors, while stronger aggregation noise caused progressively poorer
utility and weak training fits in the evaluated setting.

These stages answer different development questions and should not be collapsed
into a single model ranking. None establishes best-crop recommendation,
agronomic benefit, universal geographic generalization, a novel federated
optimizer, or an end-to-end privacy guarantee.
