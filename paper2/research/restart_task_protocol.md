# Paper 2 restart: step 2 task and evaluation design

Date: 2026-10-13. This is a **draft research specification**, not a frozen
experiment or approval for unrestricted crop advice. Machine-readable contract:
`../configs/restart_task_protocol_v1.json`. No data were split, imputed, trained
or used to select hyperparameters in this step. Step-1 audit outcomes have been
inspected; this is a development dataset, not a fresh blind confirmation set.

## Primary estimand and unit of observation

Predict the conditional harvest seed dry-matter outcome of a specified crop
system under a specified recorded context/management package:

    y_i = sum_{species c in system i} harvest_seed_dry_matter_ic
    y_hat_i = f(context_i, system_composition_i, management_i)
    training objective = mean_i (y_hat_i - y_i)^2

The unit is t/ha and the observation key is `(experiment_id, management)`.
All expected components must have one verified harvest outcome. Intercrop
components are summed; repeated trait dates are not independent samples.
Current audit coverage is 607 systems, not 34,737 independent farms. Identity
response scaling is the starting design; no log transform or test-selected
transformation is introduced. Negative predictions are reported, not silently
clipped; negative source outcomes require review rather than automatic deletion.

Crop composition is an explicit conditional input, not the output label. Fixed
public eight-species membership indicators and sole/intercrop status encode it;
unseen species remain explicitly flagged. The target is neither profit, food
quality nor a calibrated physiological suitability probability. The model
estimates association in recorded experiments, not an intervention effect or
an outcome for every untested crop. Applied crop-specific treatment choices
are bundled interventions: do not attribute their effects to species alone.

## Prediction-time scope and input contracts

The initial claim is RETROSPECTIVE CONDITIONAL OUTCOME PREDICTION. Pre-sowing
recommendation is disabled until evidence establishes which variables were
planned/known at that time. Realized applied fertilizer/irrigation cannot simply
be called pre-sowing measurements, even if their units are documented.

Proposed context fields: documented initial soil properties and previous crop,
season and sowing-calendar descriptors. Proposed candidate descriptors: species
composition and explicitly defined management quantities. These are candidates,
not an approved feature list. Cultivar, density, mixing design, fertilizer type
and crop-specific interventions need coverage and scientific interpretation
review before use or exclusion. Missing values remain unknown.

Exclude measured plant traits, actual harvest dates, yield-derived metrics,
management IDs and experiment IDs as predictors. Site IDs identify groups,
not shortcuts to the response. Coordinates, country and site effects need a
prospective ablation and spatial-proxy review, not automatic inclusion.
No full-season realized climate is enabled. If climate is later added, specify
a fixed information cutoff and verify day-level coverage; future weather must
not leak into advice. Climatology must use a documented period available before
the decision, not be fitted from held-out future observations. Predicted weather
is not measured weather and requires separate provenance.

## Outer and inner evaluation

Proposed outer evaluation: leave each of the nine sites out once, with every
management/component/date/experiment record from that site kept together.
Do not choose a favorable test site using outcome scores. Before materializing
splits, check geographic aliases, shared coordinates, repeated plots and nearby
sites; merge dependent groups if needed and amend site counts accordingly.

Within each outer training set, use only its remaining sites for inner
site-held-out validation. Fit preprocessing and perform hyperparameter/checkpoint
selection separately inside each inner fold. Never use outer test outcomes for
selection, alpha tuning, early stopping or budget adjustment. A common neural
setting across algorithms is selected through a prospectively fixed validation
rule; model-specific centralized grids must have explicit bounded budgets.
Final outer refitting requires a training-duration rule derived from inner
selection (for example a specified aggregate selected epoch count), not outer
test monitoring. This rule and tie breaking remain to be frozen before training.

Primary score: equal-site mean of site RMSE (t/ha), to avoid allowing Auzeville's
large row count to dominate. Also report equal-site MAE, pooled row-weighted
RMSE/MAE, per-site values and support counts. R2 is unavailable for too-small
or constant-target groups, not replaced with zero. Five fixed RNG seeds assess
initialization/client sampling variability; they are not five new independent
regional trials. Use paired site/seed differences descriptively. Do not treat
overlapping training folds and repeated seeds as independent evidence for
confirmatory p-values or a field-level confidence interval.

Report three panels for every outer fold: all test systems, systems whose
component species all occur in outer training, and systems whose exact species
composition occurs in outer training. Derive support using training identity
and candidate descriptors only, never outcome thresholds. All-test performance
must remain visible; supported-subset reporting must not hide hard cases.

Holding Auzeville out leaves chickpea, lentil and durum wheat unseen in training.
Those cases cannot be called validated crop-selection extrapolation. Tiny sites
with four systems have unstable site metrics. No ordinary balanced five-fold
all-species generalization claim follows. A supported-subset recommendation
analysis must abstain outside its prospectively approved candidate set.

## Secondary comparison/ranking gate

Ranking is disabled. Step 1 found 47 management-matched candidate blocks, but
they are not randomized causal contrasts and not 47 independent experiments.
Review original experiments, cultivar/density, mixture proportions, treatment
design and harvest comparability. Eleven experiments lack publication references
in the supplied reference table. Their design cannot be assumed identical.

If approved later, compare ONLY actual tested alternatives within a reviewed
block. A candidate is a defined crop/management package, not a species name with
unspecified treatment. The stated utility would be harvest seed dry matter;
it must not be described as profitability or universal agricultural benefit.
For alternative a selected by a model, descriptive observed-set regret is:

    max_{b in reviewed observed alternatives} y_b - y_a

This is a restricted observed-set evaluation, not counterfactual regret over
all crops. Candidate rules, outcome handling, ties, replicates and aggregation
must be frozen before constructing labels or evaluating rankings. Do not average
different cultivar/density packages into a species mean without justification.
Never generate suitability labels from rules and use those same rules as
independent validation. Prices/costs would require additional linked evidence
and a separate utility definition.

## Preprocessing and baselines

For each inner/outer training fold: numeric training medians, training mean/SD
and missingness indicators; training categorical vocabularies with an unknown
token. Preserve structural unknowns and audit all-missing features. Targets
are never imputed. Rule inputs, if approved later, retain separate original-unit
values and original-observation masks, not reconstructed standardized values.

Centralized candidates: training-mean control, ridge, Random Forest regressor,
XGBoost regressor and small MLP regressor. Federated candidates: existing
sample-weighted FedAvg/FedProx and restricted constant-LR plain-SGD FedNova,
with the same MLP capacity/initialization and MSE. Existing cross-entropy
classification training and Macro-F1 evaluation cannot be reused unchanged.
Do not reuse LR=.2 or 80 rounds as validated regression settings. Search spaces,
epoch/round limits, resource accounting and refit rules must be frozen later.

Logical clients are TRAINING research sites, not farms. Inner/outer held-out
sites never send updates. Default full participation exposes unequal site data
volumes without introducing dropout confounding. Minimum client size and
batch rules need explicit smoke tests; country clients are not the default.
Public training-only central preprocessing is a simulation reference, not
private preprocessing. Tensor payloads do not equal actual network traffic.

## Knowledge, privacy and claims

The old class-aligned logit bias does not directly fit a continuous response.
Do not transplant it into regression or invent physiological penalties. First
select independently sourced knowledge compatible with actual features and the
chosen target, specify its mathematical role and negative controls, and freeze
validation-only guidance-strength selection. Until then knowledge is disabled.

Secure aggregation, differential privacy, accounting and deployment are separate
unimplemented gates. Neither logical federation nor absence of raw-row uploads
establishes formal privacy. Do not finalize a privacy-preserving title yet.
No Indian field-generalization claim follows from these European research sites.

## Completion and next implementation step

This step saved the task/protocol draft and explicit execution gates. It did
not freeze a complete experiment plan, train models, choose best-crop labels,
select actual splits or alter old artifacts. Next: approve the feature/timing
contract, review geographic groups and materialize a versioned regression table
and grouped partitions; test leakage/identity and train-only preprocessing.
Then smoke-test regression before any validation search or full study. Ranking,
knowledge and privacy require their own evidence and acceptance checks.
