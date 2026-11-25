# VI. Discussion

## A. Interpreting the baseline and support results

The original grouped evaluation did not show a general advantage for neural or
federated models. XGBoost had the lowest observed equal-group and pooled RMSE,
while centralized MLP, FedAvg, FedProx, and FedNova produced severe Auzeville
errors. That outcome is scientifically useful because it shows how a seemingly
ordinary tabular preprocessing decision can interact with geographic support.
Columns that were identically zero in training could activate large,
initialization-dependent neural weights when nonzero values appeared in the
held-out group. Tree ensembles did not exhibit the same numerical failure in
this experiment, although their negative pooled R2 also cautions against a
strong generalization claim.

The training-only support guard greatly reduced fixed-model neural errors, but
its interpretation is constrained by its history. It was designed after the
outer failures were visible, and the same test folds were evaluated again.
Therefore, the guarded values identify a plausible failure mechanism and a
candidate input policy; they do not erase the original result or provide
independent confirmation. A future study should specify the handling of
all-training-constant and unseen inputs before any outer evaluation and then
test that policy on new experimental groups.

The fixed-public refit further showed that stable neural behavior was possible
under a different, prospectively frozen representation and training protocol.
However, its lower RMSE cannot be attributed to federation, the fixed transform,
equal client weighting, longer refit budgets, or any one design choice in
isolation. These factors changed together, and the stage followed earlier outer
exposure. It should therefore be interpreted as a separately documented
development study rather than a matched correction to the original baseline.

## B. Equal-group, pooled, and support-conditioned views

Equal-group and pooled metrics answered different questions. Equal-group RMSE
assigned the same influence to each of eight held-out geographies, including
groups containing only four or six systems. Pooled RMSE emphasized Auzeville,
which contained 354 of 607 systems. Neither weighting was universally preferable:
the first made small-group instability influential, whereas the second could
hide poor performance outside the dominant group. Reporting both prevented an
implicit choice of estimand from being mistaken for model improvement.

The support panels also require careful reading. All 265 systems containing
unseen species or exact compositions came from Auzeville. Their error therefore
combined biological support status with one location's target distribution and
management conditions. The apparently favorable unseen-panel result in the
fixed-public stage is not evidence that unseen crops can generally be predicted.
It instead demonstrates why support status, geography, and sample count must be
shown together.

Four fixed-public held-out groups had negative mean R2 even though the pooled R2
was positive. Kassel and Reading each contained only four systems, so their R2
and RMSE were particularly sensitive to a small number of outcomes. These
adverse group-level results limit claims based only on aggregate metrics and
reinforce that five training seeds are not five independent regional trials.

## C. Aggregation noise and privacy interpretation

The matched study isolated utility changes under the implemented clipping and
Gaussian aggregation-noise variants. The clipped control reproduced all 40
unnoised prediction arrays exactly, while error increased at noise multipliers
2, 4, and 8. At the two larger multipliers, many or all saved fits were worse
than their matched training constant and pooled R2 became strongly negative.
These outcomes were retained rather than retuned away and show the practical
cost of the evaluated noise levels under the fixed optimizer work.

This experiment does not establish an application privacy guarantee. The
conditional noisy-only composition bounds were very large, and the accounting
excluded nonprivate hyperparameter selection, unnoised controls, diagnostics,
test-metric releases, deterministic research noise seeds, and secure
aggregation. The server-side simulation also assumed a trusted curator able to
inspect client updates. Formal protection against an untrusted server would
require a vetted secure-aggregation or distributed-noise protocol, authenticated
participants, reviewed dropout/collusion assumptions, private preprocessing and
release policy, and independently validated accounting.

## D. Relation to prior agricultural federated learning

The findings complement rather than numerically compete with published
federated yield studies. The soybean-yield and localized-pruning study by
[Li et al.](https://doi.org/10.1016/j.eswa.2023.122847) considered a different
crop, forecasting setup, personalization objective, and systems budget.
[Utku et al.](https://doi.org/10.1007/s10791-026-10291-x) used country-year
cereal records and a time-aware multi-country design rather than experimental
crop-management systems and group-held-out field experiments. Work on
[Earth-observation agricultural federation](https://doi.org/10.1038/s41598-025-94244-2)
studied client imbalance through an image-segmentation proxy rather than yield
regression. These differences preclude numerical ranking across papers.

The narrower contribution of the present work is its auditable characterization
of conditional harvest regression under unequal geographic support: a nested
grouped protocol, complete baseline and adverse-result reporting, equal-group
and pooled metrics, support panels, explicit separation of post-hoc and
prospective stages, and a matched noise-utility simulation. It is not a new
federated optimizer or the first federated crop-yield study.

# VII. Limitations and Threats to Validity

The first limitation is the data-generating process. The 607 systems came from
one compiled resource containing 37 experiments rather than a new prospective
field trial. Sites, species, management packages, and sample sizes were uneven,
and Auzeville accounted for more than half of all systems. Three species were
available only through that group. Cultivar, density, mixture proportions, and
some management details were not sufficiently harmonized for causal comparison.
Recorded fertilizer and irrigation describe realized treatment bundles and are
not certified as information known before sowing. The model therefore predicts
associations for observed systems; it does not identify the best crop or the
effect of changing a treatment.

Second, the evaluation is development evidence. The initial smoke used seed 123,
and the outer outcomes were subsequently viewed during diagnosis. The support
guard was explicitly post-hoc. The fixed-public R13/R14 stages were frozen before
their corresponding executions, but they were designed after earlier outcomes
had been exposed and are not independent confirmation. Geographic folds also
share training records, and repeated seeds share the same 607 systems. Formal
independence assumptions for conventional significance tests or geographic
confidence intervals are therefore not satisfied.

Third, model selection used bounded grids. In the original neural search,
286 of 672 candidates selected the 80-round limit and four outer folds selected
the highest learning rate. In the fixed-public search, 80 of 224 candidates
reached 240 rounds, all eight choices lay on target-scale or learning-rate grid
edges, and 11 of 56 selected checkpoints were worse than their matched training
constant. Budgets were not extended after outer exposure. The study consequently
characterizes the frozen searches rather than proving convergence or global
hyperparameter optimality.

Fourth, federation was simulated locally on CPU. Logical clients were research
groups, not operational farms with independent hardware, networks, governance,
or failure patterns. Tensor payloads were not measured network traffic, and the
study did not evaluate authentication, transport security, latency under real
links, asynchronous participation, malicious clients, or institutional
deployment. No Linux server or remote client environment was available for
verification.

Fifth, privacy analysis remained a simulation. Fixed-public transforms,
clipping, Gaussian noise, event recovery, and zCDP calculations were implemented
and tested, but the complete release pipeline was not differentially private.
Conditional cumulative epsilon bounds were approximately 1297--1378 at
delta \(10^{-5}\), and unnoised controls and public deterministic noise seeds
precluded a joint production guarantee. Secure aggregation was not implemented.
The results must not be described as privacy-preserving deployment evidence.

Finally, independently validated agronomic rules were unavailable for the active
features and response, and crop-system ranking lacked comparable alternatives
and a justified utility function. Knowledge guidance and recommendation were
therefore disabled rather than filled with assumed thresholds or synthetic
optimal labels. Economic outcomes, resource use, farmer acceptance, and causal
agricultural benefit were not evaluated.

# VIII. Reproducibility

The implementation retained immutable raw inputs and versioned every prepared
dataset, split, protocol, source snapshot, runtime identity, training attempt,
checkpoint, inference intent, cached prediction, audit, analysis, and figure.
The prepared data contained eight outer and 56 inner folds. Completed evidence
included 1,120 original inner candidates, 320 original outer evaluations,
320 post-hoc guard evaluations, 224 fixed-public inner candidates, 40 unnoised
fixed-public refits, and 160 matched-noise evaluations. Completed caches were
verified without repeating model test inference.

Execution used bounded resumable runners with exclusive writer locks. Frozen
source, data, configuration, and dependency hashes were checked before recovery.
Attempt journals retained failures, and an inference intent without a valid
completion was treated as ambiguous rather than automatically retried. Noise
events committed with checkpoints and were reconciled against an append-only
ledger. These controls reduced accidental duplication and silent configuration
drift; they are not security mechanisms.

Nine accepted scientific figures are available in PNG, PDF, and SVG formats,
with exact plotted values, captions, source-analysis hashes, code hashes, and
asset hashes. Figure generation started from complete cached audits, and all
PNGs received visual inspection. Machine-readable claim, literature, manuscript,
and section validators bind numerical statements and scope qualifications to
the saved evidence.

The observed environment is recorded in `requirements-runtime-observed.txt`,
and the local recovery guide lists audit-only and regeneration commands. The
current evidence was verified in the existing Windows checkout. Absolute paths
in historical manifests and the absence of a clean-room installation mean that
portable reproduction on another machine and Linux deployment remain unverified.
Releasing a portable package would require a separate path-relocation strategy,
fresh-environment installation test, and cache/audit validation without changing
the archived evidence.

# IX. Conclusion

This study provided a reproducible characterization of conditional cereal-
legume harvest regression across eight geographically grouped field-experiment
collections. In the original frozen comparison, tree ensembles had lower
observed RMSE than the evaluated centralized and federated neural baselines,
whose held-out performance was strongly affected by unsupported input columns.
A training-only guard reduced those fixed-model errors in a clearly labelled
post-hoc sensitivity analysis. A later fixed-public protocol produced lower
unnoised neural errors, while matched Gaussian aggregation noise progressively
degraded utility and led to weak fits at larger multipliers.

The results show that aggregation choice alone is insufficient to characterize
agricultural generalization. Geographic weighting, biological support, feature
contracts, training budgets, and release assumptions materially affect the
conclusion. The evidence supports neither universal model superiority nor
best-crop recommendation, causal yield improvement, validated agronomic
knowledge guidance, independent field utility, or end-to-end privacy.

Future work should prospectively evaluate the support policy on independent
sites and seasons, harmonize cultivar and management descriptors, establish a
reviewed decision-time feature contract, and define any ranking objective using
genuinely comparable alternatives and an explicit utility. Privacy-oriented
work requires private preprocessing and selection, a limited release policy,
secret and reviewed randomness, independently checked accounting, and an
established secure-aggregation deployment. A clean-room portable reproduction
and real multi-institution infrastructure test should precede any operational
claim.
