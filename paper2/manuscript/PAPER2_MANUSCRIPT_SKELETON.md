# Paper 2 manuscript skeleton — evidence constrained

## Working title

**Geographically Grouped Federated and Centralized Regression for Conditional
Harvest Prediction in Cereal-Legume Field Experiments**

Do not add “knowledge-guided,” “crop recommendation,” “privacy-preserving,” or
“secure aggregation” to the title while claims C08--C10 remain blocked.

## Abstract

Use a structured summary of the conditional regression task, 607 systems,
eight geographic groups, nested grouped evaluation, original eight-method
comparison, fixed-public refit and matched noise-utility simulation. Include
the principal equal-group and pooled RMSE values from C02, C04 and C05. State
that the study is retrospective, simulation-based for federation/privacy work,
and does not establish crop recommendation, formal privacy or field utility.

## I. Introduction

Draft implemented and evidence-gated in
`PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md`; the outline below remains the
acceptance checklist.

1. Motivate cross-location agricultural outcome modelling under heterogeneous
   experimental coverage.
2. Define the actual gap narrowly: reproducible grouped evaluation of
   centralized and federated regression under unequal geographic support and
   controlled aggregation noise.
3. State research questions without novelty claims:
   - How do centralized and federated baselines behave under held-out groups?
   - How sensitive are conclusions to equal-group versus pooled weighting?
   - What utility degradation appears under a prospectively matched clipping/
     noise simulation?
4. List contributions as protocol, verified implementation, complete adverse
   findings and reproducible figures—not crop advice or privacy certification.

## II. Related Work

Draft implemented and source-gated in
`PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md`; the outline below remains the
acceptance checklist.

Cover established FedAvg, FedProx and FedNova definitions; agricultural FL;
grouped/geographic validation; and established DP/SecAgg separately. The old
classification-focused closest-paper table is background only and must be
updated for the regression task before submission. Externally reported scores
must be labelled and must not be compared numerically unless data, split,
target, units and metrics align.

## III. Data and Problem Formulation

Draft implemented and evidence-gated in `PAPER2_DATA_METHODS_DRAFT.md`; the
outline below remains the acceptance checklist.

- Unit: `(experiment_id, management)` crop system.
- Target: summed complete harvest seed dry matter, t/ha.
- Composition is an input, not a recommended label.
- Describe 607 systems, eight merged geographic groups and unequal support.
- Document feature timing/provenance exclusions, fixed public taxonomy,
  train-only preprocessing and missingness handling.
- State ranking and causal intervention interpretations are disabled.

## IV. Methods

Draft implemented and evidence-gated in `PAPER2_DATA_METHODS_DRAFT.md`; the
outline below remains the acceptance checklist.

### A. Grouped nested protocol

Describe 56 inner folds, eight outer held-out groups, five declared seeds,
selection/refit rules, one-time inference receipts and immutable provenance.

### B. Baselines

Training mean, ridge, random forest, XGBoost, centralized MLP, FedAvg, FedProx
and restricted FedNova. Define logical clients as training research groups.

### C. Fixed-public refit

Describe the separately frozen R12/R13 selection and repeated-seed outer refit.
Preserve the 240-round grid-edge and matched-constant limitations.

### D. Matched clipping/noise simulation

Describe equal-client aggregation, clipping, Gaussian noise variants and the
complete event ledger. Call this a utility simulation. Put threat-model and
accounting gaps in limitations; do not imply certified DP.

### E. Metrics

Primary equal-group RMSE; pooled RMSE, MAE and R2 as secondary; per-group and
species/composition support panels. Seed SD is descriptive.

## V. Results

Draft implemented and evidence-gated in `PAPER2_RESULTS_DRAFT.md`; the outline
below remains the acceptance checklist.

1. Original eight-method comparison: use `restart_baseline_figures_v2` and C02.
2. Geographic heterogeneity and support: include group counts and adverse R2.
3. Post-hoc support guard: isolate in a sensitivity subsection with C03.
4. Fixed-public unnoised refit: use `restart_public_refit_figures_v2` and C04.
5. Matched noise utility: use `restart_matched_noise_figures_v1` and C05.
6. Preserve adverse results, including high-noise degradation and convergence
   limits; do not select only favorable panels.

## VI. Discussion

Draft implemented and evidence-gated in
`PAPER2_DISCUSSION_CONCLUSION_DRAFT.md`; the outline below remains the
acceptance checklist.

Interpret why equal-group and pooled summaries differ, why Auzeville-only unseen
support limits extrapolation, and why tiny groups are unstable. Compare only
qualitatively with studies having different targets/protocols. Distinguish the
original prospective record, exposed-test post-hoc guard and later prospectively
frozen R13/R14 stages.

## VII. Limitations and Threats to Validity

Draft implemented and evidence-gated in
`PAPER2_DISCUSSION_CONCLUSION_DRAFT.md`; the outline below remains the
acceptance checklist.

Include retrospective treatment bundles, missing cultivar/density controls,
unequal/tiny groups, one dataset, prior development exposure, grid-edge/fixed
budgets, lack of independent field confirmation, disabled ranking/knowledge,
simulation-only privacy analysis and absent production SecAgg/deployment.

## VIII. Reproducibility

Draft implemented and evidence-gated in
`PAPER2_DISCUSSION_CONCLUSION_DRAFT.md`; the outline below remains the
acceptance checklist.

Reference manifests, source/data hashes, attempt journals, independent cached
audits, exact plot values, vector figures and the local recovery guide. State
that Windows-local verification is not a portable Linux deployment validation.

## IX. Conclusion

Draft implemented and evidence-gated in
`PAPER2_DISCUSSION_CONCLUSION_DRAFT.md`; the outline below remains the
acceptance checklist.

Conclude only that the study provides a reproducible characterization of
conditional regression behaviour under grouped evaluation and controlled
aggregation noise. Future work requires independent sites, validated agronomic
units/rules, reviewed ranking alternatives, established private mechanisms and
prospective field evaluation.

## Required tables and figures

- Dataset/group support table.
- Model/protocol table.
- Original baseline metrics table plus three accepted baseline figures.
- R13/R14 exact metrics table plus six accepted figures (select a journal-sized
  subset for the main paper and place the remainder in supplementary material).
- Claim-boundary table derived from `evidence_claim_matrix.json`.
