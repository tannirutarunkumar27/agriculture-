# Paper 2 research and engineering plan v2

Date: 2026-10-25. This plan supersedes no frozen protocol or completed result.
It translates the audited implementation into a defensible paper pathway.

## 1. Paper identity

**Working title:** *Geographically grouped simulation of centralized and
federated regression for cereal--legume harvest outcomes.*

The paper is a retrospective, conditional seed dry-matter regression study on
607 recorded cereal--legume systems from 37 experiments and eight geographic
groups. Clients are simulated groups processed in one trusted process. It is
not a crop-choice recommender, a new knowledge-guided learning algorithm, an
end-to-end privacy system, a secure-aggregation deployment, or a prospective
field study.

The most defensible contribution is a transparent, group-held-out comparison
of established central and simulated federated baselines, with explicit
evidence boundaries and reproducible experiment receipts.

## 2. Research questions and allowable claims

| Question | Evidence required | Allowable conclusion |
| --- | --- | --- |
| RQ1. How do established central and simulated federated regressors compare under the frozen grouped protocol? | Existing 320 audited primary outer evaluations; equal-group and pooled error reported separately | Descriptive comparison on this dataset, not universal superiority or location-level significance |
| RQ2. Is the original neural deficit attributable to federation, or to shared optimisation/representation limits? | New matched, inner-selected central/FedAvg/FedProx/restricted-FedNova development experiment | Exploratory, test-exposed mechanism evidence only |
| RQ3. Does stronger, prospectively frozen neural optimisation change the within-data comparison? | Inner-only search, fixed selection rule, convergence diagnostics, then separately labelled development outer evaluation | Development result, never independent confirmation |
| RQ4. What does the existing experiment establish about privacy or deployment? | Threat-model and operational audit | Only that neither production privacy nor deployment is established |

No knowledge-guided or crop recommendation research question is eligible until
independently sourced, regression-compatible rules and a legitimate decision
target exist.

## 3. Evidence already available

- Original primary study: 320 frozen outer evaluations, eight groups, five
  computational seeds and eight methods.
- Active source: Gaudio et al. Zenodo v2.0.0, DOI `10.5281/zenodo.14160885`.
- Validation tuning: 1,120 immutable inner results.
- Read-only diagnostic: `results/restart_primary_training_diagnosis_v1/`.
  It verified every inner receipt without inference. Of 168 candidates per
  neural method, round-cap selection occurred for 35 centralized (20.8%), 96
  FedAvg (57.1%), 96 FedProx (57.1%), and 59 restricted-FedNova (35.1%).
  The mean absolute paired FedAvg--FedProx validation-RMSE difference at
  `mu=0.01` was `1.18e-05` t/ha across 168 matched candidates. This indicates
  that the frozen setting did not meaningfully distinguish the two methods; it
  does not estimate a general FedProx effect.

## 4. Engineering sequence

1. **Preserve and verify (complete):** retain immutable historical outputs;
   never overwrite or rerun them. Use receipt/hash audits for read-only
   analyses. Remove only disposable caches.
2. **RR03 data-availability closure:** record which inputs are actually
   available at intended prediction time, resolve cultivar/density comparability
   if possible, and retain unavailable fields as exclusions rather than guesses.
3. **RR04 convergence/FedProx diagnosis (complete):** retain the new read-only
   diagnosis above. Do not claim convergence based on 80-round selection.
4. **RR05 freeze a matched ablation protocol (complete):**
   `results/restart_matched_ablation_protocol_v1/` freezes 280 inner-only
   candidate runs across 56 folds. It identifies the
   exact factors: objective (sample-weighted versus equal-client),
   representation/preprocessing, optimizer, local step budget, rounds,
   architecture, seed list and inner-selection metric. Pair folds/seeds and
   record parameters, updates and actual elapsed time. Do not use outer scores
   to choose these settings. The protocol is identity/hash verified and starts
   no training or outer inference.
5. **RR06 implement a stronger neural inner-only runner:** use train-only
   transforms and nested inner folds; compare a bounded architecture and
   optimizer grid (e.g., SGD baseline versus AdamW/schedule) with predeclared
   stopping/convergence summaries. Add synthetic tests for split isolation,
   deterministic resume, selection ties, metric units and receipt binding.
6. **Audit before outer inference:** verify every inner receipt, selection and
   convergence criterion. If the cap/grid remains active, report that outcome;
   do not silently extend the search.
7. **RR07 development outer evaluation:** only after steps 4--6, freeze a new
   identity-bound protocol and evaluate once. Label all results as development
   because these eight groups have already been observed.
8. **Independent confirmation (required for stronger claims):** collect a
   legally usable, comparable, untouched geography with identical feature/target
   semantics and a preregistered analysis. Existing split reshuffling cannot
   create this evidence.

## 5. Non-negotiable safeguards

- Every new experiment has a new output directory, immutable plan, code/config
  hashes, source snapshot, run receipt and explicit output status.
- Outer test arrays are unavailable to inner selection code; incomplete or
  ambiguous inference is never retried automatically.
- Report t/ha RMSE and MAE; show equal-group and pooled summaries separately.
  Treat seeds as computational repeats, not independent regions.
- Keep knowledge, ranking, secure aggregation and formal privacy disabled in
  the active study until their own scientific gates are met.
- Record observed wall-clock duration and transmitted payload accounting only
  when actually measured. Estimated tensor size is not a network measurement.

## 6. Paper writing boundary

Writing may start once steps 1--3 are documented, using the completed primary
study and its limitations. A stronger federated-method claim must wait for the
matched development study, and an external-generalisation claim must wait for
the independent dataset. The results section must keep the original primary,
post-hoc guard, fixed-public refit and aggregation-noise simulations separate.
