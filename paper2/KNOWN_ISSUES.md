# Resolved on 2026-11-16 (local fixes — do not reopen)

- **project_code_license**: Apache-2.0 chosen. paper2/LICENSE created.
  elease_licensing_registry_v2.json updated. DATASET_NOTICE.txt added.
- **third_party_license_inventory**: Proposed inventory written to
  configs/third_party_license_inventory_v1.txt. Requires Linux pip-licenses
  verification before the release gate is fully cleared.
- **RR03 (experiment dependence + category hash collisions)**: Fully documented
  in esearch/rr03_resolution_v1.md. 9 colliding previous_crop pairs,
  36.4% missingness, and clustering across 37 experiments are disclosed as
  limitations. No model results changed. Status: RESOLVED_WITH_DOCUMENTED_LIMITATIONS.
- **Canonical audit paths**: CANONICAL_AUDIT_PATHS.txt created. The erroneous
  estart_completion_audit_v54 reference from earlier session logs is corrected;
  canonical latest is estart_completion_audit_v10.
- **Completion audit**: Updated to v10 reflecting all above local resolutions.

---
# Active issues and gates

- The regression-scoped V4 Markdown draft now cites the dataset, numbers all
  14 working sources in first-use order, fixes the regression equation and adds
  post-hoc paired geographic differences, and places five audited figures
  without replotting. It is not submission-ready IEEE
  source. Rendered reference/figure checks, final source metadata, author and
  affiliation details, journal LaTeX conversion and main/supplement allocation
  remain open. The geographic table is descriptive; five seeds are not five
  independent sites and no confirmatory interval or p-value is established.
- Historical manifests contain checkout-specific absolute paths. Portable,
  clean-room reproduction is not yet verified and must not alter archived hashes.
  The versioned readiness audit passes all 21 local evidence assets with zero
  errors, but explicitly leaves portability BLOCKED. The current runtime list
  is an observed Windows environment, not a cross-platform lock; no fresh-env
  or Linux end-to-end run exists, and redistribution/licensing review is open.

- R14 completed and independently audited160/160;do not rerun. Its former
  progress.json starter count was a batch-end snapshot, not a live counter.
  Complete analysis retains worse noisy utility; do not tune from these scores.
- R13 figure v1 is superseded by v2 after visual inspection found a clipped
  seed marker. V2 has explicit comparable panel scales and a regression test.
  R14 five-variant plots also passed PNG visual review after complete audit.

- V3 update/noise scale is calibrated from public four-step work; all200 runs
  pass audit without numerical failure. Prediction fit remains weak:40/40
  controls underfit on matched equal-client training MSE. R12's separate
  fixed-budget inner search is now complete; this remains development evidence.
- R12 all224 candidates and eight selections pass audit.80/224 candidates
  hit240 rounds, including7/56 selected-configuration curves.11/56 selected
  checkpoints are worse than their matched equal-client training constant.
  Full-search choices are scale1/LR .01 in all8 groups: both are grid edges,
  so optimality/convergence is not established. R13 unnoised refit is frozen
  and complete; R14's separate prospective freeze/execution/audit is also complete.
  Do not tune from earlier exposed outer scores.
- The full expanded search includes four configurations already present in the
  pilot, executed under its new frozen identity; these are not independent
  replications and must not be counted as additional scientific evidence.
- Legacy frozen R12 audit did not require identity/receipt-bound attempt
  journals. Supplemental control_search_integrity.py now gates R13; all224
  actual journals pass. Do not rely solely on the legacy audit for provenance.
- R13'sfrozen unnoised study is40/40 complete and audited. Equal-group and pooled
  RMSE differ materially; report both, with descriptive seed SD not regional CI.
  Four geographic groups have negative mean outer R2, including tiny groups;
  improved saved training fit does not establish universal outer generalization.
  Unseen-species/composition panels contain265 systems from Auzeville only and
  must not be described as unseen-species validation across eight regions.
  Models are normalized-target outputs: predict_raw must restore raw units
  exactly once. Do not call legacy inference assuming arbitrary scale is1.
- R13 is unnoised and not a privacy experiment. R14 requires a separate matched
  protocol/accounting freeze (now implemented separately as R14 v1). New outer scores are exposed development data;
  do not silently alter learning rates, rounds, clipping or noise to fit them.
- Any incomplete run with inference_started.json but no valid completion is
  ambiguous and cannot be automatically retried, even with --retry-failed.
  Training-only failures can resume only after inspection and explicit retry;
  orphaned/misbound journals/checkpoints must not be silently reconstructed.
- Older pending-run notices below are historical: v2 is closed, v3 is complete.
  Do not resume v1/v2 or repeat completed runs based on those notices.

- 286/672 neural candidates selected the 80-round cap; 4/8 outer groups selected
  the highest grid LR. Fixed-budget selection is not convergence proof.
- Prior seed123 diagnostic and all outer-test exposure mean development
  evidence. Original and guard sensitivity outcomes have now been viewed.
- Unsupported zero-training columns caused extreme Auzeville neural
  extrapolation. A training-only gate fixes this connection activation;
  320 sensitivity evaluations completed. Post-hoc design requires disclosure.
  New data is needed for independent confirmation; unit/scope gates remain.
- Dataset records crop systems and harvest outcomes, not optimal-crop labels.
- Cultivar/density comparability unresolved; ranking disabled.
- Independent agronomic rule eligibility remains empty. Fertilizer application
  is not soil nutrient concentration; annual rainfall is not seasonal exposure.
- Source sand/silt metadata conflict handled by exclusion, not raw correction.
- Geography groups uneven and some species occur only at Auzeville; retain
  unseen-species/composition panels and tiny-group uncertainty.
- Public central preprocessing and CPU simulation are not private federation.
- No release-certified application DP, secure aggregation or verified remote
  deployment exists. Public transforms, clipping/Gaussian mechanisms and the
  composed accountant are tested simulation components, not deployment proof.
- Original fitted scalers/vocabularies/support gates cannot be assumed public
  under whole-client replacement. Fixed-public study assets/runner/ledger now
  exist and v3 is complete; older pending notices are superseded, not ready work.
- Hypothetical full-study noise budgets are large and must include prior events
  and overlapping releases. Unnoised controls/published simulation seeds preclude
  a joint application guarantee. Do not report per-run costs as total-study cost.
- R14 additionally excludes nonprivate inner hyperparameter/round selection,
  training diagnostics and test-metric releases from its conditional accounting.
  Observed total23174 events have per-slot hypothetical epsilon1297--1378 at
  delta1e-5; this does not support strong privacy or a pipeline DP claim.
- Historical v1 sigma4 nonfinite loss after round47 and poor v2 utility remain
  archived adverse findings. V1/v2 are closed; do not resume them. V3's8564-event
  ledger retains359 earlier events. New noisy follow-ups need new prospective
  accounting and noise identities, not edits to existing protocols/results.
- Frozen source/packages must not be edited. New outer runner is separate.
- Automation is scheduled, not proof of unattended success; inspect execution
  evidence. Failure requiring new authority must be reported, not bypassed.
# Reviewer-response issues (2026-10-15; current section)

- **Novel KG-FCR claim is unsupported.** The active task is retrospective
  conditional harvest regression, not best-crop classification/ranking.
  Independent agronomic units and counterfactual comparability remain blocked.
  See `research/reviewer_claim_evidence_matrix_v1.md`.
- **Outer-test development exposure.** Primary, guard and fixed-public results
  are known. New ablations on the same eight groups may explain mechanisms but
  cannot be called untouched confirmation. An independent geography is needed.
- **Dependence and hash collisions.** The 607 system rows cluster into 37
  experiments; all experiments have multiple rows. Fixed-public
  `previous_crop` maps 13 observed categories into seven of eight buckets,
  with nine distinct colliding pairs. See reviewer_data_audit_v1.
- **Operational and privacy gaps.** No secure aggregation, certified
  end-to-end DP, or deployed farm-client network is evidenced. Existing
  conditional epsilon is too large for a meaningful protection claim.


