# Active Paper 2 project state

Updated 2026-11-02. Canonical active ledger; STATUS.md retains historical detail.
Scope: cereal-legume conditional harvest dry-matter regression. The working
crop-recommendation/privacy title is not an established scientific claim.

| ID | Task | Status | Evidence / next action |
| --- | --- | --- | --- |
| R01 | Provenance, target and geographic splits | VERIFIED_COMPLETE | restart preparation report; 64 verified folds |
| R02 | Regression baseline client and aggregation | VERIFIED_COMPLETE | preparation/client tests and frozen search execution |
| R03 | Frozen inner-validation search | VERIFIED_COMPLETE | 1120 runs; PASS_COMPLETE audit; 14 scoped tests passed |
| R04 | Repeated-seed outer protocol and safe runner | VERIFIED_COMPLETE | 29 scoped acceptance tests pass, including package mutations and recovery rejection; complete study audit retained |
| R05 | Outer study and independent analysis | VERIFIED_COMPLETE | 320/320 audited; descriptive group/seed and support-panel analysis saved in restart_outer_analysis_v1 |
| R06 | Independent agronomic rules / regression guidance | BLOCKED | no eligible independently validated rules; no classification-prior reuse |
| R07 | Crop-system ranking | BLOCKED | cultivar/density comparability and utility unresolved |
| R08 | Threat model and privacy implementation | IN_PROGRESS | V3 execution/analysis VERIFIED_COMPLETE: 200/200, zero failures; weak controls; certified release/SecAgg pending |
| R09 | Deployment verification | BLOCKED | no Linux server credentials/infrastructure supplied; local simulator only |
| R10 | Evidence-based figures and manuscript | IN_PROGRESS | Regression-scoped V4 Markdown draft includes five pre-existing audited plots; V3 text revision binds dataset citation, ordered references, corrected equation and post-hoc paired geographic table; IEEE conversion/PDF review and portable clean-room reproduction remain open |
| R11 | Unsupported-input safeguard and sensitivity | VERIFIED_COMPLETE | Seven tests passed; 320/320 guarded evaluations audited; post-hoc scope documented |
| R12 | Fixed-public bounded control search | VERIFIED_COMPLETE | 224/224 independently audited; all8 selections verified; zero failures; fresh17 tests pass; fixed budget is not convergence proof |
| R12H | Search provenance hardening | VERIFIED_COMPLETE | Missing-journal gap reproduced; separate strict gate validates all224 journals, identities and receipts; no changed model results |
| R13 | Fixed-public unnoised repeated-seed refit | VERIFIED_COMPLETE | 40/40 audited; complete equal-group/pooled/support analysis; zero failures; fresh25 scoped tests pass; development evidence only |
| R14 | Prospective matched noise follow-up | VERIFIED_COMPLETE | 160/160 independently audited,zero failed/pending;40 cached controls reused;complete analysis;23174 events incl8564 retained prior;do not rerun |
| A01 | Scheduled continuation | PAUSED | Hourly automation was paused for audit-first review; this manual manuscript continuation does not certify automatic resumption |
| MS01 | Narrow regression manuscript and release boundary | DRAFT_VERIFIED | V4 draft and figure manifest, V3 text/evidence manifest, 22 scoped tests and 21/21 local asset audit; no archived study rerun; IEEE render, final source metadata, licensing and clean-room checks remain |
| RR01 | Reviewer novelty and claim audit | VERIFIED_COMPLETE | 20 primary studies reviewed; claim-to-evidence gate saved in research/reviewer_response_literature_v1.md and reviewer_claim_evidence_matrix_v1.md |
| RR02 | Post-hoc geographic sensitivity | VERIFIED_COMPLETE | Versioned archived-metric analysis and paired group differences; six RR02/RR03 tests pass; no new test inference |
| RR03 | Experiment dependence and category hash audit | IN_PROGRESS | 607 systems from 37 experiments; previous-crop observed category collisions quantified; prediction-time availability unresolved |
| RR06N | Matched neural search | VERIFIED_COMPLETE_CPU | 504/504 frozen candidates independently audited; final selection frozen; GPU repetition is optional performance evidence, not a scientific completion gate |
| RR07O | Development outer evaluation | VERIFIED_COMPLETE_CPU | 160/160 frozen runs independently audited; all geographic groups have prior development exposure, so this is not independent confirmation |
| RR08-RR10 | Operations/privacy and portable reproduction | IN_PROGRESS/BLOCKED by dependency | Native-Windows Flower/Ray run failed; use local WSL2 for SecAgg+/DP, then complete authenticated deployment, Linux reproduction and release gates |
| AU01 | Implementation-to-evidence audit | VERIFIED_COMPLETE for repository snapshot | `implementation_audit.md`; code/result/status cross-check; 55 scoped tests pass with workspace-local temp/import path |
| AU02 | Active regression master experiment index | VERIFIED_COMPLETE | 2,384/2,384 run receipts/assets and split sizes verified; CSV/audit under results/master_experiments_v1; primary XGBoost aggregate independently recomputed |
| AU03 | Five-area literature screening matrix | VERIFIED_COMPLETE as screening, NOT full-text systematic review | 26 primary records in research/literature_evidence_matrix_v2.md; unverified details explicitly marked |
| NM01 | Steps 4–6 method/novelty formulation gate | VERIFIED_COMPLETE as code/specification audit, NOT a validated KG-FCR method | research/novelty_method_formulation_gate_v1.md; source-matched equations, RQs, pseudocode and architecture boundary; no real-data knowledge experiment |
| DC01 | Step 7 dataset, clients and leakage validation | VERIFIED_COMPLETE for frozen regression data | 607 systems; 21 conceptual inputs; 8 geographic groups; 64 folds reverified; 392 client-fold records; 12 scoped tests pass; research/dataset_client_validation_v1.md |

Acceptance for R04: immutable seeds/settings/source/runtime bindings; train-only
refit; no test-based checkpoint choice; capped execution and writer exclusion;
exact resume, failure journaling, completed-cache integrity and ambiguous test
inference rejection; synthetic unit/integration/smoke tests pass before study.
R05 acceptance: every planned run has verified receipt and cached predictions;
independent metrics/support panels and complete matched-seed summaries audited.

Historical phase evidence (chronological; latest task table supersedes counts): research/restart_validation_completion_report.md,
results/restart_validation_verified_v2/verification.json,
results/restart_validation_selection_v1/selection_audit.json and
results/restart_search_tests_v3/verification.json. At that phase no outer test inference had occurred.
No new implementation tests or experiments were run during automation setup.
Subsequent implementation: five synthetic planner tests passed; 320-run outer
plan frozen in results/restart_outer_protocol_v1 after verifying all 1120 inner
completion caches and recomputing selections. No outer inference or training.
Evidence and commands: research/restart_outer_planning_report.md.
Next substep implemented: outer_training.py fixed-round refit, identity-bound
round checkpoints and fail-closed once-only inference/cache primitives. Seven
synthetic tests passed, including all eight method smokes and exact recovery
for four neural methods. No real outer inference. Caller must hold writer lock;
the study coordinator, failure journals and code freeze are still pending.
Details: research/restart_outer_training_report.md.
Coordinator now implemented in restart/outer_runner.py: explicit resume/caps,
writer lock, source/runtime/data binding, support context, failure journals,
safe retry and cached-completion verification. Execution package frozen at
results/restart_outer_execution_v1. Nineteen scoped tests passed. No real outer
inference. Next independently audit binding/receipt edge cases and persist test
logs, then execute bounded starter batch and independently verify outputs.
Starter batch subsequently executed: 8/320 complete, 312 pending, no observed
failure. Angers group/seed123 only; outer-test exposure now EXISTS. Do not tune
settings from these scores or describe outer tests as untouched. Independent
audit PASS_PARTIAL at results/restart_outer_verified_v1; no model reinference.
Frozen execution code unchanged. Tests/commands recorded in
research/restart_outer_starter_report.md. Full cross-group study and analysis
remain incomplete; code/package mutation-specific acceptance tests remain open.

Every continuation must inspect actual files, preserve unrelated edits, record
tests actually run and update this ledger. Implemented/tested/experimentally
validated/scientifically established are distinct. Detailed task records must
include files, dependencies, commands, outcomes, errors and next actions.

Latest bounded continuation: 72 additional frozen runs completed, total 80/320;
240 pending, zero failure files. All 80 independently audited in
results/restart_outer_verified_v2 without model reinference. Covers Angers and
Auzeville, five seeds and eight methods each. No new code/settings changes or
test rerun this batch. Next continue remaining frozen groups, then complete
matched-group analysis; do not tune from these exposed outer scores.

Latest completion (supersedes partial counts above): all remaining 240 runs
executed successfully. 320/320 complete, zero pending, zero failure files;
320 attempts, all COMPLETE. Independent audit PASS_COMPLETE in
results/restart_outer_verified_v3, without model reinference. Covers all eight
geographic groups, five seeds and eight methods. Execution subtask is verified
complete; R05 subsequently completed matched-group/seed aggregation and
comparative analysis. Report: research/restart_outer_completion_report.md.

Latest fix: training-only zero-support gate implemented, frozen and evaluated
on all 320 saved configurations without retraining. Seven tests passed;
PASS_COMPLETE_POST_HOC_SENSITIVITY saved in restart_guard_analysis_v1.
FedAvg Auzeville seed123 RMSE 36.7482 -> 0.9960. Original 320 results retained.
Post-hoc findings are not independent confirmation. Report:
research/restart_support_guard_report.md. R04 acceptance hardening subsequently
completed: 29 tests passed, zero failures/errors; persistent logs saved in
results/restart_outer_acceptance_v1. Next ready work is R08 threat-model
specification and source-supported method review; KG/ranking remain blocked.

Latest R08: privacy contract and primary-source method review saved; stable L2
clipping and full-participation Gaussian zCDP accountant implemented, eight tests
passed. Mathematical planning only, no private training or application guarantee.
Critical dependency: current fitted preprocessing/gate is data-dependent and
cannot be assumed public under whole-client replacement. Next define independent
public transforms/transcript policy before integrating noise and training.
Evidence: research/restart_privacy_threat_model.md and
results/restart_privacy_foundation_v2. No packages installed.

Next R08 stage completed: 54-dimensional data-independent transform and equal
fixed-slot clipped/Gaussian simulation aggregator implemented. Sixteen tests
passed and synthetic two-client/three-round integration passed. Evidence:
results/restart_private_primitives_v1; report:
research/restart_private_primitives_report.md. Production release mode is
rejected; this is simulation, not an application privacy guarantee. Next prepare
new fixed-transform real-data assets and matched controls, then freeze a utility
study and persistent release-accounting/recovery policy.

Latest R08 preparation: 607 systems re-encoded in 54 fixed public features;
all 64 folds verified with unchanged targets/IDs/partitions. Frozen simulation
plan has 200 matched runs (five variants, eight groups, five seeds); no training
started. Twenty scoped tests pass. Whole noisy-transcript composition produces
very large hypothetical epsilon bounds (~516--571); no application privacy or
strong full-matrix protection is claimed. Report:
research/restart_public_study_preparation_report.md. Next implement controls and
resumable round/event ledger, freeze execution code, then starter simulation.

Latest R08 runner: v1 had three completed runs and one recorded nonfinite-loss
failure after round47; all evidence retained. A new matched v2 gradient-clipping
amendment passed 24 tests and has 6/200 completed runs, zero v2 failures, with
independent audit PASS_PARTIAL. Shared ledger includes 359 events (125 v1 +
234 v2). Noise streams differ across versions. High noisy RMSE remains an
adverse finding. Next continue ONLY stable_privacy_simulation_runner.py with
restart_privacy_execution_v2 and restart_privacy_simulation_v2 --resume;
194 remain. Report: research/restart_simulation_runner_report.md.

Active follow-up (supersedes v2 continuation above): use ONLY the frozen v3
work_scaled_privacy_simulation_runner.py, restart_privacy_execution_v3 and
restart_privacy_simulation_v3. V3 has exactly four public local SGD steps,
gradient clip1 and global C=LR*4 for clipped/noisy variants. All controls have
the same optimizer work. Twenty-six tests pass; first five evaluations audited.
The full 200-run matrix is executing; check the output writer lock before any
continuation and do not start another writer or resume v2. Prior 359 events
remain in the v3 shared ledger. Poor v2 utility motivated this recorded public-
work design amendment, not a change to archived results or a privacy guarantee.

V3 completion supersedes the running notice:200/200 complete, zero pending
or failure files. No active writer remains. Audit PASS_COMPLETE and full
summaries:restart_privacy_analysis_v3. Ledger8564 events=359 prior+8205 v3.
The scale/overflow issue is addressed; equal-region RMSE3.11--3.29 t/ha.
Training-only matched-objective diagnostics show all40 controls underfit.
Two diagnostic tests passed in addition to26 pre-execution tests. Next R12
convergence study on training/inner-validation; preserve all current results.
Report:research/restart_work_scaled_completion_report.md.

Latest R12: four inner-only target-conditioning/convergence trials completed
and audited; three tests passed. With scale10 at240 rounds, both pilot folds
beat their training constant reference; best validation RMSE1.3745/1.0674.
No outer model inference. This pilot does not select final settings or prove
convergence everywhere. Expanded224-candidate inner-search package frozen:
restart_control_search_protocol_v1. Next implement/test capped coordinator and
complete-selection rules, freeze its code, then execute bounded new candidates.
Report:research/restart_control_convergence_report.md.

Latest R12 continuation: separate capped coordinator and independent checkpoint
audit implemented, tested and frozen at restart_control_search_execution_v1.
R12a runner subtask VERIFIED_COMPLETE:17 scoped tests plus26 dependency tests
passed, zero errors/failures. Four new expanded-plan candidates completed with
one COMPLETE attempt each; PASS_PARTIAL_INNER_SEARCH verifies final/best inner
models and equal-client training metrics. No outer-model inference, noise,
final setting selection or edits to previous frozen sources/packages occurred.
Expanded search is4/224 complete,220 pending; next resume this unchanged runner
in bounded batches and independently audit each batch. Do not repeat the1120
original candidates or resume closed privacy versions. Commands, acceptance,
partial results and blockers:research/restart_control_search_runner_report.md.

Latest R12 bounded continuation (supersedes4/224 above):20 more candidates
completed under unchanged frozen settings, total24/224;200 pending. Fresh17
scoped tests passed before execution, zero errors/failures. Independent audit
PASS_PARTIAL_INNER_SEARCH at restart_control_search_verified_v2 verifies all24
final/best inner checkpoints. Exactly24 attempts, all COMPLETE; the previous
four completion receipts are unchanged. No final selection or outer inference.
R12b batch subtask VERIFIED_COMPLETE; overall R12 remains IN_PROGRESS. Ten of
24 minima hit round240, so convergence is not established. Next continue the
remaining frozen candidates in bounded batches with explicit resume, then a
new independent audit. Report:research/restart_control_search_batch2_report.md.

User-authorized completion launch (now completed): fresh17 scoped tests PASS at
restart_control_search_tests_v4, then control_search_runner.py launched with
--execute --resume --cap200 on restart_control_search_v1. Existing writer lock
must be checked before any continuation; do not launch a duplicate writer.
Final audit/selection verification subsequently completed as recorded below.

R12 completion: all remaining200 candidates finished, total224/224, zero
pending/failures. PASS_COMPLETE_INNER_SEARCH at restart_control_search_verified_v3
independently verifies all saved final/best inner models, metrics and eight
complete-only selections. Exactly224 attempts, all COMPLETE; previous24
completion receipts unchanged. A completed-study resume executed0 new runs.
All eight selections use target scale1/LR .01; median refit budgets106--149.
80/224 candidates and7/56 selected-configuration curves hit round240;11/56
selected checkpoints remain worse than their matched training constant. This
is completed fixed-budget development evidence, not universal convergence,
outer generalization or privacy. No active writer remains. Next R13: separately
freeze repeated-seed refit/matched follow-up without tuning on outer scores.
Evidence/commands:research/restart_control_search_completion_report.md.

Latest repair/continuation: frozen R12 verifier did not enforce complete attempt
journal provenance. A synthetic missing-journal fixture reproduced its PASS;
new control_search_integrity.py now rejects missing/misbound/nonterminal journals
and binds the metric audit to completion receipts/selection. Fresh full metric
audit restart_control_search_verified_v4 and supplemental strict audit
restart_control_search_integrity_v1 PASS for all224 real runs; models/results
unchanged. R12H VERIFIED_COMPLETE.

R13 protocol/core subtask VERIFIED_COMPLETE: separate public_refit_protocol.py
and public_refit_training.py implemented and tested;40 unnoised controls planned
over eight groups and five declared seeds. Package restart_public_refit_protocol_v1
frozen and independently verified before any new outer inference.18 new tests,
17 existing search tests and26 dependency tests passed (61 final scoped tests).
One intermediate synthetic fixture error retained/fixed; no real-study failure.
No real refit training/inference occurred. Overall R13 remains IN_PROGRESS.
Next implement/test separate capped coordinator, once-only raw-unit inference
and cached-output audit, freeze execution code, then bounded starter. Noise
follow-up must freeze its own work/ledger/accounting before evaluation.
Report:research/restart_public_refit_preparation_report.md.

Latest R13 execution subtask VERIFIED_COMPLETE: added separate capped runner,
once-only raw-unit inference and independent cached-output/support-panel audit.
Explicit checkpoint tensor schema checks reject corrupted partial/final states
before recovery/inference; orphan assets and missing/misbound journals fail
closed.17 runner tests plus18 protocol/core and26 dependency tests passed.
One initial wrong-message test assertion was corrected; failed evidence kept.
Execution code frozen at restart_public_refit_execution_v1. Dry-run passed,
then five real Angers-holdout seeds completed with one COMPLETE attempt each.
PASS_PARTIAL_REFIT at restart_public_refit_verified_v1 verifies all5 caches
without model reinference;35/40 pending. No settings changed from outer scores,
and no noise/privacy training occurred. No writer remains. Next resume unchanged
refit runner in bounded batches, audit each to a new path, then complete-group
aggregation using existing tested aggregation. Report:
research/restart_public_refit_runner_report.md.

R13 completion supersedes starter counts above: all remaining35 runs completed;
40/40 independently audited in restart_public_refit_verified_v2, zero failed
attempts/pending runs. Exactly40 COMPLETE attempts; starter5 completion and
journal hashes unchanged. Completed-study resume executed0 new runs. Added
summarize_public_refit.py reusing existing aggregate() unchanged; eight summary
tests plus17 fresh runner tests passed. Full audit and analysis saved at
restart_public_refit_analysis_v1, no model reinference. Primary equal-group
RMSE1.067407 +/-0.021053 t/ha; pooled RMSE1.201189 +/-0.038476 (descriptive seed
SD). All40 saved final training fits beat the matched constant, but four groups
have negative mean outer R2. No universal generalization/privacy claim. R13's
unnoised task is VERIFIED_COMPLETE; matched noise follow-up remains R14 pending.
No writer remains. Report:research/restart_public_refit_completion_report.md.
