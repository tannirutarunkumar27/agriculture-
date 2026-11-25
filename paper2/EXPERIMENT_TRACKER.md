# Active experiment tracker

Manuscript-only continuation (2026-10-16): no experiment was rerun. A
regression-scoped V4 draft and V3/V4 hash manifests reuse accepted caches; the
post-hoc geographic table is from `reviewer_sensitivity_v1/analysis.json`.
See `research/regression_manuscript_release_boundary_v1.md` for open release
gates. Historical experiment entries below are unchanged.

| Experiment | Data | State | Evidence |
| --- | --- | --- | --- |
| Historical 235-run classification development | old datasets | retained, not new-study evidence | baseline_study_v1 / baseline_analysis_v1 |
| Regression diagnostic | cereal-legume | 8 configurations completed; not convergence | restart_regression_smoke_v1 |
| Frozen inner search | cereal-legume, 56 inner folds | 1120/1120 verified, zero failed | restart_validation_search_v1 / restart_validation_verified_v2 |
| Selection audit | same saved validation outputs | all 8 selections verified | restart_validation_selection_v1 |
| Repeated-seed outer evaluation | same prepared geography splits | VERIFIED_COMPLETE execution: 320/320, eight groups, five seeds, eight methods; zero pending/failures | restart_outer_study_v1 / restart_outer_verified_v3 |
| Matched-group/seed comparative summary | completed outer predictions | VERIFIED_COMPLETE descriptive analysis; three aggregation tests passed | restart_outer_analysis_v1 / research/restart_baseline_comparison.md |
| Zero-support sensitivity | unchanged saved models, training-derived gate | VERIFIED_COMPLETE 320/320; seven tests passed; POST_HOC | restart_guard_study_v1 / restart_guard_analysis_v1 |
| Noisy-aggregation integration | 12 synthetic records, two clients | VERIFIED_COMPLETE software smoke: 3 rounds; 16 tests pass; no application privacy certification | restart_private_primitives_v1 |
| Fixed-transform data | 607 systems, original geography splits | VERIFIED_COMPLETE preparation, 64 folds verified | restart_public_data_v1 / restart_public_data_verified_v1 |
| Matched noise-utility simulation v1 | fixed-transform public benchmark | Closed diagnostic: 3 complete, sigma4 failed after round47; 125 committed noise events | restart_privacy_simulation_v1 / restart_privacy_verified_v1 |
| Matched noise-utility simulation v2 | same public benchmark; matched local gradient clip1 | Closed:6 runs retained; superseded by declared v3 | restart_privacy_simulation_v2 / restart_privacy_verified_v3 |
| Matched noise-utility simulation v3 | Four public steps, C=LR*4 | VERIFIED_COMPLETE:200/200 audited, zero failures | restart_privacy_simulation_v3 / restart_privacy_analysis_v3 |
| Control-fit diagnosis | Public training inputs only, 40 controls | All40 worse than matching equal-client constant; two tests pass | restart_control_fit_diagnosis_v2 |
| Control conditioning pilot | Two inner folds, scale1/10,80/240-round reports | VERIFIED_COMPLETE diagnostic:4 candidates audited;3 tests passed | restart_control_pilot_v1 / restart_control_pilot_verified_v1 |
| Expanded fixed-transform control search | All56 inner folds,2 scales,2 LRs | VERIFIED_COMPLETE:224/224 and all8 selections independently audited;224 COMPLETE attempts,zero failures; cache resume executed0 | restart_control_search_execution_v1 / restart_control_search_v1 / restart_control_search_verified_v3 |
| R12 coordinator acceptance | Synthetic data and copied bindings | VERIFIED_COMPLETE:fresh17 scoped tests pass completion session; previously26 dependency tests passed; no errors/failures | restart_control_search_tests_v4 / restart_control_search_dependencies_tests_v1 |
| R12 supplemental provenance | Completed224 runs and journals | VERIFIED_COMPLETE:strict224-journal gate and fresh metric audit pass; no retraining | restart_control_search_integrity_v1 / restart_control_search_verified_v4 |
| R13 fixed-public unnoised refit | Completed R12 selections, eight groups x five seeds | VERIFIED_COMPLETE:40/40 audited,zero failures;40 COMPLETE attempts; cached resume executed0 | restart_public_refit_execution_v1 / restart_public_refit_study_v1 / restart_public_refit_verified_v2 |
| R13 complete descriptive summary | Audited cached predictions;607 systems per seed | VERIFIED_COMPLETE:equal-group/pooled metrics,regional and seen/unseen panels;8 summary tests pass; no reinference | restart_public_refit_analysis_v1 / restart_public_refit_summary_tests_v1 |
| R14 matched noise follow-up | Fixed-public inner-selected settings | VERIFIED_COMPLETE:160/160 independently audited,zero failures;40 cached controls;23174 events;complete matched analysis | restart_matched_noise_verified_v2 / restart_matched_noise_analysis_v1 / research/restart_matched_noise_completion_report.md |
| R13 runner/inference/cache acceptance | Synthetic recovery, schema, cache and package fixtures | VERIFIED_COMPLETE:17 runner +18 protocol/core +26 dependency tests pass; one initial assertion failure corrected/log retained | restart_public_refit_runner_tests_v3 / restart_public_refit_tests_v5 / restart_public_refit_dependency_tests_v2 |
| R13 preparation acceptance | Synthetic planner/refit/audit regression | VERIFIED_COMPLETE:18 new +17 search +26 dependency tests pass; corrected intermediate fixture error retained | restart_public_refit_tests_v4 / restart_control_search_tests_v5 / restart_public_refit_dependency_tests_v1 |
| Knowledge / ranking / privacy | not yet eligible/frozen | no validated experiment | R06 / R07 / R08 |
| R10a unnoised figures | Fresh complete R13 cached audit,40 runs | VERIFIED_COMPLETE:3 plots in PNG/PDF/SVG;PNG visual review PASS;five tests PASS;no reinference | restart_public_refit_figures_v2 / restart_publication_figure_tests_v3 |
| R10b matched figures | Fresh complete R14 audit,160 +40 reference runs | VERIFIED_COMPLETE:3 five-variant plots in PNG/PDF/SVG;PNG visual review PASS;no reinference | restart_matched_noise_figures_v1 |
| R10c baseline figures | Fresh complete original and guard cached audits,320 +320 evaluations | VERIFIED_COMPLETE:3 plots in PNG/PDF/SVG;PNG visual review PASS;four tests PASS;post-hoc guard separated;no training/reinference | restart_baseline_figures_v2 / research/restart_baseline_figures_report.md |
| R10d claim/manuscript scaffold | Completed verified studies and blocked gates | VERIFIED_COMPLETE scaffold:10 claims validated,4 blocked classes preserved;Paper-2-only skeleton;no Paper1 edit | research/evidence_claim_matrix.json / manuscript/PAPER2_MANUSCRIPT_SKELETON.md / restart_claim_matrix_validation_v1 |
| R10e regression closest-work review | Primary publisher/proceedings/author records | VERIFIED_COMPLETE: 8 study records and 3 algorithm sources; numeric-superiority and privacy boundaries validated; 3 scoped tests pass | research/restart_regression_closest_work.json / research/restart_regression_closest_work.md / restart_regression_literature_validation_v1 |
| R10f Introduction and Related Work | Validated claim registry and regression source registry | VERIFIED_COMPLETE draft: 1,271 words; 7 headings, 8 study links and 3 algorithm links pass; no Paper1 edit | manuscript/PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md / restart_manuscript_sections_validation_v1 |
| R10g Data and Methods | Frozen data/search/outer/refit/noise protocols and audits | VERIFIED_COMPLETE draft: 1,850 words; 12 headings and 14 evidence facts pass; 2 regression tests pass | manuscript/PAPER2_DATA_METHODS_DRAFT.md / restart_data_methods_validation_v1 |
| R10h Results | Accepted original/guard/refit/matched analyses and nine figures | VERIFIED_COMPLETE draft: 1,530 words; 6 headings and 17 evidence facts pass; 2 regression tests pass | manuscript/PAPER2_RESULTS_DRAFT.md / restart_results_validation_v1 |
| R10i Discussion through Conclusion | Validated results, limitations, literature and recovery evidence | VERIFIED_COMPLETE draft: 1,728 words; 8 headings and 16 scope/evidence facts pass; 2 regression tests pass | manuscript/PAPER2_DISCUSSION_CONCLUSION_DRAFT.md / restart_discussion_conclusion_validation_v1 |
| R10j full manuscript assembly | Six validated components and nine accepted figure assets | VERIFIED_COMPLETE draft assembly: 6,998 words, Sections I--IX, 13 working references; structure/source/hash gate PASS; 2 tests pass | manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md / restart_full_manuscript_v1 / restart_full_manuscript_validation_v1 |
| R10k reproducibility readiness | Essential manuscript, source, data, audit, figure and runtime evidence | LOCAL_EVIDENCE_PASS / PORTABLE_BLOCKED: 21/21 assets hashed and validated, zero errors; six declared portability blockers; 3 tests pass; no training/inference | restart/reproducibility_readiness.py / restart_reproducibility_readiness_v1 / research/restart_reproducibility_readiness_report.md |
| R10l bibliography | Thirteen manuscript sources | VERIFIED_COMPLETE working BibTeX registry: 13 full-author entries with persistent identifiers, including the six Introduction keys; source and key checks pass; citation placement still follows manuscript sections | manuscript/references.bib / tests/test_intro_bibliography.py |
| DC01 dataset/client validation | Frozen 607-system source row table and 64 group-held-out folds | VERIFIED_COMPLETE: 21 conceptual inputs; 8 geographic groups; 392 client-fold records; 64 folds independently reverified; 12 scoped tests pass; no training/inference | results/dataset_client_validation_v1 / research/dataset_client_validation_v1.md |

Directories are under results/. Existing outputs are immutable; new execution
versions must use new paths. Each new entry records exact frozen config, source
identity, split IDs, seeds, commands, attempts, tests, results and interpretation.
# Reviewer-response analyses, 2026-10-15

| Task | Evidence | Status | Boundaries / next step |
|---|---|---|---|
| RR01 literature | `research/reviewer_response_literature_v1.md` | VERIFIED_COMPLETE | Twenty primary records; no novelty claim proven |
| RR02 group sensitivity | `results/reviewer_sensitivity_v1/analysis.json` | VERIFIED_COMPLETE | Existing outer scores only; post-hoc descriptive |
| RR03 data/encoding audit | `results/reviewer_data_audit_v1/analysis.json` | PARTIAL | Source counts and collisions verified; prediction-time inputs pending |
| RR05/RR06 matched ablations and stronger neural tuning | `REVIEW_RESPONSE_IMPLEMENTATION_PLAN.md` | NOT_STARTED | Freeze tested inner-only protocol before training; no reruns of old studies |
| RR08 new-site guard confirmation | none | BLOCKED | No untouched comparable geography with validated target/units |

RR02/RR03 tests: `& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_reviewer_sensitivity paper2.restart.test_reviewer_data_audit -v` -> 6 passed, 0 failed after correcting a test lookup. No expensive training was launched.

## Implementation audit and master index

AU01/AU02: `implementation_audit.md` and `results/master_experiments_v1.csv`
with companion `.audit.json`. Read-only index of 2,384 archived completed
regression runs; all completion asset hashes and evaluation sample sizes pass.
No models were loaded or rerun. Direct CSV recomputation of primary XGBoost
gave equal-group RMSE 1.246124 and pooled RMSE 1.733538, matching archived
summary rounding. New index code and three tests:
`restart/master_experiment_index.py`, `test_master_experiment_index.py`.
Scoped audit test command in `implementation_audit.md`: 55 passed after
workspace-local temp and import-path correction; original invocation had
nine environment/import errors, not a model-code regression.

AU03: five-area screening matrix `research/literature_evidence_matrix_v2.md`
contains 26 primary records, with unknown full-text details marked as such.
No external scores treated as comparable to the project's held-out t/ha RMSE.

