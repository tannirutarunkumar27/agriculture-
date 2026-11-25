# Paper 2: geographically grouped harvest-regression study

**Current scope (2026-10-16):** retrospective conditional seed-dry-matter
regression for recorded cereal–legume systems. The completed primary study
compares centralized models and single-process, simulated horizontal
federated baselines across eight held-out geographic groups. It is not a
best-crop recommender, an evaluated knowledge-guided federated regressor, a
privacy-preserving deployment, or an end-to-end differentially private system.

The verified original comparison contains 1,120 inner-validation candidates
and 320 outer evaluations. Post-hoc support-guard, later fixed-public refit,
and matched aggregation-noise evaluations are separate development stages;
none should be rerun or merged into the original method ranking. The active
versioned manuscript is
`manuscript/PAPER2_REGRESSION_MANUSCRIPT_V4.md`; its text evidence is bound by
`results/regression_manuscript_v3/manifest.json` and its five placed audited
figures by `results/regression_manuscript_v4/manifest.json`. V2 and V3 remain
as versioned intermediates. See
`research/complete_research_validation_audit_v1.md` for the scientific audit
and `research/regression_manuscript_release_boundary_v1.md` for what remains
before a portable public release. Local evidence verification is not
independent clean-room reproduction.

## Historical design and progress notes below

The material below records earlier KG-FCR/classification ideas and interim
states. It is retained for provenance, not as the present paper's scope or
latest completion status.

Current direction (2026-10-13): user-requested research-design restart. Prior
implementation and classification results are retained as historical evidence.
Restart step 1 feasibility audit is complete:
`research/restart_feasibility_report.md`, `results/restart_feasibility_v1/` and
eight passing tests in `results/restart_audit_tests_v1/`. No new training or
best-crop labels. Define a restricted task/protocol before proceeding.

Restart step 2 now saved: `research/restart_task_protocol.md` and
`configs/restart_task_protocol_v1.json`. Proposed conditional harvest regression
and nested site evaluation are explicit; ranking/knowledge/privacy/training
remain disabled pending feature, grouping, implementation and protocol gates.

Active restart progress: new cereal/legume regression data now PREPARED and
SPLIT in `results/restart_regression_data_v1/` (607 systems, eight geographic
groups, eight outer/56 inner folds); all 64 verified. Fifteen new preparation/
client tests passed. Eight configurations ran a validation-only smoke on new
data, with no outer-test inference (`results/restart_regression_smoke_v1/`).
The fresh plan and executed evidence are in `research/restart_implementation_plan.md`
and `research/restart_preparation_phase_report.md`. Earlier paragraphs/counts
are historical; new baseline research scores still require frozen experiments.

Validation tuning is complete: all 1,120 frozen candidates verified, zero
failures or pending runs. Settings are selected for all eight outer folds;
outer-study execution subsequently completed all 320 planned runs, independently
audited with PASS_COMPLETE (restart_outer_verified_v3). Fourteen search tests passed again.
286/672 neural candidates selected round 80 and four outer folds selected the
highest grid learning rate; these bounded choices do not certify convergence.
Current ledger: `results/restart_validation_search_v1/progress.json`.
Evidence and next-stage boundary: `research/restart_validation_completion_report.md`.

Latest outer execution: **320/320 complete, zero pending/failures** across eight
groups, five seeds and eight methods. All 320 cached results independently
verified without repeated model inference. Execution report:
`research/restart_outer_completion_report.md`. Matched-group/seed comparative
analysis remains pending; outer scores must not be used to change frozen settings.

This isolated project preserves Paper 1 code and results. KG-FCR is a working
name, not a validated novelty claim. No privacy guarantee or federated result
has been established here.

## Current phase

Stage A audit completed. Focused closest-paper comparison and agronomic-source
checks are saved; exhaustive novelty/source verification remains open. The
class-aligned knowledge mechanism is provisional and disabled. Reproducible
group-preserving partitions, centralized MLP training, FedAvg, sample-weighted
FedProx, and plain-SGD FedNova are implemented for public-data CPU simulation.

From the repository root, using the existing environment:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s paper2/tests -v
& '.\.venv\Scripts\python.exe' -B paper2/scripts/audit_data.py --output paper2/results/data_audit.json
```

The audit hashes sources, records schemas and class distributions, detects
duplicate files and rows, and rejects the current unusable NASA download for
weather use. It does not change source data or fit preprocessing.

Read `research/literature_screening.md`, `research/method_specification.md`,
and `STATUS.md` before implementing training. Original archives, licenses,
citations, nutrient units, rainfall periods, and source versions are still
needed for the newly downloaded crop dataset.

## Implementation and verification

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/prepare_partition_matrix.py --config paper2/configs/partition_matrix.json --output paper2/results/partitions_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_smoke_matrix.py --partitions paper2/results/partitions_v1 --config paper2/configs/smoke.json --output paper2/results/smoke_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v2
```

These commands refuse existing output directories. Use a new version name for
a new execution, not deletion/overwrite. The earlier audit command also refuses
an existing report. To run one baseline:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.train --partition paper2/results/partitions_v1/canonical_iid_default_seed42.json --algorithm fedavg --config paper2/configs/smoke.json --output paper2/results/my_new_run
```

For interrupted training use the same command with `--resume`, preserving all
settings and pointing at its original output directory. Config target rounds
must exceed saved completed rounds. Diagnostic extension permits larger round
budgets; it is not a protocol for choosing research budgets after viewing test
scores. For publication, freeze budgets/selection before looking at the test set.

### Protocol

- Separate immutable CSV source hash, row-ID splits, crop taxonomy, and client
  distributions. Whole identical input profiles and entire Manipur fields stay
  together. Do not concatenate datasets or fabricate district labels.
- Canonical: 70/15/15. Grouped small-class datasets can deviate: actual fractions
  are saved. Manipur: three training fields plus one validation and one test
  field per crop, hence 60/20/20; this is unseen-field, same-district evaluation,
  not leave-one-district-out regional generalization.
- Train-only global mean/std and soil-color vocabulary are shared across methods
  as a PUBLIC simulation reference. No private distributed preprocessing exists.
- Dirichlet strategy draws label-specific group assignment weights; unequal
  draws a single global size vector. Draws are conditioned on the documented
  minimum client size, with at most 200 attempts. IID is random equal-group
  allocation, not guaranteed equal class proportions.
- Same initial MLP, optimizer LR, held-out rows and settings. Centralized full
  passes and federated local passes are NOT equal optimizer-step budgets;
  actual steps are recorded. Hyperparameter tuning/multiseed training is pending.
- Validation NLL selects best round. Test scores and probabilities are saved
  after selection. Short smoke scores demonstrate execution, not convergence.
- Partial participation uses uniform sampling without replacement and selected
  sample-count normalization. Missing/incompatible updates fail rather than
  silently skipping clients. Simulator has no asynchronous timeout/dropout model.
- Logs contain metrics/counts, not raw training records. Aggregation accepts only
  model update objects; the coordinator process can still access all simulation
  data. No cryptographic isolation/authentication or privacy guarantee.
- Checkpoints save last and best models, records and config/source/code identity;
  resume changes to source/partition/model settings/implementation are rejected.
- Communication counts are tensor payload sizes, not measured network bytes.

Results: source/partition hashes, client distributions, per-round validation,
local work/time, logical communication, selected test metrics, per-class report,
confusion matrix, predictions, environment manifest and resumable checkpoint.

Refer to `research/closest_paper_comparison.md` and
`research/agronomic_source_checks.md` for source links and unresolved evidence.
Observed dependency versions are recorded, not installed, in
`requirements-runtime-observed.txt`. Use `STATUS.md` for executed verification.

## Validation convergence phase

The new conventional references (`centralized.py`) and bounded pilot operate
without test evaluation. Neural tuning calls `run(..., evaluate_test=False)`;
CLI use `--validation-only`. This flag is part of resume identity: a tuning
checkpoint cannot silently become a test-evaluated run. Historical smoke
commands still intentionally evaluate their development test split.

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v4
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_convergence_pilot.py --partitions paper2/results/partitions_v1 --config paper2/configs/convergence_pilot.json --output paper2/results/convergence_pilot_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_convergence_pilot.py --pilot paper2/results/convergence_pilot_v1 --partitions paper2/results/partitions_v1 --output paper2/results/convergence_verified_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/capture_pilot_sources.py --pilot paper2/results/convergence_pilot_v1 --output paper2/results/convergence_pilot_v1/source_snapshot
```

These are already executed output names: choose NEW version directories for
fresh runs. To continue the same interrupted pilot, pass `--continue`; it checks
protocol/source/code identities and resumes neural round checkpoints. Completed
candidates are then rechecked by the independent verification command. Do not
load arbitrary joblib files: verification only loads locally generated models.

Pilot protocol: `research/convergence_protocol.md`. Results/interpretation:
`CONVERGENCE_PHASE_REPORT.md`. All 38 candidates and selected metrics are
validation-only; no final multi-seed or statistical comparison is implied.
Earlier smoke test exposure is explicitly documented. Knowledge remains
disabled and privacy mechanisms unimplemented.

## One-by-one: step 1 frozen baseline study

See `research/frozen_baseline_protocol.md` for the fixed budgets, five seeds,
client scenarios, development-holdout status, metrics and limitations.
The template and materialized 235-run plan are saved. This section records the
step-1 freeze; see step 2 below for current execution progress.

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --config paper2/configs/baseline_study_v1.json --output paper2/results/frozen_baseline_protocol_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

The first command was already executed; it refuses overwrite. Verification is
read-only and can be repeated. Changes to settings, baseline code, datasets or
dependencies require a new amended version. These freeze/verify commands never
launch training.

## One-by-one: step 2 study runner implemented

`scripts/run_baseline_study.py` consumes the frozen plan, starts no training
without `--execute`, and defaults to one new/resumed run per invocation.
Existing output requires `--resume`; settings/round budgets cannot be overridden.
See `research/study_runner_guide.md` before using recovery options.

```powershell
# Read-only inspection:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 1
# NEXT bounded batch, not executed in the step-2 report:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/run_baseline_study.py --frozen paper2/results/frozen_baseline_protocol_v1 --output paper2/results/baseline_study_v1 --max-runs 5 --execute --resume
```

Full tests: 70 passed. Real-plan starter subset: 4/235 completed; 231 remain.
Completed results are verified, not rerun, during resume. The runner records
failures and refuses ambiguous repeated test inference; inspect errors before
`--retry-failed`. No automatic silent skipping/replacement or privacy claims.
Saved commands/results: `STUDY_RUNNER_PHASE_REPORT.md` and `STATUS.md`.

## One-by-one: step 3 baseline study completed

All **235 frozen development runs** completed with zero failures. Analysis
verified all cached outputs without repeating model test inference and saved
47 complete five-seed metric groups, 182 paired comparisons and 910 paired
seed differences in `results/baseline_analysis_v1/`. Final software suite:
80 passed, zero failed (`results/tests_v9`). Step 2 counts above are historical.

See `BASELINE_STUDY_PHASE_REPORT.md` for exact commands, findings, saved evidence
and limits; `research/baseline_study_analysis_guide.md` documents the new
summarizer. Scores are development evidence, not independent confirmation,
knowledge-guided improvement, formal privacy or field validation.

Next gated step: agronomic source/unit validation and a tested class-aligned
knowledge module with prospectively frozen ablations. Existing unverified
crop rules remain disabled. Privacy and deployment are later separate phases.

## One-by-one: step 4 knowledge engineering

Class-aligned soft-bias `KnowledgeGuidedMLP`, strict `RuleBank` declarations,
explicit multi-input local training and identity-bound selected-weight
persistence are implemented. Full suite: **96 passed, zero failed**, including
16 new knowledge tests and synthetic two-round federated smoke checks.

**Real-data guidance remains blocked:** both dataset contracts lack verified
measurement metadata and the approved bank is empty. No real-data KG scores
or privacy guarantees were produced. The existing frozen runner was not changed;
a new experiment coordinator and protocol freeze are still needed after approval.

See `KNOWLEDGE_MODULE_PHASE_REPORT.md`,
`research/knowledge_module_specification.md`, and
`results/knowledge_gate_v1/gate_report.json` for evidence and next requirements.
