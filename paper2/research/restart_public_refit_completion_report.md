# R13 full unnoised refit completion and analysis

2026-10-14. R13 unnoised repeated-seed refit VERIFIED_COMPLETE:40/40 completed
and audited, zero failed attempts/pending runs. R14 matched noise follow-up is
separate and NOT_STARTED. Active target remains conditional harvest dry matter,
not best-crop classification, ranking or field-level agricultural utility.

## Acceptance and executed evidence

Inspected six project ledgers, frozen runner/protocol/auditor and starter
evidence, Git state and Python writers. No existing writer was active. Fresh
runner acceptance17 tests passed before training at
results/restart_public_refit_runner_tests_v4. All runtime/source/data/inner-
selection/provenance bindings passed without amending settings.

Resumed the existing output with --execute --resume --cap35; exactly35 new
models fitted/inferred. All40 attempt journals contain one COMPLETE attempt,
zero failures. Prior five starter completion and journal hashes unchanged.
Full audit PASS_COMPLETE_REFIT at restart_public_refit_verified_v2 validates
all40 models/checkpoints/fit and inference receipts, raw-unit cache/targets/IDs,
and seen/unseen support metrics. No model reinference. A completed-study resume
executed0 new runs; progress.json records that last cache-check invocation,
not the preceding35-run execution count.

Added summarize_public_refit.py and test_public_refit_summary.py. The existing
summarize_outer.aggregate() was reused unchanged, not rewritten. Five new
summary tests plus three existing aggregation tests passed;25 fresh scoped
tests total with the17 runner tests. Coverage includes unequal-group weighting,
incomplete/duplicate matrix rejection, per-seed row isolation, support masks,
empty panels, undefined R2 and nonfinite output rejection. No tests failed in
this session; not a full-repository suite or agronomic/privacy certification.

Analysis invokes another full cached audit at
restart_public_refit_analysis_v1/audit/verification.json, rechecks receipt and
journal identities under the study lock, then summarizes only a complete40-run
matrix. Artifact:restart_public_refit_analysis_v1/analysis.json.607 unique
systems per seed, eight geographic groups, five seeds. Seed repeats do not
create3035 independent agricultural observations. Saved training-fit diagnostics
are read from verified checkpoints, not newly inferred training predictions.

## Descriptive primary and secondary results

Values below are mean +/- sample SD over the five fixed seeds, not confidence
intervals for independent regional replications. Units:RMSE/MAE in t/ha.

| Metric | Mean | Seed SD |
| --- | ---: | ---: |
| Equal-group RMSE (primary) | 1.067407 | 0.021053 |
| Equal-group MAE | 0.881704 | 0.022902 |
| Pooled RMSE (secondary) | 1.201189 | 0.038476 |
| Pooled MAE | 0.948698 | 0.043564 |
| Pooled R2 | 0.432669 | 0.036004 |

Geographic group averages give each of eight holdouts equal weight; pooled
scores weight individual systems. Auzeville has354/607 systems, whereas
Kassel/Reading each have4 and Sanmarcoargentano6. These metrics answer different
questions; neither should be substituted silently for the other.

| Outer group | Systems per seed | Mean RMSE | Seed SD | Mean R2 |
| --- | ---: | ---: | ---: | ---: |
| Angers | 75 | 1.657205 | 0.044743 | 0.083167 |
| Auzeville | 354 | 0.989938 | 0.059142 | 0.346560 |
| Copenhagen+Taastrup | 52 | 1.170048 | 0.032327 | -0.077797 |
| Grignon | 64 | 1.707368 | 0.061959 | -0.360441 |
| Jyndevad | 48 | 1.092247 | 0.053302 | 0.254387 |
| Kassel | 4 | 0.562621 | 0.011994 | -0.313808 |
| Reading | 4 | 0.433949 | 0.005893 | 0.662469 |
| Sanmarcoargentano | 6 | 0.925882 | 0.022788 | -0.109800 |

Four groups have negative mean R2; tiny-group variability and distinct group
target distributions limit interpretation. These are retained adverse findings,
not removed or used to retune the model. No numerical superiority over previous
algorithms/configurations is claimed from this one-control summary.

## Training and support diagnostics

All40 saved final training fits are better than their matched equal-client
optimal constant (0 worse). This addresses the training-fit diagnostic for
this specific outer-refit configuration, not universal convergence or reliable
outer generalization. The earlier40/40 underfit v3 controls and11/56 inner-
selection findings remain archived and describe different protocols/partitions.

Seen-species panel:342 systems spanning all eight groups; equal-group RMSE
1.070628 +/-0.024447, pooled1.347460 +/-0.035960. Unseen-species panel:265
systems from Auzeville only; RMSE0.979646 +/-0.065301. Seen/unseen composition
panels have the same support/counts/scores under these splits, not independent
replications. Do not interpret the lower unseen score as universal unseen-
species capability: the panels have different systems/regions/distributions.

Results use fixed scale1/LR .01 and inner-selected median budgets106--149;
no outer-based checkpoint choice or new knowledge/noise mechanism. All prior
outer outcomes were already exposed; these remain development findings, not
independent field confirmation, deployment validation or a privacy guarantee.

## Commands actually executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit_runner --output paper2/results/restart_public_refit_runner_tests_v4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --output paper2/results/restart_public_refit_study_v1 --execute --resume --cap 35
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit_summary --output paper2/results/restart_public_refit_summary_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_public_refit --output paper2/results/restart_public_refit_verified_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.summarize_public_refit --output paper2/results/restart_public_refit_analysis_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_runner --output paper2/results/restart_public_refit_study_v1 --execute --resume --cap 1
```

PowerShell checks additionally verified40 COMPLETE attempts, zero failed
attempts and unchanged starter hashes. No writer remains. Authored:new summary
module/tests, six progress ledgers and this report. Generated:35 new run folders,
new tests, full audits/analysis and study progress. No frozen source/package,
original raw data, Paper1, open LaTeX or old results were edited. No dependency
installation, credentials, public ports, cloud costs or automation changes.

## Next task

R14:prospectively define/freeze matched clipped/noisy controls using the already
fixed inner selections/work, distinct noise identities, retained historical
release-event ledger and updated composed accounting. Test a separate runner
before any new noisy training. Do not tune from these outer scores or edit
archived budgets/results. Unnoised R13 outputs cannot confer application
privacy, secure aggregation or farmer benefit. Agronomic-rule/ranking,
release-certified privacy/SecAgg and remote-deployment gates remain unresolved.
