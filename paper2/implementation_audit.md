# Paper 2 implementation-to-evidence audit — 2026-10-15

## Scope and decision rule

Active Paper 2 is retrospective **conditional harvest seed dry-matter
regression** for observed cereal--legume systems, not best-crop recommendation.
The versioned source is Gaudio et al., Zenodo v2.0.0,
DOI `10.5281/zenodo.14160885`; seven downloaded source assets have recorded
hashes and published-v2 MD5 matches in
`results/restart_regression_data_v1/manifest.json`. The 607 derived system
rows are nested within 37 source field experiments and eight geographic
groups. A status below means: **implemented** = code exists; **tested** =
relevant test passes; **experimentally validated** = completed audited runs;
**scientifically supported** = evidence supports the *specific bounded claim*,
not a broad deployment or causal assertion.

| Component | Source and evidence | Implemented / tested / experimental / claim status |
|---|---|---|
| FedAvg | `federated.py` aggregation, `restart/regression_training.py`, `restart/outer_training.py`; synthetic unit/integration tests; 320 primary outer study contains 40 FedAvg runs | Yes / yes / yes / supports only a simulated baseline comparison on this task |
| FedProx | `federated.py:proximal_loss`, regression client `mu=0.01`, 40 primary outer runs | Yes / yes / yes / **does not** establish advantage over FedAvg; near-equal results need update-level diagnosis |
| Restricted FedNova | `federated.py:aggregate`, constrained plain-SGD local-step normalization; 40 primary outer runs | Yes / yes / yes / only this restricted variant, not all FedNova settings |
| Centralized references | `restart/validation_search.py`, `restart/outer_training.py`; training mean, ridge, RF, XGBoost, MLP | Yes / yes / yes / within-dataset comparison supported; primary neural search may be budget-limited |
| Proposed knowledge-guided federated recommendation | `knowledge.py` rule gate and `knowledge_training.py` provide a **classification-oriented prototype** with synthetic tests; `results/knowledge_gate_v1/gate_report.json` says `BLOCKED_FOR_REAL_DATA_GUIDANCE` | Prototype yes / synthetic tests yes / **no eligible real-data experiment** / no KG-FCR or crop-recommendation claim |
| Agricultural knowledge representation | `knowledge.py` requires class-aligned, original-unit, scope-specific documented soft bounds; no independently approved active regression rules | Gate yes / tests yes / no real-data integration / agronomic validity unestablished |
| Local training and server aggregation | `restart/regression_training.py`, `restart/work_scaled_regression_training.py`, `federated.py`, and versioned runners | Yes / yes / yes in **single-process simulation** / no operating farm clients |
| Data processing and clients | `restart/audit_cereal_legume.py`, `data_pipeline.py`, `public_preprocessing.py`, `prepare_regression.py`, `prepare_public_data.py`; 64 nested folds, train-only fitted preprocessing or separate fixed-public transform | Yes / yes / yes / geographic hold-out is supported; small groups and clustered systems limit independent inference |
| Secure aggregation | No validated secure-aggregation protocol; `restart/noisy_aggregation.py` states raw updates visible to trusted server | No / no / no / no security claim |
| Differential privacy | `restart/noisy_aggregation.py`, `privacy_accounting.py`, privacy and matched-noise simulation runners; 200 + 160 audited studies | Conditional mathematical accounting and deterministic simulation yes / primitive tests yes / utility simulation yes / **no end-to-end or release-certified DP**; conditional epsilon is very large |
| Evaluation/statistics | `restart/summarize_outer.py`, `summarize_public_refit.py`, `summarize_matched_noise.py`, `reviewer_sensitivity.py`; archived per-seed/per-group/support summaries | Yes / yes / yes / descriptive comparisons only; five seeds share the same eight locations |
| Figures | `restart/baseline_publication_figures.py`, `publication_figures.py`; versioned manifests; baseline figure assets and source-analysis hashes rechecked in this audit | Yes / existing figure tests + hash verification / generated from saved analyses / figure existence does not establish manuscript placement or scientific validity |
| Communication and runtime | `federated.py:state_bytes` and some legacy simulator counters; current regression master records do not contain measured per-run network bytes or training wall time | Partial / primitive tests / no field-network measurement / do not claim deployed efficiency |
| Server/client deployment | No authenticated network server or independent farm-client execution in active study; local single-process simulation only | No production deployment / no deployment validation / no operational claim |
| Reproducibility | Frozen protocols, source snapshots, completion receipts, saved predictions and audits; `restart_reproducibility_readiness_v1/readiness.json` verifies 21/21 local assets | Local evidence yes / local tests yes / archived execution yes / **portable clean-room replication blocked** by environment/path/licensing checks |

## Completed experiment inventory and source of truth

`results/master_experiments_v1.csv` has one row per run, with its stage, split,
method, seed, fold/groups, configuration fields, measured regression metrics,
checkpoint and hashed source paths. The companion
`results/master_experiments_v1.audit.json` records 2,384/2,384 run receipts
whose listed assets match SHA-256. Stage counts are 1,120 original inner-search,
320 primary outer, 320 post-hoc guard, 224 fixed-public inner-search, 40
fixed-public refits, 160 matched-noise evaluations, and 200 earlier privacy
simulations. There are 1,344 inner-validation and 1,040 outer-test records.
Historical classification/baseline studies on different data, exploratory
smokes, and pilot diagnostics are **not** pooled into this active regression
paper's master table. They remain on disk and are not deleted.

The active target and valid main metrics are t/ha RMSE and MAE, with group-wise
and pooled variants. Classification accuracy and macro-F1 are intentionally
blank in the master CSV. Measured network bytes and per-run training elapsed
time are also blank: a model-state byte-count helper is not a network trace.
Do not silently backfill these fields with estimates. Source config
`restart_validation_protocol_v1/config.json` documents the 80-round, 32-unit
plain-SGD primary neural search; 286/672 neural candidates reached its cap,
and four outer folds selected the grid's highest offered learning rate.
Consequently, poor neural performance cannot be attributed uniquely to
federation. Later fixed-public results change several design factors and are
development evidence, not a controlled improvement claim.

Existing independent audits report `PASS_COMPLETE` for original validation
and primary outer runs; `PASS_COMPLETE_POST_HOC_SENSITIVITY` for the guard;
`PASS_COMPLETE_INNER_SEARCH` for fixed-public selection;
`PASS_COMPLETE_REFIT` and `PASS_COMPLETE_MATCHED_STUDY` for later refit/noise.
The privacy-v3 analysis reports `PASS_COMPLETE_MATCHED_SIMULATION`, **not** a
privacy certification. The master index independently checks run-level
receipt hashes and split sample counts but does not retrain models or infer
from test predictions. Its generation script is
`restart/master_experiment_index.py` and refuses output overwrite.

## Cross-checks and observed limitations

- Duplicate stage/run identities: none in the index. All seven expected stage
  counts and every run's receipt assets match; evaluation sample counts agree
  with archived partition sizes. No new model inference was performed.
- Baseline figure-manifest assets match hashes, and its original/guard input
  analysis hashes match the saved analyses. Other figures were not visually
  re-reviewed in this audit, so full manuscript figure placement remains open.
- Outer test was used once for the frozen primary study, but its results and
  the later guard/fixed-public/noise outcomes are now known. Subsequent tuning
  on these same groups cannot yield an untouched independent test claim.
- The 607 systems are clustered into 37 experiments; all experiment IDs have
  multiple rows. Three geographic groups contain six or fewer systems. Five
  seeds are computational repeats, not independent geographic samples.
- The fixed-public hash transform maps 13 observed `previous_crop` categories
  into seven of eight bins, with nine distinct colliding pairs; future-site
  collision behavior remains unknown.
- Prediction-time availability of management variables, cultivar and density
  harmonization, new-site validity, prospective guard performance, formal
  privacy/SecAgg, and real network deployment are unresolved.

## Re-run of relevant tests during this audit

`& '.\.venv\Scripts\python.exe' -B -m unittest
paper2.tests.test_federated paper2.tests.test_knowledge
paper2.restart.test_data_pipeline paper2.restart.test_privacy_accounting
paper2.restart.test_master_experiment_index -q` -> **55 tests passed** when
the shell set workspace-local `TEMP`/`TMP` and included `paper2/restart` on
`PYTHONPATH`. An initial invocation had nine environment/import errors:
Windows sandbox denied atomic rename in its external temporary directory, and
`test_data_pipeline` expects the restart directory on the import path. No
source-code fix or weakening of a test was needed; the corrected run passed.
Three new master-index unit tests cover receipt tampering, best-validation
selection and split/metric validation.

## Scientific conclusion for writing

Supported: a **bounded, retrospective comparison** of established central
and simulated federated regressors under held-out geographic groups, plus
explicitly labeled post-hoc support and aggregation-noise analyses. Not
supported: a new knowledge-guided federated method, best-crop selection,
causal farm benefit, significant location-level superiority, production
privacy, independent new-site validation, or deployment efficiency. The
research-gap and claim gates are in `research/reviewer_response_literature_v1.md`
and `research/reviewer_claim_evidence_matrix_v1.md`. Matched ablations and
stronger neural tuning remain planned in
`REVIEW_RESPONSE_IMPLEMENTATION_PLAN.md`; they must be versioned separately.
