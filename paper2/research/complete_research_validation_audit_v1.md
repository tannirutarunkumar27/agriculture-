# Paper 2 audit-first research validation — initial report

Date: 2026-10-15. Scope: current `paper2/` checkout and its archived evidence.
This is a **read-only scientific/code audit**; it is not a correction, new
experiment, privacy certification, or independent reproduction. No source,
checkpoint, dataset, archived result, or manuscript was modified for this
report. Status labels apply to the *specified restricted claim*, not to a
broader product or a different crop-classification task.

## 1. Executive decision

**Not ready for submission under the supplied title** “Knowledge-Guided
Federated Learning for Privacy-Preserving Crop Recommendation Across
Heterogeneous Agricultural Regions.” The completed study is instead
retrospective conditional **harvest seed dry-matter regression** for 607
recorded cereal–legume management systems. It has no best-crop labels, no
eligible real-data knowledge rules, no evaluated knowledge-guided federated
regressor, no secure aggregation, and no end-to-end DP. The current assembled
manuscript already uses a more accurate regression title and explicitly denies
these claims. A bounded empirical/regression paper is possible, subject to
the methodological and portability issues below. This audit does not predict
journal acceptance.

The original frozen comparison comprises 320 outer evaluations (eight held-
out geographic groups × five seeds × eight methods); XGBoost had the lowest
observed equal-group RMSE, 1.2461 t/ha, versus 5.8373 for FedAvg. The later
fixed-public result (1.067407 t/ha) changed representation, weighting,
optimization and budget together and used already exposed groups; it cannot
be presented as a controlled improvement over the original result. Its
matched-noise experiment is a utility simulation, not production privacy.

## 2. Repository inventory and verification scope

The relevant tree contains `paper2/federated.py`, `knowledge.py`,
`knowledge_training.py`, `restart/` preparation/training/runners/verifiers,
`tests/`, frozen protocol packages, result receipts/caches, three figure
packages, a Markdown manuscript, BibTeX, status ledgers and the observed
Windows dependency record. `paper2/results/master_experiments_v1.csv` indexes
2,384 completed active-regression run records across seven stages. Its
companion audit says all listed receipt assets were verified; its current CSV
SHA-256 matches that audit. This check verifies the index's present identity,
not independent retraining or direct rehashing of every model in this turn.

`git log -5 --oneline` exposed only the repository's initial commit;
`git status --short -- paper2` reports `?? paper2/`. Thus Paper 2's current
source, manuscripts and evidence are untracked by Git in this checkout. Frozen
manifests, asset hashes and attempt journals provide local provenance, but Git
history cannot demonstrate when individual Paper 2 changes were made.

Current read-only tests: 69 algorithm/knowledge/data/outer-audit/privacy/noise
tests passed; nine baseline/publication-figure tests passed. The local
reproducibility checker returned `PASS_LOCAL_EVIDENCE_BLOCKED_PORTABLE_REPRODUCTION`
with 21/21 assets present and zero errors. These are software/evidence checks,
not proof of agronomic validity or external reproducibility. The previous DC01
audit separately rechecked all 64 folds and generated 392 client-by-fold
records; its 12 scoped tests passed. No archived training run was restarted.

## 3. Algorithm verification table

Original definitions were checked against [FedAvg (AISTATS 2017)](https://proceedings.mlr.press/v54/mcmahan17a.html),
[FedProx (MLSys 2020)](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html),
and [FedNova (NeurIPS 2020)](https://papers.nips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html).

| Component and mathematical requirement | Code and test evidence | Finding | Required correction / boundary |
| --- | --- | --- | --- |
| FedAvg: local SGD then sample-size-weighted average of compatible model states | `federated.py:aggregate`, `restart/regression_training.py:train_regression_client`, `tests/test_federated.py:test_sample_weighted_fedavg`, `test_reject_malformed_updates`; 40 primary outer FedAvg runs | **VERIFIED_CORRECT for restricted full-participation, plain-SGD simulation**. Counts, client identity, round/base/schema, tensor dtype/shape and finiteness are checked. | Missing client updates cause rejection, not a tested dropout-tolerant round. No authenticated remote clients. Do not extrapolate to production FedAvg. |
| FedProx: local MSE plus `(mu/2)||theta-theta_round_start||²`; sample-weighted server average | `federated.py:proximal_loss`, `restart/regression_training.py`, `restart/outer_training.py:fit`; positive-mu and gradient tests in `tests/test_federated.py`; 40 primary outer runs | **VERIFIED_CORRECT for the tested proximal term**, with `mu=0.01` in the original study. The reference is the round-start clone. | Report the actual near-equality to FedAvg; diagnose update-level magnitude before claiming a useful proximal effect. No variable-work/dropout experiment establishes FedProx's broader advantage here. |
| FedNova: normalize local displacement by actual plain-SGD step count; aggregate with effective weighted steps | `federated.py:aggregate`; `test_unequal_step_fednova`, `test_equal_steps_fednova_matches_fedavg`; 40 primary outer runs | **VERIFIED_CORRECT only for the constant-shared-LR, no-momentum/no-weight-decay vanilla-SGD specialization**. The implemented rule is `theta_next=theta-Σp_k tau_k Σp_k(theta-theta_k)/tau_k`. | Call it *restricted FedNova*. Do not claim support for arbitrary local solvers, momentum, schedules or the paper's full generality. |
| Knowledge-guided federated method: valid class/crop rules must alter an actually evaluated outcome path | `knowledge.py:KnowledgeGuidedMLP`, `knowledge_training.py:train_knowledge_client`; synthetic tests pass; `results/knowledge_gate_v1/gate_report.json` blocks real rules | **PARTIALLY_IMPLEMENTED prototype; MISSING from active regression experiments.** Static suitability adds `alpha(q-0.5)` to classification logits; there is no knowledge-aware server aggregation or regression KG loss. | Do not claim KG-FCR, knowledge benefit, a new optimizer, or a knowledge ablation. Require independent unit/scope-approved agronomic evidence, a regression-compatible mechanism or legitimate crop-choice target, frozen matched ablations and new-site validation. |
| Failure/identity integrity | `federated.py:aggregate`, `restart/outer_runner.py`, `restart/outer_training.py:evaluate_once`; synthetic mutation/recovery tests and run receipts | **VERIFIED_CORRECT as local fail-closed software safeguards**. Once-only inference intent/cache and immutable identities are present. | Hashes and roster checks are not cryptographic authentication; the in-process coordinator still sees individual model states. |

The manuscript's FedAvg/FedProx/FedNova equations match these restricted code
paths. One concrete typesetting defect remains: in
`manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md:335`, the regression equation has
`=mathbf{w}_2` instead of `=\mathbf{w}_2`. This does not invalidate training,
but must be fixed before LaTeX conversion. No equation for a real-data
knowledge-guided regressor can be verified because none ran.

## 4. Literature, gap and novelty

The local 26-primary-record screening matrix and eight closest agricultural
study records are useful, but are **not a systematic full-text novelty search**.
Closest contexts include [Li et al.'s localized federated yield
forecasting](https://www.sciencedirect.com/science/article/pii/S0957417423033493),
the other crop-yield FL studies in `research/restart_regression_closest_work.md`,
and the newer [X-FedAgriTwin crop-recommendation
paper](https://www.sciencedirect.com/science/article/pii/S095058492600162X).
Their datasets, splits, client definitions, metrics and budgets are not matched
to this work, so external scores are not numerical comparators. Agricultural
knowledge-guided modeling is also established; the source represented in the
matrix as “Nature Communications 2023” was actually [published 8 January
2024](https://www.nature.com/articles/s41467-023-43860-5). Correct that
screening metadata before citation.

| Candidate gap / novelty claim | Supporting and contrary evidence | Decision |
| --- | --- | --- |
| “Federated crop recommendation is new” | X-FedAgriTwin and other agricultural FL literature already address recommendation/yield contexts; active data lack choice labels | **CONTRADICTED / inapplicable** |
| “Knowledge-guided federation is the implemented contribution” | Classification-only local prototype; no eligible active rules, no real-data KG comparison | **UNSUPPORTED** |
| “Privacy-preserving regional crop recommendation” | Public-data single-process clients, visible updates, conditional noisy simulation | **CONTRADICTED for this implementation** |
| “A grouped, support-aware empirical comparison of existing regression baselines” | Nested eight-group protocol, original/guard/refit/noise archived analyses and qualified manuscript | **SUPPORTED as bounded evaluation contribution**; algorithmic novelty not established |

The precise *prospective* mechanism separating a future KG method from the
closest prior work has not been validated. The current contribution should be
classified as an application-specific evaluation/methodological audit, not a
new federated algorithm. Full-text comparison of the nearest KG+FL methods,
their actual rule representation, client protocol, data, and ablations remains
needed before any stronger novelty statement.

## 5. Dataset, clients and leakage

The official [Zenodo v2.0.0 dataset](https://zenodo.org/records/14160885)
reports 37 field experiments. Seven local raw assets match published MD5s
and recorded SHA-256s. Derived data contain 607 systems, 37 experiments, 21
conceptual inputs, eight geographic groups, 299 sole and 308 intercrop systems,
and a continuous t/ha target—**zero crop-class labels**. Feature units,
missing counts, duplicate checks and per-client species composition are in
`research/dataset_client_validation_v1.md` and
`results/dataset_client_validation_v1/`. Eight outer/56 inner folds passed
group/site/experiment isolation, once-only outer coverage, client roster and
training-only fitted-statistic reconstruction. This is stronger than a mere
random row split.

The seven training clients per outer fold are centrally simulated geographic
partitions of public research data, not deployed farms. Auzeville has 354/607
systems; Kassel and Reading have four each and Sanmarcoargentano six. All 37
experiments contribute multiple systems, so 607 is not an independent-site
sample size. `previous_crop` has 221 missing rows and category hash collisions
in the later public transform. Recorded fertilizer, irrigation and pest-control
values are retrospective management, not certified pre-sowing inputs. The
source metadata sand/silt conflict is handled by excluding those fields.

The original model selection used inner validation without outer arrays;
later post-hoc guard and fixed-public investigations occurred after outer
scores were seen. Structural leakage checks **pass**, but the latter studies
cannot claim untouched confirmation. Outcome comparability, cultivar/density,
future input availability and independent geography remain unresolved.

## 6. Fairness, result and metric verification

| Evidence-backed claim | Configuration / output / command boundary | Finding |
| --- | --- | --- |
| Original eight-method comparison | `results/restart_validation_protocol_v1/config.json`, `results/restart_outer_protocol_v1/{config,plan}.json`, 1,120 inner caches, `results/restart_outer_study_v1/runs/`, `results/restart_outer_verified_v3/verification.json`, `restart_outer_analysis_v1/analysis.json`; commands documented in `research/restart_outer_completion_report.md` | **VERIFIED_CORRECT as archived comparison**, not matched algorithmic superiority. Same outer groups/seeds; neural 32-unit architecture/SGD; per-method reference estimators differ necessarily. 286/672 neural candidates selected the 80-round cap; four outer folds selected the grid-edge LR. |
| Post-hoc zero-support guard | `results/restart_guard_study_v1/`, `restart_guard_analysis_v1/analysis.json`; `restart/guard_sensitivity.py` | **VERIFIED as post-hoc fixed-model sensitivity**, not independent gain. |
| Fixed-public unnoised refit | `restart_control_search_integrity_v1/verification.json`, `restart_public_refit_verified_v2/verification.json`, `restart_public_refit_analysis_v1/analysis.json`; protocol/runner in `restart/public_refit_*.py` | **VERIFIED as later development stage**. It changes transform, equal-client objective, LR and budget; no matched comparison to the original stage. |
| Matched aggregation noise | `restart_matched_noise_verified_v2/verification.json`, `restart_matched_noise_analysis_v1/analysis.json`, `restart/noisy_aggregation.py` | **VERIFIED utility simulation** for its specified Gaussian noise; not an end-to-end privacy claim. |
| Plots | `restart_baseline_figures_v2`, `restart_public_refit_figures_v2`, `restart_matched_noise_figures_v1` manifests and exact plotted-value files | **Verified present hashes for 33/33 listed assets; nine figure tests pass.** Current audit did not perform a new visual reading of every PDF/SVG page or final IEEE placement. |

Current independent read-only recomputation from all 320 primary fold records
matched every archived all-system mean equal-group and pooled RMSE across eight
methods to maximum absolute difference `3.55e-15`. Pooling was correctly
computed by weighting **squared** group RMSE by group size before taking a
square root, not by averaging group RMSE. The index reports actual regression
RMSE/MAE/R²; accuracy and macro-F1 are inapplicable. Measured per-run
training time and network communication bytes are **absent** in the primary
index (0/320 records). `state_bytes()` is a tensor-payload estimator, not a
network measurement. No end-to-end runtime, memory or convergence-to-target
claim should be added from unrecorded values.

Verified original all-system descriptive results (mean ± seed SD, t/ha):

| Method | Equal-group RMSE | Pooled RMSE |
| --- | ---: | ---: |
| XGBoost | 1.2461 ± 0.0000 | 1.7335 ± 0.0000 |
| Random forest | 1.3181 ± 0.0091 | 1.9958 ± 0.0360 |
| Training mean | 1.4509 ± 0.0000 | 1.9379 ± 0.0000 |
| Ridge | 1.4694 ± 0.0000 | 2.1511 ± 0.0000 |
| FedNova, restricted | 5.8134 ± 1.5956 | 29.2405 ± 9.8813 |
| FedProx | 5.8373 ± 1.6226 | 28.2904 ± 9.8478 |
| FedAvg | 5.8373 ± 1.6226 | 28.2909 ± 9.8478 |
| Centralized MLP | 6.0307 ± 1.6250 | 30.4896 ± 10.0605 |

The XGBoost–random-forest equal-group gap is about 0.0720 t/ha; it is not a
location-level significance result. Removing Auzeville changes one pooled
model ranking in the saved post-hoc sensitivity, showing interpretation
depends on group composition. The later fixed-public 1.067407 t/ha value
must remain separate.

## 7. Ablation, robustness and statistics

There is **no valid knowledge-on/off ablation on the active regression data**,
no frozen knowledge-strength test, and no independent new-site support-guard
confirmation. The original data are naturally unequal/non-IID; no synthetic
IID-vs-non-IID or participation-rate factorial was run for the claimed KG
mechanism. The guard and matched-noise panels answer narrower, clearly
labeled questions. No malicious-client, client-dropout-network or prospective
sensor-failure study supports robustness claims.

Five seeds reuse the same eight held-out geographic groups and 37 clustered
experiments. Their sample SD quantifies optimizer/data-order variability,
not a confidence interval for new regions. `restart_outer_analysis_v1` and
`reviewer_sensitivity_v1` contain matched group/seed differences and
leave-one-group summaries; the latter are exploratory after outer exposure.
No confirmatory paired t-test/Wilcoxon, valid geographic CI, or multiple-
comparison correction is established. Treating 607 systems, 40 seed×group
cells, or communication rounds as independent would be pseudoreplication.
Before an inferential claim, freeze an analysis unit (geographic group or
independent experiment), estimand, paired contrast, uncertainty method and
multiplicity policy; ideally collect untouched comparable geography. With
only eight groups, significance alone will not repair weak external validity.

## 8. Privacy, security and deployment

Threat model in `research/restart_privacy_threat_model.md`: a trusted central
curator/server may inspect all individual model updates; an external observer
would see only an approved noisy transcript **if** a complete release mechanism
were established. `restart/noisy_aggregation.py` clips and Gaussian-noises
equal fixed slots in explicit `simulation_only` mode;
`restart/privacy_accounting.py` implements a conditional zCDP calculation.
The matched ledger has 23,174 events and hypothetical noisy-only cumulative
epsilon about 1297–1378 at delta `1e-5`, while nonprivate selection,
controls, evaluation releases and deterministic research noise are not
covered. This is not a useful certified application privacy bound.

| Mechanism | Audit status |
| --- | --- |
| Raw data locality across real institutions | **MISSING**; one process reads the public dataset and constructs logical clients. |
| Transport encryption and authentication | **MISSING**; no network client/server channel was evidenced. |
| Secure aggregation | **MISSING**; server sees individual updates. Software hashes are not signatures. |
| Whole-client formal DP | **PARTIALLY_IMPLEMENTED mathematical/simulation primitives**, but release-certified implementation and full-transcript accounting **MISSING**. |
| Model-update leakage and malicious-client defenses | **MISSING** as evaluated defenses; potential leakage is an established risk. |
| Remote, recoverable multi-institution deployment | **MISSING**; local runner recovery is not remote orchestration. |

The distinction matters: [Bonawitz et al.](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/)
provide a cryptographic aggregate-only protocol, whereas this code explicitly
handles raw individual states. [Zhu et al.](https://proceedings.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)
demonstrate that shared training gradients can reveal data in other settings;
that paper is a risk example, **not** evidence of an attack on this dataset.
There are no server certificates, credentials, transport logs, dropout
reconciliation or multi-machine round receipts. No ports or infrastructure
were touched in this audit.

## 9. Manuscript claim-to-evidence matrix

| Manuscript or proposed claim | Evidence location | Status | Required change |
| --- | --- | --- | --- |
| Conditional harvest regression on recorded cereal–legume systems | Frozen data manifest, DC01 audit, current Markdown title/abstract | **SUPPORTED** | Retain retrospective/conditional wording and units. |
| Geographically held-out simulated-client comparison | 64-fold verifier, 320-run outer audit | **SUPPORTED WITH LIMITATIONS** | Say simulated groups, uneven sizes and experimental clustering. |
| Original XGBoost lowest observed RMSE among eight methods | Original cached analysis; fresh 320-record recomputation | **SUPPORTED descriptively** | No universal or inferential superiority. |
| Later fixed-public lower RMSE | 40-run verified analysis | **SUPPORTED as development result** | Do not attribute its difference from original solely to federation/knowledge. |
| Noise increases error in matched experiment | 160-run matched analysis | **SUPPORTED for specified simulation** | No production DP or privacy-benefit inference. |
| New KG-FCR method / knowledge effect | Prototype only; real-data rule gate blocked | **UNSUPPORTED** | Remove from title/contribution unless separately validated. |
| Best-crop recommendation or causal yield benefit | No choice labels/counterfactual trial | **CONTRADICTED by current target** | Do not state. |
| Secure aggregation, private regional network, end-to-end DP | Raw-update simulator and incomplete transcript accounting | **UNSUPPORTED** | Remove “privacy-preserving” title claim or implement, audit and certify separate system. |
| Broad unseen-region generalization | Same eight exposed geographic groups; tiny group counts | **NOT_YET_TESTED** | Untouched comparable sites/seasons and prospective protocol. |
| Portable public reproducibility | 21/21 local assets; no lock/clean-room/license | **PARTIALLY_SUPPORTED locally** | Fresh Windows/Linux reproduction, path relocation and redistribution review. |

The assembled Markdown manuscript is more conservative than the supplied
KG-FCR title. It still needs IEEE conversion, final citation placement,
bibliography verification, the `\mathbf` fix, and review of every rendered
figure/table against its exact source. An existing 14-entry BibTeX registry
does not mean all citations have been placed and rendered correctly. In
particular, `manuscript/references.bib` contains the exact Gaudio dataset
entry, but the assembled manuscript's 13-item reference list omits it and the
Data section currently describes the source without a citation. This is a
confirmed citation omission. `paper2/README.md` also mixes old classifier and
restart-era statements (including a claim that outer analysis is pending)
with the completed current study; it needs a scope/status cleanup before
public release. Neither file was edited during this audit.

## 10. Research-readiness ratings

| Dimension | Rating | Reason |
| --- | --- | --- |
| Technical correctness of *restricted simulated baselines* | **PASS_WITH_LIMITATIONS** | Math/code/tests/archived receipts align; no full FedNova/general network failure proof. |
| KG-FCR algorithmic novelty | **INSUFFICIENT_EVIDENCE** | Real-data mechanism and matched ablation absent; relevant prior work exists. |
| Experimental rigor for bounded retrospective regression | **PASS_WITH_LIMITATIONS** | Strong frozen/grouped audit; severe neural budget/support failures, outer exposure and no independent confirmation. |
| “Privacy-preserving”/secure deployment claim | **FAIL** | Individual updates visible; no SecAgg, TLS/auth or end-to-end DP. |
| Manuscript and portable reproducibility readiness | **PASS_WITH_LIMITATIONS locally; INSUFFICIENT_EVIDENCE for portable submission package** | Current draft is cautious, but IEEE/source/render review and clean-room/license checks remain. |

**Final decision:** The proposed knowledge-guided, privacy-preserving crop-
recommendation paper is **not scientifically supported by this implementation**.
The narrower grouped-regression manuscript is a defensible draft, not yet a
submission-ready top-tier paper. This is an evidence judgment, not a numerical
completion score.

## 11. Prioritized corrective plan and close-out evidence

| Priority | Issue / action after user review | Evidence required to close |
| --- | --- | --- |
| Critical 1 | Choose the actual paper identity: bounded regression study now, or a separately built KG recommendation method. Do not bridge them by renaming. | Title, RQs, target and contributions all map to executed code/results. For KG: legitimate crop-choice outcome, independent source-approved rules, source-matched math and on/off ablations. |
| Critical 2 | If claiming federation-specific or neural superiority, freeze matched central/FL factorials using the same representation/objective/architecture/selection budget; diagnose FedAvg–FedProx update differences and cap-hitting convergence. | Prospective protocol, synthetic acceptance, bounded new runs, complete receipts, matched grouped metrics and adverse findings; exposed outer groups labeled development. See `REVIEW_RESPONSE_IMPLEMENTATION_PLAN.md` RR04–RR07. |
| Critical 3 | Obtain untouched comparable geography/season for external confirmation and establish when management inputs are available. | Source/units/license mapping, independent holdout registered before outcomes, experiment-cluster-aware analysis and supported-feature panels. |
| Critical 4 | Remove privacy-preserving/system-security language unless a separate established mechanism is implemented. | Explicit adversary/trust/release model; vetted SecAgg with dropout/security tests if server privacy claimed; private statistics/selection/output accounting, secure RNG and externally checked end-to-end budget if DP claimed. |
| Important 5 | Add resource and statistical evidence only if those claims are desired. | Measured per-run training time, payload/network bytes and memory under a stated protocol; group/experiment-level paired contrasts with declared uncertainty and multiplicity. Never fill blank metrics from estimates. |
| Important 6 | Prepare a portable manuscript/evidence release. | Correct equation and literature year; cite the exact dataset in the manuscript; reconcile the stale README; verify all citations and IEEE-rendered figures; dependency lock, fresh environment/Linux audit, path-relocation strategy and redistribution clearance. |

No correction or new training was started. The user-requested order is to
review this audit first, then approve and address issues one by one.

## 12. Exact commands and observed outcomes

All commands were run from the repository root. The audit did not install
dependencies, write archived assets, perform model inference or launch a
long-running study.

```powershell
git log -5 --oneline
git status --short -- paper2
# one initial commit visible; paper2/ is untracked

$env:PYTHONPATH='paper2/restart;paper2'
$env:TEMP=(Resolve-Path 'paper2/results').Path
$env:TMP=$env:TEMP
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_federated paper2.tests.test_knowledge paper2.restart.test_data_pipeline paper2.restart.test_outer_summary paper2.restart.test_privacy_accounting paper2.restart.test_outer_audit paper2.restart.test_matched_noise_summary -q
# Ran 69 tests; OK
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_baseline_publication_figures paper2.restart.test_publication_figures -q
# Ran 9 tests; OK
& '.\.venv\Scripts\python.exe' -c "from paper2.restart.reproducibility_readiness import audit; r=audit(); print(r['status'],r['verified_asset_count'],r['required_asset_count'],'errors',len(r['errors']))"
# PASS_LOCAL_EVIDENCE_BLOCKED_PORTABLE_REPRODUCTION 21 21 errors 0
```

The fresh numerical comparison read `results/master_experiments_v1.csv` and
`results/restart_outer_analysis_v1/analysis.json`, grouped primary records by
method and seed, computed mean group RMSE and
`sqrt(sum(n_g*RMSE_g²)/sum(n_g))`, then compared them with all eight archived
summary records: 320 runs, eight methods/groups, five seeds, maximum difference
`3.552713678800501e-15`. The master CSV hash matched its companion audit.
All 33 assets listed by the three accepted figure manifests matched their
recorded SHA-256 hashes. Current `.venv` is Python 3.12.3 with torch 2.13.0+cpu,
numpy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, joblib 1.5.3 and XGBoost 3.3.0,
matching the observed runtime record. No Linux/fresh-environment test ran.

One inspection command initially requested nonexistent
`paper2/restart/knowledge_training.py`; the actual unchanged file is
`paper2/knowledge_training.py`. This path correction had no code/data impact.
