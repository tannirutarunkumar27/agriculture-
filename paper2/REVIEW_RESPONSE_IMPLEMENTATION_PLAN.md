# Paper 2 reviewer-response implementation plan (frozen 2026-10-15)

Scope: retrospective conditional harvest seed dry-matter regression on the
existing 607 systems / eight groups. This plan was written after exposure to
the primary outer-test results. All follow-up analyses on these groups are
development evidence, not an untouched confirmation set. Original archives
remain immutable. A lower error is not required for scientific success.

| ID | Concern / deliverable | Acceptance gate | Status |
|---|---|---|---|
| RR01 | Twenty-source novelty audit | Primary-source matrix; no invented algorithm/yield/recommendation claim | VERIFIED_COMPLETE: `research/reviewer_response_literature_v1.md` |
| RR02 | Geographic sensitivity and paired comparisons | Recompute from archived per-group metrics without reinference; exact identities, source hash, synthetic tests, all-group/leave-one/size-threshold results | VERIFIED_COMPLETE: `results/reviewer_sensitivity_v1/analysis.json`; post-hoc only |
| RR03 | Data dependence, feature availability, category collisions | Source-level audit of 37 experiment identifiers, prediction-time inputs, public-hash collisions, cultivar/density missingness; no inferred independence | IN_PROGRESS: aggregate source audit verified; prospective input availability/harmonization unresolved |
| RR04 | Primary convergence and FedProx/FedAvg diagnostic | Inspect frozen traces and algorithm code; quantify selected budget edges and actual update differences; no outer tuning | NOT_STARTED |
| RR05 | Matched ablation protocol | Freeze factor definitions and equal inner-validation budgets *before* new training; separate encoding, weighting, optimizer, duration, architecture; paired same folds/seeds and controlled compute accounting | NOT_STARTED |
| RR06 | Stronger neural inner-only search | Implement isolated tested runner; train-only preprocessing, method-specific selection, AdamW/schedule/width/depth comparisons, convergence criteria; run bounded smoke then verified inner batches | NOT_STARTED |
| RR07 | Follow-up outer evaluation | Only after RR05/06 frozen and audited; label as test-exposed development study, no independent confirmation | BLOCKED on RR05/06 |
| RR08 | Support guard independent test | Prospective policy and genuinely untouched comparable geography; report feature-specific support | BLOCKED: no qualifying untouched dataset identified |
| RR09 | Privacy and operations | Threat-model matrix; communication-byte estimates from actual payload; participation simulation only if tested; no network/DP/SecAgg claims without separate evidence | NOT_STARTED |
| RR10 | Portable reproduction and citations | Complete reference keys/dataset DOI, bounded clean-environment instructions, license check, independent fresh-env execution | IN_PROGRESS elsewhere; not yet portable |

Sequence: complete RR02 and RR03 first; freeze RR05 from *inner-training
evidence only*; implement/test RR06; execute bounded inner jobs; inspect
convergence and selection; only then conduct a versioned outer follow-up.
Compare central and federated models with the same representation, architecture,
optimizer family, local-step budget and selection resources. An equal-client
objective and a sample-weighted objective must be identified as different
estimands, not merely as optimizer settings. Factor-at-a-time comparisons
should be supplemented by specified interactions, not read as causal effects
when the factor changes the objective. Report time, updates, trainable
parameters, and estimated communication alongside error.

For RR02, seeds are paired computational replicates, *not* independent sites.
Site-deletion summaries are sensitivity checks, not geographic confidence
intervals. With only eight groups and three containing <=6 observations,
formal generalization uncertainty will remain weak. No split of these same
systems can remove previous test exposure or disentangle Auzeville from all
unseen species. New independent sites require valid feature/target mapping,
outcome units, license, and prospective registration before use.

For RR09, noise experiments are utility simulations with conditional
accounting and no secure aggregation. Do not relabel epsilon ~1300 as
practically private. Production deployment needs separate security, keys,
network, adversary and release analysis; it is not authorized by this plan.

Persistent logging: each task records exact input/version, output directory,
test command/result, deviations and failure. No historical results may be
overwritten or rerun merely to obtain favorable scores.
