# Paper 2 independent-review packet v1

Status: `REVIEW_REQUIRED_CLAIMS_DISABLED`

This packet is for an independent agronomist/privacy/research-methods reviewer.
The implementation team must not self-approve these records. Blank fields are
intentional and must be completed by the reviewer.

## Review order

### A. Agronomic rules

Use the eight-row template:

[domain_review_template.csv](../results/restart_agronomic_domain_review_gate_v2/domain_review_template.csv)

For every rule, verify:

1. source record and locator;
2. crop taxonomy and synonym mapping;
3. `soil_pH` unit compatibility with the observed feature;
4. geographic validity;
5. cultivar and growth-stage scope;
6. water-regime limitations;
7. whether the rule is a soft prior rather than a hard label;
8. independent decision, reviewer identity and affiliation, source-check date,
   conflict-of-interest disclosure, ISO review date and rationale.

All eight rules currently remain pending. No rule may enable guidance until the
domain gate and the separate audit both pass.

### B. Missing publication references

Use the 11-row template:

[independent_review_template.csv](../results/restart_ranking_reference_gate_v2/independent_review_template.csv)

An `include` decision requires an exact site/year/design match, complete
treatment coverage, an original experimental source, an evidence locator and a
written rationale. Related reviews, aggregate analyses, conference abstracts,
partial cultivar coverage and mismatched seasons must remain excluded.

The current candidate audit has one exact-but-unapproved candidate and ten
unresolved records. The frozen source workbook must not be edited during review.

### C. Candidate ranking blocks

Use the 47-block template:

[approval_template.csv](../results/restart_ranking_experiment_review_v2/approval_template.csv)

For every block, independently verify:

- original randomization;
- treatment comparability;
- harvest-date and outcome comparability;
- replicate and tie policy;
- restricted utility definition;
- source locator, reviewer, date and rationale.

The ranking gate must remain disabled if any block is pending or invalid.

## Sign-off rule

The reviewer must return the completed templates without changing block IDs,
experiment IDs, rule IDs, or candidate source URLs. The gate scripts validate
identity and fail closed on missing, altered or contradictory records.

## Current evidence

- Agronomic gate: `AGRONOMIC_DOMAIN_REVIEW_PENDING`.
- Reference gate: `RANKING_REFERENCE_SUPPLEMENT_BLOCKED_UNRESOLVED`.
- Ranking gate: `RANKING_APPROVAL_BLOCKED_PENDING_REVIEW`.
- No ranking claim, agronomic-guidance claim or final release claim is allowed.
