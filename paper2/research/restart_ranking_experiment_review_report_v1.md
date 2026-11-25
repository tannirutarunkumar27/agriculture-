# Restricted observed-set ranking review report v1

## Outcome

The cultivar, density, mixture-design, and mixing-pattern fields are present,
but the supplied data still do not support an automatic best-crop label. A
reproducible review packet now binds every mechanically matched block to the
complete observed management packages, harvest outcomes, harvest-date
signatures, and publication-reference registry. No ranking or model training
was performed.

## Source and reference audit

The reviewed version-2 source contains 37 experiment records. The supplied
reference workbook contains one matching row per experiment. Twenty-six
experiments have at least one DOI URL and 11 have a blank reference cell. The
workbook provides 15 unique DOI URLs. Presence of a DOI is a linkage check; it
does not establish randomization or treatment comparability.

The 47 mechanically matched blocks span 29 experiments and 450 observed
management packages. Thirty-two blocks have a publication reference. Fifteen
blocks from nine candidate experiments do not. Those blocks cannot be included
until an independently reviewed source locator is supplied.

## Design findings

The earlier 47-block table groups on experiment, season, sowing date, nitrogen,
phosphorus, irrigation, and pest control. It does not prove that crop systems
are randomized counterfactual alternatives.

- Thirty-nine blocks contain multiple management packages for at least one
  system-species label. Therefore, a species label is not a unique treatment.
- Seventeen blocks contain more than one component-harvest-date signature.
  Harvest comparability must be reviewed rather than assumed.
- Cultivar and density are retained in each candidate package. They are not
  averaged into a crop-level label.
- The only frozen utility is summed harvest seed dry matter in t/ha. It is not
  profit, nutrition, land-equivalent ratio, causal benefit, or prospective
  field utility.

## Fail-closed implementation

`review_ranking_experiments.py` creates experiment, block, DOI, and package
inventories plus a 47-row approval template. `ranking_approval_gate.py` requires
every block to receive an include/exclude decision. An included block must have
a publication reference, source locator, reviewer/date/rationale, and five
affirmative design approvals. Missing, partial, duplicate, or invalid reviews
leave ranking disabled.

The current gate result is
`RANKING_APPROVAL_BLOCKED_PENDING_REVIEW`: all 47 blocks are pending, zero are
included, and zero are excluded. This is the correct research state until an
independent agronomic/experimental-design reviewer completes the packet.

## Next authorized action

An independent reviewer should complete
`paper2/results/restart_ranking_experiment_review_v1/approval_template.csv`
using the original publications and experimental-design records. Run the gate
against a preserved reviewed copy in a new results directory. Do not construct
ranking labels or evaluate regret unless the gate reports
`RANKING_APPROVED_FOR_RESTRICTED_OBSERVED_SET`.
