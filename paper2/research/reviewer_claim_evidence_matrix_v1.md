# Claim-to-evidence gate for the proposed KG-FCR narrative (2026-10-15)

The user-provided KG-FCR checklist is a useful *proposal*, but its statement
that implementation is complete does not match the verified Paper 2 project
state. The active dataset supports retrospective **continuous harvest seed
dry-matter regression** for observed cereal--legume systems, not supervised
"best crop" classification or ranking. Classification accuracy/macro-F1 are
therefore not valid primary metrics for this study.

| Proposed claim or experiment | Verified state | Paper wording now |
|---|---|---|
| New knowledge-guided federated crop recommendation algorithm | No valid independent agronomic rule set or comparable crop-choice labels; R06/R07 blocked | **Do not claim** a KG-FCR method, crop ranking, or recommendation benefit |
| FedAvg/FedProx/FedNova comparison | Implemented, tested and completed on frozen simulated geographic partitions | Describe exactly as restricted regression baselines; FedNova implementation scope must be explicit |
| Centralized reference | Ridge, trees, training mean and MLP in the primary 320-run study | Valid comparison, but neural/FL methods need stronger matched tuning before a federation-specific conclusion |
| Non-IID real regional federation | Data are site-partitioned and heterogeneous, but clients are simulations of geographic groups, not operating farms | Describe simulated horizontal federation only |
| Knowledge ablation or guidance-strength sensitivity | No eligible knowledge component to ablate | **Not applicable** until a separately justified method and units exist |
| Secure aggregation | No verified protocol | **Do not claim** secure aggregation |
| Formal differential privacy | Conditional aggregation-noise simulations and large conditional epsilon; pipeline releases not covered | Report noise--utility simulation, not end-to-end DP |
| Communication/latency in a deployed network | No real client/server deployment or network timing | At most measure model-payload byte estimates separately; not real-world latency |
| Client-level metric variability | Eight held-out geographic groups, five repeated seeds; very small groups and dependence | Report group-wise regression RMSE/MAE, paired sensitivity, no geographic confidence inferred from seeds |
| Independent generalization | Existing outer groups have been exposed, and guard is post-hoc | Requires untouched comparable new geography; not established |
| Public reproducibility | Local evidence inventory passes; portable fresh environment and redistribution clearance pending | State local audit only, not independent reproduction |

Working research questions compatible with current evidence:

1. Under matched representation and tuning, how much performance difference
   remains between centralized and simulated federated neural regression?
2. Which individually controlled design changes explain the later
   fixed-public result, and how do they interact with geographic support?
3. How stable are descriptive method comparisons under group weighting,
   site deletion and experiment-level dependence?

Those questions are not yet answered by the current primary-vs-later numerical
contrast because several factors changed together. If the intended paper is
instead genuinely about crop recommendation and knowledge-guided federation,
it requires a *different validated target, agronomic source mapping, algorithm
specification, experiments and privacy/security implementation*. The current
607-system harvest dataset cannot be relabeled into counterfactual best-crop
recommendations. Do not repurpose its regression scores as classification or
recommendation evidence.
