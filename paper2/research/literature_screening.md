# Literature screening — 2026-10-12

> **Active-task update (2026-10-14):** This classification/recommendation
> screening is retained as historical project evidence. The active Paper 2 task
> is conditional cereal-legume harvest regression. Use
> `restart_regression_closest_work.md` and its validated JSON registry for the
> manuscript's closest-work comparison; do not use the classification scores
> below as regression comparators.

This is an initial screening, not a completed systematic review. Publisher and
author abstracts were checked; full-text details not inspected are explicitly
unverified. External numbers are not project results. No cross-paper numerical
superiority is claimed.

Update: the focused comparison, full-text checks and algorithm variant decisions
are now saved in `closest_paper_comparison.md`; agronomic compatibility and
approval gates are in `agronomic_source_checks.md`. Those records supersede
the pending details below where additional evidence has since been checked.

| Work / source | Verified relevance | Pending extraction / limitations |
|---|---|---|
| McMahan et al., 2017, AISTATS: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html) | Iterative model averaging; non-IID and unbalanced distributed learning; FedAvg baseline | Not an agricultural knowledge method; client counts and exact evaluation configurations require full-text extraction |
| Li et al., [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), preprint first posted 2018 | FedProx addresses statistical and systems heterogeneity | Verify final venue/version and local objective directly before coding; do not conflate preprint date with publication date |
| Wang et al., 2020, NeurIPS: [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | FedNova normalizes local updates to address objective inconsistency | Exact normalizer depends on solver; use a narrowly documented plain-SGD variant first |
| [Federated Learning: Crop classification in a smart farm decentralised network](https://www.sciencedirect.com/science/article/pii/S2772375523001065) | Publisher-indexed abstract describes FedAvg crop classification using climate inputs | Full text, dataset, clients, metrics, privacy mechanism, year/DOI and venue details still require extraction; publisher page retrieval failed |
| Cheikhrouhou, Turki, Dammak, 2026, Information and Software Technology 196, 108173: [X-FedAgriTwin](https://doi.org/10.1016/j.infsof.2026.108173) | Publisher abstract describes federated MLP crop recommendation, digital twins and SHAP | Split/client comparability and formal privacy mechanisms not established from abstract; do not reuse its external accuracy as our result |
| Bonawitz et al., 2016 workshop: [Practical Secure Aggregation for Federated Learning on User-Held Data](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/) | Secure aggregation hides individual inputs subject to protocol assumptions | This checked source is the 2016 workshop precursor, not a verified citation of the distinct CCS 2017 version; no protocol implemented here |
| Abadi et al., 2016 preprint: [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) | DP neural training requires a defined mechanism and accounting | Verify final publisher record and account for actual sampling/composition; no DP budget available here |
| Lim et al., 2025 preprint: [Personalized Federated Recommendation With Knowledge Guidance](https://arxiv.org/abs/2511.12959) | Knowledge-guided federation already exists in recommendation research | Its global/local embedding knowledge differs from agronomic rules; full-text comparison required before distinctness claim |

## Claim-to-evidence gates

| Candidate claim | Current evidence | Decision |
|---|---|---|
| First federated crop recommender | Existing agricultural papers above | Reject |
| First knowledge-guided federated recommender | FedRKG exists; broader review incomplete | Reject broad claim |
| Class-aligned agronomic guidance improves heterogeneous crop classification | No controlled Paper 2 experiment | Research hypothesis only |
| Regional farm validation | Manipur synthetic; Maharashtra provenance unverified | Not established |
| Formal privacy | No SecAgg or DP implementation/accountant | Not established |

## Research questions

1. How much does sample-weighted federation change macro-F1 and per-client
   performance relative to matched centralized neural training?
2. Does explicitly class-aligned, provenance-checked soft agronomic guidance
   help under class skew, unequal local work, and missing inputs?
3. Does any gain survive no-knowledge and corrupted-knowledge controls,
   repeated seeds, matched capacity and budget, and source-unit checks?
4. What utility and measured overhead change under separately verified
   secure aggregation and, if feasible, accounted differential privacy?

Candidate contribution: a controlled evaluation of class-aligned agronomic
guidance under documented agricultural heterogeneity, knowledge uncertainty,
and separately measured privacy mechanisms. This is not yet proven novel.

Next review: full texts and code for closest agricultural comparisons,
knowledge-aware federated optimization, update poisoning/integrity,
communication efficiency, reproducible FL evaluation, and independent
agronomic source records. Complete the dataset/client/privacy/metric matrix
before declaring Stage B complete or finalizing the KG-FCR method.
