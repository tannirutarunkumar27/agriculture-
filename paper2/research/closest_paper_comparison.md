# Closest-paper comparison and implementation decisions

> **Historical scope notice (2026-10-14):** This file records the earlier crop
> classification/KG-FCR direction. It is not the active Paper 2 related-work
> matrix. For conditional cereal-legume harvest regression, use
> `restart_regression_closest_work.md` and
> `restart_regression_closest_work.json`.

Checked 2026-10-12. This focused comparison is not proof of exhaustive novelty.
"Not verified" means unknown, not absent. External metrics are not our results.
Publisher access failures are preserved rather than filled with assumptions.

| Study / source | Problem, model and data | Client/evaluation setup | Privacy and metrics | Consequence for this project |
|---|---|---|---|---|
| Idoje, Dagiuklas, Iqbal (2023), Smart Agricultural Technology 5, 100277. [Publisher DOI](https://doi.org/10.1016/j.atech.2023.100277), [open-access full text](https://researchportal.lsbu.ac.uk/files/10539985/1-s2.0-S2772375523001065-main.pdf), Sections 3–4 | FedAvg crop classification; temperature, humidity, pH, rainfall; balanced rice/maize/chickpea subset | 80/20 train/test; common schema; Linux PySyft/Syft duet and GNS3 emulation. Exact node count and original sample count not established from inspected experimental description | Local records retained; no verified DP budget or secure-aggregation protocol in inspected methods. Table 3 externally reports accuracy 0.90 and macro-F1 0.90 for Adam LR .001. Prose/table inconsistencies mean no numerical comparison here | FedAvg crop classification already exists. Our seven-feature, 22-class experiment is not a same-protocol reproduction; control model/splits and report client heterogeneity explicitly |
| Cheikhrouhou, Turki, Dammak (2026), Information and Software Technology 196, 108173. [DOI](https://doi.org/10.1016/j.infsof.2026.108173) | X-FedAgriTwin: federated MLP, digital twins and SHAP for crop recommendation | Publisher abstract reports stable convergence within 30 communication rounds. Dataset size, exact split, client count and heterogeneity not verified; full publisher page retrieval failed | Externally reported accuracy 99.39%; formal DP/SecAgg details not established by accessible abstract | FL + MLP + SHAP is not a novelty claim. No superiority comparison to its score; obtain full text/code before final related-work assertions |
| Lim et al. (2025), preprint. [FedRKG full text](https://arxiv.org/html/2511.12959v1), Sections 3–4 | Personalized user–item recommendation using preserved local embeddings and guided global information; not physical agronomic suitability | Amazon-Video, FilmTrust, LastFM-2K, ML-1M; timestamp-based interaction splitting and implicit feedback | Recommendation setting/metrics differ from multiclass crop classification; no cross-task score comparison | Distinguish agronomic, unit-bearing crop rules from learned global/local embedding "knowledge". Broad first knowledge-guided federated recommendation claim is invalid |

## Algorithm sources and deliberately restricted variants

- [FedAvg, AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html):
  selected-client sample-weighted parameter averaging after local SGD.
- [FedProx, MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html),
  [full formulation](https://arxiv.org/html/1812.06127), Eq. 2/Algorithm 2:
  proximal penalty mu/2 ||theta-theta_t||². Original algorithm samples according
  to client weights and uses equal averaging; our explicitly recorded variant
  samples uniformly without replacement and weights selected models by size.
  It implements the defining proximal objective but is not a verbatim reproduction
  of that sampling scheme or an inherited convergence theorem.
- [FedNova, NeurIPS 2020 full text](https://arxiv.org/html/2007.07481),
  Sections 4–5: correct normalized local updates. Implemented variant uses
  constant shared LR, plain SGD, no momentum/weight decay/proximal term,
  a_k=ones(tau_k), effective tau=sum p_k tau_k. Other solvers are unsupported.

## Candidate gap and claim boundaries

Research hypothesis: class-aligned, externally traceable soft agronomic rules
may change macro-F1, rare-class behavior and client disparities under controlled
crop-distribution skew and unequal work. Existing work above does not by itself
establish that this exact design is novel; broader review remains necessary.

Implement established baselines first. Do not alter optimizer labels or claims
to make an ordinary MLP appear a new algorithm. Agricultural rules remain
disabled until the source checks in `agronomic_source_checks.md` pass.

## Outstanding evidence

X-FedAgriTwin full-text/client protocol and any formal privacy mechanism; exact
sample/client counts for the 2023 study; closest agriculture-specific
knowledge-aware federated optimization; source independence of crop labels;
attack defenses and privacy mechanisms. These are documented open items, not
reasons to fabricate a final novelty statement.
