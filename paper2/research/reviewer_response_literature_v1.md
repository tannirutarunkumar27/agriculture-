# Reviewer-response literature and novelty gate (2026-10-15)

This is a positioning review, not a meta-analysis or a numerical cross-paper
benchmark. Items 1--8 were previously checked against primary records in
`restart_regression_closest_work.json`; items 9--20 were checked against the
linked publisher/proceedings records on 2026-10-15. The 20 include methodological
and validation papers, not 20 directly comparable agricultural datasets.

| # | Primary study | Relevance to this study | Boundary |
|---|---|---|---|
| 1 | [DELCON crop-yield federation (2022)](https://doi.org/10.1109/DELCON54057.2022.9752836) | Federated soybean-yield regression already exists | Different data and protocol |
| 2 | [Li et al., FedPruning (2024)](https://doi.org/10.1016/j.eswa.2023.122847) | Localized horizontal yield forecasting | Different objective and clients |
| 3 | [Mukherjee and Buyya (2025)](https://doi.org/10.1002/spe.3416) | Federated architecture comparison in yield prediction | Different sequential data and metrics |
| 4 | [Dey et al., FLyer (2025)](https://doi.org/10.1109/TAI.2025.3534149) | Edge/cloud crop-yield federation | Different systems setting |
| 5 | [Cowlishaw et al. (2025)](https://doi.org/10.1038/s41598-025-94244-2) | Agricultural Earth-observation federation and imbalance | Crop segmentation proxy, not yield |
| 6 | [Nirosha and Vennila (2025)](https://doi.org/10.1016/j.eswa.2025.128312) | Hybrid graph/recurrent federated yield modeling | Different target and representation |
| 7 | [Utku et al. (2026)](https://doi.org/10.1007/s10791-026-10291-x) | Country-client cereal-yield forecasting | Country-year series, not field systems |
| 8 | [Koksal et al. (2026)](https://doi.org/10.1016/j.compag.2026.111851) | Crop-yield prediction across institutions | Vertical, not horizontal federation |
| 9 | [McMahan et al., FedAvg (2017)](https://proceedings.mlr.press/v54/mcmahan17a.html) | Original federated averaging baseline | Established algorithm |
| 10 | [Li et al., FedProx (2020)](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) | Heterogeneous-client proximal baseline | Established algorithm |
| 11 | [Wang et al., FedNova (2020)](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Normalized averaging under heterogeneous local steps | Established algorithm; our implementation is restricted |
| 12 | [Gorishniy et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | Strong tabular neural baselines and matched tuning | No universal tree/neural winner |
| 13 | [Grinsztajn et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) | Tree/neural benchmark and tuning rigor | Different tasks and sample sizes |
| 14 | [McElfresh et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract.html) | Dataset-specific neural/tree comparisons | Cannot infer this task's optimum |
| 15 | [Karimireddy et al., SCAFFOLD (2020)](https://proceedings.mlr.press/v119/karimireddy20a.html) | Client drift under non-IID federation | Shows drift mitigation is prior art |
| 16 | [Li et al., FedBN (2021)](https://openreview.net/pdf?id=6YEQUn0QICG) | Feature-shift non-IID and local normalization | Primarily different modalities/settings |
| 17 | [Acar et al., FedDyn (2021)](https://openreview.net/forum?id=B7v4QMR6Z9w) | Dynamic regularization for heterogeneous clients | Established optimizer, not our contribution |
| 18 | [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) | Spatial/hierarchical validation guidance | Review, not our empirical comparator |
| 19 | [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107) | Spatial/environmental blocking tools | Species-distribution task, not harvest regression |
| 20 | [Ploton et al. (2020)](https://doi.org/10.1038/s41467-020-18321-y) | Spatial validation can expose poor transfer | Ecological mapping, not this dataset |

## Novelty conclusion and testable questions

The current work does **not** establish a new FL optimizer, federated yield
prediction, non-IID treatment, privacy mechanism, or crop recommendation.
The possible contribution is a tightly controlled empirical explanation of
which changes (representation, weighting, optimizer, budget, and feature
support) account for neural/federated behavior on small heterogeneous
agricultural groups. This remains a *hypothesis* until matched ablations and
independent validation support it. A negative result is acceptable; improved
RMSE is not an acceptance criterion and cannot be promised.

Primary questions: (Q1) Under identical inputs, model architecture, search
budget, and held-out geography, what extra error is attributable to federation
versus centralized training? (Q2) Which individually varied design choices
explain the later fixed-public result? (Q3) How stable are paired conclusions
across the eight existing groups, especially when tiny groups or Auzeville are
removed? (Q4) Does a prospectively fixed support policy transfer to *new*
geographies? Q4 cannot be answered by reusing these outer tests.

No scores from these external studies should be placed beside Paper 2 scores
as a superiority claim: target units, populations, clients, splits, metrics,
and budgets are not matched.
