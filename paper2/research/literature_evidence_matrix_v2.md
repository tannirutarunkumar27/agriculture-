# Literature evidence matrix for the Paper 2 claim audit (2026-10-15)

This extends `reviewer_response_literature_v1.md` across the five requested
areas. It is a *screening and positioning* matrix from primary publisher,
proceedings or author records, not a systematic review or score comparison.
`A` means a limitation or scope explicitly reported in the source record;
`I` means our inference from different task, data or protocol. Where a detailed
metric, client distribution or security mechanism has not been verified from
the full article, it says **not verified**. That is preferable to invention.

Columns abbreviate: task/data; algorithm, knowledge, clients and privacy;
reported evaluation/finding; limitation and relevance. No external numerical
score is comparable with our t/ha held-out-site RMSE without matched units,
population, splits, metrics, clients and budget.

## A. Federated optimization

| Source | Task/data; algorithm | Knowledge; clients; privacy | Evaluation/finding | Limitation and relevance |
|---|---|---|---|---|
| [McMahan et al. 2017](https://proceedings.mlr.press/v54/mcmahan17a.html) | Decentralized benchmark learning; FedAvg | No agronomic knowledge; device-partitioned data; data locality, not formal DP | Communication rounds and model performance; iterative averaging baseline | I: not agriculture; FedAvg novelty precluded |
| [Li et al. 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) | Heterogeneous benchmark clients; FedProx | No agronomic knowledge; statistical/system heterogeneity; no inherent secure aggregation | Stability/performance under heterogeneity | I: different data; proximal method established |
| [Wang et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Variable local computation; FedNova | No agronomic knowledge; heterogeneous local steps; no inherent DP | Objective-consistency analysis and benchmarks | I: our constant-SGD variant is narrower |
| [Karimireddy et al. 2020](https://proceedings.mlr.press/v119/karimireddy20a.html) | Client drift on heterogeneous data; SCAFFOLD | No agronomic knowledge; sampled heterogeneous clients; no inherent DP | Convergence and communication comparisons | I: drift mitigation is prior art, not our novelty |
| [Li et al. 2021, FedBN](https://openreview.net/pdf?id=6YEQUn0QICG) | Feature-shift non-IID benchmarks; local normalization | No agronomic knowledge; feature-shift clients; no inherent DP | Generalization/convergence vs FedAvg/FedProx | I: site feature shift concept relevant; modality differs |
| [Acar et al. 2021, FedDyn](https://openreview.net/forum?id=B7v4QMR6Z9w) | Heterogeneous FL; dynamic regularization | No agronomic knowledge; partial/unbalanced clients; no inherent DP | Optimization/convergence studies | I: another established heterogeneity remedy; detailed dataset metrics not re-audited |

## B. Agricultural federation (yield, crop recommendation and proxy tasks)

| Source | Task/data; algorithm | Knowledge; clients; privacy | Evaluation/finding | Limitation and relevance |
|---|---|---|---|---|
| [DELCON 2022](https://doi.org/10.1109/DELCON54057.2022.9752836) | Soybean yield; federated residual models/FedAvg | Explicit agronomic rule mechanism not verified; client details and formal privacy not verified | Yield-prediction evaluation; exact metric protocol not re-audited | I: FL yield regression already exists; not matched to our systems |
| [Li et al. 2024](https://doi.org/10.1016/j.eswa.2023.122847) | Soybean yield forecasting; localized FedPruning | Pruning/personalization, not verified rule guidance; horizontal heterogeneous clients; formal DP not established here | Yield and communication/model-size outcomes | I: closest horizontal yield context but different objectives |
| [Mukherjee and Buyya 2025](https://doi.org/10.1002/spe.3416) | Crop-yield LSTM; central, ring and mesh FL | Explicit agronomic guidance not verified; topology clients; security not verified | Prediction and response-time architecture evaluation | I: network architecture and temporal target differ |
| [Dey et al. 2025](https://doi.org/10.1109/TAI.2025.3534149) | Crop-yield LSTM; edge/cloud FL | Rule guidance not verified; edge nodes; encrypted transmission described but formal DP not inherited | Yield evaluation; detailed units not re-audited | I: systems comparator, not a matched baseline |
| [Cowlishaw et al. 2025](https://doi.org/10.1038/s41598-025-94244-2) | Earth-observation crop segmentation proxy; FL | No explicit agronomic-rule mechanism verified; imbalanced clients; no Paper 2 privacy implication | Segmentation/client-imbalance analyses | A: yield data limitation motivates proxy; I: not yield regression |
| [Nirosha and Vennila 2025](https://doi.org/10.1016/j.eswa.2025.128312) | Crop-yield graph/recurrent federation | Learned graph/recurrent representation, not verified agronomic-rule mechanism; client/privacy details not re-audited | Study-specific yield metrics | I: hybrid FL crop-yield modeling predates us |
| [Utku et al. 2026](https://doi.org/10.1007/s10791-026-10291-x) | Annual multi-country cereal yield; country-client FL | No rule mechanism verified; non-IID country clients; author privacy limits noted in prior review | Time-aware later-year forecasting | I: different units and horizons; relevant heterogeneity comparator |
| [Koksal et al. 2026](https://doi.org/10.1016/j.compag.2026.111851) | Ethiopia maize/wheat yield; vertical FL | Climate/crop-model/FAO inputs; feature-holder clients; privacy specifics not re-audited | Central vs vertical prediction | I: vertical federation differs from our horizontal simulator |
| [X-FedAgriTwin 2026](https://doi.org/10.1016/j.infsof.2026.108173) | Crop recommendation; digital twin + FL MLP + SHAP | Explicit rule-guided objective not verified; simulated/physical node status requires full-text audit; data locality claimed | Source reports classification accuracy and convergence | I: FL crop recommendation itself is not new; its scores are not comparable to our regression |

## C. Knowledge-guided agricultural learning

| Source | Task/data; algorithm | Knowledge; clients; privacy | Evaluation/finding | Limitation and relevance |
|---|---|---|---|---|
| [KGML-ag-Carbon, Nature Communications 2024](https://www.nature.com/articles/s41467-023-43860-5) | County yield/carbon flux; knowledge-guided neural modeling | Process-model pretraining and customized structure/loss; centralized, not FL; no FL privacy | Source reports gains especially with limited observations and spatial/temporal checks | I: knowledge-guided yield models predate this work; knowledge source differs |
| [KALLM 2025](https://www.sciencedirect.com/science/article/pii/S0950705125002448) | Agricultural Q&A/advice; knowledge-guided LLM | Agronomic books/standards and token/sentence integration; not FL; no FL privacy | Multi-task agricultural decision evaluations | I: knowledge integration is established but task/representation differ |

## D. Privacy, security and leakage

| Source | Task/data; algorithm | Knowledge; clients; privacy | Evaluation/finding | Limitation and relevance |
|---|---|---|---|---|
| [Bonawitz et al. 2017](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/) | High-dimensional update aggregation; cryptographic SecAgg | No agricultural knowledge; dropout-tolerant protocol; server sees aggregate under stated adversaries | Protocol/security analysis and measured overhead | I: our simulator has no such protocol; aggregation noise is not SecAgg |
| [McMahan et al. 2018](https://research.google/pubs/learning-differentially-private-recurrent-language-models/) | Language-model FL; user-level DP training | No agricultural knowledge; many user clients; clipped/noisy accounting | Source reports user-level privacy/utility evaluation | A: requires sufficiently many users; I: our 7 training clients and incomplete release scope differ |
| [Zhu et al. 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html) | Gradient reconstruction on vision/language examples | No agricultural knowledge; shared gradients may leak training data | Demonstrated reconstruction attack | I: not a direct attack evaluation on our update format, but raw-update locality alone is not a guarantee |

## E. Non-IID, tabular baselines and geographic validity

| Source | Task/data; algorithm | Knowledge; clients; privacy | Evaluation/finding | Limitation and relevance |
|---|---|---|---|---|
| [Gorishniy et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | Tabular tasks; ResNet/FT-Transformer vs GBDT | No agricultural rule/FL privacy | Matched neural/tree benchmarking; no universal winner | I: motivates stronger neural references and fair tuning |
| [Grinsztajn et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) | 45 tabular datasets; tuned trees and neural nets | No agronomic rule/FL privacy | Extensive tuning and inductive-bias analysis | I: tree advantage on some tasks cannot be imported as our result |
| [McElfresh et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract.html) | 176 tabular datasets; 19 algorithms | No agronomic rule/FL privacy | Dataset properties condition tree/neural comparisons | I: reinforces dataset-specific interpretation |
| [Roberts et al. 2017](https://doi.org/10.1111/ecog.02881) | Review of spatial/temporal/hierarchical CV | Validation design, no FL/knowledge/privacy | Structured-blocking guidance | I: supports group-preserving split, not independence of our eight groups |
| [Valavi et al. 2019](https://doi.org/10.1111/2041-210X.13107) | Species-distribution spatial/environmental blocking | Validation toolkit, no FL/knowledge/privacy | Spatial vs environmental folds | I: different outcome but relevant shift protocol |
| [Ploton et al. 2020](https://doi.org/10.1038/s41467-020-18321-y) | Ecological mapping; spatial validation | No FL/knowledge/privacy | Spatial hold-out exposes weak transfer | I: geography shift is known prior art, not our discovery |

## Research-gap gate

The often-proposed gap—"no federated crop recommendation uses agricultural
knowledge"—is **not established** by this screening. Recent FL recommendation
work and non-federated agricultural knowledge-guided modeling constrain it.
The active project has no real-data KG-FCR experiment in any case. A defensible
current question is narrower: *with matched representation, model capacity,
optimization and validation budget, what explains the gap between centralized
and simulated federated neural regressors under held-out geographic support?*
This is a planned empirical mechanism study, not yet a proven novel method.

Full-text extraction of exact datasets, sample sizes, metrics, client
distributions, and source-stated limitations remains necessary before making
precise paper-level comparative statements about individual publications.
