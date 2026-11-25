# Regression-focused closest-work review

Checked 2026-10-14 against primary publisher, proceedings, or author records.
This review replaces the earlier classification-oriented comparison for the
active Paper 2 task. It is a focused positioning review, not a claim that every
federated agricultural paper has been found. External results are not project
results and are not compared numerically with Paper 2.

## What the literature already establishes

Federated crop-yield prediction is not new. Published work has considered
FedAvg with residual networks for soybean yield, localized/pruned federated
yield forecasting, recurrent edge/cloud systems, centralized and decentralized
federated architectures, graph-recurrent models, country-client cereal
forecasting, and vertical federation across feature holders. Agricultural
federation has also been studied with Earth-observation crop segmentation as a
proxy where suitable yield data were unavailable.

Consequently, Paper 2 must not claim novelty for federated yield prediction,
hybrid neural modeling, local data retention, client heterogeneity, or use of
FedAvg/FedProx/FedNova. Its contribution must be stated at the level actually
supported by the completed experiment and evidence registry.

## Comparison matrix

| Study | Relevant task and federation | Why it is useful | Why scores cannot be compared directly |
| --- | --- | --- | --- |
| DELCON 2022, [federated crop-yield prediction](https://doi.org/10.1109/DELCON54057.2022.9752836) | Soybean-yield regression; FedAvg with residual models | Confirms federated crop-yield regression predates this work | Different crop, data, clients, split, model, target scale, metrics, and budget |
| Li et al. 2024, [FedPruning](https://doi.org/10.1016/j.eswa.2023.122847) | Localized soybean-yield forecasting with client pruning | Closest horizontal heterogeneous yield-forecasting context | Adds personalization, pruning, communication and model-size objectives absent here |
| Mukherjee and Buyya 2025, [FL architectures](https://doi.org/10.1002/spe.3416) | LSTM crop-yield application comparing central, ring, and mesh federation | Shows decentralized topology evaluation already exists | Different temporal model, graph, accuracy definition, response-time objective, and protocol |
| Dey et al. 2025, [FLyer](https://doi.org/10.1109/TAI.2025.3534149) | Edge/cloud federated LSTM yield prediction | Recent agriculture-specific system comparator | Different data, infrastructure, model, metrics, and cryptographic scope |
| Cowlishaw et al. 2025, [EO agricultural predictions](https://doi.org/10.1038/s41598-025-94244-2) | Crop-type segmentation proxy; studies client count, imbalance and aggregation | Useful evidence on agricultural federation and heterogeneous clients | Proxy image segmentation, not yield regression; different inputs, support and unit of analysis |
| Nirosha and Vennila 2025, [FL-AGRN](https://doi.org/10.1016/j.eswa.2025.128312) | Graph/recurrent federated crop-yield prediction | Blocks a broad hybrid-model novelty claim | Different representation, target, clients, splits, model and metrics |
| Utku et al. 2026, [multi-country Edge-Fog-Cloud forecasting](https://doi.org/10.1007/s10791-026-10291-x) | Annual European cereal yield with country clients and time-aware evaluation | Closest cereal/non-IID context and an explicit source for privacy limitations | Country-year time series differ from experimental systems, site groups, features and target units |
| Koksal et al. 2026, [vertical federated crop-yield prediction](https://doi.org/10.1016/j.compag.2026.111851) | Ethiopia maize/wheat prediction using vertical split learning | Clarifies that vertical institutional federation is a distinct setting | Vertical feature holders rather than horizontal site clients; different data and validation |

## Positioning adopted for Paper 2

The closest contexts are Li et al. for horizontal, heterogeneous yield
forecasting and Utku et al. for non-IID multi-country cereal forecasting. Even
these are not matched benchmarks. Paper 2 predicts conditional harvest
dry matter for observed cereal-legume systems across eight held-out experimental
groups. It is not a country-yield forecast, a recommendation system, a vertical
federation study, or an independently validated field decision aid.

The defensible focus is therefore a reproducible and bounded comparison of
centralized and simulated horizontal federated baselines on this conditional
regression task. The manuscript may emphasize nested group-held-out design,
one-time outer evaluation, equal-group and pooled reporting, seen/unseen support
panels, explicit adverse results, separation of a post-hoc support guard, and a
matched aggregation-noise utility simulation. These are characteristics of the
evaluation, not evidence for a new federated optimizer or universal advantage.

## Privacy and comparison rules

Keeping records local, encrypting a communication channel, or adding simulated
aggregation noise does not by itself establish formal privacy. Paper 2 must
state the protected unit, adversary and trust assumptions, release transcript,
clipping/noise mechanism, accountant and any secure-aggregation dependency.
The present matched-noise study remains a simulation and must not be called a
production privacy guarantee.

No external score may be placed beside a Paper 2 score as a superiority test
unless target definition and units, population, client construction, train/
validation/test protocol, metric calculation and computational budget are
demonstrably comparable. The current studies do not pass that gate; they should
be compared qualitatively and cited as externally reported findings only.

The machine-readable review and enforced claim boundaries are stored in
`restart_regression_closest_work.json` and validated before manuscript use.
