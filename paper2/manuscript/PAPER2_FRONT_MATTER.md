# Geographically Grouped Federated and Centralized Regression for Conditional Harvest Prediction in Cereal-Legume Field Experiments

## Abstract

Agricultural prediction across research locations is complicated by unequal
sample sizes, heterogeneous crop support, and the risk that pooled metrics hide
failures in small or held-out groups. This study evaluates centralized and
simulated horizontal federated learning for retrospective conditional harvest
seed-dry-matter regression using 607 recorded cereal-legume crop-management
systems from eight geographic groups. A nested group-held-out protocol was used
for validation and outer evaluation, with training-only preprocessing, five
declared refit seeds, immutable execution identities, and one-time test-inference
receipts. In the original eight-method comparison, XGBoost had the lowest
observed equal-group RMSE (1.2461 t/ha) and pooled RMSE (1.7335 t/ha), while the
evaluated neural models exhibited severe extrapolation from input columns that
were unsupported during training. A training-only zero-support guard reduced
these neural errors in a post-hoc fixed-model sensitivity analysis and is not
treated as independent confirmation. A later prospectively frozen fixed-public
refit achieved equal-group RMSE 1.067407 +/- 0.021053 t/ha and pooled RMSE
1.201189 +/- 0.038476 t/ha. In its matched aggregation-noise study, equal-group
RMSE increased to 1.182853, 2.218085, and 5.628243 t/ha at noise multipliers 2,
4, and 8, respectively. Equal-group and pooled metrics, geographic results, and
seen/unseen support panels are reported together. The study provides a
reproducible characterization of grouped conditional regression and controlled
noise utility; it does not establish best-crop recommendation, causal
agricultural benefit, a novel federated optimizer, independent field
generalization, secure aggregation, or end-to-end differential privacy.

**Index Terms—** Agricultural regression, cereal-legume systems, federated
learning, grouped validation, FedAvg, FedProx, FedNova, model support,
aggregation noise, reproducibility.
