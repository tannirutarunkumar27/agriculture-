# Bounded validation convergence pilot

Frozen before execution in `configs/convergence_pilot.json` and copied into the
pilot's hashed `protocol.json`. This is a development pilot, not a final study.

## Fixed choices

Canonical IID and Maharashtra district partitions, seed 42. Neural capacity:
one shared MLP with 64 hidden units. Plain SGD, one local epoch, batch 32,
80 rounds, full participation, fixed proximal mu=.01. Three shared learning
rates (.05, .1, .2); pick ONE common neural rate per dataset by the mean best
validation NLL of centralized/FedAvg/FedProx/FedNova. Do not pick a different
capacity or favorable rate for each algorithm. Ties use candidate order.

Logistic regression: C=.1/1/10. Random forest: 200 trees with minimum leaf
size 1/2. XGBoost: 200 boosting rounds, fixed LR .1 and depth 3/6. Each conventional
model is selected by validation NLL. Search budgets differ by model and are
declared; models are performance references, not an equal-hyperparameter-cost
comparison. All use the same train IDs, training-only preprocessing and global
label schema. Soil color is encoded with training-only vocabulary.

Neural checkpoints are selected by minimum validation NLL, not by the highest
accuracy or F1. Report metrics at that selected checkpoint, even if another
round had higher F1. All candidate summaries contain null test metrics and no
test prediction files. The earlier smoke phase already evaluated these test
partitions: treat them as development holdouts, not pristine external validation.
No new test results are inspected during this pilot. A final paper needs a
clearly frozen confirmatory protocol or independent data.

## Correct interpretation

Equal epochs imply equal aggregate data exposure, not equal optimization paths
or optimizer-step count. A centralized model performs sequential updates on
all examples; federation averages separately trained models. Actual local
steps, validation curves, time and tensor payload bytes are retained. Global
train-only scaling is a public-data simulation reference, not private scaling.
District-based validation is same-district held-out data, not leave-district-out.

FedNova must match FedAvg up to floating-point differences with equal step
counts. Maharashtra unequal client sizes permit differing step normalizers;
one seed cannot establish statistical superiority. Rare-class Macro-F1 may be
unstable due to limited validation support. Regional reports include class
support and global-schema Macro-F1; absent classes contribute zero.

Best candidate at the upper learning-rate boundary, or best checkpoint near
round 80, does not prove convergence. Further fixed-budget pilots and repeated
seeds are needed before a final comparison. No agronomic guidance, formal
privacy, deployment or causal agricultural benefit is claimed by these scores.

## Reproducibility checks

Strict source/partition/code identities; no source files overwritten; all
candidate metrics recomputed from saved checkpoints/models; selected neural
round verified against validation log; no test artifacts accepted. Save the
exact source snapshot and commands for later version-compatible reproduction.

API references checked (installed environment validated through tests):
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
