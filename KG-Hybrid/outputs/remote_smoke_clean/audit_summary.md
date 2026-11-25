# Remote CPU smoke-study audit

Audit date: 2026-11-02

The retrieved receipt was checked against the local project. The SHA-256 hash of
`data/raw/Crop_recommendation.csv` and the SHA-256 hash of
`src/experiments/run_cv.py` both match the local files.

The OOF file contains 8,800 rows: 2,200 predictions for each of four models
(LogisticRegression, DecisionTree, RandomForest and XGBoostBaseline), covering
all five folds and 2,200 unique sample IDs. The run manifest records 2,200
samples, 22 classes, `include_neural: false`, and a CPU-only environment.

The reported `cv_summary.csv` is a mean and standard deviation across fold-level
metrics. A separate pooled recomputation from all OOF predictions gives the
cross-fold pooled metrics. These are different estimands: for example, the
DecisionTree fold-mean macro-F1 is 0.4672, while its pooled OOF macro-F1 is
0.517185. Neither value should replace the final audited Paper 1 evidence
without reconciling the protocol. This smoke run is deployment evidence, not a
new final result or a replacement for the audited validation package.
