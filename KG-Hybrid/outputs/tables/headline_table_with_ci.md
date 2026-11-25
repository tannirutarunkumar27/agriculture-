| Protocol | Model | Accuracy (Mean ± SD) | 95% CI | Macro-F1 | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 5-Seed 70/15/15 (N=330) | Hybrid_NoK | 99.58% ± 0.46% | [0.9901, 1.0015] | 99.58% | 0.0051 |
| 5-Seed 70/15/15 (N=330) | Hybrid_FixedK | 99.52% ± 0.27% | [0.9918, 0.9985] | 99.51% | 0.0050 |
| 5-Seed 70/15/15 (N=330) | XGBoost | 99.52% ± 0.17% | [0.9931, 0.9972] | 99.51% | 0.0185 |
| 5-Seed 70/15/15 (N=330) | Hybrid_AnchoredK | 99.45% ± 0.40% | [0.9896, 0.9995] | 99.45% | 0.0070 |
| 5-Seed 70/15/15 (N=330) | Hybrid_AnchoredK_Gate | 99.45% ± 0.25% | [0.9914, 0.9977] | 99.46% | 0.0043 |
| 5-Seed 70/15/15 (N=330) | LightGBM | 99.45% ± 0.25% | [0.9914, 0.9977] | 99.45% | 0.0068 |
| 5-Seed 70/15/15 (N=330) | CatBoost | 99.27% ± 0.51% | [0.9864, 0.9990] | 99.27% | 0.0292 |
| 5-Seed 70/15/15 (N=330) | Hybrid_ShuffledK_Gate | 99.09% ± 0.48% | [0.9850, 0.9969] | 99.09% | 0.0059 |
| 5-Seed 70/15/15 (N=330) | DecisionTree_Unconstrained | 98.79% ± 0.57% | [0.9808, 0.9949] | 98.79% | 0.0121 |
| 5-Seed 70/15/15 (N=330) | FTTransformer | 98.61% ± 0.46% | [0.9804, 0.9918] | 98.61% | 0.0100 |
| 5-Fold Stratified CV (N=2200 OOF) | XGBoostBaseline | 99.50% ± 0.30% | [0.9913, 0.9987] | 99.50% | — |
| 5-Fold Stratified CV (N=2200 OOF) | CatBoost | 99.45% ± 0.20% | [0.9920, 0.9971] | 99.45% | — |
| 5-Fold Stratified CV (N=2200 OOF) | LightGBM | 99.36% ± 0.19% | [0.9913, 0.9960] | 99.36% | — |
| 5-Fold Stratified CV (N=2200 OOF) | KG-HybridProposed | 99.32% ± 0.28% | [0.9897, 0.9966] | 99.32% | — |
| 5-Fold Stratified CV (N=2200 OOF) | RandomForest | 99.00% ± 0.34% | [0.9857, 0.9943] | 98.99% | — |
| 5-Fold Stratified CV (N=2200 OOF) | DecisionTree_Unconstrained | 98.77% ± 0.76% | [0.9782, 0.9972] | 98.77% | — |
| 5-Fold Stratified CV (N=2200 OOF) | FTTransformerBaseline | 98.18% ± 0.98% | [0.9697, 0.9940] | 98.18% | — |
| 5-Fold Stratified CV (N=2200 OOF) | LogisticRegression | 97.09% ± 0.25% | [0.9678, 0.9740] | 97.07% | — |
| 5-Fold Stratified CV (N=2200 OOF) | DecisionTree | 55.64% ± 2.33% | [0.5274, 0.5853] | 46.72% | — |
