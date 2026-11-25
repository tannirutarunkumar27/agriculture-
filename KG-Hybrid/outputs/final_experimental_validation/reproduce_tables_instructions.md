# Reproducible Audit & Table Generation Protocol

**Project:** KG-Hybrid (Crop Recommendation)  
**Auditor:** Senior ML Research Engineer & IEEE Access Reproducibility Auditor  
**Date:** 2026-10-11  

---

## 1. Overview and Reproducibility Principles

This protocol describes how to independently verify and regenerate every audited table directly from raw saved experiment prediction artifacts.
  - **No Retraining or Model Mutation:** All metrics are recomputed purely from saved raw predictions (`predictions/*.csv`, `oof_predictions/*.csv`, `probabilities/*.npy`).
- **Zero Fabrication:** The legacy incomplete neural-CV evidence is retained unchanged; the complete neural recovery is a separately stored five-fold run.

---

## 2. Directory Layout of Saved Artifacts

```
KG-Hybrid/
├── outputs/
│   ├── ablation/five_seed/
│   │   ├── predictions/
│   │   │   ├── seed_42/    (9 variant prediction CSVs: Full, A1-A8)
│   │   │   ├── seed_123/   (9 variant prediction CSVs)
│   │   │   ├── seed_2024/  (9 variant prediction CSVs)
│   │   │   ├── seed_3407/  (9 variant prediction CSVs)
│   │   │   └── seed_9999/  (9 variant prediction CSVs)
│   │   ├── ablation_per_seed.csv
│   │   ├── ablation_summary.csv
│   │   └── run_metadata.json
│   ├── cross_validation/                         (legacy partial CV; preserved)
│   ├── cross_validation_neural_recovery/
│   │   ├── oof_predictions/oof_predictions.csv (13,200 rows: 6 models × 2,200 samples)
│   │   ├── aggregate/cv_summary.csv
│   │   └── per_fold/fold_{1..5}.json
│   ├── calibration_reconciled/
│   │   └── probabilities/*.npy (9 saved probability distributions, N=330)
│   └── final_experimental_validation/
│       ├── experiment_manifest.json
│       ├── validation_report.txt
│       ├── reproduce_tables_instructions.md
│       └── tables/
│           ├── table1_ablation_5seed_validated.csv
│           ├── table2_cv_results_all_models.csv
│           ├── table3_cv_per_class_metrics.csv
│           ├── table4_calibration_recalculated.csv
│           └── confusion_matrices_cv_oof.json
```

---

## 3. Step-by-Step Regeneration Instructions

### Step 1: Environment Setup
Ensure the Python environment with required packages is active:
```powershell
python -m pip install numpy pandas scipy scikit-learn joblib
```

### Step 2: Run Independent Metric Verification
Run the reproducible audit script:
```powershell
python "KG-Hybrid/src/utils/generate_paper_ready_package.py"
```

This script:
1. Re-reads all 45 raw ablation prediction files (`y_true`, `y_pred`).
2. Calculates Top-1 accuracy and Macro-F1 per seed and variant.
3. Computes sample mean and sample standard deviation (`ddof=1`).
4. Performs paired comparisons (paired t-test and Wilcoxon signed-rank test).
5. Parses the isolated recovery OOF file across all 5 folds (2,200 samples per model) for all 6 models, including FT-Transformer and KG-Hybrid.
6. Evaluates per-class precision, recall, and F1 along with confusion matrices.
7. Evaluates calibration probabilities against axiomatic constraints and calculates ECE, Brier score, and NLL.
8. Writes all verified outputs to `KG-Hybrid/outputs/final_experimental_validation/tables/`.

---

## 4. Verification Check Commands

To run unit tests asserting cross-validation isolation, zero data leakage, and seed repeatability:
```powershell
python -m unittest KG-Hybrid/tests/test_run_cv.py
python -m unittest KG-Hybrid/tests/test_adaptive_fusion_gate.py
python -m unittest KG-Hybrid/tests/test_sample_conditioned_knowledge.py
```
All unit tests should pass with exit code 0.

The generated tables use `outputs/cross_validation_neural_recovery/`; do not overwrite or merge it with the legacy `outputs/cross_validation/` directory.
