# Agriculture App: Knowledge-Guided Hybrid (KG-Hybrid) Crop Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, publication-grade **Knowledge-Guided Hybrid (KG-Hybrid) Deep Learning Framework** for precision agricultural crop recommendation. The framework integrates empirical soil and climatic feature representations with tree-based leaf node embeddings (XGBoost) and agronomic domain rule matrices via a gated knowledge fusion mechanism.

---

## 🌟 Overview & Key Innovations

Traditional data-driven machine learning models for crop recommendation often fail to maintain domain consistency or perform poorly under sensor noise and missing input data. **KG-Hybrid** bridges machine learning representations with agronomic domain expertise.

### Key Architectural Highlights
1. **Knowledge-Guided Fusion Engine**: Blends neural representations of soil properties (N, P, K, pH) and weather parameters (Temperature, Humidity, Rainfall) with structured domain rules and XGBoost decision boundary leaf embeddings.
2. **Gated Attention Mechanism**: Dynamically controls the balance between raw empirical data embeddings and domain knowledge constraints.
3. **Lightweight Deployment Architecture**: A stream-lined **4.04M parameter** variant designed specifically for edge and resource-constrained agricultural field devices, delivering **<1.5 ms inference latency**.
4. **Dynamic Early-Exit Pipeline**: Adaptive computation mechanism allowing high-confidence predictions to exit early, reducing average inference time by up to 45%.
5. **Robustness & Fault Tolerance**: Built-in KNN imputation and noise-mitigation layer capable of handling missing sensor inputs and noisy field readings without degrading accuracy.

---

## 📁 Repository Structure

```
implementation_for _crop recommendation/
├── README.md                           # Main Project Overview & Documentation
├── .gitignore                          # Git Ignore Configuration
├── project_structure.txt               # Complete workspace file tree
│
└── KG-Hybrid/                          # Core Research & Model Module
    ├── README.md                       # Submodule overview
    ├── requirements.txt                # Python Dependencies
    ├── train.py                        # Top-level Training Script
    ├── evaluate.py                     # Global Model Evaluation Script
    │
    ├── data/                           # Dataset Directory
    │   ├── raw/                        # Raw CSV datasets (Crop_recommendation.csv)
    │   ├── interim/                    # Train/Val/Test raw splits
    │   ├── processed/                  # Normalized matrices, leaf embeddings & encoders
    │   └── models/                     # Model artifacts (kg_hybrid_model.pkl.zip)
    │
    ├── outputs/                        # Model & Benchmark Output Artifacts
    │   ├── models/                     # Trained model checkpoints (.zip archives)
    │   │   ├── kg_hybrid_best.pth.zip
    │   │   ├── lightweight_kg_hybrid_best.pth.zip
    │   │   ├── hybrid_concat_best.pth.zip
    │   │   ├── best_model.pth.zip
    │   │   └── xgboost_model.pkl.zip
    │   │
    │   ├── ablation/                   # Checkpoints for Ablation Studies (A1–A7 .zip)
    │   │   ├── a1_best.pth.zip         # (A1: No Domain Knowledge)
    │   │   ├── a2_best.pth.zip         # (A2: No XGBoost Leaves)
    │   │   ├── a3_best.pth.zip         # (A3: Concat Fusion)
    │   │   ├── a4_best.pth.zip         # (A4: No Auxiliary Loss)
    │   │   ├── a5_best.pth.zip         # (A5: Fixed Gating)
    │   │   ├── a6_best.pth.zip         # (A6: No Data Augmentation)
    │   │   └── a7_best.pth.zip         # (A7: Unscaled Knowledge)
    │   │
    │   ├── figures/                    # Publication-Grade Figures & Diagrams
    │   ├── plots/                      # Training Loss/Accuracy Curves & Radar Charts
    │   └── robustness/                 # Missing Feature & Latency Benchmark Reports
    │
    └── src/                            # Modular Source Code
        ├── models/                     # Architecture Implementations
        │   ├── baselines/              # Baseline XGBoost, Random Forest, FT-Transformer
        │   └── hybrid/                 # KG-Hybrid, Lightweight Model & Aux Losses
        ├── knowledge/                  # Agronomic Knowledge Matrix Generators
        ├── ablation/                   # Systematic Ablation Study Runners
        ├── robustness/                 # Missing Features & Stress Testing Modules
        ├── inference/                  # Production & Early-Exit Pipeline Test Suite
        └── utils/                      # Publication Visualization & Figure Generation
```

---

## 📦 Datasets & Model Artifacts

### Datasets Included
- **Raw Soil & Climate Dataset**: `KG-Hybrid/data/raw/Crop_recommendation.csv` (2,200 samples across 22 distinct crop types with Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall).
- **Processed & Interim Matrices**: Pre-split datasets, leaf node transformation matrices (`xgb_train_leaves.npy`), and normalized scaling transformers (`scaler.pkl`, `label_encoder.pkl`).

### Model Checkpoints (Zipped)
To satisfy GitHub repository limits while retaining model reproducibility, all trained weight checkpoints are provided as zipped archives inside `KG-Hybrid/outputs/models/` and `KG-Hybrid/outputs/ablation/`.

#### Extracting Model Checkpoints:
You can uncompress any model zip file using Python:
```python
import zipfile

# Example: Extracting the main KG-Hybrid checkpoint
with zipfile.ZipFile('KG-Hybrid/outputs/models/kg_hybrid_best.pth.zip', 'r') as zip_ref:
    zip_ref.extractall('KG-Hybrid/outputs/models/')
```
Or extract all models at once via command line:
```bash
python -c "import zipfile, glob; [zipfile.ZipFile(f).extractall(f.rsplit('/',1)[0]) for f in glob.glob('KG-Hybrid/outputs/**/*.zip', recursive=True)]"
```

---

## 📊 Performance & Benchmarks

> **Canonical Headline Protocol** (generated automatically from `KG-Hybrid/outputs/final_results.json`).

### 5-Seed 70/15/15 Held-Out Comparison (Mean ± SD, 95% CI)

| Model | Accuracy (Mean ± SD) | 95% CI | Macro-F1 | ECE |
| :--- | :---: | :---: | :---: | :---: |
| Hybrid_NoK | 99.58% ± 0.46% | [99.01%, 100.15%] | 99.58% | 0.0051 |
| Hybrid_FixedK | 99.52% ± 0.27% | [99.18%, 99.85%] | 99.51% | 0.0050 |
| XGBoost | 99.52% ± 0.17% | [99.31%, 99.72%] | 99.51% | 0.0185 |
| Hybrid_AnchoredK | 99.45% ± 0.40% | [98.96%, 99.95%] | 99.45% | 0.0070 |
| Hybrid_AnchoredK_Gate | 99.45% ± 0.25% | [99.14%, 99.77%] | 99.46% | 0.0043 |
| LightGBM | 99.45% ± 0.25% | [99.14%, 99.77%] | 99.45% | 0.0068 |
| CatBoost | 99.27% ± 0.51% | [98.64%, 99.90%] | 99.27% | 0.0292 |
| Hybrid_ShuffledK_Gate | 99.09% ± 0.48% | [98.50%, 99.69%] | 99.09% | 0.0059 |
| DecisionTree_Unconstrained | 98.79% ± 0.57% | [98.08%, 99.49%] | 98.79% | 0.0121 |
| FTTransformer | 98.61% ± 0.46% | [98.04%, 99.18%] | 98.61% | 0.0100 |

*5-Seed Friedman test (accuracy): $\chi^2_F = 17.6947$, $p = 0.0389$.*

### 5-Fold Stratified Cross-Validation (2,200 OOF Samples, Mean ± SD, 95% CI)

| Model | Fold Accuracy (Mean ± SD) | 95% CI | Pooled OOF Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| XGBoostBaseline | 99.50% ± 0.30% | [99.13%, 99.87%] | 99.50% | 99.50% |
| CatBoost | 99.45% ± 0.20% | [99.20%, 99.71%] | 99.45% | 99.45% |
| LightGBM | 99.36% ± 0.19% | [99.13%, 99.60%] | 99.36% | 99.36% |
| KG-HybridProposed | 99.32% ± 0.28% | [98.97%, 99.66%] | 99.32% | 99.32% |
| RandomForest | 99.00% ± 0.34% | [98.57%, 99.43%] | 99.00% | 98.99% |
| DecisionTree_Unconstrained | 98.77% ± 0.76% | [97.82%, 99.72%] | 98.77% | 98.77% |
| FTTransformerBaseline | 98.18% ± 0.98% | [96.97%, 99.40%] | 98.18% | 98.18% |
| LogisticRegression | 97.09% ± 0.25% | [96.78%, 97.40%] | 97.09% | 97.07% |
| DecisionTree | 55.64% ± 2.33% | [52.74%, 58.53%] | 55.64% | 46.72% |

*5-Fold CV Friedman test (accuracy): $\chi^2_F = 34.3090$, $p = 3.5709e-05$.*

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/tannirutarunkumar27/Agriculture_App.git
cd Agriculture_App/KG-Hybrid
pip install -r requirements.txt
```

### 2. Running Training
To train the Lightweight KG-Hybrid model:
```bash
python src/models/hybrid/train_lightweight.py
```

To extract XGBoost leaf embeddings:
```bash
python src/models/hybrid/extract_xgb_leaves.py
```

### 3. Evaluating Robustness & Early-Exit Pipeline
To run the full production test suite with KNN imputation and latency measurements:
```bash
python src/inference/test_production_pipeline.py
```

To run missing feature stress testing:
```bash
python src/robustness/test_missing_features.py
```

### 4. Running Ablation Studies
To execute the 7-phase ablation study suite:
```bash
python src/ablation/summarize_ablations.py
```

### 5. Generating Publication Figures
To regenerate all figures and visualizations:
```bash
python src/utils/generate_all_methodology_figures.py
```

---

## 📄 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
