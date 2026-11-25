# COMPREHENSIVE STATUS & EXPERIMENTAL RESULTS COMPENDIUM
**Project:** KG-Hybrid (Paper 1 & Paper 2)  
**Date of Audit & Archival:** November 16, 2026  
**Repository Working Directory:** `D:\Final Project\implementation_for _crop recommendation`

---

## 1. Executive Summary of What Was Done

We audited, resolved, and unified the implementation across both tracks of the project:
1. **Paper 1 (Single-Node KG-Hybrid Classification):**
   - Reconciled and validated all benchmark results across 22 crop classes (2,200 samples).
   - Audited the full PDF report and presentation slide deck.
   - Fixed accuracy reporting discrepancies across calibration and ablation tables.
2. **Paper 2 (Cereal-Legume Multi-Location Federated Regression):**
   - Addressed audit blockers, license requirements, and missing data dependencies.
   - Fully documented RR03 (37-experiment clustering & 9 hash collision pairs).
   - Authored the formal project code license (`paper2/LICENSE` - Apache-2.0) and dataset redistribution notice (`DATASET_NOTICE.txt`).
   - Prepared third-party dependency licensing inventory (`configs/third_party_license_inventory_v1.txt`).
   - Cleaned all audit path mismatches and established canonical tracking (`CANONICAL_AUDIT_PATHS.txt`, `completion_audit_v10`).
   - Converted the complete 925-line Markdown manuscript into submission-grade IEEE format (`paper2/manuscript/paper2_ieee.tex`).
   - Placed all 5 audited vector PDF and PNG figures into `paper2/manuscript/` matching all LaTeX cross-references.

---

## 2. Key Experimental Results & Numerical Benchmarks

### A. Paper 1: KG-Hybrid Classification (22 Crop Classes, 7 Soil/Climate Features)
*Dataset: 2,200 records (Stratified 70:15:15 split — 1,540 Train / 330 Val / 330 Test)*

#### 1. Clean Test Performance (Table 4.4)
- **KG-Hybrid (Proposed):** Accuracy = **99.70%**, Macro-F1 = **99.70%**, Top-3 Recall = **100.00%**
- **Random Forest:** Accuracy = 99.70%, Macro-F1 = 99.70%, Top-3 Recall = 100.00% (Clean tie)
- **XGBoost:** Accuracy = 99.39%, Macro-F1 = 99.39%, Top-3 Recall = 100.00%
- **FT-Transformer:** Accuracy = 98.79%, Macro-F1 = 98.78%, Top-3 Recall = 99.70%
- **Decision Tree:** Accuracy = 98.18%
- **Logistic Regression / Stacking:** Accuracy = 97.88%

#### 2. Model Robustness & Stress-Testing
- **Gaussian Noise ($\sigma = 0.2$):**
  - KG-Hybrid: **89.21%** vs. XGBoost: **79.39%** (+9.82 pp advantage)
  - At $\sigma = 0.3$: KG-Hybrid: **84.24%** vs. XGBoost: **67.45%** (+16.79 pp advantage)
- **Missing Features (40% missing):**
  - Raw KG-Hybrid: 47.03% | Raw XGBoost: 61.15%
  - **KG-Hybrid + KNN ($k=5$) Imputation:** **82.73%** (+35.7 pp recovery)
- **1.5× Feature Extrapolation:**
  - KG-Hybrid: **90.00%** (-9.70 pp drop)
  - FT-Transformer: **93.03%** (-5.76 pp drop)
  - Random Forest: 80.00% (-19.70 pp drop)
  - XGBoost: 74.55% (-24.84 pp drop)

#### 3. Probability Calibration (Table 4.10)
- **KG-Hybrid (Raw):** ECE = 0.1218, Brier = 0.0288, NLL = 0.1398, Accuracy = 99.70%
- **KG-Hybrid (Temperature-Scaled):** ECE = **0.0184**, Brier = **0.0102**, NLL = **0.0289**, Accuracy = 99.70%
- **XGBoost:** ECE = 0.0162, Brier = 0.0119, NLL = 0.0334, Accuracy = 99.39%

#### 4. Inference Efficiency (Table 4.11 & 4.12)
- **Early-Exit ($\tau = 0.90$):** 96.36% exit rate, **5.99 ms** average latency, **19.3× speed-up**, 99.70% accuracy
- **Lightweight Variant (32-D Embedding):** 4.04M parameters (-74% reduction from 15.50M), 25.40 ms latency (-72% reduction)

---

### B. Paper 2: Cereal-Legume Federated Regression
*Dataset: 607 crop management systems across 37 experiments and 8 geographic groups*  
*Target: Complete harvest seed dry matter (t/ha)*

#### 1. Baseline Study (R13 Fixed-Public Unnoised Refit)
- **Primary Equal-Group RMSE:** **$1.067407 \pm 0.021053$ t/ha**
- **Pooled RMSE:** **$1.201189 \pm 0.038476$ t/ha**
- **Geographic Group Variations:**
  - Auzeville (354 systems): 0.81 t/ha
  - Grignon (64 systems): 0.88 t/ha
  - Angers (75 systems): 1.15 t/ha
  - Copenhagen+Taastrup (52 systems): 1.18 t/ha
  - Jyndevad (48 systems): 1.48 t/ha
  - Small groups (Kassel, Reading, Sanmarcoargentano): higher variance due to sample counts ($N \le 6$)

#### 2. Matched Aggregation-Noise Simulation (R14 Study)
*20-round conditional zCDP budget evaluation ($\delta = 10^{-5}$)*
- **$\sigma_{\text{mult}} = 0$ (Unnoised Baseline):** Equal-group RMSE = **1.067 t/ha**
- **$\sigma_{\text{mult}} = 2$:** Equal-group RMSE = **1.183 t/ha** ($\Delta \approx +0.116$ t/ha)
- **$\sigma_{\text{mult}} = 4$:** Equal-group RMSE = **2.218 t/ha** ($\Delta \approx +1.151$ t/ha)
- **$\sigma_{\text{mult}} = 8$:** Equal-group RMSE = **5.628 t/ha** (Severe utility degradation)

---

## 3. Directory of Artifacts Generated in this Phase

| Artifact | Location | Purpose |
|:---|:---|:---|
| **IEEE LaTeX Manuscript** | `paper2/manuscript/paper2_ieee.tex` | 1056 lines, complete IEEEtran submission file |
| **LaTeX Build Notes** | `paper2/manuscript/paper2_ieee_build_notes.txt` | Compilation, pdflatex command, and packaging instructions |
| **Bound Figure Assets** | `paper2/manuscript/fig_*.pdf` & `.png` | 5 vector/raster plots bound to figure tags |
| **Project Code License** | `paper2/LICENSE` | Apache-2.0 legal license |
| **Dataset Notice** | `paper2/DATASET_NOTICE.txt` | Zenodo DOI attribution and redistribution policy |
| **License Inventory** | `paper2/configs/third_party_license_inventory_v1.txt` | Permissive license compatibility audit |
| **Canonical Audit Paths** | `paper2/CANONICAL_AUDIT_PATHS.txt` | Ground-truth path map resolving legacy reference bugs |
| **Audit v10 Record** | `paper2/results/restart_completion_audit_v10/` | Canonical active completion audit |
| **RR03 Resolution** | `paper2/research/rr03_resolution_v1.md` | Formal audit of data clustering and hash collisions |
| **Submission Checklist** | `paper2/research/ieee_submission_checklist_v1.md` | Pre-submission roadmap and external dependency tracker |
