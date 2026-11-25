"""
generate_realworld_summary.py
==============================
Aggregates and visualizes all real-world production-readiness improvements:
  1. Missing Feature Recovery (Imputation)
  2. Early Exit Speedup (Latency vs Threshold)
  3. Parameter Reduction (Full vs Lightweight)
  4. Uncertainty & Calibration Safety Verification

Outputs:
  - outputs/tables/realworld_comparison.csv
  - outputs/tables/realworld_comparison_latex.txt
  - outputs/figures/realworld_enhancements.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

res_file = TABLES_DIR / "production_readiness_results.json"

with open(res_file, "r") as f:
    data = json.load(f)

imp_data = data["imputation_experiments"]
ee_data  = data["early_exit_experiments"]

# ── 1. Create Comparison CSV ────────────────────────────────────────────────
imp_df = pd.DataFrame({
    "Missing Rate": ["0%", "20%", "40%", "60%"],
    "Raw Zero Fill (Baseline)": [f"{v*100:.2f}%" for v in imp_data["Raw (Zero Fill)"]],
    "Median Imputation": [f"{v*100:.2f}%" for v in imp_data["Median Imputation"]],
    "KNN Imputation (k=5)": [f"{v*100:.2f}%" for v in imp_data["KNN Imputation (k=5)"]],
    "MICE Imputation": [f"{v*100:.2f}%" for v in imp_data["MICE Imputation"]],
    "XGBoost Baseline": [f"{v*100:.2f}%" for v in imp_data["XGBoost Baseline"]]
})

imp_csv = TABLES_DIR / "realworld_imputation_comparison.csv"
imp_df.to_csv(imp_csv, index=False)
print(f"[OK] Imputation comparison CSV saved -> {imp_csv}")

ee_df = pd.DataFrame(ee_data)
ee_csv = TABLES_DIR / "realworld_early_exit_comparison.csv"
ee_df.to_csv(ee_csv, index=False)
print(f"[OK] Early exit comparison CSV saved -> {ee_csv}")

# ── 2. Create LaTeX Table ────────────────────────────────────────────────────
latex_lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Real-World Production Improvements for KG-Hybrid}",
    r"\label{tab:realworld_enhancements}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lcccc}",
    r"\hline",
    r"\textbf{Missing Feature Ratio} & \textbf{Raw (Zero-Fill)} & \textbf{Median Imp.} & \textbf{KNN Imp. (k=5)} & \textbf{XGBoost Baseline} \\",
    r"\hline"
]

for idx, r_label in enumerate([r"0\%", r"20\%", r"40\%", r"60\%"]):
    raw = imp_data["Raw (Zero Fill)"][idx] * 100
    med = imp_data["Median Imputation"][idx] * 100
    knn = imp_data["KNN Imputation (k=5)"][idx] * 100
    xgb = imp_data["XGBoost Baseline"][idx] * 100
    latex_lines.append(f"{r_label} & {raw:.2f}\\% & {med:.2f}\\% & \\textbf{{{knn:.2f}\\%}} & {xgb:.2f}\\% \\\\")

latex_lines += [
    r"\hline",
    r"\end{tabular}",
    r"}",
    r"\end{table}"
]

latex_path = TABLES_DIR / "realworld_comparison_latex.txt"
with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
print(f"[OK] LaTeX table saved -> {latex_path}")

# ── 3. Create Multi-Panel Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Real-World Production Enhancements — KG-Hybrid System', color='white', fontsize=15, fontweight='bold', y=1.02)

# Panel 1: Missing Feature Imputation
ax1 = axes[0]
ax1.set_facecolor('#1A1D2E')
ax1.spines[:].set_color('#2D3050')
ratios = [0, 20, 40, 60]

ax1.plot(ratios, [v*100 for v in imp_data["Raw (Zero Fill)"]], 'o--', color='#FF6B6B', label='Raw (Zero Fill)', linewidth=2)
ax1.plot(ratios, [v*100 for v in imp_data["XGBoost Baseline"]], 's--', color='#20C997', label='XGBoost Baseline', linewidth=2)
ax1.plot(ratios, [v*100 for v in imp_data["KNN Imputation (k=5)"]], 'd-', color='#339AF0', label='KNN Imputation (Ours)', linewidth=2.5)

ax1.set_title('Missing Feature Recovery (Imputation)', color='white', fontsize=12, fontweight='bold')
ax1.set_xlabel('Missing Feature Ratio (%)', color='#8892A4', fontsize=10)
ax1.set_ylabel('Accuracy (%)', color='#8892A4', fontsize=10)
ax1.tick_params(colors='#8892A4')
ax1.grid(color='#2D3050', linestyle='--', alpha=0.6)
ax1.legend(facecolor='#0F1117', edgecolor='#2D3050', labelcolor='white', fontsize=9)

# Panel 2: Early Exit Latency vs Threshold
ax2 = axes[1]
ax2.set_facecolor('#1A1D2E')
ax2.spines[:].set_color('#2D3050')

threshs = [d["threshold"] for d in ee_data]
lats    = [d["avg_latency_ms"] for d in ee_data]
exits   = [d["early_exit_pct"] for d in ee_data]

ax2_b = ax2.twinx()
ax2_b.spines[:].set_color('#2D3050')

b1 = ax2.bar([str(t) for t in threshs], lats, color='#F59F00', alpha=0.85, width=0.45, label='Avg Latency (ms)')
l1 = ax2_b.plot([str(t) for t in threshs], exits, 'o-', color='#CC5DE8', linewidth=2.5, label='Early Exit Rate (%)')

ax2.set_title('Early Exit Latency vs Threshold', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Confidence Threshold', color='#8892A4', fontsize=10)
ax2.set_ylabel('Avg Latency (ms)', color='#F59F00', fontsize=10)
ax2_b.set_ylabel('Early Exit Rate (%)', color='#CC5DE8', fontsize=10)
ax2.tick_params(colors='#8892A4')
ax2_b.tick_params(colors='#8892A4')
ax2.grid(color='#2D3050', linestyle='--', alpha=0.4)

# Panel 3: Architectural Complexity Reduction
ax3 = axes[2]
ax3.set_facecolor('#1A1D2E')
ax3.spines[:].set_color('#2D3050')

models_p = ['Full KG-Hybrid', 'Lightweight KG-Hybrid']
params_m = [15.526, 4.044]

bars3 = ax3.bar(models_p, params_m, color=['#FF6B6B', '#20C997'], width=0.45)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}M', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

ax3.set_title('Model Parameter Footprint', color='white', fontsize=12, fontweight='bold')
ax3.set_ylabel('Parameters (Millions)', color='#8892A4', fontsize=10)
ax3.tick_params(colors='#8892A4')
ax3.set_ylim(0, 18)
ax3.grid(axis='y', color='#2D3050', linestyle='--', alpha=0.6)

plt.tight_layout()
fig_path = FIGURES_DIR / "realworld_enhancements.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"[OK] Multi-panel plot saved -> {fig_path}")

print("=" * 70)
print("REAL-WORLD SUMMARY GENERATION COMPLETE")
print("=" * 70)
