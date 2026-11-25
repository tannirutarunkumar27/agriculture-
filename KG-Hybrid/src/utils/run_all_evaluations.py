"""
File: run_all_evaluations.py

Purpose:
Master execution script that runs all baseline models, FT-Transformer evaluation,
Hybrid-Concat ablation model, and generates all evaluation metrics, confusion matrices,
training plots, and model comparison charts.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe"

def run_script(rel_path):
    script_path = PROJECT_ROOT / rel_path
    print("=" * 70)
    print(f"RUNNING: {rel_path}")
    print("=" * 70)
    cmd = [str(PYTHON_EXE), str(script_path)]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR/LOGS:")
        print(res.stderr)
    if res.returncode != 0:
        print(f"[ERROR] {rel_path} failed with return code {res.returncode}")
    else:
        print(f"[OK] {rel_path} completed successfully.")
    print("\n")

def main():
    print("\n[START] STARTING ALL MODEL EVALUATIONS & PLOT GENERATION...\n")

    # 1. Baseline Models
    run_script("src/models/baselines/decision_tree.py")
    run_script("src/models/baselines/logistic_regression.py")
    run_script("src/models/baselines/random_forest.py")
    run_script("src/models/baselines/xgboost_model.py")

    # 2. Hybrid Baseline (Concat)
    run_script("src/models/hybrid/train_baseline_hybrid_concat.py")

    # 3. FT-Transformer Training History Plot
    run_script("src/utils/plot_training_history.py")

    # 4. Model Comparison Plots & Tables
    run_script("src/utils/model_comparison_plot.py")

    print("\n[FINISHED] ALL MODEL EVALUATIONS AND PLOTS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
