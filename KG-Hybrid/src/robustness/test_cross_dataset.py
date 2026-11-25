"""
Robustness Test 3: Cross-Dataset Generalization
================================================
Evaluates each model under:
  A. Standard test set
  B. Validation set  (slight distribution shift proxy)
  C. Extrapolation: features scaled x1.5 beyond training range
  D. Single-feature corruption: zero out one feature at a time
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.robustness.robustness_utils import DATA_PATH, load_all_models

RESULTS_PATH = PROJECT_ROOT / "outputs" / "robustness"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def acc(preds, y):
    return float(np.mean(preds == y))


def main():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("ROBUSTNESS TEST 3: Cross-Dataset / Generalization Analysis")
    print("=" * 70)

    print("\nLoading data...")
    train_df = pd.read_csv(DATA_PATH / "train.csv")
    val_df   = pd.read_csv(DATA_PATH / "validation.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_val   = val_df.drop("label",   axis=1).values.astype(np.float32)
    y_val   = val_df["label"].values.astype(np.int64)
    X_test  = test_df.drop("label",  axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    models = load_all_models(X_train, y_train)

    # ── A: Standard Test Set ──────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Standard Test Set")
    print("=" * 70)
    exp_a = {}
    for name, fn in models.items():
        a = acc(fn(X_test), y_test)
        exp_a[name] = a
        print(f"  {name:<20}: {a:.4f}")

    # ── B: Validation Set ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Validation Set (Domain Shift Proxy)")
    print("=" * 70)
    exp_b = {}
    for name, fn in models.items():
        a = acc(fn(X_val), y_val)
        exp_b[name] = a
        print(f"  {name:<20}: {a:.4f}")

    # ── C: Extrapolation x1.5 ────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Extrapolation (Features x1.5 Beyond Training Range)")
    print("=" * 70)
    X_extrap = (X_test * 1.5).astype(np.float32)
    exp_c = {}
    for name, fn in models.items():
        a = acc(fn(X_extrap), y_test)
        exp_c[name] = a
        print(f"  {name:<20}: {a:.4f}")

    # ── D: Single-Feature Corruption ─────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Single-Feature Corruption (each feature zeroed)")
    print("=" * 70)
    n_features = X_test.shape[1]
    exp_d = {name: [] for name in models}
    for fi in range(n_features):
        X_c = X_test.copy()
        X_c[:, fi] = 0.0
        for name, fn in models.items():
            exp_d[name].append(round(acc(fn(X_c), y_test), 4))
    print(f"\n  {'Model':<20} {'Min Acc':>8} {'Avg Acc':>8} {'Max Drop':>10}")
    print("  " + "-" * 52)
    for name in models:
        accs = exp_d[name]
        baseline = exp_a[name]
        min_a = min(accs)
        avg_a = float(np.mean(accs))
        max_d = min_a - baseline
        print(f"  {name:<20} {min_a:>8.4f} {avg_a:>8.4f} {max_d:>+10.4f}")

    # ── Full summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL GENERALIZATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Test':>8} {'Val':>8} {'x1.5':>8} {'Worst-Feat':>12}")
    print("-" * 64)
    for name in ["KG-Hybrid", "XGBoost", "FT-Transformer", "Random Forest"]:
        if name not in exp_a:
            continue
        wf = min(exp_d[name]) if exp_d[name] else 0.0
        print(f"{name:<20} {exp_a[name]:>8.4f} {exp_b[name]:>8.4f} "
              f"{exp_c[name]:>8.4f} {wf:>12.4f}")
    print("-" * 64)

    summary = {
        "standard_test":     {k: round(v, 4) for k, v in exp_a.items()},
        "validation_set":    {k: round(v, 4) for k, v in exp_b.items()},
        "extrapolation_1_5": {k: round(v, 4) for k, v in exp_c.items()},
        "single_feature_min":{k: round(min(v), 4) for k, v in exp_d.items()},
        "single_feature_all": exp_d
    }
    out_file = RESULTS_PATH / "cross_dataset_results.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved -> {out_file}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    main()
