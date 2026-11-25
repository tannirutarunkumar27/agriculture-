"""
Robustness Test 1: Missing Features
====================================
Tests how well each model handles randomly masked input features
at 0%, 20%, 40%, and 60% missing ratios.
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


def evaluate_with_missing(predict_fn, X_test, y_test,
                           missing_ratios=(0.0, 0.2, 0.4, 0.6),
                           n_trials=5, seed=42):
    results = []
    rng = np.random.default_rng(seed)
    for ratio in missing_ratios:
        accs = []
        for _ in range(n_trials):
            X_m = X_test.copy().astype(np.float32)
            if ratio > 0:
                mask = rng.random(X_m.shape) < ratio
                X_m[mask] = 0.0
            preds = predict_fn(X_m)
            accs.append(float(np.mean(preds == y_test)))
        acc = float(np.mean(accs))
        results.append({'missing_ratio': ratio, 'accuracy': acc})
        print(f"    Missing {int(ratio*100):3d}%: Accuracy = {acc:.4f}  (avg {n_trials} trials)")
    return results


def main():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("ROBUSTNESS TEST 1: Missing Features Analysis")
    print("=" * 70)

    print("\nLoading data...")
    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_test  = test_df.drop("label",  axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    predict_fns = load_all_models(X_train, y_train)

    missing_ratios = [0.0, 0.2, 0.4, 0.6]
    all_results    = {}

    print("\n" + "=" * 70)
    print("Running Missing-Features Tests (5 random trials each)...")
    print("=" * 70)

    for name, fn in predict_fns.items():
        print(f"\n>> {name}:")
        all_results[name] = evaluate_with_missing(fn, X_test, y_test, missing_ratios)

    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            f"{int(r['missing_ratio']*100)}%": round(r['accuracy'], 4)
            for r in results
        }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'0%':>8} {'20%':>8} {'40%':>8} {'60%':>8}  "
          f"{'Drop@40%':>10}  {'Drop@60%':>10}")
    print("-" * 80)
    for mn in ["KG-Hybrid", "XGBoost", "FT-Transformer", "Random Forest"]:
        if mn not in summary:
            continue
        row = summary[mn]
        d40 = row['40%'] - row['0%']
        d60 = row['60%'] - row['0%']
        print(f"{mn:<20} {row['0%']:>8.4f} {row['20%']:>8.4f} {row['40%']:>8.4f} "
              f"{row['60%']:>8.4f}  {d40:>+10.4f}  {d60:>+10.4f}")
    print("-" * 80)

    kg_d40  = summary['KG-Hybrid']['40%']    - summary['KG-Hybrid']['0%']
    xgb_d40 = summary['XGBoost']['40%']      - summary['XGBoost']['0%']
    rf_d40  = summary['Random Forest']['40%'] - summary['Random Forest']['0%']

    print(f"\nPerformance drop at 40% missing features:")
    print(f"   KG-Hybrid     : {kg_d40:+.4f}")
    print(f"   XGBoost       : {xgb_d40:+.4f}")
    print(f"   Random Forest : {rf_d40:+.4f}")

    if kg_d40 > xgb_d40 and kg_d40 > rf_d40:
        print("\n[WARNING] KG-Hybrid drops MORE than simpler models under missing data.")
    else:
        print("\n[OK] KG-Hybrid is MORE robust than at least one baseline!")

    out_file = RESULTS_PATH / "missing_features_results.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved -> {out_file}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    main()
