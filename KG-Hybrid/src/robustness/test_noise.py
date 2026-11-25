"""
Robustness Test 2: Gaussian Noise Injection
=============================================
Tests how well each model handles Gaussian noise at sigma = 0.0, 0.1, 0.2, 0.3.
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


def evaluate_with_noise(predict_fn, X_test, y_test,
                         noise_levels=(0.0, 0.1, 0.2, 0.3),
                         n_trials=5, seed=42):
    results = []
    rng = np.random.default_rng(seed)
    for sigma in noise_levels:
        accs = []
        for _ in range(n_trials):
            X_n = X_test.copy().astype(np.float32)
            if sigma > 0:
                X_n += rng.normal(0, sigma, X_n.shape).astype(np.float32)
            preds = predict_fn(X_n)
            accs.append(float(np.mean(preds == y_test)))
        acc = float(np.mean(accs))
        results.append({'noise_sigma': sigma, 'accuracy': acc})
        print(f"    sigma={sigma:.2f}: Accuracy = {acc:.4f}  (avg {n_trials} trials)")
    return results


def main():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("ROBUSTNESS TEST 2: Gaussian Noise Injection")
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

    noise_levels = [0.0, 0.1, 0.2, 0.3]
    all_results  = {}

    print("\n" + "=" * 70)
    print("Running Noise-Injection Tests (5 random trials each)...")
    print("=" * 70)

    for name, fn in predict_fns.items():
        print(f"\n>> {name}:")
        all_results[name] = evaluate_with_noise(fn, X_test, y_test, noise_levels)

    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            f"s={r['noise_sigma']:.1f}": round(r['accuracy'], 4)
            for r in results
        }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'s=0.0':>8} {'s=0.1':>8} {'s=0.2':>8} {'s=0.3':>8}  "
          f"{'Drop@0.2':>10}  {'Drop@0.3':>10}")
    print("-" * 82)
    for mn in ["KG-Hybrid", "XGBoost", "FT-Transformer", "Random Forest"]:
        if mn not in summary:
            continue
        row = summary[mn]
        d2  = row['s=0.2'] - row['s=0.0']
        d3  = row['s=0.3'] - row['s=0.0']
        print(f"{mn:<20} {row['s=0.0']:>8.4f} {row['s=0.1']:>8.4f} "
              f"{row['s=0.2']:>8.4f} {row['s=0.3']:>8.4f}  {d2:>+10.4f}  {d3:>+10.4f}")
    print("-" * 82)

    kg_d  = summary['KG-Hybrid']['s=0.2']    - summary['KG-Hybrid']['s=0.0']
    xgb_d = summary['XGBoost']['s=0.2']      - summary['XGBoost']['s=0.0']
    rf_d  = summary['Random Forest']['s=0.2'] - summary['Random Forest']['s=0.0']

    print(f"\nPerformance drop at sigma=0.2 noise:")
    print(f"   KG-Hybrid     : {kg_d:+.4f}")
    print(f"   XGBoost       : {xgb_d:+.4f}")
    print(f"   Random Forest : {rf_d:+.4f}")

    if kg_d > xgb_d and kg_d > rf_d:
        print("\n[WARNING] KG-Hybrid degrades MORE under noise than simpler models.")
    else:
        print("\n[OK] KG-Hybrid is MORE noise-robust than at least one baseline!")

    out_file = RESULTS_PATH / "noise_results.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved -> {out_file}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    main()
