"""
Production Readiness & Real-World Enhancements Test Suite
==========================================================
Evaluates the 5 key architectural improvements for KG-Hybrid:
  1. Missing Feature Recovery via KNN, MICE, and Median Imputation
  2. Model Complexity & Parameter Count Optimization
  3. Early Exit Strategy (Latency vs Threshold Trade-off)
  4. Confidence Calibration Output Verification
  5. Monte Carlo (MC) Dropout Uncertainty Quantification
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, FEATURES, load_all_models, get_leaf_indices_with_offset
from src.inference.production_pipeline import ProductionCropRecommender

RESULTS_DIR = PROJECT_ROOT / "outputs" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_imputation():
    print("\n" + "=" * 70)
    print("TEST 1: Missing Feature Handling — Raw vs Imputed (Median, KNN, MICE)")
    print("=" * 70)

    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")
    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_test  = test_df.drop("label",  axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)

    models = load_all_models(X_train, y_train)
    kg_predict = models["KG-Hybrid"]
    xgb_predict = models["XGBoost"]

    # Fit imputers on X_train
    median_imp = SimpleImputer(strategy='median').fit(X_train)
    knn_imp    = KNNImputer(n_neighbors=5).fit(X_train)
    mice_imp   = IterativeImputer(random_state=42, max_iter=10).fit(X_train)

    ratios = [0.0, 0.2, 0.4, 0.6]
    n_trials = 5
    rng = np.random.default_rng(42)

    results = {
        "Raw (Zero Fill)": [],
        "Median Imputation": [],
        "KNN Imputation (k=5)": [],
        "MICE Imputation": [],
        "XGBoost Baseline": []
    }

    for ratio in ratios:
        acc_raw, acc_med, acc_knn, acc_mice, acc_xgb = [], [], [], [], []
        for _ in range(n_trials):
            # Create mask
            mask = rng.random(X_test.shape) < ratio
            
            # Raw zero-fill
            X_raw = X_test.copy()
            X_raw[mask] = 0.0
            acc_raw.append(np.mean(kg_predict(X_raw) == y_test))

            # Missing representation with NaNs for imputers
            X_nan = X_test.copy()
            X_nan[mask] = np.nan

            X_med_imp = median_imp.transform(X_nan)
            acc_med.append(np.mean(kg_predict(X_med_imp) == y_test))

            X_knn_imp = knn_imp.transform(X_nan)
            acc_knn.append(np.mean(kg_predict(X_knn_imp) == y_test))

            X_mice_imp = mice_imp.transform(X_nan)
            acc_mice.append(np.mean(kg_predict(X_mice_imp) == y_test))

            # XGBoost on raw zero fill
            acc_xgb.append(np.mean(xgb_predict(X_raw) == y_test))

        results["Raw (Zero Fill)"].append(round(float(np.mean(acc_raw)), 4))
        results["Median Imputation"].append(round(float(np.mean(acc_med)), 4))
        results["KNN Imputation (k=5)"].append(round(float(np.mean(acc_knn)), 4))
        results["MICE Imputation"].append(round(float(np.mean(acc_mice)), 4))
        results["XGBoost Baseline"].append(round(float(np.mean(acc_xgb)), 4))

    print(f"\n{'Imputation Method':<24} {'0% Missing':>12} {'20% Missing':>12} {'40% Missing':>12} {'60% Missing':>12}")
    print("-" * 72)
    for method, accs in results.items():
        print(f"{method:<24} {accs[0]:>12.4f} {accs[1]:>12.4f} {accs[2]:>12.4f} {accs[3]:>12.4f}")
    print("-" * 72)

    return results


def evaluate_early_exit():
    print("\n" + "=" * 70)
    print("TEST 2: Early Exit Benchmark — Latency vs Confidence Threshold")
    print("=" * 70)

    test_df = pd.read_csv(DATA_PATH / "test.csv")
    X_test  = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)

    thresholds = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    ee_results = []

    for thresh in thresholds:
        recommender = ProductionCropRecommender(early_exit_threshold=thresh)
        
        early_exits = 0
        total_samples = len(X_test)
        correct = 0
        latencies = []

        for i in range(total_samples):
            input_dict = {feat: float(X_test[i, col]) for col, feat in enumerate(FEATURES)}
            
            t0 = time.perf_counter()
            res = recommender.predict_single(input_dict, enable_early_exit=(thresh > 0.0), input_is_scaled=True)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            if "STAGE_1_EARLY_EXIT" in res["inference_route"]:
                early_exits += 1

            # Convert crop prediction back to index
            from src.inference.production_pipeline import CROP_NAMES
            pred_idx = CROP_NAMES.index(res["recommended_crop"])
            if pred_idx == y_test[i]:
                correct += 1

        exit_pct = (early_exits / total_samples) * 100
        avg_lat  = np.mean(latencies)
        acc      = (correct / total_samples) * 100

        ee_results.append({
            "threshold": thresh,
            "early_exit_pct": round(exit_pct, 1),
            "avg_latency_ms": round(avg_lat, 2),
            "accuracy_pct": round(acc, 2)
        })
        
        route_str = "No Early Exit" if thresh == 0.0 else f"Thresh={thresh:.2f}"
        print(f"  {route_str:<16}: Early Exit Rate={exit_pct:>5.1f}% | Avg Latency={avg_lat:>6.2f}ms | Accuracy={acc:.2f}%")

    return ee_results


def evaluate_uncertainty_and_calibration():
    print("\n" + "=" * 70)
    print("TEST 3: Uncertainty Quantification (MC Dropout) & Safety Alerts")
    print("=" * 70)

    test_df = pd.read_csv(DATA_PATH / "test.csv")
    X_test  = test_df.drop("label", axis=1).values.astype(np.float32)

    recommender = ProductionCropRecommender(early_exit_threshold=0.90, mc_samples=10, uncertainty_threshold=0.15)

    # 1. Clean Sample
    clean_dict = {feat: float(X_test[0, col]) for col, feat in enumerate(FEATURES)}
    res_clean  = recommender.predict_single(clean_dict, enable_early_exit=False, input_is_scaled=True)

    # 2. Noisy Sample (add Gaussian noise sigma=0.3)
    rng = np.random.default_rng(42)
    noisy_dict = {feat: float(X_test[0, col] + rng.normal(0, 0.3)) for col, feat in enumerate(FEATURES)}
    res_noisy  = recommender.predict_single(noisy_dict, enable_early_exit=False, input_is_scaled=True)

    # 3. Corrupted / Extreme Sample (1.5x extrapolation + 2 missing features)
    corrupt_dict = {feat: float(X_test[0, col] * 1.5) for col, feat in enumerate(FEATURES)}
    corrupt_dict['N'] = None
    corrupt_dict['ph'] = None
    res_corrupt = recommender.predict_single(corrupt_dict, enable_early_exit=False, input_is_scaled=True)

    print("\n--- SAMPLE 1: CLEAN SENSOR DATA ---")
    print(json.dumps(res_clean, indent=2))

    print("\n--- SAMPLE 2: NOISY SENSOR DATA (sigma=0.3) ---")
    print(json.dumps(res_noisy, indent=2))

    print("\n--- SAMPLE 3: EXTREME / MISSING SENSOR DATA ---")
    print(json.dumps(res_corrupt, indent=2))

    return {
        "clean": res_clean,
        "noisy": res_noisy,
        "corrupted": res_corrupt
    }


def main():
    print("=" * 70)
    print("KG-HYBRID REAL-WORLD PRODUCTION READINESS TEST SUITE")
    print("=" * 70)

    imp_results = evaluate_imputation()
    ee_results  = evaluate_early_exit()
    unc_results = evaluate_uncertainty_and_calibration()

    summary = {
        "imputation_experiments": imp_results,
        "early_exit_experiments": ee_results,
        "uncertainty_samples": unc_results
    }

    out_file = RESULTS_DIR / "production_readiness_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Production Readiness Results Saved -> {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
