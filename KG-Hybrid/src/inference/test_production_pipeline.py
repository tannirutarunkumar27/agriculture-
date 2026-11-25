"""
test_production_pipeline.py
============================
Tests the full production crop recommender (KNN Imputation + Early Exit + Lightweight Model)
across the test set (330 samples) and runs complete robustness benchmark suite.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.production_pipeline import ProductionCropRecommender, CROP_NAMES
from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, FEATURES

def main():
    print("==================================================")
    print("TESTING FULL PRODUCTION PIPELINE (330 Test Samples)")
    print("==================================================")

    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")
    X_test   = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test   = test_df["label"].values.astype(np.int64)

    pipeline = ProductionCropRecommender(
        use_knn=True,
        early_exit_threshold=0.90,
        use_lightweight=True
    )

    # 1. Clean Test Set Benchmark
    correct = 0
    early_exits = 0
    latencies = []
    safety_alerts = 0

    for i in range(len(X_test)):
        sample_dict = {feat: float(X_test[i, col]) for col, feat in enumerate(FEATURES)}
        
        t0 = time.perf_counter()
        res = pipeline.predict_single(sample_dict, enable_early_exit=True, input_is_scaled=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        pred_idx = CROP_NAMES.index(res["recommended_crop"])
        if pred_idx == y_test[i]:
            correct += 1

        if "STAGE_1_EARLY_EXIT" in res["inference_route"]:
            early_exits += 1

        if res["safety_alert"] is not None:
            safety_alerts += 1

    clean_acc = (correct / len(X_test)) * 100
    avg_lat   = np.mean(latencies)
    exit_pct  = (early_exits / len(X_test)) * 100

    print(f"Clean Test Accuracy     : {clean_acc:.2f}%")
    print(f"Average Inference Latency: {avg_lat:.2f} ms")
    print(f"Early Exit Rate (Stage 1): {exit_pct:.2f}%")
    print(f"Safety Alerts Triggered  : {safety_alerts}")

    # 2. Production Pipeline Robustness Suite
    print("\n--------------------------------------------------")
    print("RUNNING ROBUSTNESS SUITE ON PRODUCTION PIPELINE")
    print("--------------------------------------------------")

    # A. 40% Missing Features WITH KNN Imputation
    rng = np.random.default_rng(42)
    missing_correct = []
    for _ in range(5):
        m_corr = 0
        for i in range(len(X_test)):
            sample_dict = {feat: float(X_test[i, col]) for col, feat in enumerate(FEATURES)}
            # Randomly mask 40% of features
            for feat in FEATURES:
                if rng.random() < 0.40:
                    sample_dict[feat] = None
            res = pipeline.predict_single(sample_dict, enable_early_exit=True, input_is_scaled=True)
            pred_idx = CROP_NAMES.index(res["recommended_crop"])
            if pred_idx == y_test[i]:
                m_corr += 1
        missing_correct.append(m_corr / len(X_test))
    missing_acc = np.mean(missing_correct) * 100

    # B. Gaussian Noise (sigma = 0.2)
    noise_correct = []
    for _ in range(5):
        n_corr = 0
        for i in range(len(X_test)):
            sample_dict = {feat: float(X_test[i, col] + rng.normal(0, 0.2)) for col, feat in enumerate(FEATURES)}
            res = pipeline.predict_single(sample_dict, enable_early_exit=True, input_is_scaled=True)
            pred_idx = CROP_NAMES.index(res["recommended_crop"])
            if pred_idx == y_test[i]:
                n_corr += 1
        noise_correct.append(n_corr / len(X_test))
    noise_acc = np.mean(noise_correct) * 100

    # C. Feature Extrapolation (x1.5)
    e_corr = 0
    for i in range(len(X_test)):
        sample_dict = {feat: float(X_test[i, col] * 1.5) for col, feat in enumerate(FEATURES)}
        res = pipeline.predict_single(sample_dict, enable_early_exit=True, input_is_scaled=True)
        pred_idx = CROP_NAMES.index(res["recommended_crop"])
        if pred_idx == y_test[i]:
            e_corr += 1
    extrap_acc = (e_corr / len(X_test)) * 100

    print(f"40% Missing + KNN Acc   : {missing_acc:.2f}%")
    print(f"Noise s=0.2 Accuracy     : {noise_acc:.2f}%")
    print(f"Extrapolation x1.5 Acc   : {extrap_acc:.2f}%")
    print("--------------------------------------------------")

    results = {
        "pipeline": "Full Production Recommender (KNN + Early Exit + Lightweight)",
        "clean_accuracy": round(clean_acc, 2),
        "missing_40pct_knn_acc": round(missing_acc, 2),
        "noise_sigma_0.2_acc": round(noise_acc, 2),
        "extrapolation_x1.5_acc": round(extrap_acc, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "early_exit_rate_pct": round(exit_pct, 2),
        "safety_alerts_count": safety_alerts
    }

    out_file = MODEL_PATH / "production_pipeline_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Full pipeline evaluation saved -> {out_file}")

if __name__ == "__main__":
    main()
