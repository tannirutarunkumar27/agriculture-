"""
test_lightweight.py
===================
Evaluates the Lightweight KG-Hybrid model across all 4 benchmark pillars:
  1. Clean Accuracy (%)
  2. Missing Features at 40% ratio WITH KNN Imputation (%)
  3. Gaussian Noise Injection (sigma = 0.2) (%)
  4. Feature Extrapolation (x1.5 Range Shift) (%)
"""

import numpy as np
import pandas as pd
import torch
import joblib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.impute import KNNImputer
from src.models.hybrid.lightweight_kg_hybrid import LightweightKGHybrid
from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, FEATURES, TOTAL_UNIQUE_LEAVES, NUM_TREES, get_leaf_indices_with_offset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("==================================================")
    print("EVALUATING LIGHTWEIGHT KG-HYBRID (4.04M Parameters)")
    print("==================================================")

    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    X_test  = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)

    xgb_model = joblib.load(MODEL_PATH / "xgboost_model.pkl")
    booster   = xgb_model.get_booster()

    km = np.load(DATA_PATH / "knowledge_matrix.npy")
    model = LightweightKGHybrid(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES,
        num_trees=NUM_TREES,
        embed_dim=32,
        num_classes=22,
        knowledge_matrix=km
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH / "lightweight_kg_hybrid_best.pth", map_location=DEVICE))
    model.eval()

    def predict_lightweight(X_arr):
        leaves_t = get_leaf_indices_with_offset(X_arr, booster).to(DEVICE)
        X_t      = torch.tensor(X_arr, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(leaves_t, X_t)
            return logits.argmax(dim=1).cpu().numpy()

    # 1. Clean Accuracy
    clean_acc = np.mean(predict_lightweight(X_test) == y_test) * 100

    # 2. Missing 40% WITH KNN Imputation
    knn_imp = KNNImputer(n_neighbors=5).fit(X_train)
    rng = np.random.default_rng(42)
    acc_missing = []
    for _ in range(5):
        mask = rng.random(X_test.shape) < 0.40
        X_nan = X_test.copy()
        X_nan[mask] = np.nan
        X_imp = knn_imp.transform(X_nan)
        acc_missing.append(np.mean(predict_lightweight(X_imp) == y_test))
    missing_acc = np.mean(acc_missing) * 100

    # 3. Noise (sigma = 0.2)
    acc_noise = []
    for _ in range(5):
        noise = rng.normal(0, 0.2, size=X_test.shape).astype(np.float32)
        X_noisy = X_test + noise
        acc_noise.append(np.mean(predict_lightweight(X_noisy) == y_test))
    noise_acc = np.mean(acc_noise) * 100

    # 4. Feature Extrapolation (x1.5)
    X_extrap = X_test * 1.5
    extrap_acc = np.mean(predict_lightweight(X_extrap) == y_test) * 100

    results = {
        "model": "Lightweight KG-Hybrid",
        "parameters": "4,044,134",
        "clean_accuracy": round(clean_acc, 2),
        "missing_40pct_knn_acc": round(missing_acc, 2),
        "noise_sigma_0.2_acc": round(noise_acc, 2),
        "extrapolation_x1.5_acc": round(extrap_acc, 2)
    }

    print("\n--------------------------------------------------")
    print(f"Clean Accuracy           : {clean_acc:.2f}%")
    print(f"Missing 40% + KNN Acc   : {missing_acc:.2f}%")
    print(f"Noise s=0.2 Accuracy     : {noise_acc:.2f}%")
    print(f"Extrapolation x1.5 Acc   : {extrap_acc:.2f}%")
    print("--------------------------------------------------")

    out_file = MODEL_PATH / "lightweight_kg_hybrid_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Lightweight metrics saved -> {out_file}")

if __name__ == "__main__":
    main()
