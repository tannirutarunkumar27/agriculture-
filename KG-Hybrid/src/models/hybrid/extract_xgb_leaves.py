import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"

# 1. Load XGBoost model
xgb_model = joblib.load(OUTPUT_MODELS / "xgboost_model.pkl")
booster = xgb_model.get_booster()

# 2. Load training data (scaled, as XGBoost was trained on scaled data)
train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
X_train = train_df.drop("label", axis=1)

# 3. Extract leaf indices
dmatrix = xgb.DMatrix(X_train)
leaf_indices = booster.predict(dmatrix, pred_leaf=True)  # Shape: (1540, 200)

# 4. Find unique leaves
num_unique_leaves = int(leaf_indices.max()) + 1
print(f"Leaf indices shape: {leaf_indices.shape}")
print(f"Unique leaves across all trees: {num_unique_leaves}")

# 5. Save leaf indices
np.save(DATA_PROCESSED / "xgb_train_leaves.npy", leaf_indices)
print(f"[OK] Leaf indices saved to {DATA_PROCESSED / 'xgb_train_leaves.npy'}")

# 6. Save metadata
with open(DATA_PROCESSED / "xgb_leaf_metadata.txt", "w") as f:
    f.write(f"num_unique_leaves = {num_unique_leaves}\n")
    f.write(f"num_trees = {leaf_indices.shape[1]}\n")
    f.write(f"num_samples = {leaf_indices.shape[0]}\n")

print(f"[OK] Metadata saved to {DATA_PROCESSED / 'xgb_leaf_metadata.txt'}")
