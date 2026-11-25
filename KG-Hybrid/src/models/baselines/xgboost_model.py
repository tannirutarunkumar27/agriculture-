"""
File: xgboost_model.py

Purpose:
Train and evaluate the XGBoost classifier.
"""

from pathlib import Path
import sys
import pandas as pd

from xgboost import XGBClassifier

# -----------------------------------------------------
# Project Root
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.utils.evaluation import evaluate_model

# -----------------------------------------------------
# Data Paths
# -----------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# -----------------------------------------------------
# Build Model
# -----------------------------------------------------

model = XGBClassifier(
    objective="multi:softmax",
    num_class=22,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    tree_method="hist"
)

# -----------------------------------------------------
# Train
# -----------------------------------------------------

print("Training XGBoost...")

model.fit(X_train, y_train)

print("Training Completed.")

# Save model artifact
import joblib
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUTPUT_MODELS / "xgboost_model.pkl")
print(f"Saved model to {OUTPUT_MODELS / 'xgboost_model.pkl'}")

# -----------------------------------------------------
# Predict
# -----------------------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

metrics = evaluate_model(
    "xgboost",
    y_test,
    predictions
)

print("\nFinal Metrics")
print(metrics)