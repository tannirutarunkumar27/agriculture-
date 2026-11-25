"""
Baseline Model:
Logistic Regression
"""

from pathlib import Path
import sys

# -----------------------------------------------------
# Add project root to Python path
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.evaluation import evaluate_model

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# -----------------------------------------------------
# Split Features and Labels
# -----------------------------------------------------

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# -----------------------------------------------------
# Build Model
# -----------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("Training Logistic Regression...")

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

model.fit(X_train, y_train)

print("Training Completed.")

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

metrics = evaluate_model(
    "logistic_regression",
    y_test,
    predictions
)

print("\nFinal Metrics")
print(metrics)