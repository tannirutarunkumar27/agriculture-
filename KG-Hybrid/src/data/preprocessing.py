"""
File: preprocessing.py

Purpose:
Preprocess the Train, Validation and Test datasets.

Pipeline:
1. Load raw split datasets
2. Fit LabelEncoder on training labels
3. Encode validation and test labels
4. Fit StandardScaler on training features only
5. Transform validation and test features
6. Save processed datasets
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Datasets
# =====================================================

train_df = pd.read_csv(INTERIM_DIR / "train_raw.csv")
validation_df = pd.read_csv(INTERIM_DIR / "validation_raw.csv")
test_df = pd.read_csv(INTERIM_DIR / "test_raw.csv")

# =====================================================
# Split Features and Labels
# =====================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_validation = validation_df.drop(columns=["label"])
y_validation = validation_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# =====================================================
# Label Encoding
# =====================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)

y_validation = label_encoder.transform(y_validation)

y_test = label_encoder.transform(y_test)

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_validation = scaler.transform(X_validation)

X_test = scaler.transform(X_test)

# =====================================================
# Convert Back to DataFrames
# =====================================================

X_train = pd.DataFrame(
    X_train,
    columns=train_df.drop(columns=["label"]).columns
)

X_validation = pd.DataFrame(
    X_validation,
    columns=validation_df.drop(columns=["label"]).columns
)

X_test = pd.DataFrame(
    X_test,
    columns=test_df.drop(columns=["label"]).columns
)

# =====================================================
# Add Labels Back
# =====================================================

X_train["label"] = y_train

X_validation["label"] = y_validation

X_test["label"] = y_test

# =====================================================
# Save Processed Datasets
# =====================================================

X_train.to_csv(
    PROCESSED_DIR / "train.csv",
    index=False
)

X_validation.to_csv(
    PROCESSED_DIR / "validation.csv",
    index=False
)

X_test.to_csv(
    PROCESSED_DIR / "test.csv",
    index=False
)

# =====================================================
# Save Preprocessing Objects
# =====================================================

joblib.dump(
    scaler,
    PROCESSED_DIR / "scaler.pkl"
)

joblib.dump(
    label_encoder,
    PROCESSED_DIR / "label_encoder.pkl"
)

# =====================================================
# Summary
# =====================================================

print("=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed Dataset Shapes")

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_validation.shape}")
print(f"Test       : {X_test.shape}")

print("\nFiles Saved")

print(PROCESSED_DIR / "train.csv")
print(PROCESSED_DIR / "validation.csv")
print(PROCESSED_DIR / "test.csv")
print(PROCESSED_DIR / "scaler.pkl")
print(PROCESSED_DIR / "label_encoder.pkl")