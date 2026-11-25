"""
File: kg_hybrid.py

Purpose:
KG-Hybrid Crop Recommendation Model.

Architecture:
A two-level stacking ensemble that combines the strengths of
multiple heterogeneous base learners with a meta-learner.

Level 1 — Base Models (Diverse Learners):
    • Logistic Regression  (linear boundary)
    • Decision Tree        (non-linear, axis-aligned)
    • Random Forest        (bagging ensemble)
    • XGBoost              (boosting ensemble)

Level 2 — Meta-Learner:
    • Logistic Regression trained on the out-of-fold predictions
      produced by the Level-1 models.

Why Stacking?
    Each base model captures different aspects of the data.
    The meta-learner learns which model to trust for which
    region of the input space, producing a stronger prediction.

Pipeline:
    ┌─────────────────────────────────────────────┐
    │           LEVEL 1 — BASE MODELS             │
    │  Logistic Reg │ Decision Tree │ RF │ XGBoost│
    └───────────┬───────────────────────────────┘
                │  Out-of-fold predictions (4 cols)
                ▼
    ┌─────────────────────────────────────────────┐
    │         LEVEL 2 — META-LEARNER              │
    │          Logistic Regression                │
    └─────────────────────────────────────────────┘
                │
                ▼
        Final Crop Prediction
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline       import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.evaluation import evaluate_model

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed — skipping XGBoost base model.")

# =====================================================
# Paths
# =====================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_SAVE_DIR = PROJECT_ROOT / "data" / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


# =====================================================
# Load Data
# =====================================================

def load_data():
    """Load processed train and test sets."""

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test  = test_df.drop(columns=["label"])
    y_test  = test_df["label"]

    return X_train, y_train, X_test, y_test


# =====================================================
# Build KG-Hybrid Stacking Model
# =====================================================

def build_kg_hybrid() -> StackingClassifier:
    """
    Build the KG-Hybrid stacking ensemble.

    Returns
    -------
    StackingClassifier
        Fully configured stacking model ready to fit.
    """

    # ─── Level 1: Base Estimators ─────────────────────────────────

    base_estimators = [

        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        ),

        (
            "decision_tree",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=15
            )
        ),

        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        ),
    ]

    if XGBOOST_AVAILABLE:
        base_estimators.append((
            "xgboost",
            XGBClassifier(
                objective="multi:softmax",
                num_class=22,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                tree_method="hist",
                verbosity=0
            )
        ))

    # ─── Level 2: Meta-Learner ────────────────────────────────────

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )

    # ─── Stacking Classifier ──────────────────────────────────────

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,                    # 5-fold cross-validation for OOF
        stack_method="predict",  # use class predictions (not proba)
        n_jobs=-1,
        passthrough=False        # only OOF predictions fed to meta
    )

    return model


# =====================================================
# Cross-Validation
# =====================================================

def cross_validate_model(model, X_train, y_train) -> np.ndarray:
    """
    Run 5-fold stratified cross-validation on training set.

    Returns
    -------
    np.ndarray
        Per-fold accuracy scores.
    """

    print("\nRunning 5-Fold Cross-Validation on training set...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"\nCV Fold Accuracies : {scores}")
    print(f"Mean CV Accuracy   : {scores.mean():.4f}")
    print(f"Std CV Accuracy    : {scores.std():.4f}")

    return scores


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 70)
    print("KG-HYBRID — STACKING ENSEMBLE")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────
    print("\nLoading processed data...")
    X_train, y_train, X_test, y_test = load_data()

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Classes          : {y_train.nunique()}")

    # ─── Build Model ──────────────────────────────────────────────
    print("\nBuilding KG-Hybrid stacking model...")
    model = build_kg_hybrid()

    n_base = len(model.estimators)
    print(f"Base models      : {n_base}")
    for name, _ in model.estimators:
        print(f"  • {name}")
    print(f"Meta-learner     : LogisticRegression")
    print(f"Cross-val folds  : 5")

    # ─── Cross-Validation ─────────────────────────────────────────
    cv_scores = cross_validate_model(model, X_train, y_train)

    # ─── Train on full training set ───────────────────────────────
    print("\n" + "=" * 70)
    print("Training KG-Hybrid on full training set...")
    print("=" * 70)

    model.fit(X_train, y_train)

    print("Training Completed.")

    # ─── Predict ──────────────────────────────────────────────────
    predictions = model.predict(X_test)

    # ─── Evaluate ─────────────────────────────────────────────────
    metrics = evaluate_model(
        "kg_hybrid",
        y_test,
        predictions
    )

    print("\nFinal Metrics")
    print(metrics)

    # ─── Save Model ───────────────────────────────────────────────
    model_path = MODEL_SAVE_DIR / "kg_hybrid_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # ─── Save CV Results ──────────────────────────────────────────
    import json
    cv_path = PROJECT_ROOT / "outputs" / "tables" / "kg_hybrid_cv_scores.json"
    with open(cv_path, "w") as f:
        json.dump({
            "cv_fold_scores": cv_scores.tolist(),
            "mean_cv_accuracy": round(float(cv_scores.mean()), 4),
            "std_cv_accuracy":  round(float(cv_scores.std()), 4),
        }, f, indent=4)
    print(f"CV scores saved : {cv_path}")

    return model, metrics


if __name__ == "__main__":
    main()
