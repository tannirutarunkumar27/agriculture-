"""
File: evaluation.py

Purpose:
Reusable evaluation utilities for all machine learning models.
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# Project Directories
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = TABLE_DIR / "model_comparison.csv"


# =====================================================
# Evaluation Function
# =====================================================

def evaluate_model(model_name, y_true, y_pred):
    """
    Evaluate a classification model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    y_true : array-like
        Actual labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }

    # -------------------------------------------------
    # Print Metrics
    # -------------------------------------------------

    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report\n")
    print(report)

    # Save TXT
    with open(
        TABLE_DIR / f"{model_name}_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # Save CSV
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    report_df.to_csv(
        TABLE_DIR / f"{model_name}_classification_report.csv",
        index=True
    )

    # -------------------------------------------------
    # Save Metrics JSON
    # -------------------------------------------------

    with open(
        TABLE_DIR / f"{model_name}_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    # -------------------------------------------------
    # Append Metrics to Comparison Table
    # -------------------------------------------------

    comparison_df = pd.DataFrame([metrics])

    if COMPARISON_FILE.exists():
        comparison_df.to_csv(
            COMPARISON_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        comparison_df.to_csv(
            COMPARISON_FILE,
            index=False
        )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        cbar=True
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print("\nResults saved successfully.")
    print(f"Figures : {FIGURE_DIR}")
    print(f"Tables  : {TABLE_DIR}")

    return metrics
