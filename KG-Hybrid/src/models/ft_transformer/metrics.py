"""
metrics.py

Evaluation metrics for FT-Transformer training.

This module computes:
- Accuracy
- Precision
- Recall
- F1-Score

These metrics are calculated during validation after each epoch.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    y_true : list or numpy.ndarray
        Ground truth labels.

    y_pred : list or numpy.ndarray
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return metrics


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        labels=sorted(set(y_true)),
        target_names=target_names,
        zero_division=0
    )


def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


def print_metrics(metrics):
    """
    Pretty-print evaluation metrics.
    """

    print("\nEvaluation Metrics")
    print("-" * 40)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")

    print("-" * 40)


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a classification report to console.
    """

    report = get_classification_report(
        y_true,
        y_pred,
        target_names=target_names
    )

    print("\nClassification Report")
    print(report)


def print_confusion_matrix(y_true, y_pred, labels=None):
    """
    Print a confusion matrix to console.
    """

    matrix = get_confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix")
    print(matrix)


if __name__ == "__main__":

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 2]

    metrics = calculate_metrics(y_true, y_pred)

    print_metrics(metrics)