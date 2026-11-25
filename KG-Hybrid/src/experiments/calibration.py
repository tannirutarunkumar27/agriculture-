"""
Model Calibration & Reliability Metrics Module (Stage 7 - Calibration)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Computes:
  - Expected Calibration Error (ECE)
  - Multi-Class Brier Score
  - Reliability Diagram Data Points (Confidence vs Accuracy per bin)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from typing import Dict, List, Tuple, Any


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Validates y_true and y_prob dimensions and values."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        raise TypeError("y_true and y_prob must be numpy arrays.")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D array of shape (N,), got shape {y_true.shape}")
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D array of shape (N, C), got shape {y_prob.shape}")
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_prob has {len(y_prob)}")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains non-finite values (NaN or Inf).")
    if np.any(y_prob < -1e-6) or np.any(y_prob > 1.0 + 1e-6):
        raise ValueError("Probabilities in y_prob must be bounded in [0.0, 1.0].")
    row_sums = np.sum(y_prob, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Rows of y_prob must sum to approximately 1.0.")


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins.

    Args:
        y_true: Ground truth class indices (N,).
        y_prob: Predicted probability distribution (N, C).
        n_bins: Number of equal-width probability bins (default: 10).

    Returns:
        ece: Float value in [0, 1].
    """
    _validate_inputs(y_true, y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in bin i
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return round(float(ece), 6)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Computes multi-class Brier Score:
      BS = (1/N) * sum_i sum_k (p_{i,k} - y_{i,k})^2
    """
    _validate_inputs(y_true, y_prob)
    n_samples, n_classes = y_prob.shape
    # One-hot encode ground truth labels
    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_one_hot[np.arange(n_samples), y_true] = 1.0

    brier_score = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
    return round(float(brier_score), 6)


def compute_nll(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    Computes Multi-Class Negative Log-Likelihood (NLL) on true class probabilities:
      NLL = -(1/N) * sum_i log(p_{i, y_i} + epsilon)

    Args:
        y_true: Ground truth class indices (N,).
        y_prob: Predicted probability distribution (N, C).
        epsilon: Numerical stability constant (default: 1e-12).

    Returns:
        nll: Float scalar value >= 0.
    """
    _validate_inputs(y_true, y_prob)
    n_samples = len(y_true)
    # Extract probability assigned to the ground-truth class
    p_true = y_prob[np.arange(n_samples), y_true]
    nll = -float(np.mean(np.log(np.clip(p_true, epsilon, 1.0))))
    return round(nll, 6)


def generate_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> List[Dict[str, float]]:
    """
    Generates reliability diagram data points (confidence vs accuracy).
    """
    _validate_inputs(y_true, y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        bin_lower = float(bin_boundaries[i])
        bin_upper = float(bin_boundaries[i + 1])

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0

        bins_data.append({
            "bin_index": i,
            "bin_lower": round(bin_lower, 2),
            "bin_upper": round(bin_upper, 2),
            "accuracy": round(bin_acc, 6),
            "confidence": round(bin_conf, 6),
            "sample_count": bin_size
        })

    return bins_data
