r"""
Agronomic Suitability Membership Functions & Knowledge Vector Generators
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Formulates the continuous Piecewise Trapezoidal-Gaussian Membership Function
S_{c, f}(x_f) \in [0.0, 1.0] and computes feature-wise knowledge gates g(x) \in [0, 1]^7.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Union, Any
from src.knowledge.knowledge_schema import CardinalBounds
from src.knowledge.knowledge_base import KNOWLEDGE_BASE, CROP_CLASSES_ORDERED

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def calculate_trapezoidal_gaussian_suitability(
    x: float,
    min_val: float,
    opt_low: float,
    opt_high: float,
    max_val: float,
    epsilon: float = 0.01
) -> float:
    """
    Computes continuous suitability S(x) in [0.0, 1.0] using a piecewise
    Trapezoidal-Gaussian membership function:
      - 0.0 if x < min_val or x > max_val
      - 1.0 if opt_low <= x <= opt_high
      - Gaussian smooth decay left: exp( -(x - opt_low)^2 / (2 * sigma_left^2) )
      - Gaussian smooth decay right: exp( -(x - opt_high)^2 / (2 * sigma_right^2) )
    """
    if x is None or math.isnan(x):
        return 0.5  # Neutral default for unobserved/missing inputs

    if x < min_val or x > max_val:
        return 0.0

    if opt_low <= x <= opt_high:
        return 1.0

    # Compute Gaussian standard deviation decay parameters
    if x < opt_low:
        denom = opt_low - min_val
        if denom <= 1e-6:
            return 0.0
        sigma_left = denom / math.sqrt(-2.0 * math.log(epsilon))
        val = math.exp(-((x - opt_low) ** 2) / (2.0 * (sigma_left ** 2)))
        return float(np.clip(val, 0.0, 1.0))
    else:  # x > opt_high
        denom = max_val - opt_high
        if denom <= 1e-6:
            return 0.0
        sigma_right = denom / math.sqrt(-2.0 * math.log(epsilon))
        val = math.exp(-((x - opt_high) ** 2) / (2.0 * (sigma_right ** 2)))
        return float(np.clip(val, 0.0, 1.0))


def compute_crop_feature_suitability(
    crop_name: str,
    feature_name: str,
    x_val: float
) -> float:
    """Computes suitability for a single crop and single feature value."""
    if crop_name not in KNOWLEDGE_BASE:
        raise ValueError(f"Crop '{crop_name}' not found in Knowledge Base.")
    profile = KNOWLEDGE_BASE[crop_name]
    bounds = profile.get_bounds_for_feature(feature_name)
    return calculate_trapezoidal_gaussian_suitability(
        x_val, bounds.min_val, bounds.opt_low, bounds.opt_high, bounds.max_val
    )


def compute_crop_suitability_matrix(x_dict_or_array: Union[Dict[str, float], np.ndarray, List[float]]) -> np.ndarray:
    """
    Evaluates input vector x against all 22 crop knowledge profiles.

    Args:
        x_dict_or_array: Either a dictionary with keys N, P, K, temperature, humidity, ph, rainfall
                         or a 7-element numerical list/array in that exact order.

    Returns:
        Matrix K(x) of shape (22, 7) with continuous suitability scores in [0.0, 1.0].
    """
    if isinstance(x_dict_or_array, (list, np.ndarray)):
        if len(x_dict_or_array) != 7:
            raise ValueError(f"Expected 7 feature values, got {len(x_dict_or_array)}")
        x_dict = {f: float(x_dict_or_array[i]) for i, f in enumerate(FEATURE_NAMES)}
    else:
        x_dict = x_dict_or_array

    matrix = np.zeros((len(CROP_CLASSES_ORDERED), 7), dtype=np.float32)

    for i, crop in enumerate(CROP_CLASSES_ORDERED):
        profile = KNOWLEDGE_BASE[crop]
        for j, feature in enumerate(FEATURE_NAMES):
            val = x_dict.get(feature, None)
            matrix[i, j] = compute_crop_feature_suitability(crop, feature, val)

    return matrix


def compute_feature_knowledge_gate(x_dict_or_array: Union[Dict[str, float], np.ndarray, List[float]]) -> np.ndarray:
    """
    Aggregates suitability across crops via max-marginal reduction to derive
    a feature-wise knowledge gating vector g(x) in [0.0, 1.0]^7.

    Returns:
        Vector g(x) of shape (7,) representing overall agronomic plausibility per feature.
    """
    matrix = compute_crop_suitability_matrix(x_dict_or_array)  # Shape (22, 7)
    gate_vector = np.max(matrix, axis=0)  # Shape (7,)
    return gate_vector
