"""
Statistical Aggregator & Significance Calculator (Stage 7 - Multi-Seed)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Computes:
  - Sample Mean, Standard Deviation, Min, Max
  - 95% Confidence Intervals via Student's t-distribution
  - Paired Wilcoxon Signed-Rank Test across seed metric vectors
  - McNemar's Non-Parametric Contingency Test on itemized test predictions
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional


def calculate_summary_statistics(
    values: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Computes mean, std, min, max, and 95% Student's t-distribution confidence interval.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    mean_val = float(np.mean(arr))
    if n == 1:
        return {
            "mean": round(mean_val, 6),
            "std": 0.0,
            "min": round(mean_val, 6),
            "max": round(mean_val, 6),
            "ci_lower": round(mean_val, 6),
            "ci_upper": round(mean_val, 6)
        }

    std_val = float(np.std(arr, ddof=1))  # Unbiased sample std
    sem = std_val / math.sqrt(n)          # Standard error of the mean

    # Student's t multiplier for 95% CI with df = n - 1
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin_err = t_crit * sem

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "ci_lower": round(mean_val - margin_err, 6),
        "ci_upper": round(mean_val + margin_err, 6)
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_proposed: np.ndarray
) -> Dict[str, Any]:
    """
    Executes McNemar's non-parametric test on itemized predictions.

    Contingency table:
      b: Baseline Incorrect & Proposed Correct
      c: Baseline Correct & Proposed Incorrect
    """
    correct_base = (y_pred_baseline == y_true)
    correct_prop = (y_pred_proposed == y_true)

    b = int(np.sum(~correct_base & correct_prop))
    c = int(np.sum(correct_base & ~correct_prop))
    a = int(np.sum(correct_base & correct_prop))
    d = int(np.sum(~correct_base & ~correct_prop))

    if (b + c) == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Edwards' continuity correction
        chi2 = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val = float(stats.chi2.sf(chi2, df=1))

    return {
        "contingency_matrix": {"a_both_correct": a, "b_base_wrong_prop_right": b, "c_base_right_prop_wrong": c, "d_both_wrong": d},
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 6),
        "statistically_significant_005": bool(p_val < 0.05)
    }


def compute_wilcoxon_test(
    metrics_baseline: List[float],
    metrics_proposed: List[float]
) -> Dict[str, Any]:
    """
    Executes paired Wilcoxon signed-rank test across 5 seed metric values.
    """
    base_arr = np.asarray(metrics_baseline, dtype=np.float64)
    prop_arr = np.asarray(metrics_proposed, dtype=np.float64)
    diff = prop_arr - base_arr

    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "statistically_significant_005": False}

    try:
        stat, p_val = stats.wilcoxon(base_arr, prop_arr, zero_method='pratt')
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant_005": bool(p_val < 0.05)
        }
    except Exception:
        # Fallback to paired t-test if diffs are too small
        stat, p_val = stats.ttest_rel(base_arr, prop_arr)
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant_005": bool(p_val < 0.05)
        }


def aggregate_experiment_results(
    raw_results: Dict[str, Dict[int, Dict[str, float]]]
) -> Dict[str, Any]:
    """
    Aggregates metrics across models and seeds into summary metrics dictionary.

    Args:
        raw_results: {model_name: {seed: {metric_name: value}}}

    Returns:
        aggregated_summary: {model_name: {metric_name: {mean, std, min, max, ci_lower, ci_upper}}}
    """
    summary = {}
    for model_name, seed_dict in raw_results.items():
        summary[model_name] = {}
        # Collect all metric keys
        first_seed = next(iter(seed_dict.values()))
        for metric_key in first_seed.keys():
            if isinstance(first_seed[metric_key], (int, float)):
                metric_vals = [seed_dict[s][metric_key] for s in seed_dict]
                summary[model_name][metric_key] = calculate_summary_statistics(metric_vals)

    return summary
