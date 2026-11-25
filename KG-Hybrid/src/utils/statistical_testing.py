"""
File: src/utils/statistical_testing.py

Purpose:
Rigorous statistical significance testing for comparing model predictions on 
out-of-distribution external datasets:
1. McNemar's Chi-Squared Test (with continuity correction and exact binomial option)
2. Paired Permutation Test for Macro F1 Differences

Citations:
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
from sklearn.metrics import f1_score


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "KG-Hybrid",
    model_b_name: str = "Baseline"
) -> Dict[str, Any]:
    """
    Perform McNemar's Test on paired predictions.
    
    Contingency Table:
                     Model B Correct  Model B Incorrect
    Model A Correct        n11               n10 (b)
    Model A Incorrect      n01 (c)           n00
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_a = np.asarray(y_pred_a).ravel()
    y_pred_b = np.asarray(y_pred_b).ravel()

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum((~correct_a) & (~correct_b)))
    b = int(np.sum(correct_a & (~correct_b)))  # A correct, B incorrect
    c = int(np.sum((~correct_a) & correct_b))  # A incorrect, B correct

    n_discordant = b + c

    if n_discordant == 0:
        statistic = 0.0
        p_value = 1.0
        test_type = "Identical predictions"
    elif n_discordant < 25:
        # Use exact binomial test
        res = binomtest(b, n_discordant, 0.5, alternative='two-sided')
        statistic = float(b)
        p_value = float(res.pvalue)
        test_type = "Exact Binomial"
    else:
        # Chi-squared test with Edwards continuity correction
        statistic = float((abs(b - c) - 1) ** 2 / n_discordant)
        p_value = float(chi2.sf(statistic, df=1))
        test_type = "Chi-Square (df=1)"

    is_significant = p_value < 0.05

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n_samples": len(y_true),
        "contingency_table": {
            "n11_both_correct": n11,
            "n00_both_incorrect": n00,
            "b_a_correct_b_incorrect": b,
            "c_a_incorrect_b_correct": c
        },
        "statistic": round(statistic, 4),
        "p_value": float(p_value),
        "test_type": test_type,
        "is_statistically_significant": is_significant
    }


def paired_permutation_test_f1(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_permutations: int = 5000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform paired permutation test for macro F1 score difference.
    """
    np.random.seed(random_seed)
    y_true = np.asarray(y_true).ravel()
    y_pred_a = np.asarray(y_pred_a).ravel()
    y_pred_b = np.asarray(y_pred_b).ravel()

    f1_a = float(f1_score(y_true, y_pred_a, average='macro', zero_division=0))
    f1_b = float(f1_score(y_true, y_pred_b, average='macro', zero_division=0))
    observed_diff = f1_a - f1_b

    n_samples = len(y_true)
    perm_diffs = []

    for _ in range(n_permutations):
        # Swap predictions per sample with probability 0.5
        swap_mask = np.random.rand(n_samples) > 0.5
        perm_a = np.where(swap_mask, y_pred_b, y_pred_a)
        perm_b = np.where(swap_mask, y_pred_a, y_pred_b)

        score_a = f1_score(y_true, perm_a, average='macro', zero_division=0)
        score_b = f1_score(y_true, perm_b, average='macro', zero_division=0)
        perm_diffs.append(score_a - score_b)

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

    return {
        "f1_model_a": round(f1_a, 4),
        "f1_model_b": round(f1_b, 4),
        "observed_f1_diff": round(observed_diff, 4),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "is_statistically_significant": p_value < 0.05
    }
