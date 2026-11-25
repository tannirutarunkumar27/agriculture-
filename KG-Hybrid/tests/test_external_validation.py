"""
File: tests/test_external_validation.py

Purpose:
Unittest test suite for validating external dataset ingestion, distribution shift metrics,
and statistical significance testing logic.
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.distribution_shift import (
    calculate_psi,
    calculate_jsd,
    calculate_mahalanobis_shift,
    compute_distribution_shift_metrics
)
from src.utils.statistical_testing import (
    run_mcnemar_test,
    paired_permutation_test_f1
)
from src.experiments.external_validation import load_external_dataset_agrifield, FEATURES


class TestExternalValidation(unittest.TestCase):

    def test_calculate_psi_identical_distributions(self):
        """Identical distributions should have PSI close to zero."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        target = ref.copy()

        psi_val = calculate_psi(ref, target)
        self.assertGreaterEqual(psi_val, 0.0)
        self.assertLess(psi_val, 0.05, f"PSI should be < 0.05 for identical distributions, got {psi_val}")

    def test_calculate_psi_shifted_distributions(self):
        """Shifted distributions should have high PSI (> 0.25)."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        target = np.random.normal(5, 1, 1000)

        psi_val = calculate_psi(ref, target)
        self.assertGreaterEqual(psi_val, 0.25, f"Shifted PSI should be >= 0.25, got {psi_val}")

    def test_calculate_jsd_bounds(self):
        """JSD should be bounded in [0, 1]."""
        ref = np.random.normal(0, 1, 500)
        target = np.random.normal(2, 1, 500)

        jsd_val = calculate_jsd(ref, target)
        self.assertTrue(0.0 <= jsd_val <= 1.0, f"JSD out of bounds: {jsd_val}")

    def test_mahalanobis_shift_calculation(self):
        """Verify Mahalanobis distance calculation on simple data."""
        ref_df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=['a', 'b', 'c'])
        target_df = pd.DataFrame(np.random.normal(3, 1, (50, 3)), columns=['a', 'b', 'c'])

        dist = calculate_mahalanobis_shift(ref_df, target_df)
        self.assertGreater(dist, 0.0, "Mahalanobis distance should be positive.")

    def test_mcnemar_test_identical_preds(self):
        """McNemar test with identical predictions should yield p_value = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred_a = np.array([0, 1, 0, 0, 1, 1])
        y_pred_b = np.array([0, 1, 0, 0, 1, 1])

        res = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertEqual(res['p_value'], 1.0)
        self.assertFalse(res['is_statistically_significant'])

    def test_paired_permutation_test(self):
        """Test paired permutation test logic."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred_a = np.array([0, 1, 1, 0, 1, 0, 0, 1])  # perfect
        y_pred_b = np.array([1, 0, 0, 1, 0, 1, 1, 0])  # all wrong

        res = paired_permutation_test_f1(y_true, y_pred_a, y_pred_b, n_permutations=200)
        self.assertGreater(res['observed_f1_diff'], 0)
        self.assertIn('p_value', res)

    def test_external_dataset_ingestion(self):
        """Verify AgriField dataset loads non-empty overlapping samples and correct features."""
        X_ext, y_ext = load_external_dataset_agrifield(align_units=False)
        self.assertFalse(X_ext.empty, "External dataset features dataframe should not be empty.")
        self.assertFalse(y_ext.empty, "External dataset target series should not be empty.")
        self.assertEqual(list(X_ext.columns), FEATURES, f"Features must match schema {FEATURES}")

    def test_saved_external_report_uses_distinct_neural_predictions(self):
        """FT and KG results must not be copied from the XGBoost baseline."""
        report_path = PROJECT_ROOT / "outputs" / "reports" / "external_validation_report.json"
        self.assertTrue(report_path.exists(), "Run external_validation.py before publication")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for regime in ("regime_a_raw", "regime_b_aligned"):
            rows = {row["Model"]: row for row in report[regime]["metrics"]}
            xgb = rows["XGBoostBaseline"]
            for neural_name in ("FTTransformerBaseline", "KG-HybridProposed"):
                neural = rows[neural_name]
                metric_vector = (
                    neural["External_Accuracy"], neural["External_F1_Macro"],
                    neural["External_Top3_Accuracy"], neural["External_ECE"],
                )
                xgb_vector = (
                    xgb["External_Accuracy"], xgb["External_F1_Macro"],
                    xgb["External_Top3_Accuracy"], xgb["External_ECE"],
                )
                self.assertNotEqual(
                    metric_vector, xgb_vector,
                    f"{regime}: {neural_name} appears to be an XGBoost prediction proxy",
                )


if __name__ == "__main__":
    unittest.main()
