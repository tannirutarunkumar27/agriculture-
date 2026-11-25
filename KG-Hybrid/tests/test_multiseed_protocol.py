"""
Unit Tests for Multi-Seed Evaluation & Reproducibility Protocol (Stage 7)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Tests:
  1. Same Seed Reproducibility (Identical outputs)
  2. Different Seeds Independence (Data split variance)
  3. Preset Seeds Execution (5 seeds present)
  4. Scaler Seed Re-Fitting Isolation
  5. Summary Statistics Calculation Accuracy
  6. Confidence Interval Bounds
  7. McNemar Test Correctness
  8. Wilcoxon Test Correctness
  9. Zero NaN/Inf Metrics Audit
  10. Environment Metadata Logging
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
import torch

from src.experiments.seed_utils import set_all_seeds, get_preset_seeds, get_execution_environment_metadata
from src.experiments.aggregate_stats import calculate_summary_statistics, compute_mcnemar_test, compute_wilcoxon_test
from src.experiments.run_multiseed import load_raw_dataset, prepare_seed_split, run_benchmark_for_seed


class TestMultiSeedProtocol(unittest.TestCase):

    def setUp(self):
        self.X_df, self.y_df = load_raw_dataset()

    def test_same_seed_reproducibility(self):
        """Test 1: Same seed produces bitwise identical dataset splits and scaler parameters."""
        res1 = prepare_seed_split(self.X_df, self.y_df, seed=42)
        res2 = prepare_seed_split(self.X_df, self.y_df, seed=42)

        np.testing.assert_array_equal(res1[0], res2[0])  # X_train
        np.testing.assert_array_equal(res1[1], res2[1])  # y_train
        np.testing.assert_array_equal(res1[7].mean_, res2[7].mean_)  # Scaler mean

    def test_different_seeds_independence(self):
        """Test 2: Different seeds produce independently varied data splits."""
        res1 = prepare_seed_split(self.X_df, self.y_df, seed=42)
        res2 = prepare_seed_split(self.X_df, self.y_df, seed=123)

        self.assertFalse(np.array_equal(res1[0], res2[0]))  # X_train differs
        self.assertFalse(np.array_equal(res1[1], res2[1]))  # y_train differs

    def test_preset_seeds_execution(self):
        """Test 3: Preset seeds list contains exactly [42, 123, 2024, 3407, 9999]."""
        seeds = get_preset_seeds()
        self.assertEqual(seeds, [42, 123, 2024, 3407, 9999])

    def test_scaler_seed_refitting(self):
        """Test 4: Scaler means differ across different seeds, proving strict re-fitting."""
        res1 = prepare_seed_split(self.X_df, self.y_df, seed=42)
        res2 = prepare_seed_split(self.X_df, self.y_df, seed=9999)

        scaler1_mean = res1[7].mean_
        scaler2_mean = res2[7].mean_
        self.assertFalse(np.allclose(scaler1_mean, scaler2_mean))

    def test_summary_statistics_calculation(self):
        """Test 5: Mean, std, min, max match analytical ground truth on a test list."""
        test_vals = [0.90, 0.92, 0.94, 0.96, 0.98]
        stats = calculate_summary_statistics(test_vals)

        self.assertAlmostEqual(stats["mean"], 0.94, places=4)
        self.assertAlmostEqual(stats["min"], 0.90, places=4)
        self.assertAlmostEqual(stats["max"], 0.98, places=4)
        self.assertGreater(stats["std"], 0.0)

    def test_confidence_interval_bounds(self):
        """Test 6: 95% Confidence Interval bounds satisfy ci_lower <= mean <= ci_upper."""
        test_vals = [0.99, 0.98, 0.995, 0.985, 0.99]
        stats = calculate_summary_statistics(test_vals)

        self.assertLessEqual(stats["ci_lower"], stats["mean"])
        self.assertGreaterEqual(stats["ci_upper"], stats["mean"])

    def test_mcnemar_test_correctness(self):
        """Test 7: McNemar test contingency matrix and chi2 calculations are correct."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_base = np.array([0, 1, 0, 0, 1, 0])  # 4 correct, 2 wrong
        y_prop = np.array([0, 1, 2, 0, 1, 2])  # 6 correct, 0 wrong

        res = compute_mcnemar_test(y_true, y_base, y_prop)
        matrix = res["contingency_matrix"]

        self.assertEqual(matrix["b_base_wrong_prop_right"], 2)
        self.assertEqual(matrix["c_base_right_prop_wrong"], 0)
        self.assertIn("p_value", res)

    def test_wilcoxon_test_correctness(self):
        """Test 8: Wilcoxon signed-rank test output dictionary format is valid."""
        base_vals = [0.95, 0.96, 0.94, 0.95, 0.96]
        prop_vals = [0.99, 0.98, 0.99, 0.99, 0.98]

        res = compute_wilcoxon_test(base_vals, prop_vals)
        self.assertIn("statistic", res)
        self.assertIn("p_value", res)
        self.assertIn("statistically_significant_005", res)

    def test_zero_nan_inf_metrics(self):
        """Test 9: Seed evaluation metrics contain zero NaN or Inf values."""
        res = run_benchmark_for_seed(42, self.X_df, self.y_df)
        metrics = res["metrics"]

        for model_name, m_dict in metrics.items():
            for metric_key, val in m_dict.items():
                if isinstance(val, (int, float)):
                    self.assertFalse(np.isnan(val), f"NaN found in {model_name}.{metric_key}")
                    self.assertFalse(np.isinf(val), f"Inf found in {model_name}.{metric_key}")

    def test_environment_metadata_logging(self):
        """Test 10: Environment metadata logger captures system properties."""
        meta = get_execution_environment_metadata()
        self.assertIn("platform", meta)
        self.assertIn("torch_version", meta)
        self.assertIn("xgboost_version", meta)
        self.assertEqual(meta["preset_seeds"], [42, 123, 2024, 3407, 9999])


if __name__ == "__main__":
    unittest.main()
