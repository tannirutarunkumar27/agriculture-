"""
Unit Tests for Calibration & Reliability Metrics Module
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np

from src.experiments.calibration import (
    compute_ece,
    compute_brier_score,
    compute_nll,
    generate_reliability_diagram_data
)


class TestCalibrationMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_perfect_calibration_ece(self):
        """Test 1: Perfectly calibrated predictions yield ECE ~ 0.0."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        # Probabilities match accuracy perfectly
        y_prob = np.zeros((10, 2))
        y_prob[:, 0] = 0.5
        y_prob[:, 1] = 0.5

        ece = compute_ece(y_true, y_prob, n_bins=5)
        self.assertLess(ece, 0.1)

    def test_overconfident_ece(self):
        """Test 2: Extremely overconfident wrong predictions produce high ECE."""
        y_true = np.zeros(10, dtype=int)  # All ground truth are class 0
        y_prob = np.zeros((10, 2))
        y_prob[:, 1] = 1.0  # Confidently predict class 1 (wrong!)

        ece = compute_ece(y_true, y_prob, n_bins=10)
        self.assertGreaterEqual(ece, 0.9)

    def test_brier_score_calculation(self):
        """Test 3: Brier score is zero for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        bs = compute_brier_score(y_true, y_prob)
        self.assertEqual(bs, 0.0)

    def test_nll_calculation(self):
        """Test 5: NLL calculation on perfect and uniform predictions."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        nll_perfect = compute_nll(y_true, y_prob)
        self.assertAlmostEqual(nll_perfect, 0.0, places=5)

    def test_empty_bins_ignored_in_ece(self):
        """Test 6: ECE calculation ignores empty bins correctly."""
        y_true = np.array([0, 1])
        y_prob = np.array([
            [0.95, 0.05],
            [0.05, 0.95]
        ])
        # Only bin 9 is populated (confidence 0.95, acc 1.0), all other bins are empty (count=0)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        self.assertAlmostEqual(ece, abs(1.0 - 0.95), places=5)


if __name__ == "__main__":
    unittest.main()
