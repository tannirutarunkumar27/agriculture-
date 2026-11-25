"""
Unit Tests for Agricultural Knowledge Representation Engine
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Tests:
  1. Optimal Range Parity (S = 1.0)
  2. Extreme Range Zeroing (S = 0.0)
  3. Gaussian Decay Smoothness (0 < S < 1)
  4. Monotonicity across bounds
  5. Tensor Dimensions (1D and 2D Batch)
  6. Missing Data Neutrality (S = 0.5 for NaN)
  7. Boundary Continuity (Zero-width edge case handling)
  8. PyTorch Projection Encoder Dimensions (B, 64)
  9. Target Leakage Isolation Audit
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import unittest
import numpy as np
import torch

from src.knowledge.knowledge_schema import CardinalBounds
from src.knowledge.suitability import calculate_trapezoidal_gaussian_suitability
from src.knowledge.representation import (
    KnowledgeRepresentationEngine,
    KnowledgeProjectionEncoder
)


class TestKnowledgeRepresentation(unittest.TestCase):

    def setUp(self):
        self.engine = KnowledgeRepresentationEngine()
        # Typical Rice optimal sample: N=80, P=40, K=40, T=25, H=80, pH=6.5, Rain=200
        self.rice_optimal_sample = [80.0, 40.0, 40.0, 25.0, 80.0, 6.5, 200.0]

    def test_optimal_range_parity(self):
        """Test 1: Values strictly within [opt_low, opt_high] must yield S = 1.0."""
        s_val = calculate_trapezoidal_gaussian_suitability(
            x=25.0, min_val=20.0, opt_low=22.0, opt_high=27.0, max_val=32.0
        )
        self.assertEqual(s_val, 1.0)

    def test_extreme_range_zeroing(self):
        """Test 2: Values outside [min_val, max_val] must yield S = 0.0."""
        s_low = calculate_trapezoidal_gaussian_suitability(
            x=15.0, min_val=20.0, opt_low=22.0, opt_high=27.0, max_val=32.0
        )
        s_high = calculate_trapezoidal_gaussian_suitability(
            x=35.0, min_val=20.0, opt_low=22.0, opt_high=27.0, max_val=32.0
        )
        self.assertEqual(s_low, 0.0)
        self.assertEqual(s_high, 0.0)

    def test_gaussian_decay_smoothness(self):
        """Test 3: Values in transition zones must yield 0.0 < S < 1.0."""
        s_val = calculate_trapezoidal_gaussian_suitability(
            x=21.0, min_val=20.0, opt_low=22.0, opt_high=27.0, max_val=32.0
        )
        self.assertGreater(s_val, 0.0)
        self.assertLess(s_val, 1.0)

    def test_monotonicity(self):
        """Test 4: Suitability must monotonically increase left and decrease right."""
        s1 = calculate_trapezoidal_gaussian_suitability(20.5, 20.0, 22.0, 27.0, 32.0)
        s2 = calculate_trapezoidal_gaussian_suitability(21.5, 20.0, 22.0, 27.0, 32.0)
        self.assertLess(s1, s2)

        s3 = calculate_trapezoidal_gaussian_suitability(28.0, 20.0, 22.0, 27.0, 32.0)
        s4 = calculate_trapezoidal_gaussian_suitability(30.0, 20.0, 22.0, 27.0, 32.0)
        self.assertGreater(s3, s4)

    def test_tensor_dimensions(self):
        """Test 5: Check single sample (22, 7) and batch (B, 22, 7) outputs."""
        # Single Sample
        mat_single = self.engine.get_compatibility_matrix(self.rice_optimal_sample)
        gate_single = self.engine.get_feature_gate_vector(self.rice_optimal_sample)
        over_single = self.engine.get_overall_compatibility(self.rice_optimal_sample)

        self.assertEqual(mat_single.shape, (22, 7))
        self.assertEqual(gate_single.shape, (7,))
        self.assertEqual(over_single.shape, (22,))

        # Batch (Batch Size = 5)
        batch_input = np.tile(self.rice_optimal_sample, (5, 1))
        mat_batch = self.engine.get_compatibility_matrix(batch_input)
        gate_batch = self.engine.get_feature_gate_vector(batch_input)
        over_batch = self.engine.get_overall_compatibility(batch_input)

        self.assertEqual(mat_batch.shape, (5, 22, 7))
        self.assertEqual(gate_batch.shape, (5, 7))
        self.assertEqual(over_batch.shape, (5, 22))

    def test_missing_data_neutrality(self):
        """Test 6: Missing NaN values return neutral suitability 0.5."""
        s_nan = calculate_trapezoidal_gaussian_suitability(
            x=math.nan, min_val=20.0, opt_low=22.0, opt_high=27.0, max_val=32.0
        )
        self.assertEqual(s_nan, 0.5)

    def test_boundary_continuity(self):
        """Test 7: Zero-width boundary opt_low == min_val does not raise DivisionByZero."""
        s_val = calculate_trapezoidal_gaussian_suitability(
            x=20.0, min_val=20.0, opt_low=20.0, opt_high=25.0, max_val=30.0
        )
        self.assertEqual(s_val, 1.0)

    def test_pytorch_projection_dimensions(self):
        """Test 8: PyTorch KnowledgeProjectionEncoder outputs shape (B, 64)."""
        encoder = KnowledgeProjectionEncoder(input_dim=154, hidden_dim=128, output_dim=64)
        dummy_matrix = torch.randn(8, 22, 7)
        embedding = encoder(dummy_matrix)
        self.assertEqual(embedding.shape, (8, 64))

    def test_leakage_audit(self):
        """Test 9: Verify representation engine runs without receiving target label vectors."""
        # Ensure method signature does not require label vectors
        sample = np.random.randn(7)
        try:
            gate = self.engine.get_feature_gate_vector(sample)
            self.assertEqual(len(gate), 7)
        except Exception as e:
            self.fail(f"Leakage audit failed: {e}")


if __name__ == "__main__":
    unittest.main()
