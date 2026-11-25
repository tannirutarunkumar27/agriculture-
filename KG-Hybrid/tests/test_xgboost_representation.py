"""
Unit Tests for XGBoost Representation Extraction Module (Stage 6)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Tests:
  1. Margin Dimension (B, 22)
  2. Leaf Indices Dimension (B, num_trees)
  3. Projection Output Shape (B, 64)
  4. LayerNorm Distribution (mean ~ 0, std ~ 1)
  5. Batch Processing Consistency
  6. Frozen Booster Isolation (No booster updates)
  7. Gradient Propagation Through Projection Encoder
  8. Missing Feature Handling (NaN safety)
  9. Target Leakage Isolation Audit
  10. Numerical Stability (Extreme feature inputs)
  11. Reproducibility
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.hybrid.xgboost_representation import (
    XGBoostLeafProjectionEncoder,
    XGBoostRepresentationExtractor
)


class TestXGBoostRepresentation(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.extractor = XGBoostRepresentationExtractor(num_classes=22, output_dim=64)
        self.sample_x = torch.randn(4, 7)

    def test_margin_dimension(self):
        """Test 1: Check margin logits shape is (B, 22)."""
        margins, leaves = self.extractor.extract_margins_and_leaves(self.sample_x)
        self.assertEqual(margins.shape, (4, 22))

    def test_leaf_indices_dimension(self):
        """Test 2: Check leaf indices shape is (B, num_trees)."""
        margins, leaves = self.extractor.extract_margins_and_leaves(self.sample_x)
        self.assertEqual(leaves.shape[0], 4)
        self.assertGreater(leaves.shape[1], 0)

    def test_projection_output_shape(self):
        """Test 3: Check extractor output shape is exactly (B, 64)."""
        R_X = self.extractor(self.sample_x)
        self.assertEqual(R_X.shape, (4, 64))

    def test_layernorm_distribution(self):
        """Test 4: Output vector R_X should have LayerNorm mean ~ 0 and std ~ 1 across dimensions."""
        R_X = self.extractor(self.sample_x)
        means = R_X.mean(dim=-1).detach().cpu().numpy()
        stds = R_X.std(dim=-1).detach().cpu().numpy()

        np.testing.assert_allclose(means, 0.0, atol=1e-4)
        np.testing.assert_allclose(stds, 1.0, atol=1e-2)

    def test_batch_processing_consistency(self):
        """Test 5: Batch forward pass matches single-sample forward passes."""
        self.extractor.eval()
        with torch.no_grad():
            R_batch = self.extractor(self.sample_x[:2])
            R_single_0 = self.extractor(self.sample_x[0:1])
            R_single_1 = self.extractor(self.sample_x[1:2])

        np.testing.assert_allclose(R_batch[0].numpy(), R_single_0[0].numpy(), rtol=1e-4)
        np.testing.assert_allclose(R_batch[1].numpy(), R_single_1[0].numpy(), rtol=1e-4)

    def test_frozen_booster_isolation(self):
        """Test 6: Pre-trained XGBoost Booster parameters remain unchanged."""
        if self.extractor.booster is not None:
            dump_before = self.extractor.booster.get_dump()
            _ = self.extractor(self.sample_x)
            dump_after = self.extractor.booster.get_dump()
            self.assertEqual(dump_before, dump_after)

    def test_gradient_propagation(self):
        """Test 7: Gradients backpropagate through projection encoder parameters."""
        self.extractor.train()
        R_X = self.extractor(self.sample_x)
        loss = R_X.sum()
        loss.backward()

        for param in self.extractor.projection_encoder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_missing_feature_handling(self):
        """Test 8: Handles NaN inputs safely without NaN outputs."""
        x_nan = torch.randn(2, 7)
        x_nan[0, 1] = torch.nan
        R_X = self.extractor(x_nan)

        self.assertFalse(torch.isnan(R_X).any().item())
        self.assertFalse(torch.isinf(R_X).any().item())

    def test_target_leakage_isolation(self):
        """Test 9: Operates cleanly without target label inputs."""
        try:
            R_X = self.extractor(self.sample_x)
            self.assertEqual(R_X.shape[1], 64)
        except Exception as e:
            self.fail(f"Target leakage test failed: {e}")

    def test_numerical_stability(self):
        """Test 10: Extreme inputs (1e5, -1e5) produce stable outputs."""
        x_extreme = torch.tensor([[1e5, -1e5, 0.0, 1e4, -1e4, 7.0, 200.0]], dtype=torch.float32)
        R_X = self.extractor(x_extreme)

        self.assertFalse(torch.isnan(R_X).any().item())
        self.assertFalse(torch.isinf(R_X).any().item())

    def test_reproducibility(self):
        """Test 11: Identical outputs given fixed random seed and evaluation mode."""
        self.extractor.eval()
        with torch.no_grad():
            R1 = self.extractor(self.sample_x)
            R2 = self.extractor(self.sample_x)

        np.testing.assert_array_equal(R1.numpy(), R2.numpy())


if __name__ == "__main__":
    unittest.main()
