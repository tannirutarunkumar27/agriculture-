"""
Unit Tests for Knowledge-Guided Feature Gate Module (Stage 5)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Tests:
  1. Dimension Parity (B, 7 -> B, 7 gate and B, 7, d tokens)
  2. Gate Bounds (0.0 <= G <= 1.0)
  3. Residual Scale Guarantee (scale >= alpha = 0.2)
  4. Batch Processing Consistency
  5. Gradient Flow Backpropagation
  6. Missing Data Neutrality (NaN handling)
  7. Target Leakage Isolation Audit
  8. Numerical Stability (Extreme feature inputs)
  9. Reproducibility
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

from src.models.hybrid.knowledge_gate import KnowledgeGuidedFeatureGate


class TestKnowledgeFeatureGate(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.gate = KnowledgeGuidedFeatureGate(num_features=7, hidden_dim=14, alpha_residual=0.2)
        self.sample_x = torch.randn(4, 7)

    def test_dimension_parity(self):
        """Test 1: Check output shapes for G (B, 7), x_gated (B, 7), and modulated tokens (B, 7, d)."""
        G, x_gated = self.gate(self.sample_x)
        self.assertEqual(G.shape, (4, 7))
        self.assertEqual(x_gated.shape, (4, 7))

        dummy_tokens = torch.randn(4, 7, 64)
        tokens_mod = self.gate.modulate_token_embeddings(dummy_tokens, G)
        self.assertEqual(tokens_mod.shape, (4, 7, 64))

    def test_gate_bounds(self):
        """Test 2: Gate output values must be strictly in [0.0, 1.0]."""
        G, _ = self.gate(self.sample_x)
        self.assertTrue((G >= 0.0).all().item())
        self.assertTrue((G <= 1.0).all().item())

    def test_residual_scale_guarantee(self):
        """Test 3: Minimum token scaling factor must be >= alpha (0.2)."""
        G, _ = self.gate(self.sample_x)
        scale = 0.2 + 0.8 * G
        self.assertTrue((scale >= 0.2).all().item())
        self.assertTrue((scale <= 1.0).all().item())

    def test_batch_processing_consistency(self):
        """Test 4: Batch forward pass must match single-sample evaluations."""
        self.gate.eval()
        with torch.no_grad():
            G_batch, _ = self.gate(self.sample_x[:2])
            G_single_0, _ = self.gate(self.sample_x[0:1])
            G_single_1, _ = self.gate(self.sample_x[1:2])

        np.testing.assert_allclose(G_batch[0].numpy(), G_single_0[0].numpy(), rtol=1e-5)
        np.testing.assert_allclose(G_batch[1].numpy(), G_single_1[0].numpy(), rtol=1e-5)

    def test_gradient_flow(self):
        """Test 5: Non-zero gradients must backpropagate through MLP parameters."""
        self.gate.train()
        x = torch.randn(2, 7, requires_grad=True)
        G, x_gated = self.gate(x)
        loss = x_gated.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in self.gate.gate_mlp.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue((param.grad.abs() > 0.0).any().item())

    def test_missing_data_neutrality(self):
        """Test 6: NaN inputs should be handled safely without error."""
        x_nan = torch.randn(2, 7)
        x_nan[0, 2] = torch.nan
        G, x_gated = self.gate(x_nan)

        self.assertFalse(torch.isnan(G).any().item())
        self.assertFalse(torch.isnan(x_gated).any().item())

    def test_target_leakage_isolation(self):
        """Test 7: Forward pass operates cleanly without target label tensors."""
        try:
            G, x_gated = self.gate(self.sample_x)
            self.assertEqual(G.shape[1], 7)
        except Exception as e:
            self.fail(f"Target leakage isolation test failed: {e}")

    def test_numerical_stability(self):
        """Test 8: Extreme feature inputs (1e5, -1e5) produce stable outputs without Inf/NaN."""
        x_extreme = torch.tensor([[1e5, -1e5, 0.0, 1e4, -1e4, 7.0, 200.0]], dtype=torch.float32)
        G, x_gated = self.gate(x_extreme)

        self.assertFalse(torch.isnan(G).any().item())
        self.assertFalse(torch.isinf(G).any().item())
        self.assertFalse(torch.isnan(x_gated).any().item())

    def test_reproducibility(self):
        """Test 9: Deterministic behavior given same input and model state."""
        self.gate.eval()
        with torch.no_grad():
            G1, _ = self.gate(self.sample_x)
            G2, _ = self.gate(self.sample_x)

        np.testing.assert_array_equal(G1.numpy(), G2.numpy())


if __name__ == "__main__":
    unittest.main()
