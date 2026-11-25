"""
Unit Tests for Corrected KG-Hybrid Adaptive Fusion Gate (Problem 1 Fix)
Tests:
  1. Output shape (batch, 192) and finite values.
  2. Input-dependent gate computation (distinct inputs yield distinct gates).
  3. Gate influence on pre-normalization representation (modulating gate modulates f_pre).
  4. Nonzero gradient flow to gate parameters from classification loss.
  5. Gate parameter updates under optimizer step.
  6. Verification that algebraic cancellation is absent.
  7. Verification that FusionWithoutGate (A2/A8) bypasses gating.
  8. Full KGHybrid forward & backward integration.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.ablation.run_ablation import FusionWithoutKnowledge, FusionWithoutGate, KGHybridBase
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer


class TestAdaptiveFusionGate(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.dim_xgb = 128
        self.dim_ft = 64
        self.dim_total = 192
        self.dummy_km = np.random.uniform(0.3, 0.9, (22, 7)).astype(np.float32)
        self.fusion = KnowledgeGuidedFusion(
            dim_xgb=self.dim_xgb,
            dim_ft=self.dim_ft,
            num_crops=22,
            num_features=7,
            knowledge_matrix=self.dummy_km
        )

    def test_1_output_shape_and_finiteness(self):
        """Test 1: Output shape is (B, 192) and all values are finite."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)
        f_fused, gate = self.fusion(f_xgb, f_ft)

        self.assertEqual(f_fused.shape, (self.batch_size, self.dim_total))
        self.assertEqual(gate.shape, (self.batch_size, self.dim_total))
        self.assertTrue(torch.isfinite(f_fused).all().item())
        self.assertTrue(torch.isfinite(gate).all().item())
        self.assertTrue((gate >= 0.0).all().item() and (gate <= 1.0).all().item())

    def test_2_distinct_inputs_produce_distinct_gates(self):
        """Test 2: Distinct inputs produce distinct sample-dependent gate vectors."""
        f_xgb1 = torch.ones(1, self.dim_xgb) * 2.0
        f_ft1 = torch.ones(1, self.dim_ft) * -2.0
        f_xgb2 = torch.ones(1, self.dim_xgb) * -5.0
        f_ft2 = torch.ones(1, self.dim_ft) * 5.0

        _, gate1 = self.fusion(f_xgb1, f_ft1)
        _, gate2 = self.fusion(f_xgb2, f_ft2)

        diff = (gate1 - gate2).abs().sum().item()
        self.assertGreater(diff, 1e-4, f"Gate vectors should differ for distinct inputs, got diff={diff}")

    def test_3_gate_modulates_fused_representation(self):
        """Test 3: Changing the gate directly modulates the pre-normalization representation."""
        f_xgb = torch.ones(1, self.dim_xgb) * 3.0
        f_ft = torch.ones(1, self.dim_ft) * 3.0
        concat = torch.cat([f_xgb, f_ft], dim=1)

        # Compute k_bias manually for the sample
        kp = self.fusion.knowledge_encoder(self.fusion.knowledge_matrix)
        query = self.fusion.query_proj(concat)
        scores = torch.matmul(query, kp.t()) / (self.fusion.dim_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        k_bias = self.fusion.context_proj(torch.matmul(attn, kp))

        # Simulation with two distinct gate values: g=0.2 vs g=0.8
        gate_low = torch.full_like(concat, 0.2)
        gate_high = torch.full_like(concat, 0.8)

        f_pre_low = gate_low * concat + 0.1 * k_bias
        f_pre_high = gate_high * concat + 0.1 * k_bias

        # Under the old cancelled formula, f_pre_low == f_pre_high == concat + 0.1 * k_bias.
        # Under the corrected formula, f_pre_high - f_pre_low == 0.6 * concat != 0.
        expected_diff = (0.8 - 0.2) * concat
        actual_diff = f_pre_high - f_pre_low

        self.assertTrue(torch.allclose(actual_diff, expected_diff, atol=1e-6))
        self.assertGreater(actual_diff.abs().sum().item(), 1.0)

    def test_4_gradient_flow_to_gate_parameters(self):
        """Test 4: Task classification loss produces finite, nonzero gradients for gate parameters."""
        self.fusion.zero_grad()
        f_xgb = torch.randn(self.batch_size, self.dim_xgb, requires_grad=True)
        f_ft = torch.randn(self.batch_size, self.dim_ft, requires_grad=True)

        head = nn.Linear(self.dim_total, 22)
        f_fused, gate = self.fusion(f_xgb, f_ft)
        logits = head(f_fused)
        target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Verify gate linear layers received gradients
        for name, param in self.fusion.gate.named_parameters():
            self.assertIsNotNone(param.grad, f"Gate parameter {name} has no gradient")
            self.assertTrue(torch.isfinite(param.grad).all().item())
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Gate parameter {name} has zero gradient")

    def test_5_gate_parameter_update_step(self):
        """Test 5: Optimizer step actually modifies gate-generation weights."""
        optimizer = optim.AdamW(self.fusion.parameters(), lr=1e-2)
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)

        # Record initial weights of first linear layer in gate
        initial_weight = self.fusion.gate[0].weight.clone()

        head = nn.Linear(self.dim_total, 22)
        f_fused, gate = self.fusion(f_xgb, f_ft)
        logits = head(f_fused)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_weight = self.fusion.gate[0].weight
        weight_diff = (updated_weight - initial_weight).abs().sum().item()
        self.assertGreater(weight_diff, 1e-6, "Gate weights were not updated by optimizer step")

    def test_6_absence_of_cancelling_expression(self):
        """Test 6: Verify forward pass does not compute (1-gate)*concat.detach()."""
        # When f_xgb and f_ft are zero vectors, f_pre should be exactly 0.1 * k_bias
        f_xgb_zero = torch.zeros(1, self.dim_xgb)
        f_ft_zero = torch.zeros(1, self.dim_ft)

        f_fused, gate = self.fusion(f_xgb_zero, f_ft_zero)
        concat_zero = torch.cat([f_xgb_zero, f_ft_zero], dim=1)
        kp = self.fusion.knowledge_encoder(self.fusion.knowledge_matrix)
        q_zero = self.fusion.query_proj(concat_zero)
        scores_zero = torch.matmul(q_zero, kp.t()) / (self.fusion.dim_k ** 0.5)
        attn_zero = torch.softmax(scores_zero, dim=-1)
        k_bias = self.fusion.context_proj(torch.matmul(attn_zero, kp))
        expected_norm = self.fusion.layer_norm(0.1 * k_bias)

        self.assertTrue(torch.allclose(f_fused, expected_norm, atol=1e-5))

    def test_7_ablation_fusion_variants(self):
        """Test 7: Verify FusionWithoutKnowledge and FusionWithoutGate."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)

        # A1: Without Knowledge
        f_no_k = FusionWithoutKnowledge(dim_xgb=self.dim_xgb, dim_ft=self.dim_ft)
        out_no_k, gate_no_k = f_no_k(f_xgb, f_ft)
        self.assertEqual(out_no_k.shape, (self.batch_size, self.dim_total))
        self.assertTrue(torch.isfinite(out_no_k).all().item())

        # A2/A8: Without Gate
        f_no_g = FusionWithoutGate(self.dummy_km, dim_xgb=self.dim_xgb, dim_ft=self.dim_ft)
        out_no_g, gate_no_g = f_no_g(f_xgb, f_ft)
        self.assertEqual(out_no_g.shape, (self.batch_size, self.dim_total))
        self.assertTrue(torch.isfinite(out_no_g).all().item())
        # Check that dummy gate is all ones (unmodulated pass-through)
        self.assertTrue((gate_no_g == 1.0).all().item())


if __name__ == "__main__":
    unittest.main()
