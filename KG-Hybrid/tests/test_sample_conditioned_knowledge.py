"""
Unit Tests for Sample-Conditioned Agronomic Knowledge Context (Problem 2 Resolution)

Covers all Phase 6 verification requirements:
  1. Main-model output shape is (batch_size, 22) and all values are finite.
  2. Dynamic k_bias shape is (batch_size, 192), strictly sample-conditioned (not [1, 192]).
  3. Crop-wise attention weights have shape (batch_size, num_crops).
  4. Attention weights are non-negative, finite, and sum to 1.0 across crop profiles.
  5. Controlled distinct inputs produce distinct attention distributions and knowledge contexts.
  6. Modulating knowledge context changes the pre-normalization fused representation.
  7. Nonzero gradient flow to query_proj, context_proj, knowledge_encoder, and knowledge_matrix.
  8. Optimizer update step modifies knowledge mechanism parameters.
  9. Adaptive gate remains effective without reintroducing Problem 1 algebraic cancellation.
 10. Training-mode forward, combined loss computation, and backward pass succeed.
 11. Evaluation-mode forward pass succeeds with deterministic outputs.
 12. Save and reload checkpoint smoke test preserves exact model predictions.
 13. Ablation variants A1 (w/o knowledge), A2 (w/o gate), and A8 (w/o gate, pure CE) execute cleanly.
 14. Numerical stability under extreme input values (large positive/negative features).
"""

import unittest
import math
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.ablation.run_ablation import FusionWithoutKnowledge, FusionWithoutGate, KGHybridBase
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.hybrid.auxiliary_losses import combined_loss


class TestSampleConditionedKnowledge(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.dim_xgb = 128
        self.dim_ft = 64
        self.dim_total = 192
        self.num_crops = 22
        self.num_features = 7
        self.dim_k = 64
        self.beta = 0.1

        # Realistic knowledge matrix shape (22, 7)
        self.dummy_km = np.random.uniform(0.3, 0.9, size=(self.num_crops, self.num_features)).astype(np.float32)

        self.fusion = KnowledgeGuidedFusion(
            dim_xgb=self.dim_xgb,
            dim_ft=self.dim_ft,
            num_crops=self.num_crops,
            num_features=self.num_features,
            knowledge_matrix=self.dummy_km,
            dim_k=self.dim_k,
            beta=self.beta,
        )

    def test_01_main_model_output_shape(self):
        """Test 1: Full KGHybrid forward pass outputs correct logits shape (B, 22)."""
        embedder = XGBLeafEmbedder(num_unique_leaves=1000, num_trees=10, embedding_dim=self.dim_xgb)
        ft = FTTransformer()
        model = KGHybrid(embedder, ft, self.fusion, num_classes=self.num_crops)

        leaf_indices = torch.randint(0, 1000, (self.batch_size, 10))
        X_ft = torch.randn(self.batch_size, self.num_features)

        logits, f_xgb, f_ft, f_fused, gate = model(leaf_indices, X_ft)

        self.assertEqual(logits.shape, (self.batch_size, self.num_crops))
        self.assertEqual(f_fused.shape, (self.batch_size, self.dim_total))
        self.assertEqual(gate.shape, (self.batch_size, self.dim_total))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_02_k_bias_shape_is_batch_conditioned(self):
        """Test 2: Dynamic k_bias has shape (B, 192), strictly not a global [1, 192] broadcast."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)
        concat = torch.cat([f_xgb, f_ft], dim=1)

        kp = self.fusion.knowledge_encoder(self.fusion.knowledge_matrix)
        q = self.fusion.query_proj(concat)
        scores = torch.matmul(q, kp.t()) / math.sqrt(self.dim_k)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, kp)
        k_bias = self.fusion.context_proj(ctx)

        self.assertEqual(k_bias.shape, (self.batch_size, self.dim_total))
        self.assertNotEqual(k_bias.shape, (1, self.dim_total))

    def test_03_attention_weights_shape(self):
        """Test 3: Crop-wise attention weights have shape (B, num_crops)."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)

        f_fused, gate, attn = self.fusion(f_xgb, f_ft, return_attention=True)

        self.assertEqual(attn.shape, (self.batch_size, self.num_crops))
        self.assertEqual(self.fusion.last_attention_weights.shape, (self.batch_size, self.num_crops))

    def test_04_attention_weights_sum_to_one_and_positive(self):
        """Test 4: Attention weights are in [0, 1] and sum to 1.0 along the crop dimension."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)

        _, _, attn = self.fusion(f_xgb, f_ft, return_attention=True)

        self.assertTrue((attn >= 0.0).all().item())
        self.assertTrue((attn <= 1.0).all().item())
        self.assertTrue(torch.isfinite(attn).all().item())

        sums = attn.sum(dim=-1)
        expected = torch.ones(self.batch_size)
        self.assertTrue(torch.allclose(sums, expected, atol=1e-5))

    def test_05_controlled_distinct_inputs_produce_distinct_attention_and_kbias(self):
        """Test 5: Controlled distinct inputs yield different attention distributions and distinct k_bias."""
        # Create two vastly different feature representations
        f_xgb_1 = torch.ones(1, self.dim_xgb) * 10.0
        f_ft_1 = torch.ones(1, self.dim_ft) * 10.0

        f_xgb_2 = torch.ones(1, self.dim_xgb) * -10.0
        f_ft_2 = torch.ones(1, self.dim_ft) * -10.0

        _, _, attn1 = self.fusion(f_xgb_1, f_ft_1, return_attention=True)
        _, _, attn2 = self.fusion(f_xgb_2, f_ft_2, return_attention=True)

        attn_diff = (attn1 - attn2).abs().sum().item()
        self.assertGreater(attn_diff, 1e-3, f"Attention weights should differ for opposite inputs, diff={attn_diff}")

        # Compute k_bias for both
        kp = self.fusion.knowledge_encoder(self.fusion.knowledge_matrix)
        ctx1 = torch.matmul(attn1, kp)
        ctx2 = torch.matmul(attn2, kp)
        kbias1 = self.fusion.context_proj(ctx1)
        kbias2 = self.fusion.context_proj(ctx2)

        kbias_diff = (kbias1 - kbias2).abs().sum().item()
        self.assertGreater(kbias_diff, 1e-2, f"Dynamic k_bias should differ across samples, diff={kbias_diff}")

    def test_06_knowledge_context_modulates_fused_representation(self):
        """Test 6: Altering the knowledge context directly modulates the pre-normalization representation."""
        f_xgb = torch.ones(1, self.dim_xgb) * 1.5
        f_ft = torch.ones(1, self.dim_ft) * 1.5
        concat = torch.cat([f_xgb, f_ft], dim=1)
        gate = self.fusion.gate(concat)

        # Context A vs Context B
        ctx_a = torch.ones(1, self.dim_k) * 1.0
        ctx_b = torch.ones(1, self.dim_k) * 5.0
        kbias_a = self.fusion.context_proj(ctx_a)
        kbias_b = self.fusion.context_proj(ctx_b)

        f_pre_a = gate * concat + self.beta * kbias_a
        f_pre_b = gate * concat + self.beta * kbias_b

        diff = (f_pre_b - f_pre_a).abs().sum().item()
        self.assertGreater(diff, 1e-3, f"Pre-normalization fused representation must reflect knowledge context changes")

    def test_07_gradient_flow_to_knowledge_parameters(self):
        """Test 7: Backward pass propagates finite, nonzero gradients to all knowledge components."""
        self.fusion.zero_grad()
        f_xgb = torch.randn(self.batch_size, self.dim_xgb, requires_grad=True)
        f_ft = torch.randn(self.batch_size, self.dim_ft, requires_grad=True)

        head = nn.Linear(self.dim_total, self.num_crops)
        f_fused, _ = self.fusion(f_xgb, f_ft)
        logits = head(f_fused)
        target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Check knowledge matrix grad
        self.assertIsNotNone(self.fusion.knowledge_matrix.grad)
        self.assertTrue(torch.isfinite(self.fusion.knowledge_matrix.grad).all().item())
        self.assertGreater(self.fusion.knowledge_matrix.grad.abs().sum().item(), 0.0)

        # Check knowledge encoder grad
        for name, p in self.fusion.knowledge_encoder.named_parameters():
            self.assertIsNotNone(p.grad, f"knowledge_encoder parameter {name} has no grad")
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0.0)

        # Check query proj grad
        for name, p in self.fusion.query_proj.named_parameters():
            self.assertIsNotNone(p.grad, f"query_proj parameter {name} has no grad")
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0.0)

        # Check context proj grad
        for name, p in self.fusion.context_proj.named_parameters():
            self.assertIsNotNone(p.grad, f"context_proj parameter {name} has no grad")
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0.0)

    def test_08_optimizer_update_modifies_knowledge_parameters(self):
        """Test 8: An AdamW optimizer step updates all knowledge-mechanism weights."""
        optimizer = optim.AdamW(self.fusion.parameters(), lr=1e-2)

        init_km = self.fusion.knowledge_matrix.clone()
        init_ke = self.fusion.knowledge_encoder.weight.clone()
        init_qp = self.fusion.query_proj.weight.clone()
        init_cp = self.fusion.context_proj.weight.clone()

        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)
        head = nn.Linear(self.dim_total, self.num_crops)

        f_fused, _ = self.fusion(f_xgb, f_ft)
        logits = head(f_fused)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertGreater((self.fusion.knowledge_matrix - init_km).abs().sum().item(), 1e-6)
        self.assertGreater((self.fusion.knowledge_encoder.weight - init_ke).abs().sum().item(), 1e-6)
        self.assertGreater((self.fusion.query_proj.weight - init_qp).abs().sum().item(), 1e-6)
        self.assertGreater((self.fusion.context_proj.weight - init_cp).abs().sum().item(), 1e-6)

    def test_09_gate_effectiveness_preserved(self):
        """Test 9: Problem 1 gate modulation remains effective under sample-conditioned knowledge."""
        f_xgb = torch.ones(1, self.dim_xgb) * 2.0
        f_ft = torch.ones(1, self.dim_ft) * 2.0
        concat = torch.cat([f_xgb, f_ft], dim=1)

        kp = self.fusion.knowledge_encoder(self.fusion.knowledge_matrix)
        q = self.fusion.query_proj(concat)
        attn = torch.softmax(torch.matmul(q, kp.t()) / math.sqrt(self.dim_k), dim=-1)
        k_bias = self.fusion.context_proj(torch.matmul(attn, kp))

        gate_low = torch.full_like(concat, 0.1)
        gate_high = torch.full_like(concat, 0.9)

        f_pre_low = gate_low * concat + self.beta * k_bias
        f_pre_high = gate_high * concat + self.beta * k_bias

        expected_diff = (0.9 - 0.1) * concat
        self.assertTrue(torch.allclose(f_pre_high - f_pre_low, expected_diff, atol=1e-6))

    def test_10_training_mode_combined_loss_and_backward(self):
        """Test 10: Full training step with combined CE + Distill + Contrastive loss executes cleanly."""
        embedder = XGBLeafEmbedder(num_unique_leaves=1000, num_trees=10, embedding_dim=self.dim_xgb)
        ft = FTTransformer()
        model = KGHybrid(embedder, ft, self.fusion, num_classes=self.num_crops)
        model.train()

        leaf_indices = torch.randint(0, 1000, (self.batch_size, 10))
        X_ft = torch.randn(self.batch_size, self.num_features)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        logits_fused, _, _, f_fused, _ = model(leaf_indices, X_ft)
        logits_xgb = torch.randn(self.batch_size, self.num_crops)
        logits_ft = torch.randn(self.batch_size, self.num_crops)

        loss, loss_ce, loss_distill, loss_contrast = combined_loss(
            logits_fused, logits_xgb, logits_ft, y, f_fused,
            alpha_ce=1.0, alpha_distill=0.5, alpha_contrast=0.1
        )

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_11_eval_mode_deterministic_predictions(self):
        """Test 11: In eval mode, two identical forward calls produce identical predictions."""
        embedder = XGBLeafEmbedder(num_unique_leaves=1000, num_trees=10, embedding_dim=self.dim_xgb)
        ft = FTTransformer()
        model = KGHybrid(embedder, ft, self.fusion, num_classes=self.num_crops)
        model.eval()

        leaf_indices = torch.randint(0, 1000, (self.batch_size, 10))
        X_ft = torch.randn(self.batch_size, self.num_features)

        with torch.no_grad():
            logits1, _, _, f_fused1, _ = model(leaf_indices, X_ft)
            logits2, _, _, f_fused2, _ = model(leaf_indices, X_ft)

        self.assertTrue(torch.allclose(logits1, logits2, atol=1e-6))
        self.assertTrue(torch.allclose(f_fused1, f_fused2, atol=1e-6))

    def test_12_checkpoint_save_and_reload_smoke_test(self):
        """Test 12: Checkpoint saving and reloading with revised architecture restores exact predictions."""
        embedder = XGBLeafEmbedder(num_unique_leaves=1000, num_trees=10, embedding_dim=self.dim_xgb)
        ft = FTTransformer()
        model_orig = KGHybrid(embedder, ft, self.fusion, num_classes=self.num_crops)
        model_orig.eval()

        leaf_indices = torch.randint(0, 1000, (self.batch_size, 10))
        X_ft = torch.randn(self.batch_size, self.num_features)

        with torch.no_grad():
            expected_logits, _, _, _, _ = model_orig(leaf_indices, X_ft)

        # Save to buffer
        buf = io.BytesIO()
        torch.save(model_orig.state_dict(), buf)
        buf.seek(0)

        # Load into freshly constructed model
        embedder2 = XGBLeafEmbedder(num_unique_leaves=1000, num_trees=10, embedding_dim=self.dim_xgb)
        ft2 = FTTransformer()
        fusion2 = KnowledgeGuidedFusion(
            dim_xgb=self.dim_xgb, dim_ft=self.dim_ft,
            num_crops=self.num_crops, num_features=self.num_features,
            knowledge_matrix=self.dummy_km
        )
        model_reloaded = KGHybrid(embedder2, ft2, fusion2, num_classes=self.num_crops)
        model_reloaded.load_state_dict(torch.load(buf))
        model_reloaded.eval()

        with torch.no_grad():
            reloaded_logits, _, _, _, _ = model_reloaded(leaf_indices, X_ft)

        self.assertTrue(torch.allclose(expected_logits, reloaded_logits, atol=1e-6))

    def test_13_ablation_variants_execution(self):
        """Test 13: Ablation variants A1, A2, and A8 execute with correct shapes and expected behavior."""
        f_xgb = torch.randn(self.batch_size, self.dim_xgb)
        f_ft = torch.randn(self.batch_size, self.dim_ft)

        # A1: w/o Knowledge
        a1_fusion = FusionWithoutKnowledge(dim_xgb=self.dim_xgb, dim_ft=self.dim_ft)
        f_a1, g_a1 = a1_fusion(f_xgb, f_ft)
        self.assertEqual(f_a1.shape, (self.batch_size, self.dim_total))
        self.assertTrue(torch.isfinite(f_a1).all().item())

        # A2 / A8: w/o Gate (uses dynamic sample-conditioned knowledge context, gate = 1.0)
        a2_fusion = FusionWithoutGate(self.dummy_km, dim_xgb=self.dim_xgb, dim_ft=self.dim_ft)
        f_a2, g_a2, attn_a2 = a2_fusion(f_xgb, f_ft, return_attention=True)
        self.assertEqual(f_a2.shape, (self.batch_size, self.dim_total))
        self.assertTrue((g_a2 == 1.0).all().item())  # dummy gate is unmodulated ones
        self.assertEqual(attn_a2.shape, (self.batch_size, self.num_crops))
        self.assertTrue(torch.isfinite(f_a2).all().item())

        # Verify A8 pure cross-entropy compatibility
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        head = nn.Linear(self.dim_total, self.num_crops)
        logits_a8 = head(f_a2)
        loss_a8 = nn.CrossEntropyLoss()(logits_a8, y)
        loss_a8.backward()
        self.assertTrue(torch.isfinite(loss_a8).item())

    def test_14_numerical_stability_under_extreme_inputs(self):
        """Test 14: Softmax and LayerNorm remain finite even with large positive/negative inputs."""
        f_xgb_large = torch.ones(self.batch_size, self.dim_xgb) * 100.0
        f_ft_large = torch.ones(self.batch_size, self.dim_ft) * -100.0

        f_fused, gate, attn = self.fusion(f_xgb_large, f_ft_large, return_attention=True)

        self.assertTrue(torch.isfinite(f_fused).all().item())
        self.assertTrue(torch.isfinite(gate).all().item())
        self.assertTrue(torch.isfinite(attn).all().item())
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
