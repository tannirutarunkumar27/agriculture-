"""
Knowledge-Guided Feature Gate Module & FT-Transformer Wrapper
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Implements the Stage 5 Knowledge-Guided Feature Gate:
  G(X) = sigmoid(W_2 * Dropout(ReLU(W_1 * [X || g_prior] + b_1)) + b_2) in [0, 1]^{B x 7}

Modulates Feature Token Embeddings in vector space:
  E'_f = (alpha + (1 - alpha) * G_f) * E_f in R^d

Zero target label leakage. 100% baseline preservation architecture.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any

from src.knowledge.representation import KnowledgeRepresentationEngine


class KnowledgeGuidedFeatureGate(nn.Module):
    """
    Stage 5: Knowledge-Guided Feature Gate.
    Calculates feature importance modulation vector G(X) in [0, 1]^7
    conditioned on standardized inputs X and domain suitability priors g_prior.
    """

    def __init__(
        self,
        num_features: int = 7,
        hidden_dim: int = 14,
        dropout_rate: float = 0.1,
        alpha_residual: float = 0.2
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.alpha_residual = alpha_residual

        # Concatenated input: [X (7d) || g_prior (7d)] = 14d
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )

        # Knowledge Representation Engine for automatic prior calculation
        self.knowledge_engine = KnowledgeRepresentationEngine()

    def _compute_g_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to compute max-marginal prior vector g_prior in PyTorch."""
        x_np = x.detach().cpu().numpy()
        g_np = self.knowledge_engine.get_feature_gate_vector(x_np)
        return torch.tensor(g_np, dtype=x.dtype, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        g_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Feature Gate.

        Args:
            x: (B, 7) standardized input features.
            g_prior: Optional (B, 7) precomputed suitability prior.

        Returns:
            G: (B, 7) feature modulation gate in [0, 1].
            x_gated: (B, 7) raw gated features for interpretability/tree branches.
        """
        # Handle NaN values safely in x
        x_safe = torch.nan_to_num(x, nan=0.0)

        if g_prior is None:
            g_prior = self._compute_g_prior(x_safe)

        g_prior_safe = torch.nan_to_num(g_prior, nan=0.5)

        # Concatenate: (B, 14)
        h_in = torch.cat([x_safe, g_prior_safe], dim=-1)

        # Forward through gate MLP: (B, 7)
        G = self.gate_mlp(h_in)

        # Compute raw scalar gated features: X * (alpha + (1 - alpha) * G)
        scale = self.alpha_residual + (1.0 - self.alpha_residual) * G
        x_gated = x_safe * scale

        return G, x_gated

    def modulate_token_embeddings(
        self,
        tokens: torch.Tensor,
        G: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulates feature token embeddings in FT-Transformer vector space.

        Args:
            tokens: (B, 7, d) feature token embeddings.
            G: (B, 7) feature modulation gate.

        Returns:
            modulated_tokens: (B, 7, d)
        """
        # Scale: (B, 7, 1)
        scale = (self.alpha_residual + (1.0 - self.alpha_residual) * G).unsqueeze(-1)
        return tokens * scale

    def get_feature_support_spectrum(
        self,
        x_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """Exposes auditable feature confidence weights for explainable AI."""
        x_list = [x_dict.get(f, 0.0) for f in self.knowledge_engine.feature_names]
        x_t = torch.tensor([x_list], dtype=torch.float32)
        with torch.no_grad():
            G, _ = self.forward(x_t)
            G_np = G.cpu().numpy()[0]

        spectrum = {}
        for feat, val, g_val in zip(self.knowledge_engine.feature_names, x_list, G_np):
            spectrum[feat] = {
                "observed_value": val,
                "feature_confidence_weight": round(float(g_val), 4),
                "residual_scaling_factor": round(float(self.alpha_residual + (1.0 - self.alpha_residual) * g_val), 4)
            }
        return spectrum


class KnowledgeGuidedFTTransformerWrapper(nn.Module):
    """
    Baseline-preserving wrapper integrating KnowledgeGuidedFeatureGate with FT-Transformer.
    """

    def __init__(self, ft_transformer_base: nn.Module, feature_gate: KnowledgeGuidedFeatureGate):
        super().__init__()
        self.ft_transformer = ft_transformer_base
        self.feature_gate = feature_gate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Knowledge-Guided Token Modulation.

        Returns:
            logits: (B, num_classes)
            G: (B, 7) gate vector for visualization/interpretability.
        """
        G, _ = self.feature_gate(x)

        # Check if base FT-Transformer exposes tokenizer directly
        if hasattr(self.ft_transformer, "feature_tokenizer"):
            tokens = self.ft_transformer.feature_tokenizer(x)  # (B, 7, d)
            tokens_modulated = self.feature_gate.modulate_token_embeddings(tokens, G)
            logits = self.ft_transformer.forward_with_tokens(tokens_modulated)
        else:
            # Fallback for standard forward
            logits = self.ft_transformer(x)

        return logits, G
