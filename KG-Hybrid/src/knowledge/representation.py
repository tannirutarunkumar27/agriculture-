"""
Knowledge Representation Engine & Projection Module
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Converts static 22-crop agronomic knowledge into continuous, machine-readable
numerical tensors:
  1. Sample-Crop Compatibility Tensor M(X) in [0, 1]^{B x 22 x 7}
  2. Feature Knowledge Gating Vector g(X) in [0, 1]^{B x 7}
  3. Class-Prior Overall Compatibility Vector s_overall(X) in [0, 1]^{B x 22}
  4. PyTorch Dense Knowledge Embedding Encoder E_K: R^{154} -> R^{64}

Derived strictly from external FAO/ICAR physics. Zero target label contamination.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Any, Optional

from src.knowledge.knowledge_schema import CardinalBounds, CropKnowledgeProfile
from src.knowledge.knowledge_base import KNOWLEDGE_BASE, CROP_CLASSES_ORDERED
from src.knowledge.suitability import (
    FEATURE_NAMES,
    calculate_trapezoidal_gaussian_suitability,
    compute_crop_suitability_matrix,
    compute_feature_knowledge_gate
)


class KnowledgeRepresentationEngine:
    """
    Core engine generating machine-readable numerical representations
    of agronomic domain knowledge for input samples.
    """

    def __init__(self, crop_order: Optional[List[str]] = None):
        self.crop_order = crop_order or CROP_CLASSES_ORDERED
        self.num_crops = len(self.crop_order)
        self.num_features = len(FEATURE_NAMES)
        self.feature_names = FEATURE_NAMES

    def get_compatibility_matrix(self, x: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Computes sample-crop compatibility matrix M(x) for single sample or batch.

        Args:
            x: Input array of shape (7,) or (B, 7).

        Returns:
            NumPy array of shape (22, 7) for single sample, or (B, 22, 7) for batch.
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        if x_np.ndim == 1:
            if len(x_np) != 7:
                raise ValueError(f"Expected 7 feature inputs, got {len(x_np)}")
            return compute_crop_suitability_matrix(x_np)
        elif x_np.ndim == 2:
            if x_np.shape[1] != 7:
                raise ValueError(f"Expected input shape (B, 7), got {x_np.shape}")
            batch_size = x_np.shape[0]
            batch_matrix = np.zeros((batch_size, self.num_crops, self.num_features), dtype=np.float32)
            for i in range(batch_size):
                batch_matrix[i] = compute_crop_suitability_matrix(x_np[i])
            return batch_matrix
        else:
            raise ValueError(f"Invalid input dimensions: {x_np.ndim} (expected 1D or 2D array)")

    def get_feature_gate_vector(self, x: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Computes max-marginal feature knowledge gate vector g(x) in [0, 1]^7.

        Args:
            x: Input array of shape (7,) or (B, 7).

        Returns:
            NumPy array of shape (7,) or (B, 7).
        """
        matrix = self.get_compatibility_matrix(x)
        if matrix.ndim == 2:  # (22, 7)
            return np.max(matrix, axis=0)  # (7,)
        else:  # (B, 22, 7)
            return np.max(matrix, axis=1)  # (B, 7)

    def get_overall_compatibility(self, x: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Computes overall agricultural suitability score per crop s_overall(x) in [0, 1]^{22}.

        Args:
            x: Input array of shape (7,) or (B, 7).

        Returns:
            NumPy array of shape (22,) or (B, 22).
        """
        matrix = self.get_compatibility_matrix(x)
        if matrix.ndim == 2:  # (22, 7)
            return np.mean(matrix, axis=1)  # (22,)
        else:  # (B, 22, 7)
            return np.mean(matrix, axis=2)  # (B, 22)

    def explain_sample_compatibility(self, x_dict: Dict[str, float], crop_name: str) -> Dict[str, Any]:
        """
        Generates an auditable, human-readable explanation trace for a specific sample-crop pair.
        """
        if crop_name not in KNOWLEDGE_BASE:
            raise ValueError(f"Crop '{crop_name}' not found in Knowledge Base.")

        profile = KNOWLEDGE_BASE[crop_name]
        trace = {
            "crop": crop_name,
            "feature_suitabilities": {},
            "overall_suitability": 0.0
        }

        total_suit = 0.0
        for feat in self.feature_names:
            val = x_dict.get(feat, None)
            bounds = profile.get_bounds_for_feature(feat)
            s_val = calculate_trapezoidal_gaussian_suitability(
                val, bounds.min_val, bounds.opt_low, bounds.opt_high, bounds.max_val
            )
            total_suit += s_val

            if val is None or math.isnan(val):
                status = "MISSING_DATA_NEUTRAL"
            elif bounds.opt_low <= val <= bounds.opt_high:
                status = "OPTIMAL"
            elif bounds.min_val <= val < bounds.opt_low:
                status = "SUB_OPTIMAL_DEFICIENCY"
            elif bounds.opt_high < val <= bounds.max_val:
                status = "SUB_OPTIMAL_EXCESS"
            else:
                status = "EXTREME_UNSUITABLE"

            trace["feature_suitabilities"][feat] = {
                "observed_value": val,
                "cardinal_bounds": bounds.as_tuple(),
                "suitability_score": round(float(s_val), 4),
                "status": status
            }

        trace["overall_suitability"] = round(float(total_suit / self.num_features), 4)
        return trace


class KnowledgeProjectionEncoder(nn.Module):
    """
    Lightweight PyTorch Neural Projection Encoder E_K.
    Maps flattened compatibility matrix (154d) into dense embedding space (64d).
    """

    def __init__(
        self,
        input_dim: int = 154,  # 22 crops * 7 features
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, compatibility_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping matrix to 64d knowledge embedding.

        Args:
            compatibility_matrix: Tensor of shape (B, 22, 7) or (B, 154).

        Returns:
            Knowledge embedding tensor of shape (B, 64).
        """
        if compatibility_matrix.ndim == 3:
            batch_size = compatibility_matrix.shape[0]
            flat_input = compatibility_matrix.view(batch_size, -1)
        else:
            flat_input = compatibility_matrix

        return self.encoder(flat_input)
