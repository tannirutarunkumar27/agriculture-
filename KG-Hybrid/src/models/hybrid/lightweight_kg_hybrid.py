"""
Lightweight KG-Hybrid Model
===========================
Optimized for edge and production deployment:
  - 32-dim compact XGB Leaf Embedding (75% parameter reduction)
  - 32-dim FT-Transformer backbone
  - Fast Knowledge-Guided Fusion layer
  - Total Parameters: ~3.8M (vs 15.5M in full KG-Hybrid)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion


class LightweightKGHybrid(nn.Module):
    """
    Lightweight version of KG-Hybrid.
    Reduces leaf embedding dimension to 32 and transformer feature dimension to 32,
    lowering parameters from 15.5M to ~3.8M while maintaining architectural structure.
    """
    def __init__(self, 
                 num_unique_leaves: int = 118800, 
                 num_trees: int = 4400,
                 embed_dim: int = 32,
                 num_classes: int = 22,
                 knowledge_matrix: np.ndarray = None,
                 knowledge_policy: str = "anchored_trainable"):
        super().__init__()
        
        self.xgb_embedder = XGBLeafEmbedder(
            num_unique_leaves=num_unique_leaves,
            num_trees=num_trees,
            embedding_dim=embed_dim
        )
        
        self.ft_transformer = FTTransformer()
        
        # Linear adapter to project FT-Transformer output (64) to compact dim (32)
        self.ft_proj = nn.Linear(64, embed_dim)
        
        if knowledge_matrix is None:
            raise ValueError(
                "LightweightKGHybrid requires the validated authoritative knowledge_matrix; "
                "a zero or random fallback would invalidate knowledge-guided inference."
            )
            
        self.fusion = KnowledgeGuidedFusion(
            dim_xgb=embed_dim,
            dim_ft=embed_dim,
            num_crops=num_classes,
            num_features=7,
            knowledge_matrix=knowledge_matrix,
            knowledge_policy=knowledge_policy,
        )
        
        dim_total = embed_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_total),
            nn.Dropout(0.1),
            nn.Linear(dim_total, num_classes)
        )
        
    def forward(self, leaf_indices, X_ft):
        f_xgb = self.xgb_embedder(leaf_indices)
        
        if hasattr(self.ft_transformer, 'extract_features'):
            f_ft_raw = self.ft_transformer.extract_features(X_ft)
        else:
            f_ft_raw = self.ft_transformer(X_ft)
            
        f_ft = self.ft_proj(f_ft_raw)
        
        f_fused, gate_weights = self.fusion(f_xgb, f_ft)
        logits = self.classifier(f_fused)
        return logits, gate_weights
