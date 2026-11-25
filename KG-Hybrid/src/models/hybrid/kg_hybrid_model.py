import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig

class KGHybrid(nn.Module):
    """
    Full Knowledge-Guided Hybrid Model combining:
    - XGBoost (decision features via leaf embeddings)
    - FT-Transformer (contextual features)
    - Knowledge-Guided Fusion
    """
    def __init__(self, xgb_embedder, ft_transformer, fusion_module, num_classes=22):
        super().__init__()
        self.xgb_embedder = xgb_embedder
        self.ft_transformer = ft_transformer
        self.fusion = fusion_module
        
        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(fusion_module.dim_total, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, leaf_indices, X_ft):
        """
        Args:
            leaf_indices: torch.LongTensor (batch, num_trees) - XGBoost leaf indices
            X_ft: torch.FloatTensor (batch, num_features) - Raw features for FT-Transformer
        Returns:
            logits: (batch, num_classes)
            f_xgb: (batch, dim_xgb)
            f_ft: (batch, dim_ft)
            f_fused: (batch, dim_total)
            gate: (batch, dim_total)
        """
        # Branch A: XGBoost decision features
        f_xgb = self.xgb_embedder(leaf_indices)  # (batch, 128)
        
        # Branch B: FT-Transformer context features
        if hasattr(self.ft_transformer, 'extract_features'):
            f_ft = self.ft_transformer.extract_features(X_ft)
        else:
            f_ft = self.ft_transformer(X_ft, return_features=True)
        
        # Fusion
        f_fused, gate = self.fusion(f_xgb, f_ft)  # (batch, 192)
        
        # Prediction
        logits = self.head(f_fused)  # (batch, 22)
        
        return logits, f_xgb, f_ft, f_fused, gate
    
    def get_xgb_logits(self, X_ft):
        """Get XGBoost logits (for distillation loss)"""
        # This is a placeholder - you'll need to pass through XGBoost separately
        # We handle this in the training loop
        pass
