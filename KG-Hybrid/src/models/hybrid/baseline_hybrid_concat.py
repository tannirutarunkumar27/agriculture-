import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig

class HybridConcat(nn.Module):
    """Simple concatenation baseline - no knowledge, no gate"""
    def __init__(self, xgb_embedder, ft_transformer, num_classes=22):
        super().__init__()
        self.xgb_embedder = xgb_embedder
        self.ft_transformer = ft_transformer
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, leaf_indices, X_ft):
        f_xgb = self.xgb_embedder(leaf_indices)
        if hasattr(self.ft_transformer, 'extract_features'):
            f_ft = self.ft_transformer.extract_features(X_ft)
        else:
            f_ft = self.ft_transformer(X_ft, return_features=True)
        concat = torch.cat([f_xgb, f_ft], dim=1)
        return self.head(concat)
