"""
Shared utilities for robustness tests.
Centralizes model loading and leaf index extraction.
"""

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig

DATA_PATH  = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FEATURES   = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# These are derived from the training procedure (tree offsets applied during training)
NUM_TREES          = 4400
MAX_LEAVES_PER_TREE = 27          # raw max leaf index + 1 per tree
TOTAL_UNIQUE_LEAVES = NUM_TREES * MAX_LEAVES_PER_TREE   # 118800
EMBEDDING_DIM       = 128

# Pre-compute tree offsets (same logic as train_kg_hybrid.py line 81)
TREE_OFFSETS = np.arange(NUM_TREES, dtype=np.int64) * MAX_LEAVES_PER_TREE


def _load_ft_weights():
    """Load FT-Transformer state dict from checkpoint (handles both formats)."""
    ckpt = torch.load(MODEL_PATH / "best_model.pth", map_location='cpu')
    return ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt


def get_leaf_indices_with_offset(X_numpy, booster):
    """
    Extract leaf indices from XGBoost booster and apply the same tree offsets
    that were used during KG-Hybrid training.

    Args:
        X_numpy: np.ndarray of shape (N, 7), float32
        booster: xgb.Booster (from sklearn XGBClassifier.get_booster())

    Returns:
        leaf_tensor: torch.LongTensor of shape (N, num_trees) with offsets
    """
    df_X = pd.DataFrame(X_numpy, columns=FEATURES)
    dmat = xgb.DMatrix(df_X)
    raw  = booster.predict(dmat, pred_leaf=True).astype(np.int64)   # (N, 4400)
    if raw.ndim != 2 or raw.shape[1] != NUM_TREES:
        raise ValueError("Booster tree count does not match the saved neural embedding layout")
    if (raw < 0).any() or (raw >= MAX_LEAVES_PER_TREE).any():
        raise ValueError("Booster leaf nodes exceed the saved neural embedding layout")
    raw += TREE_OFFSETS                                              # add offset
    return torch.tensor(raw, dtype=torch.long)


def make_rf(X_train, y_train):
    """Train Random Forest in-memory, return predict function."""
    print("  Training Random Forest in-memory...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  [OK] Random Forest ready")
    return lambda X: rf.predict(X)


def make_xgb():
    """Load sklearn XGBClassifier, return predict function."""
    clf = joblib.load(MODEL_PATH / "xgboost_model.pkl")
    print("  [OK] XGBoost loaded")
    return lambda X: clf.predict(pd.DataFrame(X, columns=FEATURES))


def make_ft():
    """Load FT-Transformer, return predict function."""
    model = FTTransformer()
    model.load_state_dict(_load_ft_weights())
    model.eval()
    print("  [OK] FT-Transformer loaded")

    def predict(X):
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()
    return predict


def make_kg():
    """Load full KG-Hybrid, return (booster, predict_fn) tuple."""
    xgb_clf  = joblib.load(MODEL_PATH / "xgboost_model.pkl")
    booster  = xgb_clf.get_booster()

    cfg      = FTConfig()
    embedder = XGBLeafEmbedder(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES,
        num_trees=NUM_TREES,
        embedding_dim=EMBEDDING_DIM
    )
    ft_model = FTTransformer()
    ft_model.load_state_dict(_load_ft_weights())

    from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
    km, _, _ = load_authoritative_knowledge_matrix(DATA_PATH)
    fusion = KnowledgeGuidedFusion(
        dim_xgb=EMBEDDING_DIM,
        dim_ft=cfg.EMBEDDING_DIM,
        num_crops=22,
        num_features=7,
        knowledge_matrix=km
    )

    kg = KGHybrid(embedder, ft_model, fusion, num_classes=22)
    kg_ckpt_path = MODEL_PATH / "kg_hybrid_corrected.pth" if (MODEL_PATH / "kg_hybrid_corrected.pth").exists() else MODEL_PATH / "kg_hybrid_best.pth"
    kg.load_state_dict(torch.load(kg_ckpt_path, map_location='cpu'), strict=True)
    kg.eval()
    print("  [OK] KG-Hybrid loaded")

    def predict(X):
        leaf_t = get_leaf_indices_with_offset(X, booster)
        x_t    = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits, _, _, _, _ = kg(leaf_t, x_t)
        return logits.argmax(dim=1).numpy()

    return predict


def load_all_models(X_train, y_train):
    """Load / build all four models. Returns dict of predict functions."""
    print("\nLoading / building models...")
    return {
        "Random Forest":  make_rf(X_train, y_train),
        "XGBoost":        make_xgb(),
        "FT-Transformer": make_ft(),
        "KG-Hybrid":      make_kg(),
    }
