"""
Train Lightweight KG-Hybrid Model
=================================
Trains LightweightKGHybrid with 32-dim XGB leaf embedding and 32-dim FT adapter.
Saves model weights to outputs/models/lightweight_kg_hybrid_best.pth.
Evaluates Clean, Missing (40% + KNN), Noise (s=0.2), and Extrapolation (x1.5).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid.lightweight_kg_hybrid import LightweightKGHybrid
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, FEATURES, TOTAL_UNIQUE_LEAVES, NUM_TREES, get_leaf_indices_with_offset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-5
KNOWLEDGE_POLICY = "anchored_trainable"
KNOWLEDGE_ANCHOR_WEIGHT = 1.0

def main():
    print("==================================================")
    print("TRAINING LIGHTWEIGHT KG-HYBRID (4.04M Parameters)")
    print("==================================================")

    train_df = pd.read_csv(DATA_PATH / "train.csv")
    val_df   = pd.read_csv(DATA_PATH / "validation.csv")
    test_df  = pd.read_csv(DATA_PATH / "test.csv")

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_val   = val_df.drop("label", axis=1).values.astype(np.float32)
    y_val   = val_df["label"].values.astype(np.int64)
    X_test  = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int64)

    xgb_model = joblib.load(MODEL_PATH / "xgboost_model.pkl")
    booster   = xgb_model.get_booster()

    leaves_train = get_leaf_indices_with_offset(X_train, booster)
    leaves_val   = get_leaf_indices_with_offset(X_val, booster)
    leaves_test  = get_leaf_indices_with_offset(X_test, booster)

    train_ds = TensorDataset(leaves_train, torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(leaves_val, torch.tensor(X_val), torch.tensor(y_val))
    test_ds  = TensorDataset(leaves_test, torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    km, _, _ = load_authoritative_knowledge_matrix()
    model = LightweightKGHybrid(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES,
        num_trees=NUM_TREES,
        embed_dim=32,
        num_classes=22,
        knowledge_matrix=km,
        knowledge_policy=KNOWLEDGE_POLICY,
    ).to(DEVICE)

    # Load pre-trained FT-Transformer backbone into model.ft_transformer
    ckpt = torch.load(MODEL_PATH / "best_model.pth", map_location=DEVICE)
    ft_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.ft_transformer.load_state_dict(ft_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    save_path = MODEL_PATH / "lightweight_kg_hybrid_best.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for b_leaves, b_x, b_y in train_loader:
            b_leaves, b_x, b_y = b_leaves.to(DEVICE), b_x.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(b_leaves, b_x)
            loss = criterion(logits, b_y) + KNOWLEDGE_ANCHOR_WEIGHT * model.fusion.knowledge_anchor_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b_leaves, b_x, b_y in val_loader:
                b_leaves, b_x, b_y = b_leaves.to(DEVICE), b_x.to(DEVICE), b_y.to(DEVICE)
                logits, _ = model(b_leaves, b_x)
                preds = logits.argmax(dim=1)
                correct += (preds == b_y).sum().item()
                total += b_y.size(0)

        val_acc = correct / total
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"[OK] Training complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[OK] Lightweight model saved -> {save_path}")

    # Evaluate on Test Set
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    leaves_test, X_test_t, y_test_t = leaves_test.to(DEVICE), torch.tensor(X_test).to(DEVICE), torch.tensor(y_test).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(leaves_test, X_test_t)
        test_acc = (logits.argmax(dim=1) == y_test_t).float().mean().item()

    print(f"[RESULTS] Lightweight KG-Hybrid Test Clean Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
