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
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.baseline_hybrid_concat import HybridConcat
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.utils.evaluation import evaluate_model

# ===================== CONFIGURATION =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models"
TABLES_PATH = PROJECT_ROOT / "outputs" / "tables"

# ===================== LOAD DATA =====================
print("Loading data for HybridConcat baseline...")
train_df = pd.read_csv(DATA_PATH / "train.csv")
val_df = pd.read_csv(DATA_PATH / "validation.csv")
test_df = pd.read_csv(DATA_PATH / "test.csv")

X_train = train_df.drop("label", axis=1).values.astype(np.float32)
y_train = train_df["label"].values.astype(np.int64)
X_val = val_df.drop("label", axis=1).values.astype(np.float32)
y_val = val_df["label"].values.astype(np.int64)
X_test = test_df.drop("label", axis=1).values.astype(np.float32)
y_test = test_df["label"].values.astype(np.int64)

# ===================== XGBOOST LEAF INDICES =====================
xgb_model = joblib.load(OUTPUT_PATH / "xgboost_model.pkl")
booster = xgb_model.get_booster()

def get_leaf_indices(X):
    df_X = pd.DataFrame(X, columns=FEATURES)
    dmat = xgb.DMatrix(df_X)
    return booster.predict(dmat, pred_leaf=True)

leaf_indices_train = get_leaf_indices(X_train)
leaf_indices_val = get_leaf_indices(X_val)
leaf_indices_test = get_leaf_indices(X_test)

max_leaves_per_tree = int(max(leaf_indices_train.max(), leaf_indices_val.max(), leaf_indices_test.max())) + 1
num_trees = leaf_indices_train.shape[1]
tree_offsets = np.arange(num_trees, dtype=np.int64) * max_leaves_per_tree

leaf_indices_train = leaf_indices_train + tree_offsets
leaf_indices_val = leaf_indices_val + tree_offsets
leaf_indices_test = leaf_indices_test + tree_offsets

total_unique_leaves = num_trees * max_leaves_per_tree

# ===================== LOAD FT-TRANSFORMER =====================
config = FTConfig()
ft_model = FTTransformer()
checkpoint = torch.load(OUTPUT_PATH / "best_model.pth", map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    ft_model.load_state_dict(checkpoint["model_state_dict"])
else:
    ft_model.load_state_dict(checkpoint)
ft_model = ft_model.to(DEVICE)

# ===================== BUILD HYBRID CONCAT =====================
xgb_embedder = XGBLeafEmbedder(num_unique_leaves=total_unique_leaves, num_trees=num_trees, embedding_dim=128).to(DEVICE)
model = HybridConcat(xgb_embedder, ft_model, num_classes=22).to(DEVICE)

train_dataset = TensorDataset(
    torch.tensor(leaf_indices_train, dtype=torch.long),
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(leaf_indices_val, dtype=torch.long),
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(leaf_indices_test, dtype=torch.long),
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0

print("Training HybridConcat baseline...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for leaves, X_ft, y in train_loader:
        leaves, X_ft, y = leaves.to(DEVICE), X_ft.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(leaves, X_ft)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(logits, 1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)
        
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for leaves, X_ft, y in val_loader:
            leaves, X_ft, y = leaves.to(DEVICE), X_ft.to(DEVICE), y.to(DEVICE)
            logits = model(leaves, X_ft)
            loss = criterion(logits, y)
            val_loss += loss.item()
            _, preds = torch.max(logits, 1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
            
    val_acc = val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)
    scheduler.step(val_loss_avg)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_PATH / "hybrid_concat_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

print(f"Evaluating HybridConcat test set...")
model.load_state_dict(torch.load(OUTPUT_PATH / "hybrid_concat_best.pth"))
model.eval()

test_preds = []
with torch.no_grad():
    for leaves, X_ft, y in test_loader:
        leaves, X_ft = leaves.to(DEVICE), X_ft.to(DEVICE)
        logits = model(leaves, X_ft)
        _, preds = torch.max(logits, 1)
        test_preds.extend(preds.cpu().numpy())

metrics = evaluate_model("hybrid_concat", y_test, np.array(test_preds))
print(f"[OK] HybridConcat Test Accuracy: {metrics['Accuracy']}")
