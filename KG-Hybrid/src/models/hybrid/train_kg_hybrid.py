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
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.hybrid.auxiliary_losses import combined_loss
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.utils.evaluation import evaluate_model

# ===================== CONFIGURATION =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 15  # Early stopping
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# The coefficient is part of the experiment specification and must be selected
# on validation data, never by repeatedly inspecting the held-out test split.
KNOWLEDGE_POLICY = "anchored_trainable"
KNOWLEDGE_ANCHOR_WEIGHT = 1.0

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"
TABLES_PATH = PROJECT_ROOT / "outputs" / "tables"

TABLES_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ===================== LOAD DATA =====================
print("Loading data...")
train_df = pd.read_csv(DATA_PATH / "train.csv")
val_df = pd.read_csv(DATA_PATH / "validation.csv")
test_df = pd.read_csv(DATA_PATH / "test.csv")

X_train = train_df.drop("label", axis=1).values.astype(np.float32)
y_train = train_df["label"].values.astype(np.int64)
X_val = val_df.drop("label", axis=1).values.astype(np.float32)
y_val = val_df["label"].values.astype(np.int64)
X_test = test_df.drop("label", axis=1).values.astype(np.float32)
y_test = test_df["label"].values.astype(np.int64)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ===================== LOAD XGBOOST LEAF INDICES =====================
print("Loading XGBoost model and extracting leaf indices...")
xgb_model = joblib.load(OUTPUT_PATH / "xgboost_model.pkl")
booster = xgb_model.get_booster()

def get_leaf_indices(X):
    df_X = pd.DataFrame(X, columns=FEATURES)
    dmat = xgb.DMatrix(df_X)
    return booster.predict(dmat, pred_leaf=True)

leaf_indices_train = get_leaf_indices(X_train)
leaf_indices_val = get_leaf_indices(X_val)
leaf_indices_test = get_leaf_indices(X_test)

# Save leaf indices
np.save(DATA_PATH / "xgb_train_leaves.npy", leaf_indices_train)
np.save(DATA_PATH / "xgb_val_leaves.npy", leaf_indices_val)
np.save(DATA_PATH / "xgb_test_leaves.npy", leaf_indices_test)

# Apply tree offsets so each tree has unique leaf embeddings (authoritative metadata: 27 leaves/tree)
metadata_file = DATA_PATH / "xgb_leaf_metadata.txt"
if metadata_file.exists():
    with open(metadata_file) as f:
        meta_dict = dict(line.strip().split(" = ") for line in f if " = " in line)
    max_leaves_per_tree = int(meta_dict.get("num_unique_leaves", 27))
else:
    max_leaves_per_tree = 27

num_trees = leaf_indices_train.shape[1]
tree_offsets = np.arange(num_trees, dtype=np.int64) * max_leaves_per_tree

leaf_indices_train = leaf_indices_train + tree_offsets
leaf_indices_val = leaf_indices_val + tree_offsets
leaf_indices_test = leaf_indices_test + tree_offsets

total_unique_leaves = num_trees * max_leaves_per_tree
print(f"Total unique leaf embeddings across {num_trees} trees: {total_unique_leaves}")

# ===================== LOAD AUTHORITATIVE KNOWLEDGE MATRIX =====================
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
knowledge_matrix, crop_names, feature_names = load_authoritative_knowledge_matrix(DATA_PATH)
initial_knowledge_matrix = knowledge_matrix.copy()
print(f"Authoritative knowledge matrix loaded with verified shape: {knowledge_matrix.shape}")

# ===================== LOAD PRE-TRAINED FT-TRANSFORMER =====================
print("Loading pre-trained FT-Transformer...")
config = FTConfig()
ft_model = FTTransformer()
checkpoint = torch.load(OUTPUT_PATH / "best_model.pth", map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    ft_model.load_state_dict(checkpoint["model_state_dict"])
else:
    ft_model.load_state_dict(checkpoint)
ft_model = ft_model.to(DEVICE)

# ===================== BUILD KG-HYBRID =====================
print("Building KG-Hybrid model...")

# 1. XGBoost Embedder
xgb_embedder = XGBLeafEmbedder(num_unique_leaves=total_unique_leaves, num_trees=num_trees, embedding_dim=128)
xgb_embedder = xgb_embedder.to(DEVICE)

# 2. Fusion Module
fusion = KnowledgeGuidedFusion(
    dim_xgb=128,
    dim_ft=config.EMBEDDING_DIM,
    num_crops=22,
    num_features=7,
    knowledge_matrix=knowledge_matrix,
    knowledge_policy=KNOWLEDGE_POLICY,
)
fusion = fusion.to(DEVICE)

# 3. Full KG-Hybrid
model = KGHybrid(xgb_embedder, ft_model, fusion, num_classes=22)
model = model.to(DEVICE)

# ===================== DATALOADERS =====================
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

# ===================== OPTIMIZER =====================
optimizer = optim.AdamW(
    list(xgb_embedder.parameters()) + 
    list(fusion.parameters()) + 
    list(model.head.parameters()) + 
    list(ft_model.parameters()),
    lr=1e-3,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ===================== TRAINING LOOP =====================
print("\nStarting KG-Hybrid Training...")
best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for leaf_indices, X_ft, y in train_loader:
        leaf_indices = leaf_indices.to(DEVICE)
        X_ft = X_ft.to(DEVICE)
        y = y.to(DEVICE)
        
        # Forward pass
        logits_fused, f_xgb, f_ft, f_fused, gate = model(leaf_indices, X_ft)
        
        # Get XGBoost and FT logits for distillation
        with torch.no_grad():
            df_batch = pd.DataFrame(X_ft.cpu().numpy(), columns=FEATURES)
            dmat = xgb.DMatrix(df_batch)
            xgb_margin = booster.predict(dmat, output_margin=True)
            logits_xgb = torch.tensor(xgb_margin, dtype=torch.float32, device=DEVICE)
            logits_ft = ft_model(X_ft)
        
        # Compute loss
        loss, loss_ce, loss_distill, loss_contrast = combined_loss(
            logits_fused, logits_xgb, logits_ft, y, f_fused,
            alpha_ce=1.0,
            alpha_distill=0.5,
            alpha_contrast=0.1,
            knowledge_anchor_loss=model.fusion.knowledge_anchor_loss(),
            alpha_knowledge=KNOWLEDGE_ANCHOR_WEIGHT,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, preds = torch.max(logits_fused, 1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)
    
    train_acc = train_correct / train_total
    train_loss_avg = train_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for leaf_indices, X_ft, y in val_loader:
            leaf_indices = leaf_indices.to(DEVICE)
            X_ft = X_ft.to(DEVICE)
            y = y.to(DEVICE)
            
            logits_fused, _, _, f_fused, _ = model(leaf_indices, X_ft)
            
            loss = nn.CrossEntropyLoss()(logits_fused, y)
            val_loss += loss.item()
            
            _, preds = torch.max(logits_fused, 1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
    
    val_acc = val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)
    
    # Save history
    history['train_loss'].append(train_loss_avg)
    history['val_loss'].append(val_loss_avg)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
    
    # LR Scheduler
    scheduler.step(val_loss_avg)
    
    # Save best model
    CORRECTED_DIR = OUTPUT_PATH / "corrected"
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTED_CKPT_PATH = OUTPUT_PATH / "kg_hybrid_corrected.pth"

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        
        # Save versioned checkpoint without overwriting legacy kg_hybrid_best.pth
        torch.save(model.state_dict(), CORRECTED_CKPT_PATH)
        torch.save(model.state_dict(), CORRECTED_DIR / "kg_hybrid_best.pth")
        
        # Capture and save learned knowledge matrix
        learned_km = model.fusion.knowledge_matrix.detach().cpu().numpy()
        np.save(CORRECTED_DIR / "knowledge_matrix_learned.npy", learned_km)
        km_df_learned = pd.DataFrame(learned_km, index=crop_names, columns=feature_names)
        km_df_learned.to_csv(TABLES_PATH / "knowledge_matrix_learned.csv")
        
        km_diff = np.abs(learned_km - initial_knowledge_matrix)
        mean_abs_change = float(np.mean(km_diff))
        max_abs_change = float(np.max(km_diff))
        
        print(f"[BEST] Corrected model saved at epoch {best_epoch} with val acc: {best_val_acc:.4f}")
        print(f"       Knowledge matrix adaptation: mean |Delta| = {mean_abs_change:.6f}, max |Delta| = {max_abs_change:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[STOP] Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n[SUMMARY] Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

# ===================== EVALUATE ON TEST SET =====================
print("\nEvaluating on Test Set with Corrected Architecture Checkpoint...")
model.load_state_dict(torch.load(CORRECTED_CKPT_PATH))
model.eval()

test_preds = []
test_probs = []
test_loss = 0.0

with torch.no_grad():
    for leaf_indices, X_ft, y in test_loader:
        leaf_indices = leaf_indices.to(DEVICE)
        X_ft = X_ft.to(DEVICE)
        y = y.to(DEVICE)
        
        logits_fused, _, _, _, _ = model(leaf_indices, X_ft)
        loss = nn.CrossEntropyLoss()(logits_fused, y)
        test_loss += loss.item()
        
        probs = torch.softmax(logits_fused, dim=1)
        _, preds = torch.max(logits_fused, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

test_acc = np.mean(np.array(test_preds) == y_test)
print(f"[OK] KG-Hybrid Test Accuracy: {test_acc:.4f}")

# ===================== SAVE METRICS & MANIFEST =====================
# Save test predictions for evaluation
test_df['kg_hybrid_pred'] = test_preds
test_df.to_csv(DATA_PATH / "test_with_predictions.csv", index=False)

# Save history
with open(TABLES_PATH / "kg_hybrid_training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# Save final metrics
final_km = model.fusion.knowledge_matrix.detach().cpu().numpy()
km_diff_final = np.abs(final_km - initial_knowledge_matrix)

metrics = {
    "architecture_version": "v2_corrected_knowledge_attention",
    "checkpoint_path": str(CORRECTED_CKPT_PATH),
    "test_accuracy": float(test_acc),
    "best_val_accuracy": float(best_val_acc),
    "best_epoch": int(best_epoch),
    "final_epoch": int(epoch + 1),
    "knowledge_matrix_policy": KNOWLEDGE_POLICY,
    "knowledge_anchor_weight": KNOWLEDGE_ANCHOR_WEIGHT,
    "knowledge_mean_abs_change": float(np.mean(km_diff_final)),
    "knowledge_max_abs_change": float(np.max(km_diff_final))
}
with open(TABLES_PATH / "kg_hybrid_final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(CORRECTED_DIR / "training_metadata.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save Manifest distinguishing legacy from corrected architecture
manifest = {
    "legacy_architecture": {
        "version": "v1_static_knowledge_bias",
        "checkpoint": "outputs/models/kg_hybrid_best.pth",
        "backup_checkpoint": "outputs/models/legacy/kg_hybrid_best_legacy_static_bias.pth",
        "description": "Problem 1 and Problem 2 affected; static k_bias broadcast [1, 192]; incompatible with v2"
    },
    "corrected_architecture": {
        "version": "v2_corrected_knowledge_attention",
        "checkpoint": "outputs/models/kg_hybrid_corrected.pth",
        "directory_checkpoint": "outputs/models/corrected/kg_hybrid_best.pth",
        "description": "Problem 1 cancelled gate removed; Problem 2 dynamic sample-conditioned crop attention active",
        "training_metadata": str(CORRECTED_DIR / "training_metadata.json"),
        "metrics": metrics
    }
}
with open(OUTPUT_PATH / "checkpoint_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

evaluate_model("kg_hybrid", y_test, np.array(test_preds))

print("\n[OK] KG-Hybrid fresh training complete!")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Corrected Model saved to: {CORRECTED_CKPT_PATH}")
print(f"Legacy model preserved at: {OUTPUT_PATH / 'kg_hybrid_best.pth'}")
