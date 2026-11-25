"""
Ablation Study Runner - Full Suite (A1 to A7)
Supports running individual ablations or the full suite sequentially.
"""

import math
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
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import base models and losses
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.auxiliary_losses import combined_loss
from src.utils.evaluation import evaluate_model

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
KNOWLEDGE_POLICY = "anchored_trainable"
KNOWLEDGE_ANCHOR_WEIGHT = 1.0

# =====================================================
# Fusion Variants for Ablations
# =====================================================

class FusionWithoutKnowledge(nn.Module):
    """A1: Removes knowledge matrix (k_bias = 0), retains adaptive gate."""
    def __init__(self, dim_xgb=128, dim_ft=64):
        super().__init__()
        self.dim_total = dim_xgb + dim_ft
        self.gate = nn.Sequential(
            nn.Linear(self.dim_total, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.dim_total),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(self.dim_total)
    
    def forward(self, f_xgb, f_ft):
        concat = torch.cat([f_xgb, f_ft], dim=1)
        gate = self.gate(concat)
        f_pre = gate * concat
        f_fused = self.layer_norm(f_pre)
        return f_fused, gate


class FusionWithoutGate(KnowledgeGuidedFusion):
    """
    A2/A8: Removes adaptive gate (unmodulated concatenation + sample-conditioned knowledge context).
    Gate is fixed to 1.0 (unmodulated pass-through).
    """
    def __init__(
        self,
        knowledge_matrix,
        dim_xgb: int = 128,
        dim_ft: int = 64,
        num_crops: int = 22,
        num_features: int = 7,
        dim_k: int = 64,
        beta: float = 0.1,
        knowledge_policy: str = "anchored_trainable",
    ):
        super().__init__(
            dim_xgb=dim_xgb,
            dim_ft=dim_ft,
            num_crops=num_crops,
            num_features=num_features,
            knowledge_matrix=knowledge_matrix,
            dim_k=dim_k,
            beta=beta,
            knowledge_policy=knowledge_policy,
        )
        
    def forward(self, f_xgb, f_ft, return_attention=False):
        concat = torch.cat([f_xgb, f_ft], dim=1)
        
        # Keep the identical input-conditioned knowledge mechanism; only the
        # feature gate is ablated.
        k_bias, attention_weights = self._compute_knowledge_bias(concat)
        
        # No adaptive gate: unmodulated pass-through + sample-conditioned knowledge context
        f_pre = concat + self.beta * k_bias
        f_fused = self.layer_norm(f_pre)
        dummy_gate = torch.ones_like(concat)
        
        if return_attention:
            return f_fused, dummy_gate, attention_weights
        return f_fused, dummy_gate


# =====================================================
# Model Variants for Ablations
# =====================================================

class KGHybridBase(nn.Module):
    """Full Model or Ablated Architecture."""
    def __init__(self, xgb_embedder, ft_transformer, fusion_module, num_classes=22, disable_xgb=False, disable_ft=False):
        super().__init__()
        self.xgb_embedder = xgb_embedder
        self.ft_transformer = ft_transformer
        self.fusion = fusion_module
        self.disable_xgb = disable_xgb
        self.disable_ft = disable_ft
        
        dim = fusion_module.dim_total if hasattr(fusion_module, 'dim_total') else 192
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, leaf_indices, X_ft):
        if self.disable_xgb:
            f_xgb = torch.zeros(leaf_indices.size(0), 128, device=leaf_indices.device)
        else:
            f_xgb = self.xgb_embedder(leaf_indices)
            
        if self.disable_ft:
            f_ft = torch.zeros(X_ft.size(0), 64, device=X_ft.device)
        else:
            if hasattr(self.ft_transformer, 'extract_features'):
                f_ft = self.ft_transformer.extract_features(X_ft)
            else:
                f_ft = self.ft_transformer(X_ft, return_features=True)
            
        f_fused, gate = self.fusion(f_xgb, f_ft)
        logits = self.head(f_fused)
        return logits, f_xgb, f_ft, f_fused, gate


# =====================================================
# General Ablation Engine
# =====================================================

def run_ablation(ablation_key):
    """
    Executes a specific ablation study experiment:
    A1: w/o Knowledge Matrix
    A2: w/o Adaptive Gate
    A3: w/o XGBoost Branch
    A4: w/o FT-Transformer Branch
    A5: w/o Distillation Loss
    A6: w/o Contrastive Loss
    A7: w/o Auxiliary Losses
    A8: w/o Adaptive Fusion (Static Equal Concatenation)
    """
    configs = {
        "A1": {"name": "w/o Knowledge Matrix", "use_knowledge": False, "use_gate": True, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.5, "alpha_contrast": 0.1},
        "A2": {"name": "w/o Adaptive Gate", "use_knowledge": True, "use_gate": False, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.5, "alpha_contrast": 0.1},
        "A3": {"name": "w/o XGBoost Branch", "use_knowledge": True, "use_gate": True, "disable_xgb": True, "disable_ft": False, "alpha_distill": 0.0, "alpha_contrast": 0.1},
        "A4": {"name": "w/o FT-Transformer Branch", "use_knowledge": True, "use_gate": True, "disable_xgb": False, "disable_ft": True, "alpha_distill": 0.5, "alpha_contrast": 0.1},
        "A5": {"name": "w/o Distillation Loss", "use_knowledge": True, "use_gate": True, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.0, "alpha_contrast": 0.1},
        "A6": {"name": "w/o Contrastive Loss", "use_knowledge": True, "use_gate": True, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.5, "alpha_contrast": 0.0},
        "A7": {"name": "w/o Auxiliary Losses", "use_knowledge": True, "use_gate": True, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.0, "alpha_contrast": 0.0},
        "A8": {"name": "w/o Adaptive Fusion", "use_knowledge": True, "use_gate": False, "disable_xgb": False, "disable_ft": False, "alpha_distill": 0.0, "alpha_contrast": 0.0},
    }

    if ablation_key not in configs:
        print(f"[ERROR] Unknown ablation key: {ablation_key}")
        return None

    cfg = configs[ablation_key]
    print("=" * 60)
    print(f"RUNNING ABLATION {ablation_key}: {cfg['name']}")
    print("=" * 60)
    
    DATA_PATH = PROJECT_ROOT / "data" / "processed"
    OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models"
    ABLATION_PATH = PROJECT_ROOT / "outputs" / "ablation"
    ABLATION_PATH.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    train_df = pd.read_csv(DATA_PATH / "train.csv")
    val_df = pd.read_csv(DATA_PATH / "validation.csv")
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    
    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_val = val_df.drop("label", axis=1).values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int64)
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)
    
    # Load XGBoost booster
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

    # Authoritative Knowledge Matrix
    from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
    knowledge_matrix, _, _ = load_authoritative_knowledge_matrix(DATA_PATH)
    
    # FT-Transformer
    ft_config = FTConfig()
    ft_model = FTTransformer()
    checkpoint_path = OUTPUT_PATH / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ft_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            ft_model.load_state_dict(checkpoint)
    ft_model = ft_model.to(DEVICE)
    
    # XGB Embedder
    xgb_embedder = XGBLeafEmbedder(
        num_unique_leaves=total_unique_leaves,
        num_trees=num_trees,
        embedding_dim=128
    ).to(DEVICE)
    
    # Fusion Selection
    if not cfg["use_knowledge"]:
        fusion = FusionWithoutKnowledge(dim_xgb=128, dim_ft=ft_config.EMBEDDING_DIM).to(DEVICE)
    elif not cfg["use_gate"]:
        fusion = FusionWithoutGate(
            knowledge_matrix,
            dim_xgb=128,
            dim_ft=ft_config.EMBEDDING_DIM,
            knowledge_policy=KNOWLEDGE_POLICY,
        ).to(DEVICE)
    else:
        fusion = KnowledgeGuidedFusion(
            dim_xgb=128,
            dim_ft=ft_config.EMBEDDING_DIM,
            knowledge_matrix=knowledge_matrix,
            knowledge_policy=KNOWLEDGE_POLICY,
        ).to(DEVICE)
        
    model = KGHybridBase(
        xgb_embedder, ft_model, fusion, num_classes=22,
        disable_xgb=cfg["disable_xgb"], disable_ft=cfg["disable_ft"]
    ).to(DEVICE)
    
    BATCH_SIZE = 32
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
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    EPOCHS = 100
    PATIENCE = 15
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for leaf_indices, X_ft, y in train_loader:
            leaf_indices, X_ft, y = leaf_indices.to(DEVICE), X_ft.to(DEVICE), y.to(DEVICE)
            
            logits_fused, f_xgb, f_ft, f_fused, gate = model(leaf_indices, X_ft)
            
            with torch.no_grad():
                df_batch = pd.DataFrame(X_ft.cpu().numpy(), columns=FEATURES)
                dmat = xgb.DMatrix(df_batch)
                xgb_margin = booster.predict(dmat, output_margin=True)
                logits_xgb = torch.tensor(xgb_margin, dtype=torch.float32, device=DEVICE)
                logits_ft = ft_model(X_ft)
            
            loss, loss_ce, loss_distill, loss_contrast = combined_loss(
                logits_fused, logits_xgb, logits_ft, y, f_fused,
                alpha_ce=1.0,
                alpha_distill=cfg["alpha_distill"],
                alpha_contrast=cfg["alpha_contrast"],
                knowledge_anchor_loss=(
                    model.fusion.knowledge_anchor_loss()
                    if hasattr(model.fusion, "knowledge_anchor_loss")
                    else None
                ),
                alpha_knowledge=(KNOWLEDGE_ANCHOR_WEIGHT if cfg["use_knowledge"] else 0.0),
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(logits_fused, 1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_acc = train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for leaf_indices, X_ft, y in val_loader:
                leaf_indices, X_ft, y = leaf_indices.to(DEVICE), X_ft.to(DEVICE), y.to(DEVICE)
                logits_fused, _, _, _, _ = model(leaf_indices, X_ft)
                loss = nn.CrossEntropyLoss()(logits_fused, y)
                val_loss += loss.item()
                _, preds = torch.max(logits_fused, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss_avg)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ABLATION_PATH / f"{ablation_key.lower()}_best.pth")
            print(f"[BEST] Model saved at epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[STOP] Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n[SUMMARY] {ablation_key} Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Evaluate Test Set
    model.load_state_dict(torch.load(ABLATION_PATH / f"{ablation_key.lower()}_best.pth"))
    model.eval()
    
    test_preds = []
    with torch.no_grad():
        for leaf_indices, X_ft, y in test_loader:
            leaf_indices, X_ft = leaf_indices.to(DEVICE), X_ft.to(DEVICE)
            logits_fused, _, _, _, _ = model(leaf_indices, X_ft)
            _, preds = torch.max(logits_fused, 1)
            test_preds.extend(preds.cpu().numpy())
    
    metrics = evaluate_model(f"ablation_{ablation_key.lower()}", y_test, np.array(test_preds))
    test_acc = metrics["Accuracy"]
    
    results = {
        "ablation": ablation_key,
        "name": cfg["name"],
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "f1_score": metrics["F1-Score"],
        "best_epoch": best_epoch,
        "device": str(DEVICE),
        "knowledge_matrix_policy": KNOWLEDGE_POLICY if cfg["use_knowledge"] else "not_applicable",
        "knowledge_anchor_weight": KNOWLEDGE_ANCHOR_WEIGHT if cfg["use_knowledge"] else 0.0,
    }
    
    with open(ABLATION_PATH / f"{ablation_key.lower()}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] {ablation_key} Complete! Test Accuracy: {test_acc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="A2", help="Ablation experiment key (A1-A7, or ALL)")
    args = parser.parse_args()
    
    if args.ablation.upper() == "ALL":
        for key in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]:
            run_ablation(key)
    else:
        run_ablation(args.ablation.upper())


if __name__ == "__main__":
    main()
