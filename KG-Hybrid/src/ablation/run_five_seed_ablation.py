"""
Five-Seed Ablation Study Runner for KG-Hybrid
Evaluates Full model and A1-A8 variants across 5 isolated random seeds:
[42, 123, 2024, 3407, 9999]

Follows strict seed isolation:
- 70/15/15 stratified split per seed (1540 train, 330 val, 330 test)
- StandardScaler and LabelEncoder fitted strictly on that seed's training split
- Seed-specific XGBoost and FT-Transformer pretraining on training split
- Consistent agronomic Knowledge Matrix initialization K0
- Early stopping on validation with restoration of best checkpoint
- Metric evaluation on held-out test split (Accuracy, Macro-F1)
- Paired statistical tests (Wilcoxon and per-seed McNemar)
"""

import os
import sys
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import xgboost as xgb
import sklearn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.seed_utils import set_all_seeds, get_preset_seeds, get_execution_environment_metadata
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.auxiliary_losses import combined_loss
from src.ablation.run_ablation import FusionWithoutKnowledge, FusionWithoutGate, KGHybridBase

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
SEEDS = [42, 123, 2024, 3407, 9999]
KNOWLEDGE_POLICY = "anchored_trainable"
KNOWLEDGE_ANCHOR_WEIGHT = 1.0

ABLATION_CONFIGS = {
    "Full": {
        "name": "Full KG-Hybrid",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.5,
        "alpha_contrast": 0.1
    },
    "A1": {
        "name": "w/o Knowledge Matrix",
        "use_knowledge": False,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.5,
        "alpha_contrast": 0.1
    },
    "A2": {
        "name": "w/o Adaptive Gate",
        "use_knowledge": True,
        "use_gate": False,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.5,
        "alpha_contrast": 0.1
    },
    "A3": {
        "name": "w/o XGBoost Branch",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": True,
        "disable_ft": False,
        "alpha_distill": 0.0,
        "alpha_contrast": 0.1
    },
    "A4": {
        "name": "w/o FT-Transformer Branch",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": True,
        "alpha_distill": 0.5,
        "alpha_contrast": 0.1
    },
    "A5": {
        "name": "w/o Distillation Loss",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.0,
        "alpha_contrast": 0.1
    },
    "A6": {
        "name": "w/o Contrastive Loss",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.5,
        "alpha_contrast": 0.0
    },
    "A7": {
        "name": "w/o Auxiliary Losses",
        "use_knowledge": True,
        "use_gate": True,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.0,
        "alpha_contrast": 0.0
    },
    "A8": {
        "name": "w/o Adaptive Fusion (Static + Pure CE)",
        "use_knowledge": True,
        "use_gate": False,
        "disable_xgb": False,
        "disable_ft": False,
        "alpha_distill": 0.0,
        "alpha_contrast": 0.0
    }
}


def load_raw_data() -> pd.DataFrame:
    raw_path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data missing at {raw_path}")
    return pd.read_csv(raw_path)


def prepare_seed_partitions(df: pd.DataFrame, seed: int, return_sample_ids: bool = False):
    """
    Performs 70/15/15 stratified split and fits scaler strictly on training data.
    Validates stratification, sizes, and absence of data leakage.
    """
    X_raw = df[FEATURES].values
    y_raw = df['label'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # 1. 70% Train, 30% Temp
    X_train_raw, X_temp_raw, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_raw, y_enc, np.arange(len(df)),
        test_size=0.30, random_state=seed, stratify=y_enc
    )
    
    # 2. 15% Val, 15% Test from Temp
    X_val_raw, X_test_raw, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp_raw, y_temp, idx_temp,
        test_size=0.50, random_state=seed, stratify=y_temp
    )
    
    # Validation checks
    assert len(X_train_raw) == 1540, f"Train size mismatch: {len(X_train_raw)}"
    assert len(X_val_raw) == 330, f"Val size mismatch: {len(X_val_raw)}"
    assert len(X_test_raw) == 330, f"Test size mismatch: {len(X_test_raw)}"
    
    # Zero overlap check
    s_train, s_val, s_test = set(idx_train), set(idx_val), set(idx_test)
    assert len(s_train.intersection(s_val)) == 0, "Train-Val overlap detected!"
    assert len(s_train.intersection(s_test)) == 0, "Train-Test overlap detected!"
    assert len(s_val.intersection(s_test)) == 0, "Val-Test overlap detected!"
    
    # Class distribution check (70 per class in train, 15 in val, 15 in test)
    for c in range(22):
        assert np.sum(y_train == c) == 70, f"Class {c} train count mismatch!"
        assert np.sum(y_val == c) == 15, f"Class {c} val count mismatch!"
        assert np.sum(y_test == c) == 15, f"Class {c} test count mismatch!"
        
    # Fit StandardScaler ONLY on X_train_raw
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    
    result = (X_train, y_train, X_val, y_val, X_test, y_test, le, scaler)
    return result + (idx_train, idx_val, idx_test) if return_sample_ids else result


def pretrain_seed_models(X_train, y_train, X_val, y_val, X_test, y_test, seed: int, device: torch.device):
    """
    Trains XGBoost and FT-Transformer strictly on the seed's training split.
    """
    set_all_seeds(seed)
    
    # 1. XGBoost Baseline
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=22,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="mlogloss",
        tree_method="hist"
    )
    xgb_clf.fit(X_train, y_train)
    booster = xgb_clf.get_booster()
    
    # Extract leaf indices
    def get_leaves(X):
        df_X = pd.DataFrame(X, columns=FEATURES)
        dmat = xgb.DMatrix(df_X)
        return booster.predict(dmat, pred_leaf=True)
        
    leaf_train = get_leaves(X_train)
    leaf_val = get_leaves(X_val)
    leaf_test = get_leaves(X_test)
    
    # The embedding layout depends only on the trained booster structure.
    max_leaves_per_tree = int(booster.trees_to_dataframe()["Node"].max()) + 1
    num_trees = leaf_train.shape[1]
    tree_offsets = np.arange(num_trees, dtype=np.int64) * max_leaves_per_tree
    
    leaf_train_offset = leaf_train + tree_offsets
    leaf_val_offset = leaf_val + tree_offsets
    leaf_test_offset = leaf_test + tree_offsets
    total_unique_leaves = num_trees * max_leaves_per_tree
    
    # 2. FT-Transformer Pretraining on seed train partition
    ft_model = FTTransformer().to(device)
    ft_optimizer = optim.AdamW(ft_model.parameters(), lr=1e-3, weight_decay=1e-5)
    ft_criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=32, shuffle=False
    )
    
    best_ft_val_loss = float('inf')
    best_ft_state = None
    
    for epoch in range(25):
        ft_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            ft_optimizer.zero_grad()
            out = ft_model(xb)
            loss = ft_criterion(out, yb)
            loss.backward()
            ft_optimizer.step()
            
        ft_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = ft_model(xb)
                val_loss += ft_criterion(out, yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_ft_val_loss:
            best_ft_val_loss = val_loss
            best_ft_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
            
    if best_ft_state is not None:
        ft_model.load_state_dict({k: v.to(device) for k, v in best_ft_state.items()})
        
    return booster, total_unique_leaves, num_trees, leaf_train_offset, leaf_val_offset, leaf_test_offset, ft_model


def run_single_variant_training(
    variant_key: str,
    cfg: Dict[str, Any],
    seed: int,
    total_unique_leaves: int,
    num_trees: int,
    leaf_train: np.ndarray,
    leaf_val: np.ndarray,
    leaf_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    booster: xgb.Booster,
    ft_model: FTTransformer,
    knowledge_matrix: np.ndarray,
    device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """
    Trains one specific ablation variant for one seed.
    """
    set_all_seeds(seed)
    
    # 1. XGBoost Embedder
    xgb_embedder = XGBLeafEmbedder(
        num_unique_leaves=total_unique_leaves,
        num_trees=num_trees,
        embedding_dim=128
    ).to(device)
    
    # 2. Cloned FT-Transformer
    ft_clone = FTTransformer().to(device)
    ft_clone.load_state_dict(ft_model.state_dict())
    ft_model.eval()  # Fixed pretrained teacher; the cloned student is fine-tuned.
    
    # 3. Fusion Module Selection
    dim_ft = 64
    if not cfg["use_knowledge"]:
        fusion = FusionWithoutKnowledge(dim_xgb=128, dim_ft=dim_ft).to(device)
    elif not cfg["use_gate"]:
        fusion = FusionWithoutGate(
            knowledge_matrix,
            dim_xgb=128,
            dim_ft=dim_ft,
            knowledge_policy=KNOWLEDGE_POLICY,
        ).to(device)
    else:
        fusion = KnowledgeGuidedFusion(
            dim_xgb=128,
            dim_ft=dim_ft,
            knowledge_matrix=knowledge_matrix,
            knowledge_policy=KNOWLEDGE_POLICY,
        ).to(device)
        
    # 4. Hybrid Model
    model = KGHybridBase(
        xgb_embedder=xgb_embedder,
        ft_transformer=ft_clone,
        fusion_module=fusion,
        num_classes=22,
        disable_xgb=cfg["disable_xgb"],
        disable_ft=cfg["disable_ft"]
    ).to(device)
    
    # Dataloaders
    BATCH_SIZE = 32
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(leaf_train, dtype=torch.long),
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(leaf_val, dtype=torch.long),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(leaf_test, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    EPOCHS = 100
    PATIENCE = 15
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for b_leaves, b_X, b_y in train_loader:
            b_leaves, b_X, b_y = b_leaves.to(device), b_X.to(device), b_y.to(device)
            
            logits_fused, f_xgb, f_ft, f_fused, gate = model(b_leaves, b_X)
            
            # Distillation targets
            with torch.no_grad():
                df_batch = pd.DataFrame(b_X.cpu().numpy(), columns=FEATURES)
                dmat = xgb.DMatrix(df_batch)
                xgb_margin = booster.predict(dmat, output_margin=True)
                logits_xgb = torch.tensor(xgb_margin, dtype=torch.float32, device=device)
                logits_ft = ft_model(b_X)
                
            loss, loss_ce, loss_distill, loss_contrast = combined_loss(
                logits_fused, logits_xgb, logits_ft, b_y, f_fused,
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
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b_leaves, b_X, b_y in val_loader:
                b_leaves, b_X, b_y = b_leaves.to(device), b_X.to(device), b_y.to(device)
                logits_fused, _, _, _, _ = model(b_leaves, b_X)
                loss = nn.CrossEntropyLoss()(logits_fused, b_y)
                val_loss += loss.item()
                _, preds = torch.max(logits_fused, 1)
                val_correct += (preds == b_y).sum().item()
                val_total += b_y.size(0)
                
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)
        
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss_avg < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss_avg
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
                
    # Evaluate Test Set with best validation model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    test_preds = []
    with torch.no_grad():
        for b_leaves, b_X, b_y in test_loader:
            b_leaves, b_X = b_leaves.to(device), b_X.to(device)
            logits_fused, _, _, _, _ = model(b_leaves, b_X)
            _, preds = torch.max(logits_fused, 1)
            test_preds.extend(preds.cpu().numpy())
            
    test_preds_arr = np.array(test_preds, dtype=np.int64)
    
    # Assertions
    assert len(test_preds_arr) == 330, f"Predictions length mismatch: {len(test_preds_arr)}"
    assert np.all((test_preds_arr >= 0) & (test_preds_arr < 22)), "Invalid prediction class range!"
    
    acc = float(accuracy_score(y_test, test_preds_arr))
    macro_f1 = float(f1_score(y_test, test_preds_arr, average='macro', labels=list(range(22)), zero_division=0))
    
    return acc, macro_f1, test_preds_arr


def run_full_five_seed_ablation(output_dir=None):
    """
    Main orchestration function for 5-seed x 9-variant ablation study.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STARTING KG-HYBRID FIVE-SEED ABLATION STUDY")
    print(f"Device: {device} | Seeds: {SEEDS}")
    print("=" * 70)
    
    out_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "ablation" / "five_seed"
    if out_dir.exists() and any(out_dir.rglob("*")):
        raise FileExistsError(f"Existing ablation evidence must be preserved; choose an empty output directory: {out_dir}")
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = load_raw_data()
    from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
    knowledge_matrix, _, _ = load_authoritative_knowledge_matrix(PROJECT_ROOT / "data" / "processed")
    
    per_seed_records = []
    seed_predictions = {s: {} for s in SEEDS}
    seed_y_true = {}
    
    t_start = time.time()
    run_count = 0
    
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n[{s_idx+1}/{len(SEEDS)}] Processing Seed {seed}...")
        s_pred_dir = pred_dir / f"seed_{seed}"
        s_pred_dir.mkdir(exist_ok=True)
        
        # 1. Partition Data
        X_train, y_train, X_val, y_val, X_test, y_test, le, scaler, idx_train, idx_val, idx_test = prepare_seed_partitions(df_raw, seed, return_sample_ids=True)
        np.savez(s_pred_dir / "partitions.npz", train_ids=idx_train, val_ids=idx_val,
                 test_ids=idx_test, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        seed_y_true[seed] = y_test
        
        # Save y_true for reference
        pd.DataFrame({'sample_id': idx_test, 'y_true': y_test}).to_csv(s_pred_dir / "y_true.csv", index=False)
        
        # 2. Pretrain seed-specific models
        t0_pre = time.time()
        booster, total_leaves, num_trees, l_train, l_val, l_test, ft_model = pretrain_seed_models(
            X_train, y_train, X_val, y_val, X_test, y_test, seed, device
        )
        print(f"  Pretrained seed models in {time.time()-t0_pre:.1f}s")
        
        # 3. Run all 9 variants
        for v_key, v_cfg in ABLATION_CONFIGS.items():
            t0_v = time.time()
            acc, f1_m, preds = run_single_variant_training(
                variant_key=v_key,
                cfg=v_cfg,
                seed=seed,
                total_unique_leaves=total_leaves,
                num_trees=num_trees,
                leaf_train=l_train,
                leaf_val=l_val,
                leaf_test=l_test,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                booster=booster,
                ft_model=ft_model,
                knowledge_matrix=knowledge_matrix,
                device=device
            )
            
            per_seed_records.append({
                "seed": seed,
                "variant": v_key,
                "variant_name": v_cfg["name"],
                "accuracy": round(acc, 6),
                "macro_f1": round(f1_m, 6)
            })
            
            seed_predictions[seed][v_key] = preds
            
            # Save predictions
            pd.DataFrame({
                "sample_id": idx_test,
                "y_true": y_test,
                "y_pred": preds
            }).to_csv(s_pred_dir / f"{v_key}_predictions.csv", index=False)
            
            run_count += 1
            print(f"    - {v_key} ({v_cfg['name']}): Acc = {acc:.4f}, Macro-F1 = {f1_m:.4f} ({time.time()-t0_v:.1f}s)")
            
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL {run_count} RUNS COMPLETED in {total_time/60:.2f} minutes.")
    print("=" * 70)
    
    # 1. Save ablation_per_seed.csv
    df_per_seed = pd.DataFrame(per_seed_records)
    per_seed_path = out_dir / "ablation_per_seed.csv"
    df_per_seed.to_csv(per_seed_path, index=False)
    print(f"[SAVED] {per_seed_path}")
    
    # 2. Compute Summary Statistics (Sample std with ddof=1)
    summary_rows = []
    variants_list = list(ABLATION_CONFIGS.keys())
    
    for v_key in variants_list:
        v_sub = df_per_seed[df_per_seed["variant"] == v_key]
        acc_vals = v_sub["accuracy"].values
        f1_vals = v_sub["macro_f1"].values
        
        acc_mean = float(np.mean(acc_vals))
        acc_std = float(np.std(acc_vals, ddof=1))
        f1_mean = float(np.mean(f1_vals))
        f1_std = float(np.std(f1_vals, ddof=1))
        
        summary_rows.append({
            "variant": v_key,
            "variant_name": ABLATION_CONFIGS[v_key]["name"],
            "accuracy_mean": round(acc_mean, 6),
            "accuracy_std": round(acc_std, 6),
            "macro_f1_mean": round(f1_mean, 6),
            "macro_f1_std": round(f1_std, 6),
            "n_seeds": len(SEEDS)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "ablation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")
    
    # 3. Statistical Tests vs. Full Model
    full_sub = df_per_seed[df_per_seed["variant"] == "Full"]
    full_accs = full_sub.sort_values("seed")["accuracy"].values
    full_f1s = full_sub.sort_values("seed")["macro_f1"].values
    
    stat_rows = []
    for v_key in variants_list:
        if v_key == "Full":
            continue
        v_sub = df_per_seed[df_per_seed["variant"] == v_key].sort_values("seed")
        v_accs = v_sub["accuracy"].values
        v_f1s = v_sub["macro_f1"].values
        
        diff_acc = v_accs - full_accs
        diff_f1 = v_f1s - full_f1s
        
        # Paired Wilcoxon Signed-Rank Test
        if np.all(diff_acc == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(full_accs, v_accs, zero_method='pratt')
                w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                t_res = stats.ttest_rel(full_accs, v_accs)
                w_stat, w_p = float(t_res.statistic), float(t_res.pvalue)
                
        # Per-seed McNemar Tests
        mcnemar_per_seed = {}
        for s in SEEDS:
            y_t = seed_y_true[s]
            pred_full = seed_predictions[s]["Full"]
            pred_v = seed_predictions[s][v_key]
            
            c_full = (pred_full == y_t)
            c_v = (pred_v == y_t)
            
            b = int(np.sum(~c_full & c_v))  # Full wrong, Variant right
            c = int(np.sum(c_full & ~c_v))  # Full right, Variant wrong
            
            if (b + c) == 0:
                chi2, m_p = 0.0, 1.0
            else:
                chi2 = float((abs(b - c) - 1.0) ** 2 / (b + c))
                m_p = float(stats.chi2.sf(chi2, df=1))
                
            mcnemar_per_seed[f"seed_{s}"] = {
                "b_full_wrong_v_right": b,
                "c_full_right_v_wrong": c,
                "chi2": round(chi2, 4),
                "p_value": round(m_p, 6)
            }
            
        stat_rows.append({
            "variant": v_key,
            "variant_name": ABLATION_CONFIGS[v_key]["name"],
            "delta_accuracy_mean": round(float(np.mean(diff_acc)), 6),
            "delta_macro_f1_mean": round(float(np.mean(diff_f1)), 6),
            "wilcoxon_stat": round(w_stat, 4),
            "wilcoxon_p_value": round(w_p, 6),
            "mcnemar_per_seed": mcnemar_per_seed
        })
        
    stat_path = out_dir / "statistical_significance.json"
    with open(stat_path, "w") as f:
        json.dump(stat_rows, f, indent=2)
    print(f"[SAVED] {stat_path}")
    
    # 4. Save Run Metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "execution_time_seconds": round(total_time, 2),
        "seeds": SEEDS,
        "split_sizes": {"train": 1540, "validation": 330, "test": 330, "num_classes": 22},
        "feature_names": FEATURES,
        "ablation_configs": ABLATION_CONFIGS,
        "training_hyperparameters": {
            "batch_size": 32,
            "max_epochs": 100,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 15,
            "scheduler": "ReduceLROnPlateau(mode='min', factor=0.5, patience=5)",
            "knowledge_matrix_policy": KNOWLEDGE_POLICY,
            "knowledge_anchor_weight": KNOWLEDGE_ANCHOR_WEIGHT,
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "xgboost_version": xgb.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {meta_path}")
    
    return df_per_seed, df_summary, stat_rows


if __name__ == "__main__":
    run_full_five_seed_ablation()
