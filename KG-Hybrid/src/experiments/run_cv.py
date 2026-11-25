"""
File: src/experiments/run_cv.py

Purpose:
Rigorous Stratified 5-Fold Cross-Validation Protocol for KG-Hybrid Crop Recommendation.

Key Features:
- StratifiedKFold (k=5, shuffle=True, random_state=42)
- Zero Data Leakage: StandardScaler fit strictly on fold-training data per fold
- Shared fold assignments across all 6 baseline and proposed models
- Fresh model parameter & seed reset per fold
- Collection of per-fold metrics, aggregate mean ± SD, per-class metrics, and complete Out-Of-Fold (OOF) predictions
"""

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb

from src.experiments.seed_utils import set_all_seeds
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
from src.models.hybrid.auxiliary_losses import combined_loss

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
KNOWLEDGE_POLICY = "anchored_trainable"
KNOWLEDGE_ANCHOR_WEIGHT = 1.0


def load_raw_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Loads raw crop recommendation CSV dataset."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset missing at {csv_path}")

    df = pd.read_csv(csv_path)
    X_df = df[FEATURES].copy()
    y_df = df["label"].copy()
    return X_df, y_df


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray,
    train_time: float,
    infer_time: float,
    num_classes: int = 22
) -> Dict[str, Any]:
    """Compute per-fold evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wei, r_wei, f1_wei, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": round(acc, 6),
        "precision_macro": round(float(p_mac), 6),
        "recall_macro": round(float(r_mac), 6),
        "f1_macro": round(float(f1_mac), 6),
        "precision_weighted": round(float(p_wei), 6),
        "recall_weighted": round(float(r_wei), 6),
        "f1_weighted": round(float(f1_wei), 6),
        "training_time_sec": round(float(train_time), 4),
        "inference_time_ms_per_sample": round(float((infer_time / len(y_true)) * 1000.0), 4),
        "per_class_precision": [round(float(v), 4) for v in p_cls],
        "per_class_recall": [round(float(v), 4) for v in r_cls],
        "per_class_f1": [round(float(v), 4) for v in f1_cls],
        "confusion_matrix": cm.tolist()
    }


def train_fold_ft_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, float, FTTransformer]:
    """Trains a fold-isolated FT-Transformer model on fold training data."""
    t0 = time.time()
    ft_model = FTTransformer().to(device)
    ft_opt = torch.optim.AdamW(ft_model.parameters(), lr=1e-3, weight_decay=1e-5)
    ft_crit = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    ft_model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            ft_opt.zero_grad()
            out = ft_model(xb)
            loss = ft_crit(out, yb)
            loss.backward()
            ft_opt.step()

    train_time = time.time() - t0

    ft_model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        logits = ft_model(x_val_t)
        probas = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probas, axis=1)

    return preds, probas, train_time, ft_model


def train_fold_kg_hybrid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    booster: xgb.Booster,
    ft_backbone: FTTransformer,
    knowledge_matrix: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 32,
    knowledge_policy: str = KNOWLEDGE_POLICY,
    knowledge_anchor_weight: float = KNOWLEDGE_ANCHOR_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Trains genuine KG-Hybrid model on fold training data without data leakage."""
    t0 = time.time()

    def get_leaves(X):
        df_X = pd.DataFrame(X, columns=FEATURES)
        dmat = xgb.DMatrix(df_X)
        return booster.predict(dmat, pred_leaf=True)

    leaf_train = get_leaves(X_train)
    leaf_val = get_leaves(X_val)

    max_leaves_per_tree = int(max(leaf_train.max(), leaf_val.max())) + 1
    num_trees = leaf_train.shape[1]
    tree_offsets = np.arange(num_trees, dtype=np.int64) * max_leaves_per_tree

    l_train_offset = leaf_train + tree_offsets
    l_val_offset = leaf_val + tree_offsets
    total_unique_leaves = num_trees * max_leaves_per_tree

    embedder = XGBLeafEmbedder(
        num_unique_leaves=total_unique_leaves,
        num_trees=num_trees,
        embedding_dim=128,
        # A multi-class 200-round booster yields 4,400 leaf positions.  Sparse
        # embedding gradients avoid a dense AdamW update of every unused leaf
        # vector on each minibatch while preserving the same leaf inputs.
        sparse=True,
    ).to(device)

    ft_clone = FTTransformer().to(device)
    ft_clone.load_state_dict(ft_backbone.state_dict())

    fusion = KnowledgeGuidedFusion(
        dim_xgb=128, dim_ft=64, num_crops=num_classes,
        num_features=7,
        knowledge_matrix=knowledge_matrix,
        knowledge_policy=knowledge_policy,
    ).to(device)

    model = KGHybrid(
        xgb_embedder=embedder,
        ft_transformer=ft_clone,
        fusion_module=fusion,
        num_classes=num_classes
    ).to(device)

    embedding_parameters = list(embedder.embeddings.parameters())
    embedding_parameter_ids = {id(parameter) for parameter in embedding_parameters}
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in embedding_parameter_ids
    ]
    optimizer = torch.optim.AdamW(dense_parameters, lr=1e-3, weight_decay=1e-5)
    embedding_optimizer = torch.optim.SparseAdam(embedding_parameters, lr=1e-3)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(l_train_offset, dtype=torch.long),
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for b_leaves, b_X, b_y in train_loader:
            b_leaves, b_X, b_y = b_leaves.to(device), b_X.to(device), b_y.to(device)
            logits_fused, f_xgb, f_ft, f_fused, gate = model(b_leaves, b_X)

            with torch.no_grad():
                df_batch = pd.DataFrame(b_X.cpu().numpy(), columns=FEATURES)
                dmat_b = xgb.DMatrix(df_batch)
                xgb_margin = booster.predict(dmat_b, output_margin=True)
                logits_xgb = torch.tensor(xgb_margin, dtype=torch.float32, device=device)
                logits_ft = ft_backbone(b_X)

            loss, _, _, _ = combined_loss(
                logits_fused, logits_xgb, logits_ft, b_y, f_fused,
                alpha_ce=1.0,
                alpha_distill=0.5,
                alpha_contrast=0.1,
                knowledge_anchor_loss=model.fusion.knowledge_anchor_loss(),
                alpha_knowledge=knowledge_anchor_weight,
            )
            optimizer.zero_grad()
            embedding_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
            optimizer.step()
            embedding_optimizer.step()

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        l_val_t = torch.tensor(l_val_offset, dtype=torch.long, device=device)
        x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        logits_fused, _, _, _, _ = model(l_val_t, x_val_t)
        probas = torch.softmax(logits_fused, dim=1).cpu().numpy()
        preds = np.argmax(probas, axis=1)

    return preds, probas, train_time


def resolve_output_dir(include_neural: bool, output_dir: str | Path | None = None) -> Path:
    """Resolve an isolated output directory for a CV run.

    The historical ``outputs/cross_validation`` directory contains only the
    classical-model OOF predictions and incomplete neural fold metadata.  A
    neural recovery run must never write into that directory, since mixing the
    old and recovered evidence would make validation ambiguous.
    """
    if output_dir is None:
        directory = (
            PROJECT_ROOT / "outputs" / "cross_validation_neural_recovery"
            if include_neural
            else PROJECT_ROOT / "outputs" / "cross_validation"
        )
    else:
        directory = Path(output_dir)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory

    return directory.resolve()


def prepare_output_dir(output_dir: Path, include_neural: bool) -> None:
    """Create a safe new run directory without replacing prior evidence."""
    contains_artifact = output_dir.exists() and any(
        path.is_file() for path in output_dir.rglob("*")
    )
    if contains_artifact:
        if include_neural:
            raise FileExistsError(
                f"Refusing to overwrite existing neural-CV evidence: {output_dir}. "
                "Choose a new --output-dir for a fresh recovery run."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("per_fold", "aggregate", "oof_predictions", "logs"):
        (output_dir / name).mkdir(exist_ok=True)


def run_stratified_5fold_cv(
    include_neural: bool = False,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Execute an isolated stratified five-fold CV experiment suite.

    ``include_neural=True`` defaults to a new recovery directory rather than
    the legacy classical-only directory.  Pass ``output_dir`` to name a
    specific, empty destination for a recovery run.
    """
    print("=" * 70)
    print("EXECUTING STRATIFIED 5-FOLD CROSS-VALIDATION EXPERIMENT")
    print("=" * 70)

    X_df, y_df = load_raw_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_df)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # 1. Stratified 5-Fold Partition
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(X_df, y_encoded))

    if include_neural:
        models_list = [
            "LogisticRegression",
            "DecisionTree",
            "RandomForest",
            "XGBoostBaseline",
            "FTTransformerBaseline",
            "KG-HybridProposed"
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        km, _, _ = load_authoritative_knowledge_matrix()
    else:
        models_list = [
            "LogisticRegression",
            "DecisionTree",
            "RandomForest",
            "XGBoostBaseline"
        ]

    output_dir = resolve_output_dir(include_neural, output_dir)
    prepare_output_dir(output_dir, include_neural)

    per_fold_results = {m: [] for m in models_list}
    oof_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n--- Running Fold {fold_idx}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        set_all_seeds(42 + fold_idx)

        # Zero Leakage Preprocessing: Fit scaler ONLY on fold training data
        X_train_raw = X_df.iloc[train_idx].values
        y_train = y_encoded[train_idx]
        X_val_raw = X_df.iloc[val_idx].values
        y_val = y_encoded[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        fold_preds = {}
        fold_probas = {}

        # 1. Logistic Regression
        t0 = time.time()
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        fold_preds["LogisticRegression"] = lr.predict(X_val)
        fold_probas["LogisticRegression"] = lr.predict_proba(X_val)
        t_infer = time.time() - t0

        res_lr = evaluate_predictions(y_val, fold_preds["LogisticRegression"], fold_probas["LogisticRegression"], t_train, t_infer, num_classes)
        per_fold_results["LogisticRegression"].append(res_lr)

        # 2. Decision Tree
        t0 = time.time()
        dt = DecisionTreeClassifier(max_depth=6, random_state=42)
        dt.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        fold_preds["DecisionTree"] = dt.predict(X_val)
        fold_probas["DecisionTree"] = dt.predict_proba(X_val)
        t_infer = time.time() - t0

        res_dt = evaluate_predictions(y_val, fold_preds["DecisionTree"], fold_probas["DecisionTree"], t_train, t_infer, num_classes)
        per_fold_results["DecisionTree"].append(res_dt)

        # 3. Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        fold_preds["RandomForest"] = rf.predict(X_val)
        fold_probas["RandomForest"] = rf.predict_proba(X_val)
        t_infer = time.time() - t0

        res_rf = evaluate_predictions(y_val, fold_preds["RandomForest"], fold_probas["RandomForest"], t_train, t_infer, num_classes)
        per_fold_results["RandomForest"].append(res_rf)

        # 4. XGBoost Baseline
        t0 = time.time()
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=42
        )
        xgb_clf.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        fold_preds["XGBoostBaseline"] = xgb_clf.predict(X_val)
        fold_probas["XGBoostBaseline"] = xgb_clf.predict_proba(X_val)
        t_infer = time.time() - t0

        res_xgb = evaluate_predictions(y_val, fold_preds["XGBoostBaseline"], fold_probas["XGBoostBaseline"], t_train, t_infer, num_classes)
        per_fold_results["XGBoostBaseline"].append(res_xgb)

        # 5. FT-Transformer Baseline & 6. KG-Hybrid Proposed
        if include_neural:
            print(f"  [Fold {fold_idx}] Training genuine FT-Transformer backbone...")
            preds_ft, probas_ft, t_ft_train, ft_backbone = train_fold_ft_transformer(
                X_train, y_train, X_val, num_classes, device
            )
            fold_preds["FTTransformerBaseline"] = preds_ft
            fold_probas["FTTransformerBaseline"] = probas_ft
            res_ft = evaluate_predictions(y_val, preds_ft, probas_ft, t_ft_train, 0.0, num_classes)
            per_fold_results["FTTransformerBaseline"].append(res_ft)

            print(f"  [Fold {fold_idx}] Training genuine KG-Hybrid model...")
            preds_kg, probas_kg, t_kg_train = train_fold_kg_hybrid(
                X_train, y_train, X_val, xgb_clf.get_booster(), ft_backbone,
                km, num_classes, device
            )
            fold_preds["KG-HybridProposed"] = preds_kg
            fold_probas["KG-HybridProposed"] = probas_kg
            res_kgh = evaluate_predictions(y_val, preds_kg, probas_kg, t_kg_train, 0.0, num_classes)
            per_fold_results["KG-HybridProposed"].append(res_kgh)
        else:
            print(f"  [Fold {fold_idx}] Neural models (FT-Transformer & KG-Hybrid) marked incomplete pending dedicated compute.")

        # Save fold metrics JSON
        fold_json = {m: per_fold_results[m][-1] for m in models_list}
        with open(output_dir / "per_fold" / f"fold_{fold_idx}.json", "w") as f:
            json.dump(fold_json, f, indent=4)

        # Record OOF predictions
        for i_pos, global_sample_idx in enumerate(val_idx):
            for m in models_list:
                pred_c = int(fold_preds[m][i_pos])
                prob_vec = fold_probas[m][i_pos].tolist()
                oof_rows.append({
                    "sample_id": int(global_sample_idx),
                    "fold_id": fold_idx,
                    "model_id": m,
                    "true_label": int(y_val[i_pos]),
                    "true_crop": class_names[y_val[i_pos]],
                    "predicted_label": pred_c,
                    "predicted_crop": class_names[pred_c],
                    "top_confidence": round(float(np.max(prob_vec)), 6)
                })

    # Save Out-Of-Fold predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_path = output_dir / "oof_predictions" / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[SUCCESS] Out-of-Fold predictions saved to {oof_path}")

    # Aggregate Metrics Calculation (Mean ± SD)
    agg_rows = []
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]

    for m in models_list:
        row = {"Model": m}
        for k in metric_keys:
            vals = [fold_res[k] for fold_res in per_fold_results[m]]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{k}_mean"] = round(mean_v, 4)
            row[f"{k}_std"] = round(std_v, 4)
            row[f"{k}_formatted"] = f"{mean_v:.4f} ± {std_v:.4f}"
        agg_rows.append(row)

    summary_df = pd.DataFrame(agg_rows)
    summary_path = output_dir / "aggregate" / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SUCCESS] Aggregate CV Summary saved to {summary_path}")

    # Per-Class Aggregation Across Folds
    per_class_rows = []
    for m in models_list:
        p_matrix = np.array([fold_res["per_class_precision"] for fold_res in per_fold_results[m]])
        r_matrix = np.array([fold_res["per_class_recall"] for fold_res in per_fold_results[m]])
        f1_matrix = np.array([fold_res["per_class_f1"] for fold_res in per_fold_results[m]])

        p_means, p_stds = np.mean(p_matrix, axis=0), np.std(p_matrix, axis=0, ddof=1)
        r_means, r_stds = np.mean(r_matrix, axis=0), np.std(r_matrix, axis=0, ddof=1)
        f1_means, f1_stds = np.mean(f1_matrix, axis=0), np.std(f1_matrix, axis=0, ddof=1)

        for c_idx, c_name in enumerate(class_names):
            per_class_rows.append({
                "Model": m,
                "Crop": c_name,
                "Precision_Mean_SD": f"{p_means[c_idx]:.4f} ± {p_stds[c_idx]:.4f}",
                "Recall_Mean_SD": f"{r_means[c_idx]:.4f} ± {r_stds[c_idx]:.4f}",
                "F1_Mean_SD": f"{f1_means[c_idx]:.4f} ± {f1_stds[c_idx]:.4f}",
                "F1_Mean": round(float(f1_means[c_idx]), 4),
                "F1_SD": round(float(f1_stds[c_idx]), 4)
            })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = output_dir / "aggregate" / "cv_per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"[SUCCESS] Per-class CV metrics saved to {per_class_path}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "models": models_list,
        "include_neural": include_neural,
        "n_samples": int(len(X_df)),
        "n_classes": int(num_classes),
        "features": FEATURES,
        "oof_predictions": str(oof_path.relative_to(PROJECT_ROOT)),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "per_class_metrics": str(per_class_path.relative_to(PROJECT_ROOT)),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Run manifest saved to {manifest_path}")

    print("\n5-Fold Cross-Validation Aggregate Metrics (Mean ± SD):")
    display_cols = ["Model", "accuracy_formatted", "precision_macro_formatted", "recall_macro_formatted", "f1_macro_formatted"]
    print(summary_df[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return {
        "summary": agg_rows,
        "oof_count": len(oof_df),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stratified 5-Fold Cross-Validation Protocol")
    parser.add_argument(
        "--include_neural", action="store_true", default=False,
        help="Train genuine neural models (FT-Transformer & KG-Hybrid) within each fold."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Empty directory for this CV run, relative to the project root or absolute. "
            "Neural recovery runs must use a new directory."
        ),
    )
    args = parser.parse_args()
    run_stratified_5fold_cv(
        include_neural=args.include_neural,
        output_dir=args.output_dir,
    )
