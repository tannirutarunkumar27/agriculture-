"""
File: src/experiments/run_calibration.py

Purpose:
Fully dynamic, reproducible calibration evaluation pipeline for the final KG-Hybrid paper.
Evaluates all 6 models on the held-out test partition (data/processed/test.csv, N=330):
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost Baseline
  5. FT-Transformer Baseline
  6. KG-Hybrid (Proposed)

Generates:
  - outputs/tables/table2_calibration_metrics.csv (Table II)
  - outputs/tables/calibration_bins.csv (Reliability bin points)
  - outputs/figures/reliability_diagram.png
  - outputs/figures/reliability_diagram.pdf
  - outputs/figures/fig_f1_reliability_diagram.png
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.experiments.calibration import compute_ece, compute_brier_score, compute_nll, generate_reliability_diagram_data
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.robustness.robustness_utils import (
    get_leaf_indices_with_offset,
    FEATURES,
    NUM_TREES,
    TOTAL_UNIQUE_LEAVES,
    EMBEDDING_DIM
)

DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CLASSES = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]


def load_and_validate_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and validates processed dataset partitions."""
    train_csv = DATA_PATH / "train.csv"
    test_csv = DATA_PATH / "test.csv"
    encoder_pkl = DATA_PATH / "label_encoder.pkl"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Processed datasets missing in {DATA_PATH}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    le = joblib.load(encoder_pkl)
    actual_classes = list(le.classes_)
    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(f"Class ordering mismatch!\nExpected: {EXPECTED_CLASSES}\nActual: {actual_classes}")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)

    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    if len(X_test) != 330 or len(y_test) != 330:
        raise ValueError(f"Expected 330 test samples, got {len(X_test)}")
    if len(X_train) != 1540 or len(y_train) != 1540:
        raise ValueError(f"Expected 1540 train samples, got {len(X_train)}")

    print(f"[DATA OK] Train: {X_train.shape}, Test: {X_test.shape}, Classes: {len(actual_classes)}")
    return train_df, test_df, X_train, y_train, X_test, y_test


def evaluate_model_probabilities(
    name: str,
    y_true: np.ndarray,
    probas: np.ndarray
) -> Dict[str, Any]:
    """Runs strict sanity checks and computes dynamic calibration metrics."""
    # 1. Sanity Checks
    if probas.shape != (330, 22):
        raise ValueError(f"Model {name}: Expected probas shape (330, 22), got {probas.shape}")
    if not np.all(np.isfinite(probas)):
        raise ValueError(f"Model {name}: probas contains NaN/Inf!")
    if np.any(probas < -1e-6) or np.any(probas > 1.0 + 1e-6):
        raise ValueError(f"Model {name}: probas out of bounds [{probas.min()}, {probas.max()}]")
    
    if not np.allclose(probas.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"Model {name}: probability rows must sum to one")
    # Clip tiny float errors to [0, 1] and re-normalize rows
    probas_clean = np.clip(probas, 0.0, 1.0)
    probas_clean = probas_clean / np.sum(probas_clean, axis=1, keepdims=True)

    preds = np.argmax(probas_clean, axis=1)
    acc = float(accuracy_score(y_true, preds))
    ece = compute_ece(y_true, probas_clean, n_bins=10)
    brier = compute_brier_score(y_true, probas_clean)
    nll = compute_nll(y_true, probas_clean, epsilon=1e-12)

    bin_data = generate_reliability_diagram_data(y_true, probas_clean, n_bins=10)

    print(f"{name:<26s} | Acc: {acc:.4f} | ECE: {ece:.6f} | Brier: {brier:.6f} | NLL: {nll:.6f} | Prob Min/Max: [{probas_clean.min():.4f}, {probas_clean.max():.4f}]")

    return {
        "model_name": name,
        "n_samples": len(y_true),
        "accuracy": acc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "bin_data": bin_data,
        "probas": probas_clean
    }


def run_all_calibration(fresh_predictions: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use shared saved evidence by default; explicitly request fresh inference."""
    if not fresh_predictions:
        from src.experiments.publication_evidence import rebuild_calibration
        return rebuild_calibration()
    print("=" * 80)
    print("STARTING DYNAMIC CALIBRATION EVALUATION PIPELINE")
    print("=" * 80)

    train_df, test_df, X_train, y_train, X_test, y_test = load_and_validate_data()
    df_X_test = pd.DataFrame(X_test, columns=FEATURES)

    results = []
    all_bins_data = []

    # 1. Logistic Regression
    print("\n[1/6] Evaluating Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_probas = lr.predict_proba(X_test)
    res_lr = evaluate_model_probabilities("Logistic Regression", y_test, lr_probas)
    results.append(res_lr)

    # 2. Decision Tree
    print("\n[2/6] Evaluating Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_probas = dt.predict_proba(X_test)
    res_dt = evaluate_model_probabilities("Decision Tree", y_test, dt_probas)
    results.append(res_dt)

    # 3. Random Forest
    print("\n[3/6] Evaluating Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probas = rf.predict_proba(X_test)
    res_rf = evaluate_model_probabilities("Random Forest", y_test, rf_probas)
    results.append(res_rf)

    # 4. XGBoost Baseline
    print("\n[4/6] Evaluating XGBoost Baseline...")
    xgb_path = MODEL_PATH / "xgboost_model.pkl"
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGBoost model missing at {xgb_path}")
    xgb_clf = joblib.load(xgb_path)
    xgb_probas = xgb_clf.predict_proba(df_X_test)
    res_xgb = evaluate_model_probabilities("XGBoost Baseline", y_test, xgb_probas)
    results.append(res_xgb)

    # 5. FT-Transformer Baseline
    print("\n[5/6] Evaluating FT-Transformer Baseline...")
    ft_path = MODEL_PATH / "best_model.pth"
    if not ft_path.exists():
        raise FileNotFoundError(f"FT-Transformer checkpoint missing at {ft_path}")
    ft_model = FTTransformer()
    ckpt_ft = torch.load(ft_path, map_location="cpu")
    ft_state = ckpt_ft["model_state_dict"] if isinstance(ckpt_ft, dict) and "model_state_dict" in ckpt_ft else ckpt_ft
    ft_model.load_state_dict(ft_state)
    ft_model.eval()

    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits_ft = ft_model(x_test_tensor)
        ft_probas = torch.softmax(logits_ft, dim=1).numpy()
    res_ft = evaluate_model_probabilities("FT-Transformer Baseline", y_test, ft_probas)
    results.append(res_ft)

    # 6. KG-Hybrid Proposed Model
    print("\n[6/6] Evaluating KG-Hybrid (Proposed)...")
    kg_path = MODEL_PATH / "kg_hybrid_corrected.pth" if (MODEL_PATH / "kg_hybrid_corrected.pth").exists() else MODEL_PATH / "kg_hybrid_best.pth"
    if not kg_path.exists():
        raise FileNotFoundError(f"KG-Hybrid checkpoint missing at {kg_path}")
    
    from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
    km, _, _ = load_authoritative_knowledge_matrix(DATA_PATH)
    embedder = XGBLeafEmbedder(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES, # 118,800
        num_trees=NUM_TREES,                  # 4,400
        embedding_dim=EMBEDDING_DIM           # 128
    )
    ft_backbone = FTTransformer()
    fusion = KnowledgeGuidedFusion(
        dim_xgb=EMBEDDING_DIM,
        dim_ft=FTConfig.EMBEDDING_DIM,
        num_crops=22,
        num_features=7,
        knowledge_matrix=km
    )
    kg_model = KGHybrid(embedder, ft_backbone, fusion, num_classes=22)
    kg_ckpt = torch.load(kg_path, map_location="cpu")
    kg_model.load_state_dict(kg_ckpt, strict=True)
    kg_model.eval()

    # Cached leaves can belong to a different booster. Use the actual checkpoint.
    leaf_t = get_leaf_indices_with_offset(X_test, xgb_clf.get_booster())

    with torch.no_grad():
        out = kg_model(leaf_t, x_test_tensor)
        logits_kg = out[0] if isinstance(out, tuple) else out
        kg_probas = torch.softmax(logits_kg, dim=1).numpy()
    res_kg = evaluate_model_probabilities("KG-Hybrid (Proposed)", y_test, kg_probas)
    results.append(res_kg)

    # =========================================================================
    # BUILD OUTPUT DATAFRAMES
    # =========================================================================
    # Table II DataFrame
    table2_rows = []
    for r in results:
        table2_rows.append({
            "Model": r["model_name"],
            "N Samples": r["n_samples"],
            "Top-1 Accuracy": round(r["accuracy"], 4),
            "ECE": round(r["ece"], 6),
            "Brier Score": round(r["brier"], 6),
            "NLL": round(r["nll"], 6)
        })
    df_table2 = pd.DataFrame(table2_rows)
    table2_csv = TABLES_DIR / "table2_calibration_metrics.csv"
    df_table2.to_csv(table2_csv, index=False)
    print(f"\n[SAVED] Table II -> {table2_csv}")

    # Calibration Bins DataFrame
    bins_rows = []
    for r in results:
        for b in r["bin_data"]:
            bins_rows.append({
                "Model": r["model_name"],
                "Bin": b["bin_index"],
                "Bin Lower": b["bin_lower"],
                "Bin Upper": b["bin_upper"],
                "Count": b["sample_count"],
                "Mean Confidence": b["confidence"],
                "Accuracy": b["accuracy"]
            })
    df_bins = pd.DataFrame(bins_rows)
    bins_csv = TABLES_DIR / "calibration_bins.csv"
    df_bins.to_csv(bins_csv, index=False)
    print(f"[SAVED] Calibration Bins -> {bins_csv}")

    # =========================================================================
    # GENERATE DYNAMIC RELIABILITY DIAGRAM
    # =========================================================================
    plot_reliability_diagram(results)

    return df_table2, df_bins


def plot_reliability_diagram(results: List[Dict[str, Any]]) -> None:
    """Plots multi-model reliability curves using actual dynamic predictions."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.75, linewidth=1.75)

    model_styles = {
        "KG-Hybrid (Proposed)": {"color": "#16A34A", "marker": "s", "lw": 2.5, "ms": 7},
        "XGBoost Baseline":     {"color": "#D97706", "marker": "o", "lw": 2.0, "ms": 6},
        "FT-Transformer Baseline": {"color": "#4F46E5", "marker": "^", "lw": 2.0, "ms": 6},
        "Random Forest":        {"color": "#DC2626", "marker": "d", "lw": 1.8, "ms": 5},
        "Logistic Regression":  {"color": "#0284C7", "marker": "v", "lw": 1.5, "ms": 5},
        "Decision Tree":        {"color": "#9333EA", "marker": "x", "lw": 1.5, "ms": 5},
    }

    for r in results:
        name = r["model_name"]
        style = model_styles.get(name, {"color": "#64748B", "marker": ".", "lw": 1.5, "ms": 4})
        bin_data = r["bin_data"]

        confs = [b["confidence"] for b in bin_data if b["sample_count"] > 0]
        accs  = [b["accuracy"] for b in bin_data if b["sample_count"] > 0]

        if len(confs) > 0:
            label_str = f"{name} (ECE={r['ece']:.6f})"
            ax.plot(confs, accs, f"{style['marker']}-", color=style["color"],
                    label=label_str, linewidth=style["lw"], markersize=style["ms"])

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title("Reliability Diagram (Dynamic Model Calibration Comparison)", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_png = FIGURES_DIR / "reliability_diagram.png"
    out_pdf = FIGURES_DIR / "reliability_diagram.pdf"
    out_f1  = FIGURES_DIR / "fig_f1_reliability_diagram.png"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.savefig(out_f1, dpi=300)
    plt.close()

    print(f"[SAVED] Reliability Diagrams -> {out_png}, {out_pdf}, {out_f1}")


if __name__ == "__main__":
    df_t2, df_bins = run_all_calibration()
    print("\n" + "=" * 80)
    print("FINAL DYNAMIC TABLE II CALIBRATION METRICS")
    print("=" * 80)
    print(df_t2.to_string(index=False))
