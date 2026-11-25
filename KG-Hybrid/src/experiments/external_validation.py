"""
File: src/experiments/external_validation.py

Purpose:
Out-of-Distribution (OOD) External Dataset Validation Protocol for KG-Hybrid.

Evaluates models under two distinct experimental regimes:
  Regime A: Raw zero-shot OOD evaluation.
  Regime B: Exploratory rainfall-scaled sensitivity analysis.

The second regime is not external field validation or a validated unit
conversion.  It is retained only as a clearly labelled sensitivity check until
the measurement periods and units are documented by the data provider.

Outputs:
  - outputs/tables/external_validation_metrics.csv
  - outputs/reports/external_validation_report.json
  - outputs/figures/external_distribution_shift.png
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

from src.utils.distribution_shift import (
    compute_distribution_shift_metrics,
    calculate_mahalanobis_shift,
    plot_distribution_shift
)
from src.utils.statistical_testing import (
    run_mcnemar_test,
    paired_permutation_test_f1
)
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.robustness.robustness_utils import (
    MODEL_PATH, NUM_TREES, TOTAL_UNIQUE_LEAVES, EMBEDDING_DIM,
    get_leaf_indices_with_offset,
)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _state_dict_from_checkpoint(path: Path) -> dict:
    """Load either a raw state dict or the trainer checkpoint wrapper."""
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_saved_neural_models(xgb_clf: Any) -> Tuple[FTTransformer, KGHybrid]:
    """Load the actual Paper 1 neural checkpoints for external evaluation.

    External validation must never proxy FT-Transformer or KG-Hybrid predictions
    with the XGBoost baseline.  The XGBoost checkpoint is also the source of
    leaf indices used by the saved hybrid checkpoint.
    """
    ft_path = MODEL_PATH / "best_model.pth"
    kg_path = (MODEL_PATH / "kg_hybrid_corrected.pth"
               if (MODEL_PATH / "kg_hybrid_corrected.pth").exists()
               else MODEL_PATH / "kg_hybrid_best.pth")
    if not ft_path.exists() or not kg_path.exists():
        missing = [str(path) for path in (ft_path, kg_path) if not path.exists()]
        raise FileNotFoundError("External validation requires saved neural checkpoints: " + ", ".join(missing))

    ft_model = FTTransformer()
    ft_model.load_state_dict(_state_dict_from_checkpoint(ft_path), strict=True)
    ft_model.eval()

    from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
    matrix, _, _ = load_authoritative_knowledge_matrix(PROJECT_ROOT / "data" / "processed")
    embedder = XGBLeafEmbedder(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES,
        num_trees=NUM_TREES,
        embedding_dim=EMBEDDING_DIM,
    )
    fusion = KnowledgeGuidedFusion(
        dim_xgb=EMBEDDING_DIM,
        dim_ft=FTConfig.EMBEDDING_DIM,
        num_crops=22,
        num_features=len(FEATURES),
        knowledge_matrix=matrix,
    )
    kg_model = KGHybrid(embedder, FTTransformer(), fusion, num_classes=22)
    kg_model.load_state_dict(_state_dict_from_checkpoint(kg_path), strict=True)
    kg_model.eval()
    return ft_model, kg_model


def predict_saved_neural_probabilities(
    X_scaled: np.ndarray, xgb_clf: Any, ft_model: FTTransformer, kg_model: KGHybrid
) -> Tuple[np.ndarray, np.ndarray]:
    """Return probabilities from the saved FT and KG-Hybrid checkpoints."""
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    leaves = get_leaf_indices_with_offset(X_scaled, xgb_clf.get_booster())
    with torch.no_grad():
        ft_probs = torch.softmax(ft_model(x_tensor), dim=1).cpu().numpy()
        kg_out = kg_model(leaves, x_tensor)
        kg_logits = kg_out[0] if isinstance(kg_out, tuple) else kg_out
        kg_probs = torch.softmax(kg_logits, dim=1).cpu().numpy()
    return ft_probs, kg_probs


def load_internal_data() -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """Load preprocessed internal train/test data and fitted scaler/label encoder."""
    proc_dir = PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_csv(proc_dir / "train.csv")
    test_df = pd.read_csv(proc_dir / "test.csv")

    scaler = joblib.load(proc_dir / "scaler.pkl")
    label_encoder = joblib.load(proc_dir / "label_encoder.pkl")

    return train_df, test_df, scaler, label_encoder


def load_external_dataset_agrifield(align_units: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ingest and map AgriField-Manipur dataset.
    """
    file_path = PROJECT_ROOT / "data" / "raw" / "AgriField-Manipur_v2025.1" / "data" / "ml_dataset.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"AgriField dataset missing at {file_path}")

    df = pd.read_csv(file_path)

    # Label normalization mapping to match internal 22 crops
    crop_mapping = {
        'banana': 'banana',
        'black gram': 'blackgram',
        'chickpea': 'chickpea',
        'lentil': 'lentil',
        'maize': 'maize',
        'mango': 'mango',
        'orange': 'orange',
        'papaya': 'papaya',
        'rice': 'rice'
    }

    df['crop_clean'] = df['crop'].str.strip().str.lower()
    df['target_crop'] = df['crop_clean'].map(crop_mapping)

    # Filter samples belonging to overlapping crop subset
    df_overlapping = df.dropna(subset=['target_crop']).copy()
    df_overlapping['label'] = df_overlapping['target_crop']

    X_ext_raw = df_overlapping[FEATURES].copy()

    if align_units:
        # Exploratory sensitivity only.  The source-data metadata does not
        # establish a valid seasonal conversion, so this must not be presented
        # as domain alignment or field-calibrated preprocessing.
        X_ext_raw['rainfall'] = X_ext_raw['rainfall'] * 9.0

    y_ext_raw = df_overlapping['label'].copy()

    return X_ext_raw, y_ext_raw


def calculate_top3_accuracy(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Compute Top-3 Accuracy."""
    top3_preds = np.argsort(probas, axis=1)[:, -3:]
    correct = [y_true[i] in top3_preds[i] for i in range(len(y_true))]
    return float(np.mean(correct))


def calculate_ece(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def evaluate_regime(
    regime_name: str,
    align_units: bool,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: Any,
    label_encoder: Any
) -> Dict[str, Any]:
    """Evaluate external dataset performance under a specified alignment regime."""
    print(f"\n" + "-" * 60)
    print(f"EVALUATING REGIME: {regime_name}")
    print("-" * 60)

    X_train_internal = train_df[FEATURES]
    y_train_internal = train_df['label'].values
    X_test_internal = test_df[FEATURES]
    y_test_internal = test_df['label'].values

    X_ext_raw, y_ext_raw = load_external_dataset_agrifield(align_units=align_units)
    X_ext_scaled = scaler.transform(X_ext_raw)
    X_ext_scaled_df = pd.DataFrame(X_ext_scaled, columns=FEATURES)
    y_ext_encoded = label_encoder.transform(y_ext_raw)

    # Covariate-shift metrics require identical units.  The processed internal
    # partition is standardized, whereas the external data are in physical
    # units, so invert the training scaler before comparing distributions.
    X_train_raw = pd.DataFrame(
        scaler.inverse_transform(X_train_internal), columns=FEATURES
    )
    shift_table = compute_distribution_shift_metrics(X_train_raw, X_ext_raw, FEATURES)
    mahalanobis_dist = calculate_mahalanobis_shift(X_train_raw, X_ext_raw)

    # Baseline models
    models = {}
    preds_ext = {}
    probas_ext = {}

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_internal, y_train_internal)
    models['LogisticRegression'] = lr
    preds_ext['LogisticRegression'] = lr.predict(X_ext_scaled_df)
    probas_ext['LogisticRegression'] = lr.predict_proba(X_ext_scaled_df)

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train_internal, y_train_internal)
    models['DecisionTree'] = dt
    preds_ext['DecisionTree'] = dt.predict(X_ext_scaled_df)
    probas_ext['DecisionTree'] = dt.predict_proba(X_ext_scaled_df)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train_internal, y_train_internal)
    models['RandomForest'] = rf
    preds_ext['RandomForest'] = rf.predict(X_ext_scaled_df)
    probas_ext['RandomForest'] = rf.predict_proba(X_ext_scaled_df)

    xgb_path = MODEL_PATH / "xgboost_model.pkl"
    if not xgb_path.exists():
        raise FileNotFoundError(f"Saved XGBoost checkpoint is required: {xgb_path}")
    xgb_clf = joblib.load(xgb_path)
    models['XGBoostBaseline'] = xgb_clf
    preds_ext['XGBoostBaseline'] = xgb_clf.predict(X_ext_scaled_df)
    probas_ext['XGBoostBaseline'] = xgb_clf.predict_proba(X_ext_scaled_df)

    # Neural models must use their own saved checkpoints.  Do not replace these
    # outputs with XGBoost predictions: that would fabricate a model comparison.
    ft_model, kg_model = load_saved_neural_models(xgb_clf)
    ft_probas, kg_probas = predict_saved_neural_probabilities(
        X_ext_scaled, xgb_clf, ft_model, kg_model
    )
    preds_ext['FTTransformerBaseline'] = ft_probas.argmax(axis=1)
    probas_ext['FTTransformerBaseline'] = ft_probas
    preds_ext['KG-HybridProposed'] = kg_probas.argmax(axis=1)
    probas_ext['KG-HybridProposed'] = kg_probas

    # Metrics evaluation
    results_rows = []
    for model_name, y_pred in preds_ext.items():
        proba = probas_ext[model_name]

        acc_ext = float(accuracy_score(y_ext_encoded, y_pred))
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_ext_encoded, y_pred, average="macro", zero_division=0
        )
        top3_acc = calculate_top3_accuracy(y_ext_encoded, proba)
        ece_val = calculate_ece(y_ext_encoded, proba)

        if model_name in models:
            y_pred_int = models[model_name].predict(X_test_internal)
        elif model_name == 'FTTransformerBaseline':
            with torch.no_grad():
                y_pred_int = ft_model(torch.tensor(X_test_internal.values, dtype=torch.float32)).argmax(dim=1).numpy()
        else:
            _, internal_kg_probas = predict_saved_neural_probabilities(
                X_test_internal.values.astype(np.float32), xgb_clf, ft_model, kg_model
            )
            y_pred_int = internal_kg_probas.argmax(axis=1)
        acc_int = float(accuracy_score(y_test_internal, y_pred_int))
        _, _, f1_int, _ = precision_recall_fscore_support(y_test_internal, y_pred_int, average="macro", zero_division=0)

        results_rows.append({
            "Regime": regime_name,
            "Model": model_name,
            "External_Accuracy": round(acc_ext, 4),
            "External_F1_Macro": round(float(f1_mac), 4),
            "External_Top3_Accuracy": round(top3_acc, 4),
            "External_ECE": round(ece_val, 4),
            "Internal_F1_Macro": round(float(f1_int), 4),
            "Generalization_Gap_F1": round(f1_int - f1_mac, 4),
            "Generalization_Gap_Accuracy": round(acc_int - acc_ext, 4)
        })

    eval_df = pd.DataFrame(results_rows)
    print(eval_df.to_string(index=False))

    mcnemar_res = run_mcnemar_test(
        y_ext_encoded,
        preds_ext['KG-HybridProposed'],
        preds_ext['XGBoostBaseline'],
        model_a_name="KG-Hybrid",
        model_b_name="XGBoostBaseline"
    )

    perm_res = paired_permutation_test_f1(
        y_ext_encoded,
        preds_ext['KG-HybridProposed'],
        preds_ext['XGBoostBaseline']
    )

    return {
        "regime": regime_name,
        "sample_count": len(X_ext_raw),
        "mahalanobis_distance": round(mahalanobis_dist, 4),
        "feature_shift": shift_table.to_dict(orient="records"),
        "metrics": results_rows,
        "mcnemar_test": mcnemar_res,
        "permutation_test": perm_res
    }


def run_external_validation_pipeline() -> Dict[str, Any]:
    """Execute complete external validation across both experimental regimes."""
    print("=" * 70)
    print("EXECUTING OOD EXTERNAL DATASET VALIDATION PROTOCOL")
    print("=" * 70)

    train_df, test_df, scaler, label_encoder = load_internal_data()

    # Plot initial raw distribution shift
    X_train_internal = train_df[FEATURES]
    X_ext_raw, _ = load_external_dataset_agrifield(align_units=False)
    plot_path = PROJECT_ROOT / "outputs" / "figures" / "external_distribution_shift.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_distribution_shift(X_train_internal, X_ext_raw, FEATURES, str(plot_path), "AgriField-Manipur (Raw)")
    print(f"[SUCCESS] Distribution shift plot saved to {plot_path}")

    # Evaluate Regime A (Raw Zero-Shot OOD)
    res_regime_a = evaluate_regime("Regime A (Raw Zero-Shot)", False, train_df, test_df, scaler, label_encoder)

    # Evaluate Regime B (explicitly exploratory; not domain-aligned validation)
    res_regime_b = evaluate_regime(
        "Regime B (Exploratory Rainfall-Scaled Sensitivity)", True,
        train_df, test_df, scaler, label_encoder,
    )

    # Combined Table Export
    combined_rows = res_regime_a["metrics"] + res_regime_b["metrics"]
    combined_df = pd.DataFrame(combined_rows)

    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "external_validation_metrics.csv"
    combined_df.to_csv(table_path, index=False)
    print(f"\n[SUCCESS] Combined external metrics saved to {table_path}")

    # JSON Report Export
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "external_validation_report.json"

    full_report = {
        "dataset_name": "AgriField-Manipur v2025.1",
        "regime_a_raw": res_regime_a,
        "regime_b_aligned": res_regime_b
    }

    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=4)
    print(f"[SUCCESS] External validation JSON report saved to {report_path}")

    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION PROTOCOL COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return full_report


if __name__ == "__main__":
    run_external_validation_pipeline()
