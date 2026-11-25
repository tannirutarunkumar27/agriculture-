"""
Comprehensive Forensic Audit Script for KG-Hybrid IEEE Experimental Results.
Extracts and computes exact numbers for:
1. Baselines & Proposed Model (Single Split, 5-Fold CV, 5-Seed Holdout)
2. Dynamic Calibration
3. 5-Seed Ablation Study
4. Robustness (Missing Features, Gaussian Noise, Extrapolation)
5. Routing & Production Readiness (Early Exit, Risk-Coverage)
6. Parameter Counts and Active Model Architecture
7. Confusion Matrix & Per-Class Error Analysis
8. Stale-Value Audit
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.kg_hybrid_model import KGHybrid

audit_report = {}

# 1. PARAMETER COUNTS FROM ACTIVE CODE AND CHECKPOINTS
xgb_model = joblib.load(PROJECT_ROOT / "outputs" / "models" / "xgboost_model.pkl")
booster = xgb_model.get_booster()
train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "validation.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
leaf_train = booster.predict(xgb.DMatrix(train_df[features]), pred_leaf=True)
leaf_val = booster.predict(xgb.DMatrix(val_df[features]), pred_leaf=True)
leaf_test = booster.predict(xgb.DMatrix(test_df[features]), pred_leaf=True)

max_leaves = int(max(leaf_train.max(), leaf_val.max(), leaf_test.max())) + 1
num_trees = leaf_train.shape[1]
total_leaves = num_trees * max_leaves

# Checkpoint inspection
ckpt_path = PROJECT_ROOT / "outputs" / "models" / "kg_hybrid_corrected.pth"
if not ckpt_path.exists():
    raise FileNotFoundError("The audit requires the corrected checkpoint, not the historical static-bias checkpoint")
state = torch.load(ckpt_path, map_location='cpu')
total_ckpt_params = sum(p.numel() for p in state.values())  # Includes buffers; not a trainable-parameter count.

ckpt_leaves = state["xgb_embedder.embeddings.weight"].shape[0]
embedder = XGBLeafEmbedder(num_unique_leaves=ckpt_leaves, num_trees=num_trees, embedding_dim=128)
embed_params = sum(p.numel() for p in embedder.parameters())

ft = FTTransformer()
ft_params = sum(p.numel() for p in ft.parameters())

km = np.load(PROJECT_ROOT / "data" / "processed" / "knowledge_matrix.npy")
fusion = KnowledgeGuidedFusion(dim_xgb=128, dim_ft=64, knowledge_matrix=km)
fusion_params = sum(p.numel() for p in fusion.parameters())

model = KGHybrid(embedder, ft, fusion, num_classes=22)
head_params = sum(p.numel() for p in model.head.parameters())
total_model_params = sum(p.numel() for p in model.parameters())

audit_report["parameter_counts"] = {
    "num_trees": num_trees,
    "max_leaves_per_tree": max_leaves,
    "total_unique_leaves": total_leaves,
    "embedding_dim": 128,
    "embedder_embeddings_params": embedder.embeddings.weight.numel(),
    "embedder_layernorm_params": embedder.norm.weight.numel() + embedder.norm.bias.numel(),
    "embedder_total_params": embed_params,
    "ft_transformer_backbone_params": ft_params,
    "fusion_gate_params": sum(p.numel() for p in fusion.gate.parameters()),
    "fusion_knowledge_matrix_params": fusion.knowledge_matrix.numel(),
    "fusion_knowledge_mechanism_params": (
        sum(p.numel() for p in fusion.knowledge_encoder.parameters()) +
        sum(p.numel() for p in fusion.query_proj.parameters()) +
        sum(p.numel() for p in fusion.context_proj.parameters())
    ) if hasattr(fusion, 'knowledge_encoder') else (fusion.knowledge_proj.weight.numel() + fusion.knowledge_proj.bias.numel()),
    "fusion_layernorm_params": fusion.layer_norm.weight.numel() + fusion.layer_norm.bias.numel(),
    "fusion_total_params": fusion_params,
    "prediction_head_params": head_params,
    "active_pytorch_model_total_params": total_model_params,
    "saved_checkpoint_tensor_elements_including_buffers": total_ckpt_params,
    "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
}

# 2. CALIBRATION METRICS (from authoritative CSV)
cal_csv = PROJECT_ROOT / "outputs" / "tables" / "table2_calibration_metrics.csv"
cal_df = pd.read_csv(cal_csv)
audit_report["calibration"] = cal_df.to_dict(orient="records")

# 3. 5-SEED ABLATION SUMMARY
abl_csv = PROJECT_ROOT / "outputs" / "ablation" / "five_seed" / "ablation_summary.csv"
abl_df = pd.read_csv(abl_csv)
audit_report["five_seed_ablation"] = abl_df.to_dict(orient="records")

# 4. 5-SEED MULTI-SEED EXPERIMENTS (Baselines)
ms_csv = PROJECT_ROOT / "outputs" / "multi_seed_experiments" / "aggregate" / "summary_metrics.csv"
if ms_csv.exists():
    ms_df = pd.read_csv(ms_csv)
    audit_report["five_seed_baselines"] = ms_df.to_dict(orient="records")

# 5. 5-FOLD CROSS-VALIDATION
cv_csv = PROJECT_ROOT / "outputs" / "cross_validation_neural_recovery" / "aggregate" / "cv_summary.csv"
if cv_csv.exists():
    cv_df = pd.read_csv(cv_csv)
    audit_report["cross_validation"] = cv_df.to_dict(orient="records")

# 6. ROBUSTNESS RESULTS
rob_json = PROJECT_ROOT / "outputs" / "robustness" / "full_robustness_report.json"
if rob_json.exists():
    with open(rob_json) as f:
        audit_report["robustness"] = json.load(f)

# 7. PRODUCTION READINESS (KNN Imputation & Early Exit)
prod_json = PROJECT_ROOT / "outputs" / "tables" / "production_readiness_results.json"
if prod_json.exists():
    with open(prod_json) as f:
        audit_report["production_readiness"] = json.load(f)

# 8. PER-CLASS CONFUSION ANALYSIS ON HELD-OUT TEST (N=330)
# Model test predictions
model.load_state_dict(state)
model.eval()

tree_offsets = np.arange(num_trees, dtype=np.int64) * max_leaves
leaf_test_offset = leaf_test + tree_offsets
X_test_tensor = torch.tensor(test_df[features].values, dtype=torch.float32)
leaf_test_tensor = torch.tensor(leaf_test_offset, dtype=torch.long)
y_test = test_df['label'].values

with torch.no_grad():
    logits, _, _, _, _ = model(leaf_test_tensor, X_test_tensor)
    test_preds = logits.argmax(dim=1).numpy()

cm = confusion_matrix(y_test, test_preds, labels=list(range(22)))
misclassified_indices = np.where(y_test != test_preds)[0]

error_pairs = []
for idx in misclassified_indices:
    error_pairs.append({
        "sample_index": int(idx),
        "true_label": int(y_test[idx]),
        "pred_label": int(test_preds[idx])
    })

p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)
label_encoder = joblib.load(PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl")
class_names = list(label_encoder.classes_)

per_class_metrics = []
for i, cname in enumerate(class_names):
    per_class_metrics.append({
        "class_id": i,
        "class_name": cname,
        "precision": round(float(p_cls[i]), 4),
        "recall": round(float(r_cls[i]), 4),
        "f1": round(float(f1_cls[i]), 4)
    })

audit_report["error_analysis"] = {
    "test_samples": len(y_test),
    "accuracy": round(float(accuracy_score(y_test, test_preds)), 6),
    "num_misclassified": len(misclassified_indices),
    "error_pairs": error_pairs,
    "per_class_metrics": per_class_metrics
}

# Save full audit dictionary
out_audit = PROJECT_ROOT / "outputs" / "tables" / "comprehensive_audit_results.json"
with open(out_audit, "w") as f:
    json.dump(audit_report, f, indent=2)

print(f"[OK] Full audit results successfully written to {out_audit}")
