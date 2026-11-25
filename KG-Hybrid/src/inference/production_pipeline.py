"""
Production Crop Recommendation Pipeline (KG-Hybrid)
===================================================
Incorporates real-world production enhancements & reconciled performance bounds:
  1. KNN / Iterative Imputation for missing sensor inputs (Handling sensor faults)
  2. Two-Stage Confidence-Based Early Exit Routing:
     - Stage 1: XGBoost predictions; route-specific metrics require measured evidence.
     - Stage 2: KG-Hybrid predictions plus repeated MC Dropout passes.
  3. Monte Carlo (MC) Dropout Uncertainty Quantification (10 forward passes)
  4. Structured JSON Output with Data Quality, Sensor Health, Model Metadata & Timestamp
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.impute import KNNImputer, SimpleImputer
from src.models.ft_transformer.model import FTTransformer
from src.models.ft_transformer.config import FTConfig
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.hybrid.lightweight_kg_hybrid import LightweightKGHybrid
from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, get_leaf_indices_with_offset, FEATURES, NUM_TREES, MAX_LEAVES_PER_TREE, TOTAL_UNIQUE_LEAVES, EMBEDDING_DIM

CROP_NAMES = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]


class ProductionCropRecommender:
    """
    Production-ready wrapper for KG-Hybrid with Imputation, Early Exit,
    Lightweight Model Option, MC Dropout Uncertainty Quantification, and Structured Diagnostics.
    """
    def __init__(self,
                 use_knn: bool = True,
                 early_exit_threshold: float = 0.90,
                 use_lightweight: bool = True,
                 mc_samples: int = 10,
                 uncertainty_threshold: float = 0.15,
                 region_context: str = "SouthAsia/Tropical"):

        self.early_exit_threshold = early_exit_threshold
        if not 0 <= early_exit_threshold <= 1 or mc_samples < 1:
            raise ValueError("early_exit_threshold must be in [0, 1] and mc_samples positive")
        self.use_lightweight = use_lightweight
        self.mc_samples = mc_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.region_context = region_context
        self.model_version = "v1.2.0-ProductionLightweight" if use_lightweight else "v1.0.0-ProductionFull"

        # 1. Initialize Imputer
        train_df = pd.read_csv(DATA_PATH / 'train.csv')
        X_train = train_df.drop('label', axis=1).values.astype(np.float32)
        self.scaler = joblib.load(DATA_PATH / "scaler.pkl")

        if use_knn:
            self.imputer = KNNImputer(n_neighbors=5).fit(X_train)
        else:
            self.imputer = SimpleImputer(strategy='median').fit(X_train)

        # 2. Load XGBoost Model
        self.xgb_clf = joblib.load(MODEL_PATH / 'xgboost_model.pkl')
        self.booster = self.xgb_clf.get_booster()

        # 3. Load Neural Model (Lightweight or Full)
        from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
        km, _, _ = load_authoritative_knowledge_matrix(DATA_PATH)
        if use_lightweight:
            self.kg_model = LightweightKGHybrid(
                num_unique_leaves=TOTAL_UNIQUE_LEAVES,
                num_trees=NUM_TREES,
                embed_dim=32,
                num_classes=22,
                knowledge_matrix=km
            )
            lw_path = MODEL_PATH / "lightweight_kg_hybrid_best.pth"
            if not lw_path.exists():
                raise FileNotFoundError(f"Trained lightweight checkpoint required: {lw_path}")
            self.kg_model.load_state_dict(torch.load(lw_path, map_location='cpu'), strict=True)
            self.kg_model.eval()
        else:
            cfg = FTConfig()
            embedder = XGBLeafEmbedder(
                num_unique_leaves=TOTAL_UNIQUE_LEAVES,
                num_trees=NUM_TREES,
                embedding_dim=EMBEDDING_DIM
            )
            ft_model = FTTransformer()
            ckpt = torch.load(MODEL_PATH / "best_model.pth", map_location='cpu')
            ft_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            ft_model.load_state_dict(ft_state)

            fusion = KnowledgeGuidedFusion(
                dim_xgb=EMBEDDING_DIM,
                dim_ft=cfg.EMBEDDING_DIM,
                num_crops=22,
                num_features=7,
                knowledge_matrix=km
            )

            self.kg_model = KGHybrid(embedder, ft_model, fusion, num_classes=22)

            # Prioritize corrected v2 architecture checkpoint
            corrected_path = MODEL_PATH / "kg_hybrid_corrected.pth"
            if corrected_path.exists():
                ckpt_to_load = corrected_path
            elif (MODEL_PATH / "corrected" / "kg_hybrid_best.pth").exists():
                ckpt_to_load = MODEL_PATH / "corrected" / "kg_hybrid_best.pth"
            else:
                ckpt_to_load = MODEL_PATH / "kg_hybrid_best.pth"

            self.kg_model.load_state_dict(torch.load(ckpt_to_load, map_location='cpu'), strict=True)
            self.kg_model.eval()

    def enable_mc_dropout(self):
        """Enable dropout layers during evaluation for MC Dropout sampling."""
        for m in self.kg_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_single(self, input_dict: dict, enable_early_exit: bool = True, input_is_scaled: bool = False) -> dict:
        """
        Execute inference on sensor values in original units. Benchmarks using
        already standardized test.csv features must set input_is_scaled=True.
        """
        t_start = time.perf_counter()
        now_utc = datetime.now(timezone.utc).isoformat()

        # Parse inputs & detect missing values
        raw_vals = []
        missing_feats = []
        for feat in FEATURES:
            val = input_dict.get(feat, None)
            if val is None or val == "":
                raw_vals.append(np.nan)
                missing_feats.append(feat)
            else:
                try:
                    numeric = float(val)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid numeric sensor value for {feat}") from error
                if np.isnan(numeric):
                    raw_vals.append(np.nan)
                    missing_feats.append(feat)
                elif not np.isfinite(numeric):
                    raise ValueError(f"Nonfinite sensor value for {feat}")
                else:
                    raw_vals.append(numeric)

        raw_arr = np.array(raw_vals, dtype=np.float32).reshape(1, -1)
        model_arr = raw_arr if input_is_scaled else self.scaler.transform(raw_arr).astype(np.float32)

        # Stage 0: Data Quality Audit & Imputation
        is_imputed = len(missing_feats) > 0
        if is_imputed:
            X_proc = self.imputer.transform(model_arr)
            data_status = f"IMPUTED ({len(missing_feats)} missing features: {', '.join(missing_feats)})"
            sensor_health = f"WARNING: Sensor fault detected for {', '.join(missing_feats)}"
        else:
            X_proc = model_arr
            data_status = "HEALTHY (All 7 features provided)"
            sensor_health = "OK"

        df_X = pd.DataFrame(X_proc, columns=FEATURES)

        # Stage 1: XGBoost Fast Pass & Early Exit Evaluation
        xgb_probs = self.xgb_clf.predict_proba(df_X)[0]
        top_idx_xgb = int(np.argmax(xgb_probs))
        top_conf_xgb = float(xgb_probs[top_idx_xgb])

        if enable_early_exit and top_conf_xgb >= self.early_exit_threshold and not is_imputed:
            t_end = time.perf_counter()
            top3_idx = np.argsort(xgb_probs)[::-1][:3]
            top3 = [
                {"crop": CROP_NAMES[idx], "probability": round(float(xgb_probs[idx]), 4)}
                for idx in top3_idx
            ]
            return {
                "timestamp": now_utc,
                "model_version": self.model_version,
                "region_context": self.region_context,
                "recommended_crop": CROP_NAMES[top_idx_xgb],
                "confidence": round(top_conf_xgb, 4),
                "uncertainty_std": 0.0,
                "top_3_crops": top3,
                "inference_route": "STAGE_1_EARLY_EXIT (XGBoost)",
                "data_quality": data_status,
                "sensor_health": sensor_health,
                "latency_ms": round((t_end - t_start) * 1000, 2),
                "safety_alert": None
            }

        # Stage 2: Neural KG-Hybrid Pass
        leaf_t = get_leaf_indices_with_offset(X_proc, self.booster)
        x_t    = torch.tensor(X_proc, dtype=torch.float32)

        with torch.no_grad():
            out = self.kg_model(leaf_t, x_t)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1).numpy()[0]

        # Stage 3: Monte Carlo Dropout Uncertainty Estimation
        mc_probs = []
        self.enable_mc_dropout()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                mc_out = self.kg_model(leaf_t, x_t)
                mc_logits = mc_out[0] if isinstance(mc_out, tuple) else mc_out
                mc_probs.append(torch.softmax(mc_logits, dim=1).numpy()[0])
        self.kg_model.eval()

        mc_arr = np.array(mc_probs)
        mean_probs = mc_arr.mean(axis=0)
        std_probs  = mc_arr.std(axis=0)

        top_idx  = int(np.argmax(mean_probs))
        top_conf = float(mean_probs[top_idx])
        top_std  = float(std_probs[top_idx])

        top3_idx = np.argsort(mean_probs)[::-1][:3]
        top3 = [
            {
                "crop": CROP_NAMES[idx],
                "probability": round(float(mean_probs[idx]), 4),
                "uncertainty_std": round(float(std_probs[idx]), 4)
            }
            for idx in top3_idx
        ]

        safety_alert = None
        if top_std > self.uncertainty_threshold or top_conf < 0.60:
            safety_alert = "HIGH UNCERTAINTY DETECTED: Soil parameters are at boundary conditions. Verification recommended."

        t_end = time.perf_counter()

        model_name = "Lightweight KG-Hybrid" if self.use_lightweight else "Full KG-Hybrid"
        route = f"STAGE_2_{model_name.upper().replace(' ', '_')} (Imputed Input)" if is_imputed else f"STAGE_2_{model_name.upper().replace(' ', '_')} (Low Confidence Early Pass)"

        return {
            "timestamp": now_utc,
            "model_version": self.model_version,
            "region_context": self.region_context,
            "recommended_crop": CROP_NAMES[top_idx],
            "confidence": round(top_conf, 4),
            "uncertainty_std": round(top_std, 4),
            "top_3_crops": top3,
            "inference_route": route,
            "data_quality": data_status,
            "sensor_health": sensor_health,
            "latency_ms": round((t_end - t_start) * 1000, 2),
            "safety_alert": safety_alert
        }


if __name__ == "__main__":
    recommender = ProductionCropRecommender(use_knn=True, early_exit_threshold=0.90, use_lightweight=True)
    sample_input = {
        "N": None,  # Faulty/missing sensor
        "P": 55.0,
        "K": 40.0,
        "temperature": 24.5,
        "humidity": 82.0,
        "ph": None, # Faulty/missing sensor
        "rainfall": 210.0
    }
    result = recommender.predict_single(sample_input, enable_early_exit=True)
    print(json.dumps(result, indent=2))

