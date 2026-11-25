"""
File: knowledge_feature_gate.py

Step 4 & 5 — Knowledge Representation + Knowledge-Guided Feature Gate

Citation:
Arik, S. O., & Pfister, T. (2021). "TabNet: Attentive Interpretable Tabular Learning."
Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679-6687.

Purpose:
    Converts agronomic expert knowledge into machine-readable feature
    importance weights, then applies a soft gate to scale input features
    before they enter the XGBoost and FT-Transformer branches.

Pipeline:
    Raw Features (7)
          │
    Knowledge Feature Gate
       • Load structured knowledge base
       • Compute per-feature importance scores
       • Apply element-wise soft gating
          │
    Knowledge-Aware Feature Representation (7)
          │
    ┌─────┴──────┐
    XGBoost    FT-Transformer
"""

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"
TABLES_DIR     = OUTPUT_DIR / "tables"
FIGURES_DIR    = OUTPUT_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature & Crop Constants ────────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROPS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

# ─── Structured Agronomic Knowledge Base ─────────────────────────────────────
# Based on FAO/ICAR guidelines: each feature has a range [min, max] and optimal
KNOWLEDGE_BASE = {
    "rice":        {"N": [60,140], "P": [30,60],  "K": [35,60],  "temperature": [20,35], "humidity": [70,100], "ph": [5.5,7.0], "rainfall": [150,300], "season": "Kharif"},
    "maize":       {"N": [60,120], "P": [30,60],  "K": [20,50],  "temperature": [18,35], "humidity": [50,80],  "ph": [5.5,7.5], "rainfall": [50,100],  "season": "Kharif"},
    "chickpea":    {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [15,30], "humidity": [30,60],  "ph": [6.0,8.0], "rainfall": [40,80],   "season": "Rabi"},
    "kidneybeans": {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [15,27], "humidity": [40,70],  "ph": [6.0,7.5], "rainfall": [40,90],   "season": "Kharif"},
    "pigeonpeas":  {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [18,35], "humidity": [40,70],  "ph": [5.5,7.5], "rainfall": [60,150],  "season": "Kharif"},
    "mothbeans":   {"N": [10,30],  "P": [30,60],  "K": [20,40],  "temperature": [25,40], "humidity": [30,60],  "ph": [5.5,8.0], "rainfall": [25,60],   "season": "Kharif"},
    "mungbean":    {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [25,35], "humidity": [50,80],  "ph": [6.0,7.5], "rainfall": [50,100],  "season": "Kharif"},
    "blackgram":   {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [25,35], "humidity": [50,80],  "ph": [6.0,7.5], "rainfall": [60,100],  "season": "Kharif"},
    "lentil":      {"N": [10,40],  "P": [40,80],  "K": [20,50],  "temperature": [10,25], "humidity": [35,65],  "ph": [6.0,8.0], "rainfall": [25,60],   "season": "Rabi"},
    "pomegranate": {"N": [20,60],  "P": [30,60],  "K": [40,80],  "temperature": [25,38], "humidity": [30,60],  "ph": [5.5,7.5], "rainfall": [50,75],   "season": "Annual"},
    "banana":      {"N": [80,150], "P": [30,60],  "K": [100,200],"temperature": [20,35], "humidity": [70,90],  "ph": [5.5,7.0], "rainfall": [100,200], "season": "Annual"},
    "mango":       {"N": [20,60],  "P": [20,50],  "K": [30,60],  "temperature": [24,38], "humidity": [50,80],  "ph": [5.5,7.5], "rainfall": [75,125],  "season": "Annual"},
    "grapes":      {"N": [20,50],  "P": [30,60],  "K": [40,80],  "temperature": [15,35], "humidity": [40,70],  "ph": [5.5,7.0], "rainfall": [50,80],   "season": "Annual"},
    "watermelon":  {"N": [30,80],  "P": [40,80],  "K": [50,100], "temperature": [25,40], "humidity": [70,90],  "ph": [6.0,7.5], "rainfall": [40,80],   "season": "Summer"},
    "muskmelon":   {"N": [30,80],  "P": [40,80],  "K": [50,100], "temperature": [28,42], "humidity": [70,90],  "ph": [6.0,7.5], "rainfall": [30,60],   "season": "Summer"},
    "apple":       {"N": [20,60],  "P": [30,60],  "K": [40,80],  "temperature": [5,25],  "humidity": [50,90],  "ph": [5.5,7.0], "rainfall": [100,150], "season": "Annual"},
    "orange":      {"N": [20,60],  "P": [30,60],  "K": [40,80],  "temperature": [15,35], "humidity": [60,90],  "ph": [5.5,7.5], "rainfall": [75,125],  "season": "Annual"},
    "papaya":      {"N": [30,80],  "P": [30,60],  "K": [40,80],  "temperature": [22,35], "humidity": [70,90],  "ph": [5.5,7.0], "rainfall": [100,200], "season": "Annual"},
    "coconut":     {"N": [20,60],  "P": [30,60],  "K": [50,100], "temperature": [25,38], "humidity": [70,90],  "ph": [5.0,8.0], "rainfall": [100,300], "season": "Annual"},
    "cotton":      {"N": [60,120], "P": [30,60],  "K": [30,60],  "temperature": [25,40], "humidity": [40,70],  "ph": [6.0,8.0], "rainfall": [50,100],  "season": "Kharif"},
    "jute":        {"N": [60,120], "P": [20,50],  "K": [20,50],  "temperature": [24,38], "humidity": [70,90],  "ph": [5.5,7.5], "rainfall": [150,200], "season": "Kharif"},
    "coffee":      {"N": [20,60],  "P": [20,50],  "K": [20,50],  "temperature": [15,30], "humidity": [70,90],  "ph": [5.0,6.5], "rainfall": [150,250], "season": "Annual"},
}


def build_structured_knowledge_base():
    """
    Build and save the complete structured knowledge base as JSON.
    Returns the structured dict.
    """
    kb = {
        "description": "Agricultural crop knowledge base based on FAO/ICAR guidelines",
        "features": FEATURES,
        "crops": CROPS,
        "rules": {}
    }

    for crop in CROPS:
        if crop in KNOWLEDGE_BASE:
            info = KNOWLEDGE_BASE[crop]
            kb["rules"][crop] = {
                "N_range":           info["N"],
                "P_range":           info["P"],
                "K_range":           info["K"],
                "temperature_range": info["temperature"],
                "humidity_range":    info["humidity"],
                "ph_range":          info["ph"],
                "rainfall_range":    info["rainfall"],
                "season":            info["season"],
            }
        else:
            kb["rules"][crop] = {
                "N_range": [20, 80], "P_range": [20, 60], "K_range": [20, 60],
                "temperature_range": [20, 35], "humidity_range": [50, 80],
                "ph_range": [5.5, 7.5], "rainfall_range": [50, 150],
                "season": "Kharif"
            }

    # Save JSON
    kb_path = DATA_PROCESSED / "knowledge_base.json"
    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2)
    print(f"[OK] Knowledge base saved: {kb_path}")
    return kb


def compute_feature_importance_from_knowledge(knowledge_base: dict) -> np.ndarray:
    """
    Compute per-feature global importance scores from the knowledge base.

    Logic:
        For each feature, measure the average range width (max - min) across
        all crops. A wider range means this feature is more discriminative
        across crops → higher importance.

    Returns:
        importance: np.ndarray of shape (7,) — normalized to [0, 1]
    """
    rules = knowledge_base["rules"]
    feature_keys = {
        "N":           "N_range",
        "P":           "P_range",
        "K":           "K_range",
        "temperature": "temperature_range",
        "humidity":    "humidity_range",
        "ph":          "ph_range",
        "rainfall":    "rainfall_range",
    }

    importance = np.zeros(len(FEATURES))
    for i, feat in enumerate(FEATURES):
        key = feature_keys[feat]
        ranges = []
        for crop_rules in rules.values():
            lo, hi = crop_rules[key]
            ranges.append(hi - lo)
        importance[i] = np.mean(ranges)

    # Normalize to [0, 1]
    importance = importance / (importance.max() + 1e-8)
    return importance


def compute_per_crop_suitability(knowledge_base: dict) -> np.ndarray:
    """
    Build the 22×7 suitability matrix as normalized range widths.
    Each entry represents how wide the acceptable range is for crop i, feature j.
    Wider range = more tolerant = higher suitability score.
    """
    rules = knowledge_base["rules"]
    feature_keys = ["N_range", "P_range", "K_range", "temperature_range",
                    "humidity_range", "ph_range", "rainfall_range"]

    # Reference max ranges for normalization
    ref_max = [280, 80, 200, 35, 80, 3.0, 300]

    matrix = np.zeros((22, 7))
    for i, crop in enumerate(CROPS):
        if crop in rules:
            for j, (key, ref) in enumerate(zip(feature_keys, ref_max)):
                lo, hi = rules[crop][key]
                width = (hi - lo) / (ref + 1e-8)
                matrix[i, j] = min(width, 1.0)

    return matrix


# ─── PyTorch Feature Gate Module ─────────────────────────────────────────────

class KnowledgeFeatureGate(nn.Module):
    """
    Knowledge-Guided Feature Gate (Steps 4 & 5).

    Takes raw input features and applies a learnable soft gate
    biased by domain knowledge importance scores.

    The gate output is a knowledge-aware feature representation
    that emphasizes agronomically important features.

    Args:
        num_features (int): Number of input features (default: 7)
        importance_prior (np.ndarray): Pre-computed importance vector (7,)
    """

    def __init__(self, num_features: int = 7, importance_prior: np.ndarray = None):
        super().__init__()
        self.num_features = num_features

        # Learnable gate: linear → sigmoid
        self.gate_net = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()
        )

        # Knowledge prior as a fixed buffer (not trained)
        if importance_prior is not None:
            prior = torch.tensor(importance_prior, dtype=torch.float32)
        else:
            prior = torch.ones(num_features)
        self.register_buffer("knowledge_prior", prior)

        # Learnable blend coefficient (how much to weight knowledge vs learned)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, num_features) — raw input features

        Returns:
            x_gated: (batch, num_features) — knowledge-aware features
            gate:    (batch, num_features) — gate values for interpretability
        """
        # Learned gate from data
        learned_gate = self.gate_net(x)  # (batch, 7)

        # Knowledge prior broadcast to batch
        prior = self.knowledge_prior.unsqueeze(0).expand_as(learned_gate)  # (batch, 7)

        # Blend: alpha controls learned vs prior
        alpha = torch.sigmoid(self.alpha)
        gate = alpha * learned_gate + (1.0 - alpha) * prior

        # Apply gate
        x_gated = x * gate

        return x_gated, gate


# ─── Standalone Analysis & Saving ─────────────────────────────────────────────

def run_feature_gate_analysis():
    """
    Run standalone feature gate analysis:
    1. Build structured knowledge base
    2. Compute feature importance
    3. Save importance table (CSV + JSON)
    4. Save gate weight visualization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("KNOWLEDGE-GUIDED FEATURE GATE ANALYSIS (Steps 4 & 5)")
    print("=" * 70)

    # 1. Build knowledge base
    print("\n[1/4] Building structured knowledge base...")
    kb = build_structured_knowledge_base()

    # 2. Compute feature importance
    print("\n[2/4] Computing feature importance from agronomic rules...")
    importance = compute_feature_importance_from_knowledge(kb)

    print("\nFeature Importance Scores (knowledge-derived):")
    for feat, imp in zip(FEATURES, importance):
        bar = "#" * int(imp * 20)
        print(f"  {feat:<15}: {imp:.4f}  {bar}")

    # 3. Save importance table
    print("\n[3/4] Saving feature importance table...")
    imp_df = pd.DataFrame({
        "Feature": FEATURES,
        "Knowledge_Importance": importance,
        "Rank": pd.Series(importance).rank(ascending=False).astype(int).values
    }).sort_values("Rank")

    imp_csv = TABLES_DIR / "knowledge_feature_importance.csv"
    imp_df.to_csv(imp_csv, index=False)
    print(f"  Saved: {imp_csv}")

    imp_json = TABLES_DIR / "knowledge_feature_importance.json"
    with open(imp_json, "w") as f:
        json.dump({
            "feature_importance": dict(zip(FEATURES, importance.tolist())),
            "ranked_features": imp_df["Feature"].tolist()
        }, f, indent=2)
    print(f"  Saved: {imp_json}")

    # 4. Build suitability matrix and save
    suitability = compute_per_crop_suitability(kb)
    np.save(DATA_PROCESSED / "knowledge_matrix_enhanced.npy", suitability)
    df_suit = pd.DataFrame(suitability, index=CROPS, columns=FEATURES)
    df_suit.to_csv(DATA_PROCESSED / "knowledge_matrix_enhanced.csv")
    print(f"  Enhanced matrix saved: {DATA_PROCESSED / 'knowledge_matrix_enhanced.npy'}")

    # 5. Visualize
    print("\n[4/4] Generating feature gate visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel A: Feature importance bar chart
    ax = axes[0]
    ax.set_facecolor("#F8FAFC")
    colors = ["#DC2626" if imp == importance.max() else "#0284C7" for imp in importance]
    bars = ax.barh(FEATURES, importance, color=colors, edgecolor="#0F172A", linewidth=0.8, height=0.6)
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#0F172A", fontsize=11, fontweight="bold")
    ax.set_title("Knowledge-Derived Feature Importance", color="#0F172A",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score (Normalized)", color="#0F172A", fontsize=12, fontweight="bold")
    ax.set_yticklabels(FEATURES, color="#0F172A", fontsize=11.5, fontweight="bold")
    ax.tick_params(colors="#0F172A", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1.20)
    ax.grid(True, alpha=0.5, color="#CBD5E1", linestyle="--", axis="x")

    # Panel B: Suitability heatmap
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    im = ax2.imshow(suitability, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(FEATURES, rotation=35, ha="right", color="#0F172A", fontsize=11, fontweight="bold")
    ax2.set_yticks(range(22))
    ax2.set_yticklabels(CROPS, color="#0F172A", fontsize=9.5, fontweight="bold")
    ax2.set_title("Crop–Feature Suitability Matrix", color="#0F172A",
                  fontsize=14, fontweight="bold", pad=12)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#0F172A", labelsize=10)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#94A3B8")

    fig.suptitle("Knowledge-Guided Feature Gate — Agronomic Suitability & Importance",
                 color="#0F172A", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = FIGURES_DIR / "knowledge_feature_gate.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n" + "=" * 70)
    print("KNOWLEDGE FEATURE GATE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Knowledge Base  : {DATA_PROCESSED / 'knowledge_base.json'}")
    print(f"  Enhanced Matrix : {DATA_PROCESSED / 'knowledge_matrix_enhanced.npy'}")
    print(f"  Importance CSV  : {imp_csv}")
    print(f"  Importance JSON : {imp_json}")
    print(f"  Gate Figure     : {fig_path}")

    return importance, kb


if __name__ == "__main__":
    run_feature_gate_analysis()
