"""
generate_all_paper_figures.py
==============================
Generates all 8 high-resolution publication-quality figures for the paper:
  - Figure 1: KG-Hybrid System Architecture Flowchart
  - Figure 2: Comprehensive Ablation Study Results Bar Chart
  - Figure 3: Missing Feature Recovery: Raw vs KNN Imputed
  - Figure 4: Early Exit Mechanism: Latency vs Threshold & Accuracy
  - Figure 5: Comprehensive Robustness Comparison (Missing, Noise, Extrapolation)
  - Figure 6: Feature Importance & Knowledge Prior Analysis
  - Figure 7: Calibrated Production API Output Payload Schema
  - Figure 8: Parameter Count Footprint vs Inference Latency
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import save_figure, copy_artifact
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR   = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Apply global publication style (dark luxury aesthetic)
plt.style.use('dark_background')
BG_COLOR   = '#0B0E14'
CARD_COLOR = '#161B22'
BORDER_COL = '#30363D'
TEXT_COLOR = '#F0F6FC'
ACCENT_BLUE = '#58A6FF'
ACCENT_GREEN = '#3FB950'
ACCENT_ORANGE = '#D29922'
ACCENT_RED   = '#F85149'
ACCENT_PURPLE = '#BC8CFF'

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': BORDER_COL,
    'axes.facecolor': CARD_COLOR,
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': BORDER_COL,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})


def fig1_architecture():
    """Figure 1: Architecture Diagram Flowchart"""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'KG-Hybrid System Architecture', ha='center', va='center', fontsize=16, fontweight='bold', color=TEXT_COLOR)
    ax.text(0.5, 0.90, 'Stage 1: XGBoost Decision Tree Leaf Extraction | Stage 2: FT-Transformer Context + Knowledge-Guided Fusion', 
            ha='center', va='center', fontsize=10, color='#8B949E')

    # Draw boxes
    boxes = [
        (0.08, 0.65, 0.22, 0.18, 'Input Features X\n(7 Soil & Climate Factors)', ACCENT_BLUE),
        (0.38, 0.72, 0.24, 0.14, 'XGBoost Booster\n(200 Trees, 4400 Leaves)', ACCENT_ORANGE),
        (0.38, 0.52, 0.24, 0.14, 'FT-Transformer Backbone\n(Feature Tokenizer + LayerNorm)', ACCENT_PURPLE),
        (0.68, 0.62, 0.24, 0.18, 'Knowledge-Guided Fusion\n(Adaptive Gate + Matrix Prior)', ACCENT_GREEN),
        (0.38, 0.20, 0.54, 0.18, 'Production Pipeline Wrapper\n(KNN Imputer + Early Exit τ=0.90 + MC Dropout)', ACCENT_RED),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=CARD_COLOR, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    # Arrows
    ax.annotate('', xy=(0.38, 0.79), xytext=(0.30, 0.74), arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=2))
    ax.annotate('', xy=(0.38, 0.59), xytext=(0.30, 0.74), arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=2))
    ax.annotate('', xy=(0.68, 0.71), xytext=(0.62, 0.79), arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=2))
    ax.annotate('', xy=(0.68, 0.71), xytext=(0.62, 0.59), arrowprops=dict(arrowstyle="->", color=ACCENT_PURPLE, lw=2))
    ax.annotate('', xy=(0.65, 0.38), xytext=(0.80, 0.62), arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2))

    plt.tight_layout()
    p = FIGURES_DIR / "fig1_architecture.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] Figure 1 saved -> {p}")


def fig2_ablation():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import ablation_plot
    ablation_plot()


def fig3_missing_features():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import missing_plot
    missing_plot()


def fig4_early_exit():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import routing_plot
    routing_plot()


def fig5_robustness():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import robustness_plot
    robustness_plot()


def fig6_shap_importance():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import xai_figures
    xai_figures()


def fig7_api_schema():
    """Figure 7: Calibrated Production API Output Payload Schema"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    sample_json = (
        "{\n"
        '  "timestamp": "2026-08-17T16:45:00.000Z",\n'
        '  "model_version": "v1.2.0-ProductionLightweight",\n'
        '  "recommended_crop": "coffee",\n'
        '  "confidence": 0.9956,\n'
        '  "uncertainty_std": 0.0020,\n'
        '  "top_3_crops": [\n'
        '    {"crop": "coffee", "probability": 0.9956, "uncertainty_std": 0.0020},\n'
        '    {"crop": "jute", "probability": 0.0012, "uncertainty_std": 0.0004},\n'
        '    {"crop": "rice", "probability": 0.0009, "uncertainty_std": 0.0004}\n'
        '  ],\n'
        '  "inference_route": "STAGE_1_EARLY_EXIT (XGBoost)",\n'
        '  "data_quality": "HEALTHY (All 7 features provided)",\n'
        '  "sensor_health": "OK",\n'
        '  "latency_ms": 5.99,\n'
        '  "safety_alert": null\n'
        "}"
    )

    rect = plt.Rectangle((0.05, 0.05), 0.90, 0.88, facecolor=CARD_COLOR, edgecolor=ACCENT_GREEN, linewidth=2)
    ax.add_patch(rect)

    ax.text(0.10, 0.86, 'Calibrated Production API Payload Schema (JSON)', fontsize=12, fontweight='bold', color=ACCENT_GREEN)
    ax.text(0.10, 0.12, sample_json, fontsize=9.5, fontfamily='monospace', color='#E6EDE3', va='bottom')

    plt.tight_layout()
    p = FIGURES_DIR / "fig7_api_schema.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] Figure 7 saved -> {p}")


def fig8_param_latency_tradeoff():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import parameter_plot
    parameter_plot()


def main():
    print("==================================================")
    print("GENERATING ALL 8 PUBLICATION PAPER FIGURES")
    print("==================================================")

    fig1_architecture()
    fig2_ablation()
    fig3_missing_features()
    fig4_early_exit()
    fig5_robustness()
    fig6_shap_importance()
    fig7_api_schema()
    fig8_param_latency_tradeoff()

    print("\n" + "=" * 70)
    print(f"[OK] ALL 8 FIGURES GENERATED SUCCESSFULLY IN -> {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
