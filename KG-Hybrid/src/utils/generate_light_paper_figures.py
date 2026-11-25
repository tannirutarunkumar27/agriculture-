"""
generate_light_paper_figures.py
================================
Generates publication-quality figures for Robustness and Production sections
with clean, light academic backgrounds (IEEE style).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import save_figure, copy_artifact
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define Light Academic Theme
BG_COLOR    = '#FFFFFF'
CARD_COLOR  = '#F8FAFC'
BORDER_COL  = '#CBD5E1'
TEXT_COLOR  = '#0F172A'
SUBTEXT_COL = '#475569'

PRIMARY_BLUE   = '#1E40AF' # KG-Hybrid / Primary
SECONDARY_BLUE = '#3B82F6'
ACCENT_GREEN   = '#16A34A' # Imputation / Production
ACCENT_RED     = '#DC2626' # Baseline / Raw / High Risk
ACCENT_ORANGE  = '#D97706' # XGBoost / Latency
ACCENT_PURPLE  = '#7C3AED' # Attention / Transformer

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#94A3B8',
    'axes.linewidth': 1.2,
    'axes.facecolor': CARD_COLOR,
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': '#E2E8F0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})


def fig3_missing_features_light():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import missing_plot
    missing_plot()


def fig4_early_exit_light():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import routing_plot
    routing_plot()


def fig5_robustness_light():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import robustness_plot
    robustness_plot()


def fig6_feature_complementarity_light():
    """Use measured gain, input gradients and TreeSHAP; no invented attention."""
    import sys
    import shutil
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.generate_paper_tables_and_plots import generate_fig_f5_feature_importance_3way
    generate_fig_f5_feature_importance_3way()
    for name in ("fig6_feature_complementarity.png", "fig6_feature_complementarity_light.png"):
        copy_artifact(FIGURES_DIR / "fig_f5_feature_importance_3way.png", FIGURES_DIR / name)


def fig7_api_schema_light():
    """Figure 7: Calibrated Production API Payload Schema (Light Academic Box)"""
    print("[Figure] Generating Fig 7: Production API Schema (Light)...")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
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
        '    {"crop": "jute",   "probability": 0.0012, "uncertainty_std": 0.0004},\n'
        '    {"crop": "rice",   "probability": 0.0009, "uncertainty_std": 0.0004}\n'
        '  ],\n'
        '  "inference_route": "STAGE_1_EARLY_EXIT (XGBoost)",\n'
        '  "data_quality": "HEALTHY (All 7 features provided)",\n'
        '  "sensor_health": "OK",\n'
        '  "latency_ms": 5.99,\n'
        '  "safety_alert": null\n'
        "}"
    )

    rect = patches.FancyBboxPatch((0.04, 0.04), 0.92, 0.90, boxstyle="round,pad=0.03", 
                                facecolor='#F8FAFC', edgecolor='#16A34A', linewidth=2)
    ax.add_patch(rect)

    ax.text(0.08, 0.88, 'Production REST API Payload Schema (JSON Diagnostics)', fontsize=12, fontweight='bold', color='#16A34A')
    ax.text(0.08, 0.09, sample_json, fontsize=9.5, fontfamily='monospace', color='#0F172A', va='bottom', fontweight='bold')

    plt.tight_layout()
    p1 = FIGURES_DIR / "fig7_api_schema.png"
    p2 = FIGURES_DIR / "fig7_api_schema_light.png"
    save_figure(plt.gcf(), p1, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    save_figure(plt.gcf(), p2, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"    [OK] Saved -> {p1}")


def fig8_param_latency_tradeoff_light():
    """Generate from empirical records; no manually specified outcome metrics."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.review_figures import parameter_plot
    parameter_plot()


def main():
    print("==================================================")
    print("GENERATING LIGHT ACADEMIC FIGURES (ROBUSTNESS & PRODUCTION)")
    print("==================================================")
    fig3_missing_features_light()
    fig4_early_exit_light()
    fig5_robustness_light()
    fig6_feature_complementarity_light()
    fig7_api_schema_light()
    fig8_param_latency_tradeoff_light()
    print("==================================================")
    print(f"[OK] ALL LIGHT ACADEMIC FIGURES SAVED TO -> {FIGURES_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()
