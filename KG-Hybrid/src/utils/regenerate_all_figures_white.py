"""
regenerate_all_figures_white.py
=================================
Regenerates ALL figures with:
  1. White background (paper-ready)
  2. Large, readable font sizes
  3. Bold, vivid colour palette (no dark theme)
  4. Subtitle/caption removed from fig1 architecture
  5. All other titles kept as original

Run from KG-Hybrid directory:
    python src/utils/regenerate_all_figures_white.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import save_figure, copy_artifact
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
PLOTS_DIR    = PROJECT_ROOT / "outputs" / "plots"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BG_COLOR    = '#FFFFFF'
CARD_COLOR  = '#F7F9FC'
BORDER_COL  = '#CBD5E1'
TEXT_COLOR  = '#0F172A'

COL_BLUE   = '#1565C0'
COL_GREEN  = '#1B7E4A'
COL_ORANGE = '#D84315'
COL_RED    = '#C62828'
COL_PURPLE = '#6A1B9A'
COL_CYAN   = '#00838F'
COL_AMBER  = '#E65100'

BAR_BLUE   = '#42A5F5'
BAR_GREEN  = '#66BB6A'
BAR_ORANGE = '#FFA726'
BAR_RED    = '#EF5350'
BAR_PURPLE = '#AB47BC'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.edgecolor': '#90A4AE',
    'axes.linewidth': 1.4,
    'axes.facecolor': CARD_COLOR,
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': '#CFD8DC',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'axes.titlepad': 14,
})


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    # Upper caption removed per paper publication formatting
    boxes = [
        (0.05, 0.60, 0.22, 0.22, 'Input Features X\n(7 Soil & Climate Factors)', COL_BLUE),
        (0.35, 0.68, 0.24, 0.16, 'XGBoost Booster\n(200 Trees, 4400 Leaves)', COL_AMBER),
        (0.35, 0.46, 0.24, 0.16, 'FT-Transformer Backbone\n(Feature Tokenizer + LayerNorm)', COL_PURPLE),
        (0.65, 0.57, 0.26, 0.22, 'Knowledge-Guided Fusion\n(Adaptive Gate + Matrix Prior)', COL_GREEN),
        (0.35, 0.15, 0.54, 0.20, 'Production Pipeline Wrapper\n(KNN Imputer + Early Exit tau=0.90 + MC Dropout)', COL_RED),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor='#FAFAFA', edgecolor=color, linewidth=2.8,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color,
                transform=ax.transAxes, zorder=3)
    arrow_kw = dict(arrowstyle='->', lw=2.5, mutation_scale=20)
    ax.annotate('', xy=(0.35, 0.76), xytext=(0.27, 0.76),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(color=COL_AMBER, **arrow_kw))
    ax.annotate('', xy=(0.35, 0.54), xytext=(0.27, 0.71),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(color=COL_PURPLE, **arrow_kw))
    ax.annotate('', xy=(0.65, 0.68), xytext=(0.59, 0.76),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(color=COL_AMBER, **arrow_kw))
    ax.annotate('', xy=(0.65, 0.68), xytext=(0.59, 0.54),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(color=COL_PURPLE, **arrow_kw))
    ax.annotate('', xy=(0.62, 0.35), xytext=(0.78, 0.57),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(color=COL_GREEN, **arrow_kw))
    plt.tight_layout()
    p = FIGURES_DIR / "fig1_architecture.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] fig1_architecture -> {p}")


def fig2_ablation():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import ablation_plot
    return ablation_plot()


def fig3_missing_features():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import missing_plot
    return missing_plot()


def fig4_early_exit():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import routing_plot
    return routing_plot()


def fig5_robustness():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import robustness_plot
    return robustness_plot()


def fig6_shap():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import xai_figures
    return xai_figures()


def fig7_api():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis('off')
    sample_json = ('{\n'
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
        '}')
    rect = patches.FancyBboxPatch((0.04, 0.04), 0.92, 0.90, boxstyle="round,pad=0.03",
                                  facecolor='#F8FAFC', edgecolor=COL_GREEN, linewidth=2.5)
    ax.add_patch(rect)
    # Upper title removed per publication formatting
    ax.text(0.08, 0.12, sample_json, fontsize=10.5, fontfamily='monospace',
            color=TEXT_COLOR, va='bottom', fontweight='bold')
    plt.tight_layout()
    p = FIGURES_DIR / "fig7_api_schema.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] fig7_api_schema -> {p}")


def fig8_param():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import parameter_plot
    return parameter_plot()


def _load_model_metrics():
    """Compute all comparison scores from canonical saved predictions."""
    from sklearn.metrics import precision_recall_fscore_support
    from src.experiments.publication_evidence import load_evidence
    labels, evidence = load_evidence()
    results = {}
    for record in evidence:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, record["probas"].argmax(1), average="macro", zero_division=0)
        results[record["model_name"]] = {
            "Accuracy": record["accuracy"], "Precision": precision, "Recall": recall, "F1-Score": f1}
    return results


def plot_radar(results):
    if not results:
        print("[SKIP] No metrics for radar chart.")
        return
    labels = ["Accuracy","Precision","Recall","F1-Score"]
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist() + [0]
    COLORS_RADAR = [COL_CYAN, COL_RED, BAR_ORANGE, COL_GREEN, COL_PURPLE, COL_AMBER, COL_BLUE]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor('#F0F4F8')
    for (model, metrics), color in zip(results.items(), COLORS_RADAR):
        values = [metrics[m] for m in labels] + [metrics[labels[0]]]
        ax.plot(angles, values, color=color, linewidth=2.8, label=model.replace("\n", " "))
        ax.fill(angles, values, color=color, alpha=0.07)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylim(0.90, 1.0)
    # Upper caption removed per publication formatting
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.spines["polar"].set_color('#B0BEC5')
    plt.tight_layout()
    p = PLOTS_DIR / "model_comparison_radar.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] model_comparison_radar -> {p}")


def plot_model_bar(results):
    if not results:
        return
    models  = list(results.keys())
    metrics = ["Accuracy","Precision","Recall","F1-Score"]
    x = np.arange(len(models))
    width = 0.18
    offsets = np.linspace(-(len(metrics)-1)/2, (len(metrics)-1)/2, len(metrics)) * width
    COLORS  = [BAR_BLUE, BAR_RED, BAR_ORANGE, BAR_GREEN]
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)
    for metric, color, offset in zip(metrics, COLORS, offsets):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x+offset, values, width=width, color=color, alpha=0.92, label=metric, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0.90, 1.015)
    ax.set_ylabel("Score", fontsize=14, fontweight='bold')
    # Upper caption removed per publication formatting
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, axis="y", zorder=0, linestyle='--', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = PLOTS_DIR / "model_comparison_bar_chart.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] model_comparison_bar_chart -> {p}")


def plot_accuracy_horizontal(results):
    if not results:
        return
    models = list(results.keys())
    acc_vals = [results[m]["Accuracy"] for m in models]
    sorted_pairs = sorted(zip(acc_vals, models))
    acc_vals, models = zip(*sorted_pairs)
    PALETTE = [COL_BLUE, COL_BLUE, COL_BLUE, COL_RED, COL_PURPLE, COL_AMBER, COL_CYAN]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    y = np.arange(len(models))
    bars = ax.barh(y, acc_vals, color=PALETTE[:len(models)], alpha=0.92, height=0.55, zorder=3)
    for bar, val in zip(bars, acc_vals):
        ax.text(val+0.0003, bar.get_y()+bar.get_height()/2, f"{val*100:.2f}%",
                va="center", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("\n", " ") for m in models], fontsize=12, fontweight='bold')
    ax.set_xlim(0.90, 1.008)
    ax.set_xlabel("Accuracy", fontsize=14, fontweight='bold')
    # Upper caption removed per publication formatting
    ax.grid(True, axis="x", zorder=0, linestyle='--', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = PLOTS_DIR / "model_accuracy_comparison.png"
    save_figure(plt.gcf(), p, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"[OK] model_accuracy_comparison -> {p}")


def fig5_early_exit_flowchart():
    fig, ax = plt.subplots(figsize=(15, 12), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    C_BLUE   = {'bg': '#E3F2FD', 'border': '#1565C0', 'text': '#0D47A1'}
    C_GREEN  = {'bg': '#E8F5E9', 'border': '#1B7E4A', 'text': '#1B5E20'}
    C_ORANGE = {'bg': '#FFF3E0', 'border': '#D84315', 'text': '#BF360C'}
    C_PURPLE = {'bg': '#F3E5F5', 'border': '#6A1B9A', 'text': '#4A148C'}
    C_SLATE  = {'bg': '#ECEFF1', 'border': '#455A64', 'text': '#263238'}
    def draw_node(x, y, w, h, title, subtitle=None, style=C_BLUE, rx=0.6):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rx}",
                                      lw=2.5, edgecolor=style['border'], facecolor=style['bg'])
        ax.add_patch(rect)
        cy = y + h/2 + (1.0 if subtitle else 0)
        ax.text(x+w/2, cy, title, ha='center', va='center', fontsize=11, fontweight='bold', color=style['text'])
        if subtitle:
            ax.text(x+w/2, y+h/2-1.0, subtitle, ha='center', va='center', fontsize=10, color='#37474F')
    def draw_diamond(cx, cy, w, h, title, style=C_ORANGE):
        pts = [[cx, cy+h/2],[cx+w/2, cy],[cx, cy-h/2],[cx-w/2, cy]]
        poly = patches.Polygon(pts, closed=True, lw=2.5, edgecolor=style['border'], facecolor=style['bg'])
        ax.add_patch(poly)
        ax.text(cx, cy, title, ha='center', va='center', fontsize=11, fontweight='bold', color=style['text'])
    def draw_edge(start, end, label=None, color='#546E7A', rad=0.0):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.2, mutation_scale=16,
                                    connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2+1.0
            ax.text(mx, my, label, ha='center', va='center', fontsize=10, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor=color, lw=1))
    # Upper title and subtitle banner removed per publication formatting
    draw_node(30, 84, 40, 7, "Start: New Agricultural Sample", "(N, P, K, Temp, Humidity, pH, Rainfall)", style=C_BLUE)
    draw_diamond(50, 72, 34, 11, "Missing Features?", style=C_SLATE)
    draw_node(78, 68, 18, 7, "KNN Imputation", "(k=5, fitted)", style=C_SLATE)
    draw_edge((67, 72), (78, 71.5), label="Yes", color='#455A64')
    draw_edge((87, 68), (87, 57), color='#455A64')
    draw_edge((87, 57), (67, 57), color='#455A64')
    draw_edge((50, 66.5), (50, 60), label="No / Imputed", color=C_BLUE['border'])
    draw_node(33, 52, 34, 8, "Stage 1: XGBoost Tree Ensemble", "Extract Leaf Indices & Predict Confidence", style=C_BLUE)
    draw_edge((50, 52), (50, 45.5), color=C_BLUE['border'])
    draw_diamond(50, 38.5, 36, 12, "Confidence >= 0.90 ?", style=C_ORANGE)
    draw_node(4, 19, 38, 10, "XGBoost high-confidence path",
              "Return XGBoost Fast Prediction\nHigh confidence; latency not implied", style=C_GREEN)
    draw_edge((32, 38.5), (23, 29), label="YES (High Conf)", color=C_GREEN['border'], rad=-0.1)
    draw_node(58, 19, 38, 10, "STAGE 2 FULL KG-HYBRID (3.64%)",
              "Execute FT-Transformer + Knowledge Fusion\nLatency: ~90ms | Refined Uncertainty", style=C_ORANGE)
    draw_edge((68, 38.5), (77, 29), label="NO (Uncertain)", color=C_ORANGE['border'], rad=0.1)
    draw_node(25, 3, 50, 8, "Final Recommendation Output",
              "Recommended Crop + Confidence Score + Uncertainty", style=C_PURPLE)
    draw_edge((23, 19), (40, 11), color=C_GREEN['border'], rad=0.05)
    draw_edge((77, 19), (60, 11), color=C_ORANGE['border'], rad=-0.05)
    stat_rect = patches.FancyBboxPatch((2, 69), 22, 23, boxstyle="round,pad=0.4,rounding_size=0.5",
                                        lw=2, edgecolor=C_GREEN['border'], facecolor=C_GREEN['bg'])
    ax.add_patch(stat_rect)
    ax.text(13, 89.5, "PERFORMANCE STATS", ha='center', va='center', fontsize=12, fontweight='bold', color=C_GREEN['text'])
    from src.experiments.publication_evidence import rebuild_routing_evidence
    audit = rebuild_routing_evidence()
    route = audit.loc[audit.threshold == .9].iloc[0]
    for i, s in enumerate([f"XGBoost exits: {route.xgb_exit_fraction:.2%}",
                           f"Hybrid calls: {int(route.hybrid_count)}/330",
                           f"Cascade accuracy: {route.cascade_accuracy:.2%}",
                           "Latency: not measured in this audit",
                           "Healthy inputs; deterministic routes"]):
        ax.text(4, 86 - i*3.3, "  " + s, ha='left', va='center', fontsize=10, fontweight='bold', color=C_GREEN['text'])
    plt.tight_layout()
    p1 = FIGURES_DIR / "fig5_early_exit_flow_diagram.png"
    save_figure(plt.gcf(), p1, dpi=200, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"[OK] fig5_early_exit_flow_diagram -> {p1}")


def fig6_feature_complementarity():
    """Measured sensitivity comparison; no assumed complementarity claim."""
    from src.utils.generate_paper_tables_and_plots import generate_fig_f5_feature_importance_3way
    import shutil
    generate_fig_f5_feature_importance_3way()
    for name in ("fig6_feature_complementarity.png", "fig6_feature_complementarity_light.png"):
        copy_artifact(FIGURES_DIR / "fig_f5_feature_importance_3way.png", FIGURES_DIR / name)


def ablation_accuracy_plot():
    """Regenerate from recorded experimental evidence."""
    from src.utils.review_figures import ablation_plot
    return ablation_plot()


def main():
    print("="*70)
    print("  REGENERATING ALL FIGURES - WHITE BG, LARGE FONTS, BOLD COLOURS")
    print("="*70)
    print("\n[Paper Figures 1-8]")
    fig1_architecture()
    fig2_ablation()
    fig3_missing_features()
    fig4_early_exit()
    fig5_robustness()
    fig6_shap()
    fig7_api()
    fig8_param()
    print("\n[Model Comparison Plots]")
    results = _load_model_metrics()
    if results:
        print(f"  Loaded metrics for {len(results)} models.")
        plot_radar(results)
        plot_model_bar(results)
        plot_accuracy_horizontal(results)
    else:
        print("  [WARN] No model metric JSON files found - skipping comparison plots.")
    print("\n[Methodology Figures]")
    fig5_early_exit_flowchart()
    fig6_feature_complementarity()
    print("\n[Extra Plots]")
    ablation_accuracy_plot()
    print("\n"+"="*70)
    print(f"  [DONE] All figures regenerated!")
    print(f"  Figures -> {FIGURES_DIR}")
    print(f"  Plots   -> {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
