"""
File: model_comparison_plot.py

Purpose:
Dynamically load evaluation metrics for all trained models (Baselines, FT-Transformer,
Hybrid-Concat, and KG-Hybrid) and generate comparative plots and summary tables.
Saves plots to outputs/plots/ and tables to outputs/tables/.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
TABLE_DIR      = PROJECT_ROOT / "outputs" / "tables"
PLOT_DIR       = PROJECT_ROOT / "outputs" / "plots"
REPORT_DIR     = PROJECT_ROOT / "outputs" / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CSV      = TABLE_DIR / "model_comparison_clean.csv"

# Pretty display names
MODEL_NAME_MAP = {
    "logistic_regression": "Logistic\nRegression",
    "decision_tree": "Decision\nTree",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
    "ft_transformer": "FT-\nTransformer",
    "hybrid_concat": "Hybrid-\nConcat",
    "kg_hybrid": "KG-\nHybrid"
}

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]

def load_all_metrics():
    """Dynamically load metrics JSON files for all models."""
    results = {}
    
    # Pre-defined load order
    order = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "ft_transformer",
        "hybrid_concat",
        "kg_hybrid"
    ]
    
    for key in order:
        json_path = TABLE_DIR / f"{key}_metrics.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            display_name = MODEL_NAME_MAP.get(key, key.replace("_", "\n").title())
            results[display_name] = {
                "Accuracy": data.get("Accuracy", 0.0),
                "Precision": data.get("Precision", 0.0),
                "Recall": data.get("Recall", 0.0),
                "F1-Score": data.get("F1-Score", 0.0)
            }
            
    # Fallback to defaults if empty
    if not results:
        print("Warning: No metrics JSON files found in outputs/tables/")
        
    return results

def save_clean_csv(results):
    """Save clean model comparison CSV."""
    rows = []
    for model, metrics in results.items():
        row = {"Model": model.replace("\n", " ")}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV saved: {CLEAN_CSV}")
    return df

def plot_grouped_bar(results):
    """Grouped bar chart — all models x all metrics."""
    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(METRICS)

    x = np.arange(n_models)
    width = 0.18 / (n_models / 5.0) if n_models > 5 else 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    COLORS = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77"]

    STYLE = {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#2a2a4a",
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(15, 7))

        for idx, (metric, color, offset) in enumerate(zip(METRICS, COLORS, offsets)):
            values = [results[m][metric] for m in models]
            bars = ax.bar(
                x + offset, values,
                width=width,
                color=color,
                alpha=0.88,
                label=metric,
                zorder=3
            )

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=7.0, color="#e0e0e0",
                    rotation=90
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0.90, 1.01)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            "Model Comparison — Crop Recommendation (22 Classes)",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = PLOT_DIR / "model_comparison_bar_chart.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

def plot_accuracy_horizontal(results):
    """Clean horizontal bar chart sorted by accuracy."""
    models = list(results.keys())
    acc_vals = [results[m]["Accuracy"] for m in models]

    sorted_pairs = sorted(zip(acc_vals, models))
    acc_vals, models = zip(*sorted_pairs)

    PALETTE = ["#3a86ff", "#3a86ff", "#3a86ff", "#ff6b6b", "#ff006e", "#833ab4", "#00b4d8"]
    if len(models) > len(PALETTE):
        PALETTE = PALETTE * 2

    STYLE = {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#2a2a4a",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        y = np.arange(len(models))

        bars = ax.barh(
            y, acc_vals,
            color=PALETTE[:len(models)],
            alpha=0.9,
            height=0.5,
            zorder=3
        )

        for bar, val in zip(bars, acc_vals):
            ax.text(
                val + 0.0003,
                bar.get_y() + bar.get_height() / 2,
                f"{val*100:.2f}%",
                va="center", fontsize=10,
                fontweight="bold", color="#e0e0e0"
            )

        ax.set_yticks(y)
        ax.set_yticklabels(
            [m.replace("\n", " ") for m in models],
            fontsize=11
        )
        ax.set_xlim(0.90, 1.005)
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title(
            "Model Accuracy Comparison — Crop Recommendation",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.grid(True, axis="x", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = PLOT_DIR / "model_accuracy_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

def plot_radar(results):
    """Radar / Spider chart comparing all models."""
    labels = METRICS
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    COLORS_RADAR = [
        "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#ff6fc8", "#9b59b6", "#1abc9c"
    ]

    STYLE = {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(
            figsize=(8, 8),
            subplot_kw=dict(polar=True)
        )
        ax.set_facecolor("#16213e")

        for (model, metrics), color in zip(results.items(), COLORS_RADAR):
            values = [metrics[m] for m in labels]
            values += values[:1]
            ax.plot(
                angles, values,
                color=color, linewidth=2,
                label=model.replace("\n", " ")
            )
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_thetagrids(
            np.degrees(angles[:-1]), labels,
            fontsize=12
        )
        ax.set_ylim(0.90, 1.0)
        ax.set_title(
            "Model Comparison — Radar Chart",
            fontsize=14, fontweight="bold", pad=20,
            color="#e0e0e0"
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1),
            fontsize=9
        )
        ax.tick_params(colors="#e0e0e0")
        ax.spines["polar"].set_color("#2a2a4a")

        plt.tight_layout()
        path = PLOT_DIR / "model_comparison_radar.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

def main():
    print("=" * 60)
    print("GENERATING MODEL COMPARISON PLOTS & TABLES")
    print("=" * 60)

    results = load_all_metrics()
    print(f"Loaded metrics for {len(results)} models.")

    save_clean_csv(results)
    plot_grouped_bar(results)
    plot_accuracy_horizontal(results)
    plot_radar(results)

    print(f"\nAll plots and clean tables successfully saved to: {PLOT_DIR} and {TABLE_DIR}")

if __name__ == "__main__":
    main()
