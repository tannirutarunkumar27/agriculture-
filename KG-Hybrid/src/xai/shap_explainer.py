"""
Step 9 — Explainable AI Module (SHAP + Feature Importance Analysis)
Saves all results to outputs/figures/xai/ and outputs/tables/
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH   = PROJECT_ROOT / "data" / "processed"
MODEL_PATH  = PROJECT_ROOT / "outputs" / "models"
XAI_FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "xai"
TABLE_DIR   = PROJECT_ROOT / "outputs" / "tables"
XAI_FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
CROPS = [
    'apple','banana','blackgram','chickpea','coconut','coffee',
    'cotton','grapes','jute','kidneybeans','lentil','maize',
    'mango','mothbeans','mungbean','muskmelon','orange','papaya',
    'pigeonpeas','pomegranate','rice','watermelon'
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(DATA_PATH / "train.csv")
test_df  = pd.read_csv(DATA_PATH / "test.csv")
X_train  = train_df[FEATURES].values
X_test   = test_df[FEATURES].values
y_train  = train_df["label"].values
y_test   = test_df["label"].values

le = joblib.load(DATA_PATH / "label_encoder.pkl")
crop_names = le.classes_.tolist()

# ── Load XGBoost ───────────────────────────────────────────────────────────
print("Loading XGBoost model...")
try:
    xgb_model = joblib.load(MODEL_PATH / "xgboost_model.pkl")
    XGB_OK = True
except Exception as e:
    print(f"  [WARN] XGBoost not found: {e}")
    XGB_OK = False

# ── Load sklearn baselines ─────────────────────────────────────────────────
print("Loading baseline models...")
BASELINE_MODELS = {}
for name, fname in [("Random Forest","random_forest_model.pkl"),
                    ("Decision Tree","decision_tree_model.pkl"),
                    ("Logistic Regression","logistic_regression_model.pkl")]:
    p = MODEL_PATH / fname
    if p.exists():
        BASELINE_MODELS[name] = joblib.load(p)

# ══════════════════════════════════════════════════════════════════════════════
# PART A — XGBoost SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════════
shap_results = {}

if XGB_OK:
    try:
        import shap
        print("\n[A] XGBoost SHAP Analysis...")

        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_test)

        # Handle shape: some versions return (n_samples, n_features, n_classes)
        if isinstance(shap_values_xgb, list):
            # List of arrays per class — take mean abs across classes
            shap_mean = np.mean([np.abs(sv) for sv in shap_values_xgb], axis=0)
        elif shap_values_xgb.ndim == 3:
            shap_mean = np.mean(np.abs(shap_values_xgb), axis=2)
        else:
            shap_mean = np.abs(shap_values_xgb)

        # Global importance: mean |SHAP| per feature
        global_imp = shap_mean.mean(axis=0)
        shap_results["XGBoost"] = dict(zip(FEATURES, global_imp.tolist()))

        # ── Bar chart ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        sorted_idx = np.argsort(global_imp)
        colors = ["#e74c3c" if i == sorted_idx[-1] else "#3498db" for i in sorted_idx]
        ax.barh([FEATURES[i] for i in sorted_idx], global_imp[sorted_idx],
                color=colors, edgecolor="#30363d", linewidth=0.5)
        for i, (idx, val) in enumerate(zip(sorted_idx, global_imp[sorted_idx])):
            ax.text(val + 0.0005, i, f"{val:.4f}", va="center",
                    color="#c9d1d9", fontsize=8)
        ax.set_title("XGBoost — SHAP Feature Importance (Mean |SHAP|)",
                     color="#f0f6fc", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.grid(True, alpha=0.2, color="#30363d", axis="x")
        plt.tight_layout()
        plt.savefig(XAI_FIG_DIR / "shap_summary_xgboost.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print("  [OK] XGBoost SHAP bar chart saved.")

        # ── Beeswarm (dot plot) ───────────────────────────────────────────
        # Flatten shap_mean to 2-D for beeswarm
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        # Simulate beeswarm with scatter
        for j, feat in enumerate(FEATURES):
            vals = shap_mean[:, j]
            feat_vals = X_test[:, j]
            # Normalize feature value for color
            range_val = feat_vals.max() - feat_vals.min()
            norm = (feat_vals - feat_vals.min()) / (range_val + 1e-8)
            sc = ax.scatter(vals, np.full_like(vals, j) + np.random.uniform(-0.2, 0.2, len(vals)),
                            c=norm, cmap="coolwarm", alpha=0.4, s=8)
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(FEATURES, color="#c9d1d9")
        ax.set_xlabel("SHAP Value (impact on model output)", color="#c9d1d9")
        ax.set_title("XGBoost — SHAP Beeswarm Plot", color="#f0f6fc",
                     fontsize=12, fontweight="bold")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.axvline(0, color="#c9d1d9", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.15, color="#30363d", axis="x")
        cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(colors="#c9d1d9")
        cb.set_label("Feature Value (low → high)", color="#c9d1d9", fontsize=8)
        plt.tight_layout()
        plt.savefig(XAI_FIG_DIR / "shap_beeswarm_xgboost.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print("  [OK] XGBoost SHAP beeswarm saved.")

        # ── Waterfall for one sample ──────────────────────────────────────
        sample_idx = 0
        sample_shap = shap_mean[sample_idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        colors_wf = ["#e74c3c" if v > 0 else "#3498db" for v in sample_shap]
        ax.barh(FEATURES, sample_shap, color=colors_wf, edgecolor="#30363d", linewidth=0.5)
        ax.set_title(f"XGBoost — SHAP Waterfall (Sample #{sample_idx})",
                     color="#f0f6fc", fontsize=12, fontweight="bold")
        ax.set_xlabel("SHAP Value", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.axvline(0, color="#c9d1d9", linewidth=0.8)
        ax.grid(True, alpha=0.2, color="#30363d", axis="x")
        plt.tight_layout()
        plt.savefig(XAI_FIG_DIR / "shap_waterfall_xgboost.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print("  [OK] XGBoost SHAP waterfall saved.")

        np.save(XAI_FIG_DIR / "shap_values_xgboost.npy", shap_mean)

    except ImportError:
        print("  [WARN] shap not installed. Install with: pip install shap")
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — XGBoost Built-in Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
if XGB_OK:
    print("\n[B] XGBoost Built-in Feature Importance...")
    try:
        booster = xgb_model.get_booster()
        imp_types = {"weight": {}, "gain": {}, "cover": {}}
        for imp_type in imp_types:
            scores = booster.get_score(importance_type=imp_type)
            for k, v in scores.items():
                if k in FEATURES:
                    imp_types[imp_type][k] = v
                elif k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(FEATURES):
                        imp_types[imp_type][FEATURES[idx]] = v

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0d1117")
        titles = {"weight": "Feature Weight (Frequency)",
                  "gain": "Feature Gain (Information)",
                  "cover": "Feature Cover (Samples)"}
        for ax, (imp_type, scores) in zip(axes, imp_types.items()):
            ax.set_facecolor("#161b22")
            if scores:
                feats = list(scores.keys())
                vals = list(scores.values())
                vals_arr = np.array(vals)
                vals_norm = vals_arr / (vals_arr.max() + 1e-8)
                sorted_idx = np.argsort(vals_norm)
                ax.barh([feats[i] for i in sorted_idx], vals_norm[sorted_idx],
                        color="#3498db", edgecolor="#30363d", linewidth=0.5)
            ax.set_title(titles[imp_type], color="#f0f6fc", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#c9d1d9")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
            ax.grid(True, alpha=0.2, color="#30363d", axis="x")
        fig.suptitle("XGBoost Feature Importance — Three Metrics",
                     color="#f0f6fc", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(XAI_FIG_DIR / "xgboost_feature_importance_3panel.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print("  [OK] XGBoost 3-panel feature importance saved.")
    except Exception as e:
        print(f"  [WARN] Built-in importance failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Random Forest Feature Importance (Gini / MDI)
# ══════════════════════════════════════════════════════════════════════════════
if "Random Forest" in BASELINE_MODELS:
    print("\n[C] Random Forest Feature Importance (Gini/MDI)...")
    rf = BASELINE_MODELS["Random Forest"]
    rf_imp = rf.feature_importances_
    shap_results["Random Forest"] = dict(zip(FEATURES, rf_imp.tolist()))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    sorted_idx = np.argsort(rf_imp)
    colors = ["#e74c3c" if i == sorted_idx[-1] else "#9b59b6" for i in sorted_idx]
    ax.barh([FEATURES[i] for i in sorted_idx], rf_imp[sorted_idx],
            color=colors, edgecolor="#30363d", linewidth=0.5)
    for i, (idx, val) in enumerate(zip(sorted_idx, rf_imp[sorted_idx])):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center",
                color="#c9d1d9", fontsize=8)
    ax.set_title("Random Forest — Gini Feature Importance (MDI)",
                 color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.2, color="#30363d", axis="x")
    plt.tight_layout()
    plt.savefig(XAI_FIG_DIR / "feature_importance_random_forest.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Random Forest importance saved.")


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Decision Tree Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
if "Decision Tree" in BASELINE_MODELS:
    print("\n[D] Decision Tree Feature Importance...")
    dt = BASELINE_MODELS["Decision Tree"]
    dt_imp = dt.feature_importances_
    shap_results["Decision Tree"] = dict(zip(FEATURES, dt_imp.tolist()))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    sorted_idx = np.argsort(dt_imp)
    ax.barh([FEATURES[i] for i in sorted_idx], dt_imp[sorted_idx],
            color="#2ecc71", edgecolor="#30363d", linewidth=0.5)
    ax.set_title("Decision Tree — Feature Importance",
                 color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Importance", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.2, color="#30363d", axis="x")
    plt.tight_layout()
    plt.savefig(XAI_FIG_DIR / "feature_importance_decision_tree.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Decision Tree importance saved.")


# ══════════════════════════════════════════════════════════════════════════════
# PART E — Logistic Regression Coefficients
# ══════════════════════════════════════════════════════════════════════════════
if "Logistic Regression" in BASELINE_MODELS:
    print("\n[E] Logistic Regression Coefficient Analysis...")
    lr = BASELINE_MODELS["Logistic Regression"]
    # coef_ shape: (n_classes, n_features) — take mean absolute across classes
    lr_imp = np.mean(np.abs(lr.coef_), axis=0)
    lr_imp_norm = lr_imp / (lr_imp.max() + 1e-8)
    shap_results["Logistic Regression"] = dict(zip(FEATURES, lr_imp_norm.tolist()))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    sorted_idx = np.argsort(lr_imp_norm)
    ax.barh([FEATURES[i] for i in sorted_idx], lr_imp_norm[sorted_idx],
            color="#e67e22", edgecolor="#30363d", linewidth=0.5)
    ax.set_title("Logistic Regression — Mean |Coefficient| per Feature",
                 color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Mean |Coefficient|", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.2, color="#30363d", axis="x")
    plt.tight_layout()
    plt.savefig(XAI_FIG_DIR / "feature_importance_logistic_regression.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Logistic Regression importance saved.")


# ══════════════════════════════════════════════════════════════════════════════
# PART F — Cross-Model Feature Importance Comparison
# ══════════════════════════════════════════════════════════════════════════════
if shap_results:
    print("\n[F] Cross-Model Feature Importance Comparison...")

    # Build comparison DataFrame
    imp_df = pd.DataFrame(shap_results, index=FEATURES)
    # Normalize each column to [0, 1]
    imp_df_norm = imp_df.apply(lambda col: col / (col.max() + 1e-8))

    # Save CSV
    imp_csv = TABLE_DIR / "shap_feature_importance.csv"
    imp_df_norm.to_csv(imp_csv)
    print(f"  Saved: {imp_csv}")

    # Save JSON
    imp_json = TABLE_DIR / "shap_feature_importance.json"
    with open(imp_json, "w") as f:
        json.dump({
            "feature_importance_by_model": {
                model: dict(zip(FEATURES, imp_df_norm[model].tolist()))
                for model in imp_df_norm.columns
            },
            "average_importance": dict(zip(
                FEATURES,
                imp_df_norm.mean(axis=1).tolist()
            ))
        }, f, indent=2)
    print(f"  Saved: {imp_json}")

    # Grouped bar chart
    models_present = list(imp_df_norm.columns)
    x = np.arange(len(FEATURES))
    width = 0.8 / len(models_present)
    model_colors = {
        "XGBoost": "#3498db",
        "Random Forest": "#9b59b6",
        "Decision Tree": "#2ecc71",
        "Logistic Regression": "#e67e22",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for i, model in enumerate(models_present):
        color = model_colors.get(model, "#95a5a6")
        vals = imp_df_norm[model].values
        ax.bar(x + i * width - (len(models_present) - 1) * width / 2,
               vals, width, label=model, color=color,
               edgecolor="#30363d", linewidth=0.4, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, color="#c9d1d9", rotation=15, ha="right")
    ax.set_ylabel("Normalized Feature Importance", color="#c9d1d9")
    ax.set_title("Feature Importance Comparison Across All Models",
                 color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax.grid(True, alpha=0.2, color="#30363d", axis="y")
    plt.tight_layout()
    plt.savefig(XAI_FIG_DIR / "feature_importance_comparison_all_models.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Cross-model comparison chart saved.")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    im = ax.imshow(imp_df_norm.T.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, color="#c9d1d9", rotation=30, ha="right")
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels(models_present, color="#c9d1d9")
    ax.set_title("Feature Importance Heatmap (Normalized)",
                 color="#f0f6fc", fontsize=12, fontweight="bold")
    for i in range(len(models_present)):
        for j in range(len(FEATURES)):
            val = imp_df_norm.T.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if val > 0.5 else "#c9d1d9", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#c9d1d9")
    plt.tight_layout()
    plt.savefig(XAI_FIG_DIR / "feature_importance_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Feature importance heatmap saved.")


# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("XAI MODULE COMPLETE — Step 9")
print("=" * 70)
print(f"\nOutputs saved to: {XAI_FIG_DIR}")
out_files = list(XAI_FIG_DIR.glob("*.png")) + list(XAI_FIG_DIR.glob("*.npy"))
for f in sorted(out_files):
    print(f"  {f.name}")
print(f"\nTables saved to: {TABLE_DIR}")
for fname in ["shap_feature_importance.csv", "shap_feature_importance.json"]:
    p = TABLE_DIR / fname
    if p.exists():
        print(f"  {p.name}")
