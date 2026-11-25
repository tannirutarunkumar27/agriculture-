"""Publication tables and figures from saved evidence; missing evidence fails."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import rebuild_calibration, rebuild_routing_evidence, load_evidence, save_figure
from src.utils.validate_final_experiments import validate_ablation
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix

TABLES_DIR = PROJECT_ROOT / "outputs/tables"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def generate_table1():
    frame = pd.read_csv(PROJECT_ROOT / "outputs/cross_validation_neural_recovery/aggregate/cv_summary.csv")
    result = frame[["Model", "accuracy_formatted", "f1_macro_formatted"]].copy()
    result["Evaluation"] = "5-fold CV mean +/- sample SD; not independent-seed confidence intervals"
    result.to_csv(TABLES_DIR / "table1_master_results_ci.csv", index=False)
    return result


def generate_table2():
    return rebuild_calibration()[0]


def generate_table3():
    frame = pd.read_csv(PROJECT_ROOT / "outputs/final_experimental_validation/tables/table3_cv_per_class_metrics.csv")
    result = frame[frame.Model == "KG-HybridProposed"].copy()
    if len(result) != 22:
        raise ValueError("Complete hybrid OOF per-class metrics required")
    result.to_csv(TABLES_DIR / "table3_per_class_metrics.csv", index=False)
    return result


def generate_table4():
    per_seed, summary, evidence = validate_ablation()
    if evidence["validated_metric_rows"] != 45 or evidence["metric_mismatches"]:
        raise ValueError("Ablation predictions do not match recorded metrics")
    full = per_seed[per_seed.variant == "Full"].set_index("seed")["macro_f1_recomputed"]
    rows = []
    for variant in summary.variant:
        values = per_seed[per_seed.variant == variant].set_index("seed")["macro_f1_recomputed"]
        difference = values.loc[full.index] - full
        pvalue = float(stats.wilcoxon(difference).pvalue) if (difference != 0).any() else 1.0
        rows.append({"variant": variant, "n_seeds": len(values), "macro_f1_mean": values.mean(),
                     "macro_f1_sd": values.std(ddof=1), "delta_macro_f1_vs_full": difference.mean(),
                     "wilcoxon_pvalue": pvalue, "significant_unadjusted_p_lt_0_05": pvalue < .05})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "table4_ablation_statistical_significance.csv", index=False)
    return result


def generate_table5():
    matrix, crops, features = load_authoritative_knowledge_matrix()
    result = pd.DataFrame(matrix, index=crops, columns=features)
    result.index.name = "crop"
    result["provenance_status"] = "Manual weights; institutional derivation unverified"
    result["definition"] = "Explicit crop-feature weights; no range-to-weight transformation implemented"
    result.to_csv(TABLES_DIR / "table5_knowledge_matrix_full.csv")
    return result


def generate_table6():
    recorded = json.loads((TABLES_DIR / "inference_params.json").read_text(encoding="utf-8"))
    result = pd.DataFrame([{"Model": model, **values,
        "Timing_Context": "Recorded direct-model batch inference; not cascade latency",
        "Edge_Latency": "Not measured", "Peak_RAM": "Not measured"}
        for model, values in recorded.items()])
    result.to_csv(TABLES_DIR / "table6_computational_cost.csv", index=False)
    return result


def generate_table7():
    per_seed, _, evidence = validate_ablation()
    result = per_seed[["seed", "variant", "accuracy_recomputed", "macro_f1_recomputed",
                       "accuracy_matches_record", "macro_f1_matches_record"]].copy()
    result["sample_identity_limitation"] = evidence["limitation"]
    result.to_csv(TABLES_DIR / "table7_numerical_consistency.csv", index=False)
    return result


def generate_fig_f1_reliability_diagram():
    rebuild_calibration()


def generate_fig_f2_confusion_matrix():
    labels, results = load_evidence()
    kg = next(r for r in results if r["model_name"] == "KG-Hybrid (Proposed)")
    _, crops, _ = load_authoritative_knowledge_matrix()
    cm = confusion_matrix(labels, kg["probas"].argmax(1), labels=np.arange(22))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap="YlGnBu", xticklabels=crops, yticklabels=crops, ax=ax)
    ax.set(xlabel="Predicted crop", ylabel="True crop", title="Corrected KG-Hybrid test predictions (N=330)")
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig_f2_confusion_matrix_annotated.png", dpi=300)
    plt.close(fig)


def generate_fig_f3_learning_curves():
    history = json.loads((TABLES_DIR / "kg_hybrid_training_history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, keys in zip(axes, (("train_loss", "val_loss"), ("train_acc", "val_acc"))):
        for key in keys:
            values = np.asarray(history[key], dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Missing/nonfinite recorded training series: {key}")
            ax.plot(np.arange(1, len(values) + 1), values, label=key)
        ax.set(xlabel="Epoch", ylabel="Recorded loss" if "loss" in keys[0] else "Accuracy")
        ax.legend()
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig_f3_learning_curves.png", dpi=300)
    plt.close(fig)


def generate_fig_f4_risk_coverage():
    rebuild_routing_evidence()


def generate_fig_f5_feature_importance_3way():
    import torch
    import joblib
    from src.models.ft_transformer.model import FTTransformer
    model = FTTransformer()
    checkpoint = torch.load(PROJECT_ROOT / "outputs/models/best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    features = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv")[FEATURES].to_numpy(dtype=np.float32)
    inputs = torch.tensor(features, requires_grad=True)
    logits = model(inputs)
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    sensitivity = inputs.grad.abs().mean(0).numpy()
    booster = joblib.load(PROJECT_ROOT / "outputs/models/xgboost_model.pkl")
    shap_values = np.load(FIGURES_DIR / "xai/shap_values_xgboost.npy")
    if shap_values.ndim < 2 or shap_values.shape[1] != len(FEATURES):
        raise ValueError("Expected saved SHAP layout (samples, features[, classes])")
    shap_importance = np.abs(shap_values).mean(axis=tuple(i for i in range(shap_values.ndim) if i != 1))
    series = {"XGBoost gain": booster.feature_importances_, "FT input gradients": sensitivity,
              "Saved XGBoost SHAP": shap_importance}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    evidence = {}
    for i, (name, values) in enumerate(series.items()):
        values = np.asarray(values)
        values = values / values.sum() if values.sum() else values
        evidence[name] = values
        ax.bar(np.arange(7) + (i - 1) * .25, values, width=.25, label=name)
    pd.DataFrame(evidence, index=FEATURES).to_csv(TABLES_DIR / "feature_importance_3way_measured.csv")
    ax.set(xticks=np.arange(7), xticklabels=FEATURES, ylabel="Normalized measured importance")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig_f5_feature_importance_3way.png", dpi=300)
    plt.close(fig)


def generate_fig_f6_architecture_shapes():
    return FIGURES_DIR / "fig1_kg_hybrid_detailed_architecture.png"


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for generator in (generate_table1, generate_table2, generate_table3, generate_table4,
                      generate_table5, generate_table6, generate_table7,
                      generate_fig_f1_reliability_diagram, generate_fig_f2_confusion_matrix,
                      generate_fig_f3_learning_curves, generate_fig_f4_risk_coverage,
                      generate_fig_f5_feature_importance_3way):
        generator()
        print(f"Generated from evidence: {generator.__name__}")
