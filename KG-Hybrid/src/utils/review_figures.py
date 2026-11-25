"""Shared empirical replacements for legacy illustration generators."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

from src.experiments.publication_evidence import ROOT, SOURCES, load_evidence, rebuild_routing_evidence, save_figure, copy_artifact
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
from src.utils.validate_final_experiments import validate_ablation


def confusion_plots():
    from sklearn.metrics import confusion_matrix
    labels, results = load_evidence()
    _, crops, _ = load_authoritative_knowledge_matrix()
    filenames = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
                 "Random Forest": "random_forest", "XGBoost Baseline": "xgboost",
                 "FT-Transformer Baseline": "ft_transformer", "KG-Hybrid (Proposed)": "kg_hybrid"}
    for result in results:
        matrix = confusion_matrix(labels, result["probas"].argmax(1), labels=np.arange(22))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=crops, yticklabels=crops, ax=ax)
        ax.set(xlabel="Predicted crop", ylabel="True crop", title=result["model_name"])
        fig.tight_layout()
        save_figure(fig, FIGURES / f"{filenames[result['model_name']]}_confusion_matrix.png", dpi=300)
        plt.close(fig)


def comparison_plot():
    _, results = load_evidence()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [r["model_name"] for r in results]
    axes[0].barh(names, [r["accuracy"] for r in results])
    axes[0].set_xlabel("Direct-model test accuracy")
    axes[1].barh(names, [r["ece"] for r in results])
    axes[1].set_xlabel("ECE from the same saved predictions")
    fig.tight_layout()
    save_figure(fig, FIGURES / "comprehensive_comparison.png", dpi=300)
    plt.close(fig)

FIGURES = ROOT / "outputs/figures"
TABLES = ROOT / "outputs/tables"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def ablation_plot():
    _, summary, evidence = validate_ablation()
    if evidence["validated_metric_rows"] != 45:
        raise ValueError("Complete ablation predictions required")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.variant, summary.macro_f1_mean * 100, yerr=summary.macro_f1_std * 100, capsize=3)
    ax.set(ylabel="Macro-F1 (%)", title="Saved five-seed ablation: mean and sample SD")
    save_figure(fig, FIGURES / "fig2_ablation.png", dpi=300)
    plt.close(fig)


def missing_plot():
    recorded = json.loads((ROOT / "outputs/robustness/full_robustness_report.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in recorded["missing_features"].items():
        ratios = [float(key.rstrip("%")) for key in values]
        ax.plot(ratios, np.asarray(list(values.values())) * 100, "o-", label=name)
    ax.set(xlabel="Missing feature fraction (%)", ylabel="Recorded direct-model accuracy (%)")
    ax.legend()
    save_figure(fig, FIGURES / "fig3_missing_features.png", dpi=300)
    plt.close(fig)


def routing_plot():
    frame = rebuild_routing_evidence()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(frame.threshold, frame.xgb_exit_fraction * 100, "o-")
    axes[0].set(xlabel="XGBoost confidence threshold", ylabel="XGBoost exit coverage (%)")
    for key in ("cascade_accuracy", "direct_xgb_accuracy", "direct_hybrid_accuracy"):
        axes[1].plot(frame.threshold, frame[key] * 100, label=key)
    axes[1].set(xlabel="XGBoost confidence threshold", ylabel="Accuracy (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Deterministic saved-model routes; MC Dropout and latency excluded")
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig4_early_exit.png", dpi=300)
    save_figure(fig, FIGURES / "realworld_enhancements.png", dpi=300)
    plt.close(fig)


def robustness_plot():
    recorded = json.loads((ROOT / "outputs/robustness/full_robustness_report.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in recorded["noise_injection"].items():
        sigma = [float(key.split("=")[1]) for key in values]
        ax.plot(sigma, np.asarray(list(values.values())) * 100, "o-", label=name)
    ax.set(xlabel="Recorded standardized Gaussian noise SD", ylabel="Direct-model accuracy (%)")
    ax.legend()
    save_figure(fig, FIGURES / "fig5_robustness.png", dpi=300)
    plt.close(fig)


def distribution_plot():
    from src.experiments.external_validation import load_external_dataset_agrifield, load_internal_data
    train, _, scaler, _ = load_internal_data()
    external, _ = load_external_dataset_agrifield(align_units=False)
    external = pd.DataFrame(scaler.transform(external), columns=FEATURES)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, name in zip(axes.flat, FEATURES):
        sns.kdeplot(train[name], ax=ax, label="Internal training")
        sns.kdeplot(external[name], ax=ax, label="Observed external data")
        ax.set_title(name)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[7:]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGURES / "external_distribution_shift.png", dpi=300)
    plt.close(fig)


def parameter_plot():
    recorded = json.loads((TABLES / "inference_params.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in recorded.items():
        count = str(values["params"]).replace(",", "")
        if not count.isdigit():
            continue  # Tree/node counts are not neural trainable parameters.
        ax.scatter(int(count) / 1e6, values["inference_ms"])
        ax.annotate(name, (int(count) / 1e6, values["inference_ms"]), fontsize=8)
    ax.set(xlabel="Recorded model parameter count (millions)", ylabel="Recorded batch inference latency (ms)")
    save_figure(fig, FIGURES / "fig8_param_latency_tradeoff.png", dpi=300)
    plt.close(fig)


def history_plot():
    from src.utils.generate_paper_tables_and_plots import generate_fig_f3_learning_curves
    generate_fig_f3_learning_curves()
    for name in ("kg_hybrid_training_history.png", "fig9_training_history.png"):
        copy_artifact(FIGURES / "fig_f3_learning_curves.png", FIGURES / name)


def comparative_history_plot():
    files = {"KG-Hybrid": TABLES / "kg_hybrid_training_history.json",
             "FT-Transformer": TABLES / "ft_transformer_training_history.json"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, path) in zip(axes, files.items()):
        ax.set_title(name)
        if not path.exists():
            ax.text(.5, .5, "Training trace not retained", ha="center", transform=ax.transAxes)
            continue
        history = json.loads(path.read_text())
        for key in ("train_acc", "val_acc"):
            if key in history:
                ax.plot(np.arange(len(history[key])) + 1, history[key], label=key)
        ax.set(xlabel="Epoch", ylabel="Recorded accuracy")
        ax.legend()
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_training_history_all_models.png", dpi=300)
    plt.close(fig)


def knowledge_plot():
    matrix, crops, features = load_authoritative_knowledge_matrix()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, ax=ax, xticklabels=features, yticklabels=crops, vmin=0, vmax=1)
    ax.set(title="Active manually specified prior weights (source derivation unverified)")
    fig.tight_layout()
    save_figure(fig, FIGURES / "knowledge_feature_gate.png", dpi=300)
    plt.close(fig)


def xai_figures():
    """Native TreeSHAP on the saved booster; signed explanations, no simulation."""
    xai = FIGURES / "xai"
    xai.mkdir(exist_ok=True)
    model_path = ROOT / "outputs/models/xgboost_model.pkl"
    model = joblib.load(model_path)
    inputs = pd.read_csv(ROOT / "data/processed/test.csv")[FEATURES]
    booster = model.get_booster()
    data = xgb.DMatrix(inputs)
    contributions = booster.predict(data, pred_contribs=True)
    margins = booster.predict(data, output_margin=True)
    if contributions.ndim != 3 or contributions.shape != (len(inputs), 22, 8):
        raise ValueError("Expected TreeSHAP tensor (samples, classes, features+base)")
    np.testing.assert_allclose(contributions.sum(axis=2), margins, atol=2e-5, rtol=2e-5)
    selected = margins.argmax(1)
    signed = contributions[np.arange(len(inputs)), selected, :-1]
    base = contributions[np.arange(len(inputs)), selected, -1]
    absolute_mean = np.abs(contributions[:, :, :-1]).mean(axis=1)
    np.save(xai / "shap_values_xgboost.npy", absolute_mean)
    np.save(xai / "shap_signed_predicted_class.npy", signed)
    np.save(xai / "shap_base_predicted_class.npy", base)
    np.save(xai / "shap_full_class_contributions.npy", contributions)
    importance = absolute_mean.mean(0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(FEATURES)[order], importance[order])
    ax.set_xlabel("Mean absolute TreeSHAP contribution over samples and classes")
    save_figure(fig, xai / "shap_summary_xgboost.png", dpi=300)
    save_figure(fig, FIGURES / "fig6_shap_importance.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)
    for feature in range(7):
        points = ax.scatter(signed[:, feature], feature + rng.uniform(-.15, .15, len(inputs)),
                            c=inputs.iloc[:, feature], cmap="coolwarm", s=9, alpha=.6)
    ax.set(yticks=np.arange(7), yticklabels=FEATURES, xlabel="Signed TreeSHAP for each sample's predicted class")
    save_figure(fig, xai / "shap_beeswarm_xgboost.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    starts = base[0] + np.r_[0, np.cumsum(signed[0])[:-1]]
    ax.barh(FEATURES, signed[0], left=starts, color=np.where(signed[0] >= 0, "tab:red", "tab:blue"))
    ax.axvline(base[0], color="gray", linestyle="--", label=f"Base margin {base[0]:.4f}")
    ax.axvline(margins[0, selected[0]], color="black", label=f"Predicted-class margin {margins[0, selected[0]]:.4f}")
    ax.set_xlabel("Additive margin contributions: saved test row 0")
    ax.legend(fontsize=8)
    save_figure(fig, xai / "shap_waterfall_xgboost.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("weight", "gain", "cover")):
        recorded = booster.get_score(importance_type=metric)
        values = [recorded.get(f, recorded.get(f"f{i}", 0)) for i, f in enumerate(FEATURES)]
        ax.barh(FEATURES, values)
        ax.set_title(f"Recorded XGBoost {metric}")
    fig.tight_layout()
    save_figure(fig, xai / "xgboost_feature_importance_3panel.png", dpi=300)
    plt.close(fig)
    # Cross-model importance is measured from refitted, explicitly named baselines.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    values = {"Saved XGBoost": model.feature_importances_}
    for name, estimator in (("Refitted Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)),
                            ("Refitted Decision Tree", DecisionTreeClassifier(random_state=42)),
                            ("Refitted Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))):
        estimator.fit(train[FEATURES], train.label)
        values[name] = estimator.feature_importances_ if hasattr(estimator, "feature_importances_") else np.abs(estimator.coef_).mean(0)
    frame = pd.DataFrame(values, index=FEATURES)
    frame = frame / frame.sum(axis=0)
    frame.to_csv(TABLES / "measured_baseline_feature_importance.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot.bar(ax=ax)
    ax.set_ylabel("Normalized measured importance (different estimators use different definitions)")
    fig.tight_layout()
    save_figure(fig, xai / "feature_importance_comparison_all_models.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(frame.T, annot=True, ax=ax)
    fig.tight_layout()
    save_figure(fig, xai / "feature_importance_heatmap.png", dpi=300)
    plt.close(fig)
