"""Rebuild calibration and routing outputs from one saved probability source.

No training, fitted probability calibration, or manually specified metric values.
The corrected checkpoint's results supersede the older manuscript's numbers.
"""
from pathlib import Path
import hashlib
import json
import shutil
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.calibration import (
    compute_ece, compute_brier_score, compute_nll,
    generate_reliability_diagram_data, _validate_inputs,
)

SOURCES = {
    "Logistic Regression": "logistic_regression_test_probas.npy",
    "Decision Tree": "decision_tree_unconstrained_test_probas.npy",
    "Random Forest": "random_forest_unconstrained_test_probas.npy",
    "XGBoost Baseline": "xgboost_baseline_saved_checkpoint_test_probas.npy",
    "FT-Transformer Baseline": "ft-transformer_baseline_test_probas.npy",
    "KG-Hybrid (Proposed)": "kg-hybrid_proposed_corrected_v2_test_probas.npy",
}


def copy_artifact(source, destination):
    """Copy a generated artifact atomically, including files previewed on Windows."""
    source, destination = Path(source), Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(source, temporary_path)
        for attempt in range(5):
            try:
                os.replace(temporary_path, destination)
                break
            except OSError:
                if attempt == 4:
                    # Some Windows sandbox/preview configurations deny a rename
                    # over an existing file even when both paths share a
                    # directory.  Preserve the completed destination through a
                    # normal overwrite as a portable fallback.  This is not
                    # atomic, so it is intentionally used only after atomic
                    # replacement has failed repeatedly.
                    shutil.copy2(temporary_path, destination)
                    break
                time.sleep(.2)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_figure(fig, destination, **kwargs):
    """Publish rendered figures atomically so concurrent readers see whole files."""
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        fig.savefig(temporary_path, **kwargs)
        for attempt in range(5):
            try:
                os.replace(temporary_path, destination)
                break
            except OSError:
                if attempt == 4:
                    # See copy_artifact: retain a Windows-compatible fallback
                    # for environments that prohibit replacing an existing
                    # previewed file with os.replace.
                    shutil.copy2(temporary_path, destination)
                    break
                time.sleep(.2)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_evidence():
    labels = pd.read_csv(ROOT / "data/processed/test.csv")["label"].to_numpy(dtype=int)
    directory = ROOT / "outputs/calibration_reconciled/probabilities"
    results = []
    for name, filename in SOURCES.items():
        path = directory / filename
        probabilities = np.load(path).astype(np.float64)
        _validate_inputs(labels, probabilities)
        if probabilities.shape != (len(labels), 22):
            raise ValueError(f"Unexpected probability dimensions: {path}")
        # Only correct floating-point row-sum roundoff after validating the input.
        probabilities = np.clip(probabilities, 0, 1)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        bins = generate_reliability_diagram_data(labels, probabilities)
        ece = compute_ece(labels, probabilities)
        bin_ece = sum(b["sample_count"] * abs(b["accuracy"] - b["confidence"])
                      for b in bins) / len(labels)
        if not np.isclose(ece, bin_ece, atol=1e-6):
            raise ValueError(f"Bin/table ECE disagreement for {name}")
        results.append({
            "model_name": name, "probas": probabilities, "bin_data": bins,
            "accuracy": float(np.mean(probabilities.argmax(1) == labels)),
            "ece": ece, "brier": compute_brier_score(labels, probabilities),
            "nll": compute_nll(labels, probabilities), "source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return labels, results


def rebuild_calibration(sync_manuscript=True):
    labels, results = load_evidence()
    tables = ROOT / "outputs/tables"
    figures = ROOT / "outputs/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{
        "Model": r["model_name"], "N Samples": len(labels),
        "Top-1 Accuracy": round(r["accuracy"], 4), "ECE": r["ece"],
        "Brier Score": r["brier"], "NLL": r["nll"],
    } for r in results])
    # Keep the historical ``*_reconciled`` filename synchronized with the
    # authoritative table.  Previous workflows allowed these two files to
    # diverge, which made it possible for a manuscript to cite a metric that
    # was not supported by the saved probability arrays.
    table.to_csv(tables / "table2_calibration_metrics.csv", index=False)
    table.to_csv(tables / "table2_calibration_metrics_reconciled.csv", index=False)
    bins = pd.DataFrame([{
        "Model": r["model_name"], "Bin": b["bin_index"],
        "Bin Lower": b["bin_lower"], "Bin Upper": b["bin_upper"],
        "Count": b["sample_count"], "Mean Confidence": b["confidence"],
        "Accuracy": b["accuracy"],
    } for r in results for b in r["bin_data"]])
    bins.to_csv(tables / "calibration_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
    for r in results:
        populated = [b for b in r["bin_data"] if b["sample_count"]]
        ax.plot([b["confidence"] for b in populated], [b["accuracy"] for b in populated],
                "o-", label=f"{r['model_name']} (ECE={r['ece']:.6f})")
    ax.set(xlabel="Mean predicted confidence", ylabel="Empirical accuracy",
           xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for filename in ("reliability_diagram.png", "reliability_diagram.pdf", "fig_f1_reliability_diagram.png"):
        save_figure(fig, figures / filename, dpi=300)
        if sync_manuscript and (ROOT.parent / "manuscript").exists():
            copy_artifact(figures / filename, ROOT.parent / "manuscript" / filename)
    plt.close(fig)
    manifest = {
        "partition": "Saved processed test partition, 330 samples; row order preserved",
        "ece_bins": 10, "metric_source": "saved corrected-v2 probability arrays",
        "sources": [{k: r[k] for k in ("model_name", "source", "sha256", "ece")} for r in results],
        "note": "Older manuscript metric values must be updated to this recomputed table; they are not enforced as targets.",
    }
    (tables / "calibration_evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return table, bins


def rebuild_routing_evidence():
    """Separate actual XGBoost exits from direct hybrid and cascade accuracy."""
    labels, results = load_evidence()
    by_name = {r["model_name"]: r["probas"] for r in results}
    xgb = by_name["XGBoost Baseline"]
    kg = by_name["KG-Hybrid (Proposed)"]
    rows = []
    for threshold in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0):
        exit_mask = xgb.max(1) >= threshold
        cascade = np.where(exit_mask, xgb.argmax(1), kg.argmax(1))
        rows.append({
            "evaluation_mode": "deterministic saved checkpoint probabilities; healthy inputs; no MC Dropout",
            "threshold": threshold, "xgb_exit_count": int(exit_mask.sum()),
            "hybrid_count": int((~exit_mask).sum()), "xgb_exit_fraction": float(exit_mask.mean()),
            "xgb_exit_accuracy": float(np.mean(xgb.argmax(1)[exit_mask] == labels[exit_mask])) if exit_mask.any() else np.nan,
            "hybrid_fallback_accuracy": float(np.mean(kg.argmax(1)[~exit_mask] == labels[~exit_mask])) if (~exit_mask).any() else np.nan,
            "cascade_accuracy": float(np.mean(cascade == labels)),
            "direct_xgb_accuracy": float(np.mean(xgb.argmax(1) == labels)),
            "direct_hybrid_accuracy": float(np.mean(kg.argmax(1) == labels)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "outputs/tables/early_exit_route_audit.csv", index=False)
    # A selective-confidence curve describes the direct hybrid, not the cascade.
    order = np.argsort(-kg.max(1), kind="stable")
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    accuracy = np.cumsum(kg.argmax(1)[order] == labels[order]) / np.arange(1, len(labels) + 1)
    pd.DataFrame({"coverage": coverage, "accuracy": accuracy, "risk": 1 - accuracy}).to_csv(
        ROOT / "outputs/tables/hybrid_risk_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coverage, 1 - accuracy)
    ax.set(xlabel="Direct hybrid prediction coverage", ylabel="Error rate", xlim=(0, 1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for filename in ("fig_f4_risk_coverage.png", "fig4_risk_coverage.png"):
        save_figure(fig, ROOT / "outputs/figures" / filename, dpi=300)
    plt.close(fig)
    return frame


def sync_publication_assets():
    """Synchronize an explicit set of regenerated assets; never edit paper text."""
    manuscript = ROOT.parent / "manuscript"
    if not manuscript.exists():
        return []
    filenames = [
        "reliability_diagram.png", "reliability_diagram.pdf", "fig_f1_reliability_diagram.png",
        "fig_f2_confusion_matrix_annotated.png", "fig_f3_learning_curves.png",
        "fig_f4_risk_coverage.png", "fig_f5_feature_importance_3way.png",
        "fig2_ablation.png", "fig3_missing_features.png", "fig4_early_exit.png",
        "fig5_robustness.png", "fig6_shap_importance.png", "fig8_param_latency_tradeoff.png",
        "realworld_enhancements.png", "knowledge_feature_gate.png",
        "knowledge_guided_fusion_architecture.png", "fig1_kg_hybrid_detailed_architecture.png",
        "fig4_training_inference_pipeline.png", "fig5_early_exit_flow_diagram.png",
        "fig6_feature_complementarity.png", "comprehensive_comparison.png",
        "kg_hybrid_training_history.png", "fig_training_history_all_models.png",
        "external_distribution_shift.png", "logistic_regression_confusion_matrix.png",
        "decision_tree_confusion_matrix.png", "random_forest_confusion_matrix.png",
        "xgboost_confusion_matrix.png", "ft_transformer_confusion_matrix.png",
        "kg_hybrid_confusion_matrix.png",
        "xai/shap_summary_xgboost.png", "xai/shap_beeswarm_xgboost.png",
        "xai/shap_waterfall_xgboost.png", "xai/xgboost_feature_importance_3panel.png",
        "xai/feature_importance_comparison_all_models.png", "xai/feature_importance_heatmap.png",
    ]
    records = []
    for filename in filenames:
        source = ROOT / "outputs/figures" / filename
        if not source.exists():
            raise FileNotFoundError(f"Generate the evidence asset before synchronizing: {source}")
        destination = manuscript / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_artifact(source, destination)
        records.append({"asset": filename, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (ROOT / "outputs/tables/publication_asset_manifest.json").write_text(
        json.dumps({"assets": records, "manuscript_text_and_pdf_modified": False}, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    print(rebuild_calibration()[0].to_string(index=False))
    print(rebuild_routing_evidence().to_string(index=False))
