"""Generate the three reviewed manuscript figures from validated saved evidence.

No model training, synthetic observations, or changes to existing paper assets.
Run from KG-Hybrid: python -m src.utils.generate_updated_manuscript_figures
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.experiments.cv_evidence import validate_cv_directory
from src.experiments.publication_evidence import ROOT, load_evidence, save_figure


OUTPUT = ROOT.parent / "manuscript" / "updated_figures"


def source_record(path):
    return {
        "path": str(path.relative_to(ROOT.parent)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export_figure(fig, stem):
    fig.tight_layout()
    save_figure(fig, OUTPUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    save_figure(fig, OUTPUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    labels, results = load_evidence()
    if len(labels) != 330:
        raise ValueError("The saved-checkpoint figures require the designated 330 rows")
    by_name = {result["model_name"]: result for result in results}
    expected_table_path = ROOT / "outputs/final_experimental_validation/tables/table4_calibration_recalculated.csv"
    expected_table = pd.read_csv(expected_table_path).set_index("Model_File")
    selected = [
        ("KG-Hybrid (Proposed)", "KG-Hybrid (noncompact)", "#0072B2", "o"),
        ("XGBoost Baseline", "XGBoost (saved)", "#D55E00", "s"),
        ("FT-Transformer Baseline", "FT-Transformer", "#009E73", "^"),
        ("Random Forest", "Random Forest (unrestricted)", "#CC79A7", "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.2, label="Ideal calibration")
    bin_rows = []
    for key, display_name, color, marker in selected:
        result = by_name[key]
        reference = expected_table.loc[Path(result["source"]).name]
        for metric, column in (("ece", "ECE"), ("brier", "Brier_Score"), ("nll", "NLL")):
            np.testing.assert_allclose(result[metric], reference[column], atol=1e-6, rtol=0)
        populated = [entry for entry in result["bin_data"] if entry["sample_count"] > 0]
        if sum(entry["sample_count"] for entry in populated) != len(labels):
            raise ValueError("Reliability bins do not cover the entire test partition")
        ax.plot(
            [entry["confidence"] for entry in populated],
            [entry["accuracy"] for entry in populated],
            color=color, marker=marker, markersize=5, lw=1.3,
            label=f"{display_name}: ECE={result['ece']:.6f}",
        )
        bin_rows.extend({"model": display_name, **entry} for entry in result["bin_data"])
    ax.set(xlabel="Mean predicted confidence", ylabel="Empirical accuracy", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), borderaxespad=0,
              fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    export_figure(fig, "fig1_reliability_saved_checkpoint")
    pd.DataFrame(bin_rows).to_csv(OUTPUT / "fig1_reliability_bins.csv", index=False)

    cv_path = ROOT / "outputs/cross_validation_neural_recovery"
    matrix_path = ROOT / "outputs/final_experimental_validation/tables/confusion_matrices_cv_oof.json"
    matrix_evidence = json.loads(matrix_path.read_text(encoding="utf-8"))
    cv_validation = validate_cv_directory(cv_path, ROOT, matrix_evidence["models"])
    oof_path = cv_path / "oof_predictions/oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    kg_oof = oof[oof.model_id == "KG-HybridProposed"].sort_values("sample_id")
    crops = matrix_evidence["classes"]
    true_labels = kg_oof.true_label.to_numpy(dtype=int)
    predicted_labels = kg_oof.predicted_label.to_numpy(dtype=int)
    np.testing.assert_array_equal(kg_oof.true_crop, np.asarray(crops)[true_labels])
    np.testing.assert_array_equal(kg_oof.predicted_crop, np.asarray(crops)[predicted_labels])
    counts = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(crops)))
    np.testing.assert_array_equal(counts, matrix_evidence["confusion_matrices"]["KG-HybridProposed"])
    np.testing.assert_array_equal(counts.sum(axis=1), np.full(22, 100))
    normalized = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for row, column in zip(*np.nonzero(counts)):
        value = normalized[row, column]
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.55 else "black")
    ax.set(xticks=np.arange(22), yticks=np.arange(22), xticklabels=crops, yticklabels=crops,
           xlabel="Predicted crop", ylabel="True crop")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 22, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 22, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Fraction of true-class records")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    export_figure(fig, "fig2_confusion_oof_normalized")
    pd.DataFrame(counts, index=crops, columns=crops).rename_axis("true_crop").to_csv(OUTPUT / "fig2_confusion_counts.csv")
    pd.DataFrame(normalized, index=crops, columns=crops).rename_axis("true_crop").to_csv(OUTPUT / "fig2_confusion_normalized.csv")

    kg = by_name["KG-Hybrid (Proposed)"]["probas"]
    confidence = kg.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    ranked_confidence = confidence[order]
    ranked_errors = (kg.argmax(axis=1) != labels)[order]
    # Thresholding cannot split confidence ties. Retain only complete tied groups.
    end_of_group = np.flatnonzero(np.r_[ranked_confidence[:-1] != ranked_confidence[1:], True])
    retained = end_of_group + 1
    errors = np.cumsum(ranked_errors)[end_of_group]
    coverage = retained / len(labels)
    risk = errors / retained
    np.testing.assert_allclose(risk[-1], 1 - by_name["KG-Hybrid (Proposed)"]["accuracy"], atol=1e-12)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(coverage * 100, risk * 100, color="#0072B2", lw=1.8, label="Direct KG-Hybrid (noncompact)")
    ax.scatter([100], [risk[-1] * 100], color="#0072B2", s=30, zorder=3)
    ax.annotate(f"100% coverage: {risk[-1] * 100:.3f}% risk\n{int(errors[-1])} error / {len(labels)} records",
                xy=(100, risk[-1] * 100), xytext=(42, 0.22),
                arrowprops={"arrowstyle": "->", "color": "0.4"}, fontsize=9)
    ax.set(xlabel="Retained prediction coverage (%)", ylabel="Selective risk: error rate (%)",
           xlim=(0, 102), ylim=(-0.015, max(0.35, float(risk.max() * 100) * 1.15)))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    export_figure(fig, "fig3_risk_coverage_direct_hybrid")
    pd.DataFrame({"confidence_threshold": ranked_confidence[end_of_group], "retained_count": retained,
                  "error_count": errors, "coverage": coverage, "risk": risk}).to_csv(
        OUTPUT / "fig3_risk_coverage_points.csv", index=False)

    manifest = {
        "no_retraining": True, "existing_manuscript_and_images_modified": False,
        "figure1": {"n": len(labels), "bins": 10, "ece": {name: by_name[name]["ece"] for name, *_ in selected},
                    "sources": [{"model": name, "source": by_name[name]["source"], "sha256": by_name[name]["sha256"]} for name, *_ in selected]},
        "figure2": {"n": int(counts.sum()), "correct": int(counts.trace()), "errors": int(counts.sum() - counts.trace()),
                    "support_per_class": counts.sum(axis=1).tolist(), "class_order": crops,
                    "normalization": "row", "validation": cv_validation, "source": source_record(oof_path)},
        "figure3": {"n": len(labels), "errors": int(errors[-1]), "full_coverage_risk": float(risk[-1]),
                    "selection": "direct hybrid maximum class probability >= threshold; ties retained together",
                    "excludes": ["XGBoost early-exit cascade", "MC-Dropout", "latency"],
                    "probability_source_sha256": by_name["KG-Hybrid (Proposed)"]["sha256"]},
        "reference_sources": [source_record(expected_table_path), source_record(matrix_path),
                              source_record(ROOT / "data/processed/test.csv"),
                              source_record(ROOT / "data/raw/Crop_recommendation.csv")],
    }
    (OUTPUT / "figure_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "ece": manifest["figure1"]["ece"],
                      "oof_correct": manifest["figure2"]["correct"], "oof_errors": manifest["figure2"]["errors"],
                      "full_coverage_risk_percent": risk[-1] * 100}, indent=2))


if __name__ == "__main__":
    main()
