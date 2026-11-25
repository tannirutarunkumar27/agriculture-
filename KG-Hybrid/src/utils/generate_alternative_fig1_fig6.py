"""Generate alternative evidence-bound visualizations for manuscript Figs. 1 and 6.

Uses the same reconciled 330-record saved-probability evidence. No training,
probability modification, interpolation, or synthetic observations.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from src.experiments.publication_evidence import ROOT, load_evidence


OUTPUT = ROOT.parent / "manuscript" / "alternative_figures_fig1_fig6_v2"
MODELS = [
    ("KG-Hybrid (Proposed)", "KG-Hybrid (noncompact)", "#0072B2", "o"),
    ("XGBoost Baseline", "XGBoost (saved)", "#D55E00", "s"),
    ("FT-Transformer Baseline", "FT-Transformer", "#009E73", "^"),
    ("Random Forest", "Random Forest (unrestricted)", "#CC79A7", "D"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, stem):
    fig.tight_layout()
    files = []
    for ext in ("png", "pdf", "svg"):
        path = OUTPUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=400 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        files.append(path)
    plt.close(fig)
    return files


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    labels, results = load_evidence()
    if len(labels) != 330:
        raise ValueError("Expected the designated 330-record partition")
    by_name = {r["model_name"]: r for r in results}
    assets = []

    # Fig. 1 alternative A: small multiples prevent crossing-line and legend clutter.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), sharex=True, sharey=True)
    gap_rows = []
    for ax, (key, display, color, marker) in zip(axes.flat, MODELS):
        result = by_name[key]
        populated = [b for b in result["bin_data"] if b["sample_count"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1)
        sizes = [22 + 2.4 * np.sqrt(b["sample_count"]) for b in populated]
        ax.scatter([b["confidence"] for b in populated], [b["accuracy"] for b in populated],
                   s=sizes, color=color, marker=marker, edgecolor="white", linewidth=.5, zorder=3)
        for b in populated:
            gap_rows.append({"model": display, **b, "calibration_gap": b["accuracy"] - b["confidence"]})
            if b["sample_count"] >= 20:
                x_offset = -24 if b["bin_index"] == 8 else 5
                ax.annotate(f"n={b['sample_count']}", (b["confidence"], b["accuracy"]),
                            xytext=(x_offset, -12), textcoords="offset points", fontsize=7, color="0.25")
        ax.set_title(f"{display}\nECE = {result['ece']:.6f}")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(alpha=.18)
    fig.supxlabel("Mean predicted confidence"); fig.supylabel("Empirical accuracy")
    fig.suptitle("Saved-checkpoint reliability by model (bubble size reflects bin count)", y=1.02, fontsize=11)
    assets += save(fig, "fig1_alt_small_multiples_reliability")

    # Fig. 1 alternative B: signed calibration gaps expose direction and sparse bins.
    gaps = pd.DataFrame(gap_rows)
    matrix = np.full((len(MODELS), 10), np.nan); counts = np.zeros((len(MODELS), 10), int)
    for i, (_, display, _, _) in enumerate(MODELS):
        rows = gaps[gaps.model == display]
        for row in rows.itertuples():
            matrix[i, int(row.bin_index)] = row.calibration_gap
            counts[i, int(row.bin_index)] = int(row.sample_count)
    limit = max(.05, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    image = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if counts[i, j]:
                ax.text(j, i, f"{matrix[i,j]:+.2f}\nn={counts[i,j]}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(matrix[i, j]) > .55 * limit else "black")
            else:
                ax.text(j, i, "-", ha="center", va="center", color="0.55")
    ax.set_xticks(range(10), [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)], rotation=45, ha="right")
    ax.set_yticks(range(4), [m[1] for m in MODELS]); ax.set_xlabel("Confidence bin")
    ax.set_title("Signed calibration gap by confidence bin (accuracy - confidence)")
    cb = fig.colorbar(image, ax=ax, fraction=.03, pad=.02); cb.set_label("Calibration gap")
    assets += save(fig, "fig1_alt_calibration_gap_heatmap")

    # Shared exact data for Fig. 6 alternatives.
    kg = by_name["KG-Hybrid (Proposed)"]["probas"]
    confidence = kg.max(axis=1); prediction = kg.argmax(axis=1); error = prediction != labels
    if int(error.sum()) != 1:
        raise ValueError("Expected exactly one error in the corrected KG-Hybrid evidence")
    error_confidence = float(confidence[error][0])
    order = np.argsort(-confidence, kind="stable")
    ranked_confidence = confidence[order]; ranked_error = error[order]
    error_rank = int(np.flatnonzero(ranked_error)[0]) + 1

    # Fig. 6 alternative A: confidence distribution plus ranked error location.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [1.2, 1]}, sharex=False)
    axes[0].hist(confidence[~error], bins=np.linspace(.5, 1, 21), color="#56B4E9", alpha=.82,
                 edgecolor="white", label=f"Correct (n={int((~error).sum())})")
    axes[0].axvline(error_confidence, color="#D55E00", lw=2, label=f"Single error, confidence={error_confidence:.3f}")
    axes[0].set_ylabel("Records"); axes[0].set_xlabel("Maximum predicted probability")
    axes[0].set_title("Confidence distribution and the single misclassification")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=.18)
    ranks = np.arange(1, len(labels) + 1)
    axes[1].plot(ranks, ranked_confidence, color="#0072B2", lw=1.4)
    axes[1].scatter([error_rank], [ranked_confidence[error_rank-1]], color="#D55E00", s=42, zorder=3)
    axes[1].annotate(f"error enters at rank {error_rank}\ncoverage={error_rank/len(labels)*100:.1f}%",
                     (error_rank, ranked_confidence[error_rank-1]), xytext=(-110, 25), textcoords="offset points",
                     arrowprops={"arrowstyle": "->", "color": "0.35"}, fontsize=8)
    axes[1].set(xlabel="Confidence rank (most to least confident)", ylabel="Confidence", xlim=(1, 330), ylim=(.48, .99))
    axes[1].grid(alpha=.18)
    assets += save(fig, "fig6_alt_confidence_error_distribution")

    # Fig. 6 alternative B: an operating table/curve over practical thresholds.
    thresholds = np.array([.50, .55, .60, .65, .70, .75, .80, .85, .90, .925, .95])
    retained = np.array([(confidence >= t).sum() for t in thresholds])
    errors = np.array([error[confidence >= t].sum() for t in thresholds])
    coverage = retained / len(labels) * 100
    risk = np.divide(errors, retained, out=np.zeros_like(errors, dtype=float), where=retained > 0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    axes[0].step(thresholds, coverage, where="post", color="#0072B2", lw=2)
    axes[0].scatter(thresholds, coverage, color="#0072B2", s=22)
    axes[0].set(xlabel="Confidence threshold", ylabel="Coverage (%)", ylim=(0, 105))
    axes[0].grid(alpha=.18); axes[0].set_title("Coverage retained")
    axes[1].step(thresholds, errors, where="post", color="#D55E00", lw=2, label="Errors retained")
    axes[1].scatter(thresholds, errors, color="#D55E00", s=22)
    chosen = {0: (-5, 12), 2: (8, 12), 8: (-18, 12), 10: (10, 12)}
    for index, (x, e, n, r) in enumerate(zip(thresholds, errors, retained, risk)):
        if index in chosen:
            axes[1].annotate(f"{int(e)}/{int(n)}\n({r:.3f}%)", (x, e), xytext=chosen[index],
                             textcoords="offset points", ha="center", fontsize=6.5)
    axes[1].set(xlabel="Confidence threshold", ylabel="Retained errors", ylim=(-.08, 1.35), yticks=[0, 1])
    axes[1].grid(alpha=.18); axes[1].set_title("Observed selective errors")
    fig.suptitle("Threshold operating characteristics for direct KG-Hybrid predictions", y=1.03, fontsize=11)
    assets += save(fig, "fig6_alt_threshold_operating_curve")

    gaps.to_csv(OUTPUT / "fig1_alt_calibration_gap_values.csv", index=False)
    pd.DataFrame({"threshold": thresholds, "retained": retained, "coverage_percent": coverage,
                  "errors": errors, "risk_percent": risk}).to_csv(OUTPUT / "fig6_alt_threshold_values.csv", index=False)
    manifest = {
        "status": "GENERATED_FROM_RECONCILED_SAVED_EVIDENCE", "n": 330,
        "no_retraining": True, "probabilities_modified": False,
        "figure1_ece": {m[1]: by_name[m[0]]["ece"] for m in MODELS},
        "figure6": {"errors": 1, "error_confidence": error_confidence, "error_rank": error_rank,
                    "full_coverage_risk_percent": 100 / 330},
        "assets": {p.name: sha(p) for p in assets},
        "source_probability_hashes": {m[1]: by_name[m[0]]["sha256"] for m in MODELS},
        "visual_review": "pending"
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "figures": 4, "errors": 1, "error_rank": error_rank}, indent=2))


if __name__ == "__main__":
    main()
