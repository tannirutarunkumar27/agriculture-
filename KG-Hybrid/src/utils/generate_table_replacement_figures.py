"""Export table IX-XIII replacements from existing, configuration-specific evidence.

Run from the repository root with the project Python environment. No training,
source-data editing, model loading, or changes to the manuscript are performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
OUTPUT = WORKSPACE / "manuscript" / "table_replacement_figures"
SEEDS = (42, 123, 2024, 3407, 9999)
MODELS = ("Random Forest", "XGBoost", "FT-Transformer", "KG-Hybrid")
COLORS = ("#009E73", "#E69F00", "#0072B2", "#D55E00")
MARKERS = ("s", "^", "D", "o")
LINESTYLES = ("--", "-.", ":", "-")
STEMS = (
    "table_ix_registered_parameters",
    "table_x_feature_masking",
    "table_xi_imputation",
    "table_xii_gaussian_noise",
    "table_xiii_feature_scaling",
    "combined_robustness_three_panel",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_record(path):
    return {
        "path": str(path.relative_to(WORKSPACE)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export(fig, stem):
    fig.canvas.draw()
    # Check explicitly placed text against the canvas before exporting.
    renderer = fig.canvas.get_renderer()
    bbox = fig.bbox
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if bounds.width and bounds.height:
                if bounds.x0 < bbox.x0 - 3 or bounds.y0 < bbox.y0 - 3 or bounds.x1 > bbox.x1 + 3 or bounds.y1 > bbox.y1 + 3:
                    raise ValueError(f"Text outside canvas in {stem}: {text.get_text()!r}")
    for legend in fig.legends:
        legend_bounds = legend.get_window_extent(renderer)
        for ax in fig.axes:
            for label in (ax.xaxis.label, ax.yaxis.label):
                if label.get_visible() and label.get_text() and legend_bounds.overlaps(label.get_window_extent(renderer)):
                    raise ValueError(f"Legend overlaps an axis label in {stem}")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=600, facecolor="white")
    # A lightweight preview is only for visual QA, not publication.
    fig.savefig(OUTPUT / f"{stem}_preview.png", dpi=180, facecolor="white")
    plt.close(fig)


def accuracy_axes(ax):
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def model_lines(ax, x, values):
    for i, name in enumerate(MODELS):
        ax.plot(x, values[name], label=name, color=COLORS[i], marker=MARKERS[i],
                linestyle=LINESTYLES[i], linewidth=1.5, markersize=4.6)


def load_evidence():
    robustness_path = PROJECT / "outputs/robustness/full_robustness_report.json"
    robustness = read_json(robustness_path)
    mask_x = [0, 20, 40, 60]
    noise_x = [0.0, 0.1, 0.2, 0.3]
    masking = {m: [100 * robustness["missing_features"][m][f"{x}%"] for x in mask_x] for m in MODELS}
    noise = {m: [100 * robustness["noise_injection"][m][f"s={x:.1f}"] for x in noise_x] for m in MODELS}
    clean = {m: 100 * robustness["cross_dataset"]["standard_test"][m] for m in MODELS}
    scaling = {m: 100 * robustness["cross_dataset"]["extrapolation_1_5"][m] for m in MODELS}

    # Do not propagate the misleading legacy "cross_dataset" name into captions.
    for m in MODELS:
        np.testing.assert_allclose(masking[m][0], clean[m], rtol=0, atol=1e-10)
        np.testing.assert_allclose(noise[m][0], clean[m], rtol=0, atol=1e-10)

    imputation_path = PROJECT / "outputs/tables/realworld_imputation_comparison.csv"
    methods = {
        "Zero replacement": "Raw Zero Fill (Baseline)",
        "Median imputation": "Median Imputation",
        "KNN imputation (k=5)": "KNN Imputation (k=5)",
        "Iterative imputation": "MICE Imputation",
    }
    with imputation_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["Missing Rate"].rstrip("%")) for row in rows] != mask_x:
        raise ValueError("Unexpected masking levels in the imputation evidence")
    imputation = {display: [float(row[column].rstrip("%")) for row in rows]
                  for display, column in methods.items()}
    # Intentionally exclude the separate XGBoost column: Table XI is within-model.
    if any(values[0] != 97.88 for values in imputation.values()):
        raise ValueError("Imputation checkpoint no longer has the expected clean reference")

    variants = (
        ("FTTransformer", "FT-Transformer"),
        ("Hybrid_NoK", "Hybrid: no knowledge, no gate"),
        ("Hybrid_FixedK", "Hybrid: fixed knowledge, no gate"),
        ("Hybrid_AnchoredK", "Hybrid: anchored knowledge, no gate"),
        ("Hybrid_AnchoredK_Gate", "Full anchored and gated hybrid"),
        ("Hybrid_ShuffledK_Gate", "Hybrid: shuffled fixed knowledge and gate"),
    )
    sources = [source_record(robustness_path), source_record(imputation_path)]
    parameter_rows = []
    for variant, display in variants:
        counts = []
        for seed in SEEDS:
            path = PROJECT / "outputs/knowledge_comparison_review_fixes" / f"seed_{seed}" / f"{variant}_metrics.json"
            metric = read_json(path)
            if metric["seed"] != seed or metric["model"] != variant or metric["n_test"] != 330:
                raise ValueError(f"Unexpected model-seed metadata in {path}")
            counts.append(metric["trainable_parameters"])
            sources.append(source_record(path))
        parameter_rows.append({"configuration": display, "variant": variant,
                               "per_seed_counts": counts, "mean": float(np.mean(counts)),
                               "sample_sd": float(np.std(counts, ddof=1))})
    np.testing.assert_equal([round(row["mean"]) for row in parameter_rows],
                            [212054, 1666668, 1692012, 1692166, 1741638, 1741484])
    np.testing.assert_equal([round(row["sample_sd"]) for row in parameter_rows[1:]], [8537] * 5)

    for section in (masking, noise, imputation):
        if any(not np.isfinite(x) or not 0 <= x <= 100 for values in section.values() for x in values):
            raise ValueError("Invalid percentage in the recorded evidence")
    return {"mask_x_percent": mask_x, "noise_sd_standardized": noise_x,
            "masking_accuracy_percent": masking, "noise_accuracy_percent": noise,
            "clean_accuracy_percent": clean, "scaled_accuracy_percent": scaling,
            "imputation_accuracy_percent": imputation, "parameter_counts": parameter_rows,
            "sources": sources}


def plot_parameters(evidence):
    rows = evidence["parameter_counts"]
    fig, ax = plt.subplots(figsize=(7.16, 3.3))
    fig.subplots_adjust(left=0.36, right=0.985, bottom=0.22, top=0.86)
    y = np.arange(len(rows))
    means = np.array([row["mean"] for row in rows]) / 1e6
    sd = np.array([row["sample_sd"] for row in rows]) / 1e6
    colors = ["#77818A"] * len(rows)
    colors[4] = COLORS[3]
    ax.barh(y, means, xerr=sd, height=0.56, color=colors,
            error_kw={"ecolor": "0.18", "capsize": 3, "elinewidth": 0.9})
    displays = ["FT-Transformer", "Hybrid: no knowledge, no gate", "Hybrid: fixed prior, no gate",
                "Hybrid: anchored prior, no gate", "KG-Hybrid: anchored prior and gate",
                "Hybrid: shuffled fixed prior and gate"]
    ax.set_yticks(y, displays, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.6)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Registered neural parameters (millions)")
    ax.set_title("Compact-model parameter allocation", fontsize=10, pad=12)
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        value = f"{row['mean']:,.0f}"
        if i:
            value += f" $\\pm$ {row['sample_sd']:,.0f}"
        ax.text(means[i] + sd[i] + 0.045, i, value, va="center", fontsize=8.3)
    fig.text(0.5, 0.025, "Hybrid whiskers: sample SD across five seeds. XGBoost ensemble excluded.",
             ha="center", fontsize=8)
    export(fig, STEMS[0])


def plot_line(evidence, stem, kind):
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.30, top=0.82)
    if kind == "masking":
        x, values = evidence["mask_x_percent"], evidence["masking_accuracy_percent"]
        title, xlabel = "Random mean replacement", "Masking probability (%)"
        limits, ticks = (-2, 62), x
    else:
        x, values = evidence["noise_sd_standardized"], evidence["noise_accuracy_percent"]
        title, xlabel = "Additive Gaussian noise", "Noise SD in standardized coordinates"
        limits, ticks = (-0.01, 0.31), x
    model_lines(ax, x, values)
    accuracy_axes(ax)
    ax.set(xlabel=xlabel, xlim=limits, xticks=ticks)
    ax.set_title(title, fontsize=10, pad=9)
    fig.text(0.56, 0.90, "Earlier noncompact-model evaluation", ha="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.04),
               ncol=2, frameon=False, fontsize=8, columnspacing=1.0, handlelength=2.0)
    fig.text(0.5, 0.025, "Mean of five perturbation trials; SD unavailable.", ha="center", fontsize=7.5)
    export(fig, stem)


def plot_imputation(evidence):
    fig, ax = plt.subplots(figsize=(3.5, 3.8))
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.32, top=0.80)
    for i, (method, values) in enumerate(evidence["imputation_accuracy_percent"].items()):
        color = ("#626B73", "#E69F00", "#0072B2", "#009E73")[i]
        ax.plot(evidence["mask_x_percent"], values, color=color, marker=MARKERS[i],
                linestyle=LINESTYLES[i], linewidth=1.5, markersize=4.6, label=method)
    accuracy_axes(ax)
    ax.set(xlabel="Missing-entry probability (%)", xlim=(-2, 62), xticks=[0, 20, 40, 60])
    ax.set_title("Within-model missing-input recovery", fontsize=9.5, pad=9)
    fig.text(0.56, 0.88, "Separate earlier KG-Hybrid checkpoint", ha="center", fontsize=8)
    # One exact, source-derived operating point rather than cluttering every mark.
    ax.annotate("81.33%", xy=(40, evidence["imputation_accuracy_percent"]["KNN imputation (k=5)"][2]),
                xytext=(28, 94), fontsize=8, color="0.15",
                arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7})
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ["Zero replacement", "Median", "KNN (k=5)", "Iterative"],
               loc="lower center", bbox_to_anchor=(0.53, 0.075), ncol=2,
               frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.0)
    fig.text(0.5, 0.025, "Five-trial means; SD unavailable. Clean: 97.88%.",
             ha="center", fontsize=7.5)
    export(fig, STEMS[2])


def plot_scaling(evidence):
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.32, top=0.80)
    y = np.arange(len(MODELS))
    for offset, values, color, label, hatch in (
        (-0.18, evidence["clean_accuracy_percent"], "#B9C7D1", "Clean inputs", "//"),
        (0.18, evidence["scaled_accuracy_percent"], "#0072B2", r"Standardized scaling ($1.5\times$)", None),
    ):
        numbers = [values[model] for model in MODELS]
        ax.barh(y + offset, numbers, height=0.31, color=color, hatch=hatch,
                edgecolor="white", linewidth=0, label=label)
        for position, value in zip(y + offset, numbers):
            ax.text(value + 1.9, position, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, ["Random\nForest", "XGBoost", "FT-\nTransformer", "KG-Hybrid"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 121)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Synthetic feature scaling", fontsize=10, pad=9)
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.55, 0.88, "Earlier noncompact-model evaluation", ha="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.045),
               ncol=1, frameon=False, fontsize=8)
    fig.text(0.5, 0.022, "Single transformed partition; original labels retained.",
             ha="center", fontsize=7.5)
    export(fig, STEMS[4])


def plot_combined(evidence):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.28, top=0.76, wspace=0.25)
    model_lines(axes[0], evidence["mask_x_percent"], evidence["masking_accuracy_percent"])
    model_lines(axes[1], evidence["noise_sd_standardized"], evidence["noise_accuracy_percent"])
    axes[0].set(xlabel="Masking probability (%)", xlim=(-2, 62), xticks=[0, 20, 40, 60])
    axes[1].set(xlabel="Noise SD (standardized)", xlim=(-0.01, 0.31), xticks=[0, 0.1, 0.2, 0.3])
    for i, model in enumerate(MODELS):
        axes[2].bar(i - 0.18, evidence["clean_accuracy_percent"][model], width=0.34,
                    color=COLORS[i], alpha=0.25, hatch="//", edgecolor=COLORS[i], linewidth=0.5)
        axes[2].bar(i + 0.18, evidence["scaled_accuracy_percent"][model], width=0.34,
                    color=COLORS[i], linewidth=0)
    axes[2].set(xlabel="Classifier", xticks=range(4), xticklabels=["RF", "XGB", "FT", "KG"])
    for i, ax in enumerate(axes):
        accuracy_axes(ax)
        if i:
            ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        ax.set_title(("(a) Mean replacement", "(b) Gaussian noise", "(c) Feature scaling")[i], fontsize=9)
    fig.text(0.5, 0.98, "Earlier noncompact evaluations - not compact-model results", ha="center", va="top", fontsize=9)
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                      linewidth=1.4, label=model) for i, model in enumerate(MODELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4,
               frameon=False, fontsize=8, columnspacing=1.4, handlelength=2.0)
    fig.legend(handles=[Patch(facecolor="0.85", edgecolor="0.5", hatch="//", label="Panel (c): clean"),
                        Patch(facecolor="0.5", label=r"Panel (c): $1.5\times$ scaling")],
               loc="lower center", bbox_to_anchor=(0.5, 0.07), frameon=False, ncol=2, fontsize=8)
    fig.text(0.5, 0.018, "Masking/noise: mean of five trials, SD unavailable. Scaling: one deterministic partition.",
             ha="center", fontsize=7.6)
    export(fig, STEMS[5])


CAPTIONS = {
    STEMS[0]: "Registered neural parameter counts in the compact controlled comparison. Hybrid bars show means and whiskers show sample standard deviations across five seed-specific runs. The FT-Transformer count is constant across runs. Counts include the retained standalone FT classification head, which is bypassed by hybrid prediction; the separate XGBoost ensemble is excluded.",
    STEMS[1]: "Earlier cross-model zero-replacement experiment. Each standardized feature entry is independently replaced by zero at the indicated masking probability. Points are mean accuracies over five perturbation trials; trial-level standard deviations are unavailable. These are not results for the compact five-seed models.",
    STEMS[2]: "Earlier within-model KG-Hybrid imputation comparison using a separate checkpoint with clean accuracy 97.88\\%. Missing entries are represented as NaNs for recovery, and all methods share the same mask within each trial. Imputers are fitted on standardized training data. Points are means over five trials; trial-level standard deviations are unavailable. This comparison does not establish an architectural advantage over imputed baselines.",
    STEMS[3]: "Earlier cross-model Gaussian-perturbation experiment. Independent additive noise is applied in standardized feature coordinates. Points show mean accuracies over five trials; trial-level standard deviations are unavailable. These are not compact-model results or sensor-calibrated physical noise levels.",
    STEMS[4]: "Earlier synthetic feature-scaling experiment comparing clean inputs with standardized test vectors multiplied by $1.5\\times$. Values are accuracies on one deterministic evaluation partition with the original labels retained. This transformation is not external-dataset validation and does not guarantee out-of-range measurements.",
    STEMS[5]: "Earlier noncompact-model stress tests: (a) independent zero replacement of standardized feature entries, (b) independent additive Gaussian noise in standardized coordinates, and (c) clean inputs versus $1.5\\times$ standardized feature scaling. Panels (a) and (b) show means over five perturbation trials, without available trial-level standard deviations; panel (c) uses one deterministic transformed partition. Original benchmark labels are retained. These results do not describe the compact five-seed models. The separate within-model imputation comparison is not included.",
}


def write_support(evidence):
    (OUTPUT / "plot_source_data.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    blocks = ["% Add \\usepackage{graphicx} to your preamble if not already present.",
              "% Place the table_replacement_figures directory beside your main .tex file.",
              "% Use either the individual robustness figures OR the combined figure, not both.",
              "% Figure numbers are assigned by LaTeX; update old table cross-references.", ""]
    labels = ("fig:registered_parameters", "fig:masking_replacement", "fig:imputation_recovery",
              "fig:gaussian_perturbations", "fig:feature_scaling", "fig:combined_robustness")
    for i, stem in enumerate(STEMS):
        wide = i in (0, 5)
        environment = "figure*" if wide else "figure"
        blocks.extend([f"% Replaces {'Table IX' if i == 0 else 'Table X' if i == 1 else 'Table XI' if i == 2 else 'Table XII' if i == 3 else 'Table XIII' if i == 4 else 'Tables X, XII and XIII'}.",
                       f"\\begin{{{environment}}}[t]", "\\centering",
                       f"\\includegraphics[width=\\{'textwidth' if wide else 'columnwidth'}]{{table_replacement_figures/{stem}.pdf}}",
                       f"\\caption{{{CAPTIONS[stem]}}}", f"\\label{{{labels[i]}}}",
                       f"\\end{{{environment}}}", ""])
    (OUTPUT / "replacement_figures.tex").write_text("\n".join(blocks), encoding="utf-8")
    readme = """# Table-to-figure replacements

Original mapping refers to IEEE_Access_Tarun (4).pdf. Tables/figures will be
renumbered after editing the LaTeX source.

| Original table | Figure stem | Recommended width |
| --- | --- | --- |
| IX: registered neural parameters | table_ix_registered_parameters | two columns |
| X: zero replacement | table_x_feature_masking | one column |
| XI: imputation comparison | table_xi_imputation | one column |
| XII: Gaussian noise | table_xii_gaussian_noise | one column |
| XIII: feature scaling | table_xiii_feature_scaling | one column |
| X + XII + XIII combined alternative | combined_robustness_three_panel | two columns |

Use the combined robustness figure OR its three individual counterparts, not
both. Keep the imputation figure separate because it uses a different checkpoint.
Remove the duplicated Table VI (same data as X), and Table VII (summary of XII and
XIII), and replace their references with references to the retained figures.
Keep Tables IV and V for exact classification and calibration values, and Table
VIII for route counts. Tables I-III can also remain tables.

PDF is the preferred vector input to LaTeX. PNG files are 600 DPI; SVG files are
editable vector exports. Preview PNGs are only for quick inspection.

Use replacement_figures.tex for captions, labels, and placement. Put this folder
beside the main .tex file, update any former Table references, and retain the
classification/calibration tables. Existing manuscript files are not edited.

All plots use existing evidence, without training or invented observations.
Hybrid parameter counts are independently aggregated from the five seed JSON
files, using sample SD (ddof=1); counts exclude the external XGBoost ensemble.
The old robustness report labels scaling as cross_dataset internally; the new
figures intentionally call it standardized feature scaling, not external data.
The imputation CSV's separate XGBoost column is omitted to preserve the
within-model comparison in Table XI. Robustness trial-level SDs are not available,
so these plots contain no fabricated uncertainty bars. Lines connect recorded
operating points; they are not estimates from additional experiments.

Sources and SHA-256 hashes are in plot_source_data.json and validation_manifest.json.
Reproduce from the repository root:

    .venv\\Scripts\\python.exe KG-Hybrid\\src\\utils\\generate_table_replacement_figures.py
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "no_retraining": True, "existing_manuscript_modified": False,
        "source_evidence_modified": False, "original_pdf": r"C:\Users\tanne\Downloads\IEEE_Access_Tarun (4).pdf",
        "sources": evidence["sources"], "parameter_sd_estimator": "sample SD; ddof=1; five seeds",
        "robustness_error_bars": "not shown: trial-level SD unavailable",
        "separate_imputation_checkpoint": True,
        "neural_counts_exclude_external_xgboost": True,
        "scaling_is_not_external_dataset": True,
        "outputs": {stem: {"replaces": ("IX", "X", "XI", "XII", "XIII", "X+XII+XIII")[i],
                           "caption": CAPTIONS[stem],
                           "sha256": {suffix: hashlib.sha256((OUTPUT / f"{stem}.{suffix}").read_bytes()).hexdigest()
                                      for suffix in ("pdf", "svg", "png")}}
                    for i, stem in enumerate(STEMS)},
    }
    (OUTPUT / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    package = OUTPUT.parent / "table_replacement_figure_package.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            if path.is_file():
                archive.write(path, f"table_replacement_figures/{path.name}")
        archive.write(Path(__file__), "generate_table_replacement_figures.py")
    print(json.dumps({"output_directory": str(OUTPUT), "package": str(package),
                      "vector_pdf_figures": len(STEMS), "source_count": len(evidence["sources"])}, indent=2))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "hatch.linewidth": 0.55})
    evidence = load_evidence()
    plot_parameters(evidence)
    plot_line(evidence, STEMS[1], "masking")
    plot_imputation(evidence)
    plot_line(evidence, STEMS[3], "noise")
    plot_scaling(evidence)
    plot_combined(evidence)
    write_support(evidence)


if __name__ == "__main__":
    main()
