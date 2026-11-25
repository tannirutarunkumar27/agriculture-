"""Publication figures for the original baseline and post-hoc guard studies.

The two studies are deliberately kept separate: the guard was designed after
outer-test failures were visible and is not an independent confirmation.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

from paper2.restart import guard_sensitivity, summarize_outer
from paper2.restart.validation_search import ROOT, atomic_json, read, sha


ALGORITHMS = (
    "xgboost", "random_forest", "training_mean", "ridge",
    "fednova", "fedavg", "fedprox", "centralized",
)
NEURAL = ("centralized", "fedavg", "fedprox", "fednova")
LABELS = {
    "xgboost": "XGBoost", "random_forest": "Random forest",
    "training_mean": "Training mean", "ridge": "Ridge",
    "fednova": "FedNova", "fedavg": "FedAvg", "fedprox": "FedProx",
    "centralized": "Centralized neural",
}
COLORS = {"equal": "#266B8E", "pooled": "#D17C2F", "guard": "#16836A"}


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative metric required")
    return float(value)


def extract(original, guarded):
    """Validate complete analyses and return only values used by the plots."""
    if original.get("status") != "COMPLETE_DESCRIPTIVE_ANALYSIS" or original.get("verified_runs") != 320:
        raise ValueError("Complete 320-run original analysis required")
    if guarded.get("status") != "PASS_COMPLETE_POST_HOC_SENSITIVITY" or guarded.get("verified") != 320:
        raise ValueError("Complete 320-run guard analysis required")
    if "POST_HOC_SENSITIVITY" not in guarded.get("scope", ""):
        raise ValueError("Guard analysis must retain its post-hoc scope")
    if guarded.get("training_performed") is not False or guarded.get("inference_repeated_during_audit") is not False:
        raise ValueError("Unexpected guard execution provenance")

    original_all = {r["algorithm"]: r for r in original["summary"] if r["panel"] == "all"}
    guarded_all = {r["algorithm"]: r for r in guarded["summary"] if r["panel"] == "all"}
    if set(original_all) != set(ALGORITHMS) or set(guarded_all) != set(ALGORITHMS):
        raise ValueError("Exact eight-algorithm inventory required")

    overview = []
    for algorithm in ALGORITHMS:
        row = original_all[algorithm]
        seeds = [r for r in original["per_seed"] if r["algorithm"] == algorithm and r["panel"] == "all"]
        if len(seeds) != 5 or row["seeds"] != 5 or row["samples_per_seed"] != 607 or row["groups_per_seed"] != 8:
            raise ValueError("Complete five-seed, 607-system, eight-group baseline required")
        entry = {"algorithm": algorithm, "label": LABELS[algorithm]}
        for metric, getter in (
            ("equal_group_rmse", lambda r: r["metrics"]["equal_group_rmse"]),
            ("pooled_rmse", lambda r: r["metrics"]["pooled"]["rmse"]),
        ):
            values = [_finite(getter(r)) for r in seeds]
            measured = row[metric]
            if not np.isclose(measured["mean"], np.mean(values), rtol=1e-12, atol=1e-12):
                raise ValueError("Baseline mean differs from seed values")
            if not np.isclose(measured["seed_sd"], np.std(values, ddof=1), rtol=1e-12, atol=1e-12):
                raise ValueError("Baseline SD differs from seed values")
            entry[metric] = {"mean": _finite(measured["mean"]), "seed_sd": _finite(measured["seed_sd"]),
                             "seed_values": values}
        overview.append(entry)

    geographic = []
    region_rows = [r for r in original["per_region"] if r["algorithm"] == "xgboost" and r["panel"] == "all"]
    groups = sorted({r["group"] for r in region_rows})
    if len(region_rows) != 40 or len(groups) != 8:
        raise ValueError("Complete XGBoost geographic results required")
    for group in groups:
        rows = [r for r in region_rows if r["group"] == group]
        values = [_finite(r["metrics"]["rmse"]) for r in rows]
        samples = {r["metrics"]["samples"] for r in rows}
        if len(rows) != 5 or len(samples) != 1:
            raise ValueError("Five seeds and stable sample count required per group")
        geographic.append({"group": group, "samples": samples.pop(), "mean": float(np.mean(values)),
                           "seed_sd": float(np.std(values, ddof=1)), "seed_values": values})
    if sum(r["samples"] for r in geographic) != 607:
        raise ValueError("Geographic sample counts must sum to 607")

    sensitivity = []
    for algorithm in NEURAL:
        before = original_all[algorithm]
        after = guarded_all[algorithm]
        guarded_seeds = [r["metrics"]["equal_group_rmse"] for r in guarded["per_seed"]
                         if r["algorithm"] == algorithm and r["panel"] == "all"]
        if len(guarded_seeds) != 5:
            raise ValueError("Five guarded seeds required")
        if not np.isclose(after["equal_group_rmse_mean"], np.mean(guarded_seeds), rtol=1e-12, atol=1e-12):
            raise ValueError("Guard mean differs from seed values")
        if not np.isclose(after["equal_group_rmse_seed_sd"], np.std(guarded_seeds, ddof=1), rtol=1e-12, atol=1e-12):
            raise ValueError("Guard SD differs from seed values")
        sensitivity.append({
            "algorithm": algorithm, "label": LABELS[algorithm],
            "original_mean": _finite(before["equal_group_rmse"]["mean"]),
            "original_seed_sd": _finite(before["equal_group_rmse"]["seed_sd"]),
            "guarded_mean": _finite(after["equal_group_rmse_mean"]),
            "guarded_seed_sd": _finite(after["equal_group_rmse_seed_sd"]),
        })
    return {
        "scope": "Original prospective outer evaluation plus separately labelled post-hoc sensitivity",
        "uncertainty": "Descriptive SD across five seeds; not a geographic confidence interval",
        "overview": overview, "geographic": geographic, "posthoc_sensitivity": sensitivity,
        "guard_scope": guarded["scope"],
    }


def draw(values, output):
    output = Path(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    assets, captions = [], []

    def save(fig, name, caption):
        for ext in ("png", "pdf", "svg"):
            path = output / f"{name}.{ext}"
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            assets.append(path.name)
        plt.close(fig)
        captions.append({"name": name, "caption": caption})

    rows = values["overview"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), layout="constrained", sharey=True)
    for ax, metric, title, color, ticks in zip(
        axes, ("equal_group_rmse", "pooled_rmse"),
        ("Equal geographic groups", "Pooled held-out systems"),
        (COLORS["equal"], COLORS["pooled"]),
        ((1, 2, 5, 10), (1, 2, 5, 10, 20, 50)),
    ):
        for i, row in enumerate(rows):
            score = row[metric]
            ax.scatter(score["seed_values"], np.linspace(i - .10, i + .10, 5), s=14, color=color, alpha=.42)
            ax.errorbar(score["mean"], i, xerr=score["seed_sd"], fmt="D", color=color, capsize=3, markersize=5)
        ax.set_xscale("log"); ax.set_xlim(ticks[0] * .9, ticks[-1]);
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
        ax.set_xlabel("RMSE (t/ha; log scale)"); ax.set_title(title)
        ax.grid(axis="x", which="both", alpha=.18)
        ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    axes[0].set_ylim(len(rows) - .5, -.5)
    fig.suptitle("Original held-out baseline evaluation | 607 systems, 8 groups, 5 seeds", fontsize=11)
    save(fig, "original_baseline_rmse",
         "Original prospectively frozen outer evaluation. Dots are five seed estimates; diamonds and bars are the mean and descriptive seed SD. The logarithmic axis makes the large neural-model errors visible without hiding differences among conventional baselines. Equal-group and pooled RMSE answer different weighting questions.")

    geo = values["geographic"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    for i, row in enumerate(geo):
        ax.scatter(row["seed_values"], np.linspace(i - .10, i + .10, 5), s=14, color=COLORS["equal"], alpha=.40)
        ax.errorbar(row["mean"], i, xerr=row["seed_sd"], fmt="D", color=COLORS["equal"], capsize=3)
    ax.set_yticks(range(8), [f"{r['group']}  (n={r['samples']})" for r in geo]); ax.invert_yaxis()
    ax.set_xlim(left=0); ax.set_xlabel("XGBoost held-out RMSE (t/ha)"); ax.grid(axis="x", alpha=.18)
    ax.set_title("Original XGBoost error by held-out geographic group")
    save(fig, "original_xgboost_geographic_rmse",
         "Original XGBoost geographic holdouts with their unequal system counts. Dots are seed estimates; diamonds and bars show mean and descriptive seed SD. Regions are evaluation groups, not independent replications, and these bars are not geographic confidence intervals.")

    sens = values["posthoc_sensitivity"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    y = np.arange(len(sens))
    for i, row in enumerate(sens):
        ax.plot([row["guarded_mean"], row["original_mean"]], [i, i], color="#AEB7BF", linewidth=2, zorder=1)
        ax.errorbar(row["original_mean"], i, xerr=row["original_seed_sd"], fmt="o", color="#A84C45",
                    capsize=3, label="Original" if i == 0 else None, zorder=2)
        ax.errorbar(row["guarded_mean"], i, xerr=row["guarded_seed_sd"], fmt="D", color=COLORS["guard"],
                    capsize=3, label="Training-only zero-support guard" if i == 0 else None, zorder=3)
    ax.set_yticks(y, [r["label"] for r in sens]); ax.invert_yaxis(); ax.set_xlim(left=0)
    ax.set_xlabel("Equal-group RMSE (t/ha), mean ± seed SD"); ax.grid(axis="x", alpha=.18)
    ax.set_title("Post-hoc sensitivity after outer-test failures were exposed")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2)
    ax.text(.01, -.27, "Sensitivity only — not independent confirmation and not a replacement for the original result.",
            transform=ax.transAxes, fontsize=8, color="#7A2E2A")
    save(fig, "posthoc_guard_sensitivity",
         "Post-hoc sensitivity for the four neural methods. The zero-support guard was specified after outer-test failures were observed; it reused fixed trained models and therefore cannot be interpreted as independent confirmation. Original results remain the primary prospective record.")
    return assets, captions


def generate(output):
    output = Path(output).resolve()
    if ROOT / "paper2/results" not in output.parents:
        raise ValueError("Use a new Paper 2 results directory")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Fresh cached audits only. Neither call trains models; the guard analysis also
    # verifies that audit did not repeat inference.
    summarize_outer.summarize(output / "verified_original_analysis")
    guard_sensitivity.analyze(
        ROOT / "paper2/results/restart_guard_protocol_v1",
        ROOT / "paper2/results/restart_guard_study_v1",
        output / "verified_guard_analysis",
    )
    original_path = output / "verified_original_analysis/analysis.json"
    guarded_path = output / "verified_guard_analysis/analysis.json"
    values = extract(read(original_path), read(guarded_path))
    atomic_json(output / "plot_values.json", values)
    assets, captions = draw(values, output)
    atomic_json(output / "captions.json", captions)
    atomic_json(output / "manifest.json", {
        "status": "GENERATED_FROM_COMPLETE_SEPARATELY_AUDITED_ANALYSES",
        "original_analysis_sha256": sha(original_path), "guard_analysis_sha256": sha(guarded_path),
        "code_sha256": sha(__file__), "matplotlib": matplotlib.__version__,
        "assets": {name: sha(output / name) for name in assets + ["plot_values.json", "captions.json"]},
        "visual_review": "pending", "model_training": False, "model_reinference": False,
        "posthoc_guard_separated": True,
    })
    return {"status": "GENERATED_FROM_COMPLETE_SEPARATELY_AUDITED_ANALYSES", "plots": 3,
            "formats": ["png", "pdf", "svg"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(generate(parser.parse_args().output))
