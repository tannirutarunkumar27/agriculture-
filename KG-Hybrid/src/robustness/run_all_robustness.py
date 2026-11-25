"""
Robustness Test Runner — Runs all three robustness experiments sequentially
and generates a comprehensive visualization report.

Usage:
    cd "D:\\Final Project\\implementation_for _crop recommendation\\KG-Hybrid"
    python src/robustness/run_all_robustness.py
"""

import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "outputs" / "robustness"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# =============================================================
# Run each script as a subprocess (avoids __file__ path issues)
# =============================================================

def run_script(script_name, label):
    script_path = PROJECT_ROOT / "src" / "robustness" / script_name
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    result = subprocess.run(
        [PYTHON, "-X", "utf8", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,       # stream output live
        text=True,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        return False
    return True


# =============================================================
# Visualization (reads saved JSON files)
# =============================================================

MODEL_ORDER  = ["KG-Hybrid", "XGBoost", "FT-Transformer", "Random Forest"]
MODEL_COLORS = {
    "KG-Hybrid":      "#DC2626",
    "XGBoost":        "#2563EB",
    "FT-Transformer": "#059669",
    "Random Forest":  "#7C3AED",
}
MODEL_MARKERS = {
    "KG-Hybrid":      "o",
    "XGBoost":        "s",
    "FT-Transformer": "^",
    "Random Forest":  "D",
}


def make_plots(miss_r, noise_r, cross_r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 11), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    for ax in axes.flat:
        ax.set_facecolor('#F8FAFC')
        ax.tick_params(colors='#0F172A', labelsize=10)
        ax.xaxis.label.set_color('#0F172A')
        ax.xaxis.label.set_fontweight('bold')
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_color('#0F172A')
        ax.yaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontsize(11)
        ax.title.set_color('#0F172A')
        ax.title.set_fontweight('bold')
        ax.title.set_fontsize(12.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── (0,0) Missing features accuracy vs ratio ──────────────
    ax = axes[0, 0]
    ax.set_title("Missing Features: Accuracy vs Mask Ratio", pad=10)
    for mn in MODEL_ORDER:
        if mn not in miss_r:
            continue
        row    = miss_r[mn]
        ratios = sorted(row.keys(), key=lambda x: int(x.replace('%','')))
        vals   = [row[k] * 100 for k in ratios]
        xs     = [int(r.replace('%','')) for r in ratios]
        ax.plot(xs, vals, marker=MODEL_MARKERS[mn], color=MODEL_COLORS[mn],
                linewidth=2.8, markersize=8, label=mn)
    ax.set_xlabel("Missing Feature Ratio (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-2, 65)
    ax.legend(fontsize=10, facecolor='#FFFFFF', edgecolor='#CBD5E1', labelcolor='#0F172A', framealpha=0.95)
    ax.grid(True, alpha=0.5, color='#CBD5E1', linestyle='--')

    # ── (0,1) Missing features: drop at 40% ───────────────────
    ax = axes[0, 1]
    ax.set_title("Missing Features: Accuracy Drop at 40%", pad=10)
    drops  = []
    labels = []
    colors = []
    for mn in MODEL_ORDER:
        if mn not in miss_r:
            continue
        row  = miss_r[mn]
        base = row.get('0%', list(row.values())[0])
        d40  = (row.get('40%', base) - base) * 100
        drops.append(d40)
        labels.append(mn)
        colors.append(MODEL_COLORS[mn])
    bars = ax.bar(labels, drops, color=colors, edgecolor='#0F172A', linewidth=0.8, width=0.55)
    ax.set_ylabel("Accuracy Drop Δ (%)")
    ax.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax.axhline(0, color='#64748B', linewidth=1.0, linestyle='--')
    for bar, val in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 3.5 if val < 0 else bar.get_height() + 1.0,
                f'{val:+.1f}%', ha='center', va='bottom', color='#0F172A', fontsize=10.5, fontweight='bold')
    ax.grid(True, alpha=0.5, axis='y', color='#CBD5E1', linestyle='--')

    # ── (0,2) Noise: accuracy vs sigma ────────────────────────
    ax = axes[0, 2]
    ax.set_title("Gaussian Noise: Accuracy vs Noise Level (σ)", pad=10)
    for mn in MODEL_ORDER:
        if mn not in noise_r:
            continue
        row    = noise_r[mn]
        sigmas = sorted(row.keys())
        vals   = [row[k] * 100 for k in sigmas]
        xs = [float(s.split('=')[-1]) for s in sigmas]
        ax.plot(xs, vals, marker=MODEL_MARKERS[mn], color=MODEL_COLORS[mn],
                linewidth=2.8, markersize=8, label=mn)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=10, facecolor='#FFFFFF', edgecolor='#CBD5E1', labelcolor='#0F172A', framealpha=0.95)
    ax.grid(True, alpha=0.5, color='#CBD5E1', linestyle='--')

    # ── (1,0) Noise: drop at sigma=0.2 ────────────────────────
    ax = axes[1, 0]
    ax.set_title("Gaussian Noise: Accuracy Drop at σ = 0.2", pad=10)
    drops2  = []
    labels2 = []
    for mn in MODEL_ORDER:
        if mn not in noise_r:
            continue
        row  = noise_r[mn]
        keys = list(row.keys())
        base_key = [k for k in keys if k.endswith('0.0') or k.endswith('=0.0')]
        mid_key  = [k for k in keys if k.endswith('0.2') or k.endswith('=0.2')]
        base = row[base_key[0]] if base_key else list(row.values())[0]
        mid  = row[mid_key[0]]  if mid_key  else list(row.values())[2]
        drops2.append((mid - base) * 100)
        labels2.append(mn)
    bars2 = ax.bar(labels2, drops2,
                   color=[MODEL_COLORS[m] for m in labels2],
                   edgecolor='#0F172A', linewidth=0.8, width=0.55)
    ax.set_ylabel("Accuracy Drop Δ (%)")
    ax.set_xticklabels(labels2, rotation=15, ha='right', fontweight='bold')
    ax.axhline(0, color='#64748B', linewidth=1.0, linestyle='--')
    for bar, val in zip(bars2, drops2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 2.5 if val < 0 else bar.get_height() + 0.8,
                f'{val:+.1f}%', ha='center', va='bottom', color='#0F172A', fontsize=10.5, fontweight='bold')
    ax.grid(True, alpha=0.5, axis='y', color='#CBD5E1', linestyle='--')

    # ── (1,1) Cross-dataset: 4 scenarios ──────────────────────
    ax = axes[1, 1]
    ax.set_title("Generalization Across Evaluation Scenarios", pad=10)
    scenarios = {
        "Test":         cross_r.get("standard_test", {}),
        "Val":          cross_r.get("validation_set", {}),
        "Extrap x1.5":  cross_r.get("extrapolation_1_5", {}),
        "Worst Feat":   cross_r.get("single_feature_min", {}),
    }
    x_pos = np.arange(len(scenarios))
    width = 0.18
    valid_models = [mn for mn in MODEL_ORDER if mn in cross_r.get("standard_test", {})]
    for i, mn in enumerate(valid_models):
        vals = [scenarios[s].get(mn, 0.0) * 100 for s in scenarios]
        ax.bar(x_pos + i * width, vals, width, label=mn,
               color=MODEL_COLORS[mn], edgecolor='#0F172A', linewidth=0.6)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(list(scenarios.keys()), rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9, facecolor='#FFFFFF', edgecolor='#CBD5E1', labelcolor='#0F172A', framealpha=0.95)
    ax.grid(True, alpha=0.5, axis='y', color='#CBD5E1', linestyle='--')

    # ── (1,2) Composite robustness score ──────────────────────
    ax = axes[1, 2]
    ax.set_title("Composite Robustness Ranking", pad=10)

    composite = {}
    for mn in MODEL_ORDER:
        if mn not in miss_r or mn not in noise_r or mn not in cross_r.get("standard_test",{}):
            continue
        base_miss  = miss_r[mn].get('0%', 1.0)
        miss_keys  = [k for k in noise_r[mn] if k.endswith('0.0') or k.endswith('=0.0')]
        base_noise = noise_r[mn][miss_keys[0]] if miss_keys else 1.0
        miss_drop  = abs(miss_r[mn].get('40%', base_miss) - base_miss)
        mid_keys   = [k for k in noise_r[mn] if k.endswith('0.2') or k.endswith('=0.2')]
        mid_noise  = noise_r[mn][mid_keys[0]] if mid_keys else base_noise
        noise_drop = abs(mid_noise - base_noise)
        extrap_acc = cross_r.get("extrapolation_1_5", {}).get(mn, 0.0)
        wf_acc     = cross_r.get("single_feature_min", {}).get(mn, 0.0)
        score = np.mean([1 - miss_drop, 1 - noise_drop, extrap_acc, wf_acc])
        composite[mn] = round(float(score), 4)

    if composite:
        sorted_models = sorted(composite.keys(), key=lambda k: composite[k], reverse=False)
        bars3 = ax.barh(sorted_models,
                        [composite[m] for m in sorted_models],
                        color=[MODEL_COLORS[m] for m in sorted_models],
                        edgecolor='#0F172A', linewidth=0.8, height=0.55)
        ax.set_xlabel("Composite Robustness Score [0 - 1]")
        ax.set_xlim(0, 1.15)
        for bar, mn in zip(bars3, sorted_models):
            ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2,
                    f'{composite[mn]:.4f}', va='center', color='#0F172A', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5, axis='x', color='#CBD5E1', linestyle='--')
        ax.set_yticklabels(sorted_models, fontweight='bold', color='#0F172A')

    fig.suptitle("Comprehensive Model Robustness Analysis Suite",
                 color='#0F172A', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_PATH / "robustness_report.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"\nRobustness plot saved -> {out_path}")
    return composite


# =============================================================
# Main
# =============================================================

def main():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("KG-HYBRID ROBUSTNESS ANALYSIS - FULL SUITE")
    print("=" * 70)

    # Run each experiment as a subprocess
    ok1 = run_script("test_missing_features.py", "EXPERIMENT 1: MISSING FEATURES")
    ok2 = run_script("test_noise.py",            "EXPERIMENT 2: GAUSSIAN NOISE")
    ok3 = run_script("test_cross_dataset.py",    "EXPERIMENT 3: CROSS-DATASET")

    # Load saved JSON results
    print("\n" + "=" * 70)
    print("LOADING RESULTS AND GENERATING VISUALIZATIONS")
    print("=" * 70)

    miss_r  = {}
    noise_r = {}
    cross_r = {}

    mf_file = RESULTS_PATH / "missing_features_results.json"
    if mf_file.exists():
        with open(mf_file) as f:
            miss_r = json.load(f)
        print(f"  Loaded: {mf_file.name}")
    else:
        print(f"  [WARN] {mf_file.name} not found — skipping")

    ns_file = RESULTS_PATH / "noise_results.json"
    if ns_file.exists():
        with open(ns_file) as f:
            noise_r = json.load(f)
        print(f"  Loaded: {ns_file.name}")
    else:
        print(f"  [WARN] {ns_file.name} not found — skipping")

    cd_file = RESULTS_PATH / "cross_dataset_results.json"
    if cd_file.exists():
        with open(cd_file) as f:
            cross_r = json.load(f)
        print(f"  Loaded: {cd_file.name}")
    else:
        print(f"  [WARN] {cd_file.name} not found — skipping")

    if miss_r and noise_r and cross_r:
        composite = make_plots(miss_r, noise_r, cross_r)

        # Final verdict
        print("\n" + "=" * 70)
        print("COMPOSITE ROBUSTNESS SCORES")
        print("=" * 70)
        sorted_models = sorted(composite.keys(), key=lambda k: composite[k], reverse=True)
        for rank, mn in enumerate(sorted_models, 1):
            badge = " <- OUR MODEL" if mn == "KG-Hybrid" else ""
            print(f"  #{rank}  {mn:<20}: {composite[mn]:.4f}{badge}")

        kg_rank = sorted_models.index("KG-Hybrid") + 1
        if kg_rank == 1:
            print("\nVERDICT: KG-Hybrid is the MOST ROBUST model!")
            print("  -> Paper narrative: competitive accuracy + superior robustness")
        elif kg_rank == 2:
            print("\nVERDICT: KG-Hybrid is 2nd most robust.")
            print("  -> Still supportable robustness narrative.")
        else:
            print(f"\nVERDICT: KG-Hybrid ranks #{kg_rank} in robustness.")
            print("  -> Consider simplifying the model or revising the narrative.")

        # Save full report
        all_results = {
            "missing_features": miss_r,
            "noise_injection":  noise_r,
            "cross_dataset":    cross_r,
            "composite_scores": composite,
        }
        out_file = RESULTS_PATH / "full_robustness_report.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull report saved -> {out_file}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
