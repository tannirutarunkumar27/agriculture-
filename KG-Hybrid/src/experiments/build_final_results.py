"""Build canonical outputs/final_results.json and headline tables from saved artifacts.

Single source of truth for Paper 1:
- Reads ONLY from saved artifacts in outputs/. Never invents numbers.
- Computes 95% confidence intervals (t-distribution, df=n-1) for 5-seed and
  5-fold CV protocols.
- Computes Friedman omnibus test and Nemenyi post-hoc pairwise exact p-values.
- Keeps the single 70/15/15 split strictly as a labelled example.
- Generates outputs/final_results.json, headline_table_with_ci.{csv,md},
  friedman_nemenyi_posthoc.csv, and updates README.md performance table.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent


def _ci95(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        se = std / math.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        half = t_crit * se
    else:
        half = 0.0
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def _friedman_nemenyi(matrix: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """Run Friedman test and Nemenyi post-hoc on (blocks x models) DataFrame."""
    models = list(matrix.columns)
    k = len(models)
    n_blocks = len(matrix)
    if k < 2 or n_blocks < 2:
        return {"error": "insufficient blocks or models"}

    # Friedman test
    stat, p_val = stats.friedmanchisquare(*[matrix[m].values for m in models])
    # Ranks per block: rank 1 = best
    ranks = matrix.rank(axis=1, ascending=not higher_is_better, method="average")
    mean_ranks = ranks.mean(axis=0).to_dict()

    # Nemenyi post-hoc exact p-values via studentized range distribution
    se_nemenyi = math.sqrt(k * (k + 1) / (6.0 * n_blocks))
    cd_05 = float(stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2.0)) * se_nemenyi
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            m1, m2 = models[i], models[j]
            diff = abs(mean_ranks[m1] - mean_ranks[m2])
            q_stat = (diff / se_nemenyi) * math.sqrt(2.0)
            p_nem = float(stats.studentized_range.sf(q_stat, k, np.inf))
            pairwise.append({
                "model_a": m1,
                "model_b": m2,
                "mean_rank_a": float(mean_ranks[m1]),
                "mean_rank_b": float(mean_ranks[m2]),
                "abs_rank_diff": float(diff),
                "nemenyi_q": float(q_stat),
                "nemenyi_p_value": p_nem,
                "significant_alpha_05": bool(p_nem < 0.05),
            })

    return {
        "n_blocks": int(n_blocks),
        "n_models": int(k),
        "friedman_chi2": float(stat) if not math.isnan(stat) else 0.0,
        "friedman_p_value": float(p_val) if not math.isnan(p_val) else 1.0,
        "nemenyi_critical_diff_alpha_05": float(cd_05),
        "mean_ranks": {m: float(v) for m, v in mean_ranks.items()},
        "pairwise": pairwise,
    }


def load_five_seed() -> tuple[dict, pd.DataFrame]:
    kc_path = ROOT / "outputs" / "knowledge_comparison_review_fixes" / "per_seed_metrics.csv"
    sa_path = ROOT / "outputs" / "stage_a_baselines" / "five_seed" / "per_seed_metrics.csv"
    frames = [pd.read_csv(kc_path)]
    if sa_path.exists():
        frames.append(pd.read_csv(sa_path))
    df = pd.concat(frames, ignore_index=True)

    summary = {}
    for model, grp in df.groupby("model"):
        summary[model] = {
            "accuracy": _ci95(grp["accuracy"].values),
            "macro_f1": _ci95(grp["macro_f1"].values),
            "ece": _ci95(grp["ece"].values),
            "brier": _ci95(grp["brier"].values),
            "nll": _ci95(grp["nll"].values),
            "training_seconds": _ci95(grp["training_seconds"].values),
        }
    pivot_acc = df.pivot(index="seed", columns="model", values="accuracy")
    stats_acc = _friedman_nemenyi(pivot_acc, higher_is_better=True)
    return {"summary": summary, "friedman_nemenyi_accuracy": stats_acc}, df


def load_five_fold_cv() -> tuple[dict, pd.DataFrame]:
    cv_summary_path = ROOT / "outputs" / "final_experimental_validation" / "cv_validated_summary.csv"
    cv_oof_df = pd.read_csv(cv_summary_path)

    # Load per-fold accuracies from cross_validation_neural_recovery
    fold_rows = []
    per_fold_dir = ROOT / "outputs" / "cross_validation_neural_recovery" / "per_fold"
    for fold_idx in range(1, 6):
        fpath = per_fold_dir / f"fold_{fold_idx}.json"
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for model_name, metrics in data.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                fold_rows.append({
                    "fold": fold_idx,
                    "model": model_name,
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics.get("f1_macro", metrics.get("macro_f1", np.nan))),
                })

    # Load Stage A new CV baselines
    sa_fold_path = ROOT / "outputs" / "stage_a_baselines" / "five_fold_cv" / "per_fold_metrics.csv"
    sa_summary_path = ROOT / "outputs" / "stage_a_baselines" / "five_fold_cv" / "cv_summary.csv"
    if sa_fold_path.exists():
        sa_df = pd.read_csv(sa_fold_path)
        for _, r in sa_df.iterrows():
            fold_rows.append({
                "fold": int(r["fold"]),
                "model": str(r["model"]),
                "accuracy": float(r["accuracy"]),
                "macro_f1": float(r["macro_f1"]),
            })

    folds_df = pd.DataFrame(fold_rows)
    oof_map = {
        str(r["model"]): {
            "oof_accuracy": float(r["oof_accuracy_recomputed"]),
            "oof_macro_f1": float(r["oof_macro_f1_recomputed"]),
        }
        for _, r in cv_oof_df.iterrows()
    }
    if sa_summary_path.exists():
        for _, r in pd.read_csv(sa_summary_path).iterrows():
            oof_map[str(r["model"])] = {
                "oof_accuracy": float(r["accuracy"]),
                "oof_macro_f1": float(r["macro_f1"]),
            }

    summary = {}
    for model, grp in folds_df.groupby("model"):
        entry = {
            "fold_accuracy": _ci95(grp["accuracy"].values),
            "fold_macro_f1": _ci95(grp["macro_f1"].values),
        }
        if model in oof_map:
            entry["pooled_oof"] = oof_map[model]
        summary[model] = entry

    pivot_acc = folds_df.pivot(index="fold", columns="model", values="accuracy")
    stats_acc = _friedman_nemenyi(pivot_acc, higher_is_better=True)
    return {"summary": summary, "friedman_nemenyi_accuracy": stats_acc}, folds_df


def load_supporting_artifacts() -> dict:
    artifacts: dict = {}

    # Single 70/15/15 split example (labelled explicitly as example only)
    cal_path = ROOT / "outputs" / "tables" / "table2_calibration_metrics_reconciled.csv"
    if cal_path.exists():
        artifacts["single_split_example_calibration"] = pd.read_csv(cal_path).to_dict(orient="records")

    # Computational cost
    cost_path = ROOT / "outputs" / "tables" / "table6_computational_cost.csv"
    if cost_path.exists():
        artifacts["computational_cost"] = pd.read_csv(cost_path).to_dict(orient="records")

    # Early-exit & production readiness
    ee_path = ROOT / "outputs" / "tables" / "realworld_early_exit_comparison.csv"
    if ee_path.exists():
        artifacts["early_exit_sweep"] = pd.read_csv(ee_path).to_dict(orient="records")

    prod_path = ROOT / "outputs" / "tables" / "production_readiness_results.json"
    if prod_path.exists():
        artifacts["production_readiness"] = json.loads(prod_path.read_text(encoding="utf-8"))

    # Lightweight model
    lw_path = ROOT / "outputs" / "models" / "lightweight_kg_hybrid_results.json"
    if lw_path.exists():
        artifacts["lightweight_model"] = json.loads(lw_path.read_text(encoding="utf-8"))

    # 3-way feature importance
    fi_path = ROOT / "outputs" / "tables" / "feature_importance_3way_measured.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path, index_col=0)
        artifacts["feature_importance_3way"] = fi_df.to_dict(orient="index")

    # Stage A tuning manifest
    sa_manifest = ROOT / "outputs" / "stage_a_baselines" / "manifest.json"
    if sa_manifest.exists():
        artifacts["stage_a_baselines_manifest"] = json.loads(sa_manifest.read_text(encoding="utf-8"))

    return artifacts


def write_headline_tables(final_results: dict) -> None:
    rows = []
    five_seed = final_results["five_seed"]["summary"]
    five_fold = final_results["five_fold_cv"]["summary"]

    for model, d in sorted(five_seed.items(), key=lambda kv: -kv[1]["accuracy"]["mean"]):
        acc = d["accuracy"]
        f1 = d["macro_f1"]
        ece = d["ece"]
        rows.append({
            "Protocol": "5-Seed 70/15/15 (N=330)",
            "Model": model,
            "Accuracy_Mean": round(acc["mean"], 4),
            "Accuracy_SD": round(acc["std"], 4),
            "Accuracy_95CI": f"[{acc['ci95_low']:.4f}, {acc['ci95_high']:.4f}]",
            "MacroF1_Mean": round(f1["mean"], 4),
            "ECE_Mean": round(ece["mean"], 4),
        })

    for model, d in sorted(five_fold.items(), key=lambda kv: -kv[1]["fold_accuracy"]["mean"]):
        acc = d["fold_accuracy"]
        f1 = d["fold_macro_f1"]
        rows.append({
            "Protocol": "5-Fold Stratified CV (N=2200 OOF)",
            "Model": model,
            "Accuracy_Mean": round(acc["mean"], 4),
            "Accuracy_SD": round(acc["std"], 4),
            "Accuracy_95CI": f"[{acc['ci95_low']:.4f}, {acc['ci95_high']:.4f}]",
            "MacroF1_Mean": round(f1["mean"], 4),
            "ECE_Mean": None,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ROOT / "outputs" / "tables" / "headline_table_with_ci.csv", index=False)

    # Markdown version
    md_lines = [
        "| Protocol | Model | Accuracy (Mean ± SD) | 95% CI | Macro-F1 | ECE |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_out.iterrows():
        ece_str = f"{r['ECE_Mean']:.4f}" if pd.notna(r["ECE_Mean"]) else "—"
        md_lines.append(
            f"| {r['Protocol']} | {r['Model']} | {r['Accuracy_Mean']*100:.2f}% ± {r['Accuracy_SD']*100:.2f}% | "
            f"{r['Accuracy_95CI']} | {r['MacroF1_Mean']*100:.2f}% | {ece_str} |"
        )
    (ROOT / "outputs" / "tables" / "headline_table_with_ci.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Friedman + Nemenyi CSV
    nem_rows = []
    for proto_key, proto_label in [("five_seed", "5-Seed"), ("five_fold_cv", "5-Fold CV")]:
        fn = final_results[proto_key]["friedman_nemenyi_accuracy"]
        for p in fn.get("pairwise", []):
            nem_rows.append({"protocol": proto_label, "friedman_chi2": fn["friedman_chi2"], "friedman_p": fn["friedman_p_value"], **p})
    pd.DataFrame(nem_rows).to_csv(ROOT / "outputs" / "tables" / "friedman_nemenyi_posthoc.csv", index=False)


def update_readme_from_json(final_results: dict) -> None:
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")

    five_seed = final_results["five_seed"]["summary"]
    five_fold = final_results["five_fold_cv"]["summary"]

    lines = [
        "## 📊 Performance & Benchmarks",
        "",
        "> **Canonical Headline Protocol** (generated automatically from `KG-Hybrid/outputs/final_results.json`).",
        "",
        "### 5-Seed 70/15/15 Held-Out Comparison (Mean ± SD, 95% CI)",
        "",
        "| Model | Accuracy (Mean ± SD) | 95% CI | Macro-F1 | ECE |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for model, d in sorted(five_seed.items(), key=lambda kv: -kv[1]["accuracy"]["mean"]):
        acc, f1, ece = d["accuracy"], d["macro_f1"], d["ece"]
        lines.append(
            f"| {model} | {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}% | "
            f"[{acc['ci95_low']*100:.2f}%, {acc['ci95_high']*100:.2f}%] | "
            f"{f1['mean']*100:.2f}% | {ece['mean']:.4f} |"
        )

    fn5 = final_results["five_seed"]["friedman_nemenyi_accuracy"]
    lines.extend([
        "",
        f"*5-Seed Friedman test (accuracy): $\\chi^2_F = {fn5['friedman_chi2']:.4f}$, $p = {fn5['friedman_p_value']:.4f}$.*",
        "",
        "### 5-Fold Stratified Cross-Validation (2,200 OOF Samples, Mean ± SD, 95% CI)",
        "",
        "| Model | Fold Accuracy (Mean ± SD) | 95% CI | Pooled OOF Accuracy | Macro-F1 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for model, d in sorted(five_fold.items(), key=lambda kv: -kv[1]["fold_accuracy"]["mean"]):
        acc = d["fold_accuracy"]
        f1 = d["fold_macro_f1"]
        oof = d.get("pooled_oof", {}).get("oof_accuracy", acc["mean"])
        lines.append(
            f"| {model} | {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}% | "
            f"[{acc['ci95_low']*100:.2f}%, {acc['ci95_high']*100:.2f}%] | "
            f"{oof*100:.2f}% | {f1['mean']*100:.2f}% |"
        )

    fncv = final_results["five_fold_cv"]["friedman_nemenyi_accuracy"]
    lines.extend([
        "",
        f"*5-Fold CV Friedman test (accuracy): $\\chi^2_F = {fncv['friedman_chi2']:.4f}$, $p = {fncv['friedman_p_value']:.4e}$.*",
        "",
    ])

    new_block = "\n".join(lines)
    pattern = re.compile(r"## 📊 Performance & Benchmarks.*?(?=\n---\n)", re.DOTALL)
    if pattern.search(text):
        updated = pattern.sub(lambda _: new_block, text, count=1)
        readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    five_seed_res, _ = load_five_seed()
    five_fold_res, _ = load_five_fold_cv()
    supporting = load_supporting_artifacts()

    final_results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Built strictly from saved artifacts in outputs/ without invented values",
        "five_seed": five_seed_res,
        "five_fold_cv": five_fold_res,
        "supporting": supporting,
    }

    out_path = ROOT / "outputs" / "final_results.json"
    out_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    write_headline_tables(final_results)
    update_readme_from_json(final_results)
    print(f"Wrote {out_path}")
    print(f"5-seed Friedman: chi2={five_seed_res['friedman_nemenyi_accuracy']['friedman_chi2']:.4f}, p={five_seed_res['friedman_nemenyi_accuracy']['friedman_p_value']:.4f}")
    print(f"5-fold CV Friedman: chi2={five_fold_res['friedman_nemenyi_accuracy']['friedman_chi2']:.4f}, p={five_fold_res['friedman_nemenyi_accuracy']['friedman_p_value']:.4e}")


if __name__ == "__main__":
    main()
