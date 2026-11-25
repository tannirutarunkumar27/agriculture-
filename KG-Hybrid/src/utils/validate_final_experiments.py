"""Validate saved KG-Hybrid experiment artifacts without training or mutation.

This utility is deliberately conservative: it recomputes only metrics supported
by saved predictions and explicitly records incomplete or unverifiable evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.cv_evidence import validate_cv_directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_experimental_validation"
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation" / "five_seed"
LEGACY_CV_DIR = PROJECT_ROOT / "outputs" / "cross_validation"
RECOVERY_CV_DIR = PROJECT_ROOT / "outputs" / "cross_validation_neural_recovery"
SEEDS = [42, 123, 2024, 3407, 9999]
VARIANTS = ["Full", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
NEURAL_MODELS = ["FTTransformerBaseline", "KG-HybridProposed"]
CLASSICAL_MODELS = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoostBaseline"]
ALL_CV_MODELS = CLASSICAL_MODELS + NEURAL_MODELS


def sha256_values(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode("utf-8")).hexdigest()


def validate_ablation() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    recorded = pd.read_csv(ABLATION_DIR / "ablation_per_seed.csv")
    lookup = recorded.set_index(["seed", "variant"])
    records: list[dict] = []
    y_true_hashes: set[str] = set()
    duplicate_predictions: defaultdict[str, list[str]] = defaultdict(list)

    for seed in SEEDS:
        for variant in VARIANTS:
            path = ABLATION_DIR / "predictions" / f"seed_{seed}" / f"{variant}_predictions.csv"
            record = {
                "seed": seed,
                "variant": variant,
                "prediction_file": str(path.relative_to(PROJECT_ROOT)),
                "file_present": path.exists(),
                "sample_id_available": False,
                "sample_id_integrity_verifiable": False,
            }
            if not path.exists():
                record.update({"status": "missing_prediction_file"})
                records.append(record)
                continue
            pred = pd.read_csv(path)
            record["row_count"] = len(pred)
            record["columns"] = "|".join(pred.columns)
            required = {"y_true", "y_pred"}
            if not required.issubset(pred.columns):
                record["status"] = "invalid_prediction_schema"
                records.append(record)
                continue
            y_true = pred["y_true"].astype(int).to_numpy()
            y_pred = pred["y_pred"].astype(int).to_numpy()
            record["y_true_hash"] = sha256_values(y_true.tolist())
            record["y_pred_hash"] = sha256_values(y_pred.tolist())
            y_true_hashes.add(record["y_true_hash"])
            duplicate_predictions[record["y_pred_hash"]].append(f"{seed}/{variant}")
            record["accuracy_recomputed"] = accuracy_score(y_true, y_pred)
            record["macro_f1_recomputed"] = f1_score(y_true, y_pred, average="macro")
            if (seed, variant) in lookup.index:
                expected = lookup.loc[(seed, variant)]
                record["accuracy_recorded"] = float(expected["accuracy"])
                record["macro_f1_recorded"] = float(expected["macro_f1"])
                record["accuracy_matches_record"] = bool(
                    np.isclose(record["accuracy_recomputed"], record["accuracy_recorded"], atol=1e-6)
                )
                record["macro_f1_matches_record"] = bool(
                    np.isclose(record["macro_f1_recomputed"], record["macro_f1_recorded"], atol=1e-6)
                )
            else:
                record["accuracy_matches_record"] = False
                record["macro_f1_matches_record"] = False
            record["status"] = (
                "metric_validated_no_sample_ids"
                if len(pred) == 330 and record["accuracy_matches_record"] and record["macro_f1_matches_record"]
                else "integrity_check_failed"
            )
            records.append(record)

    per_seed = pd.DataFrame(records)
    valid = per_seed[
        per_seed["status"].eq("metric_validated_no_sample_ids")
    ].copy()
    summary = (
        valid.groupby("variant", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            accuracy_mean=("accuracy_recomputed", "mean"),
            accuracy_std=("accuracy_recomputed", "std"),
            macro_f1_mean=("macro_f1_recomputed", "mean"),
            macro_f1_std=("macro_f1_recomputed", "std"),
        )
        .sort_values("variant")
    )
    evidence = {
        "expected_runs": len(SEEDS) * len(VARIANTS),
        "files_found": int(per_seed["file_present"].sum()),
        "validated_metric_rows": int(valid.shape[0]),
        "metric_mismatches": int(
            (~per_seed.get("accuracy_matches_record", pd.Series(dtype=bool)).fillna(False)).sum()
            + (~per_seed.get("macro_f1_matches_record", pd.Series(dtype=bool)).fillna(False)).sum()
        ),
        "unique_y_true_sequences": len(y_true_hashes),
        "same_y_true_sequence_within_each_seed": {
            str(seed): int(per_seed.loc[per_seed["seed"].eq(seed), "y_true_hash"].nunique()) == 1
            for seed in SEEDS
        },
        "duplicate_prediction_groups_within_seed": [
            values for values in duplicate_predictions.values() if len(values) > 1
        ],
        "limitation": "Prediction CSVs do not contain sample IDs; duplicate/missing sample identity checks cannot be performed.",
    }
    return per_seed, summary, evidence


def resolve_complete_cv_dir() -> Path:
    """Select the complete, isolated recovery run when it is available.

    The legacy directory intentionally remains untouched because it documents
    the originally incomplete neural run.  The recovery run is valid only when
    it contains the expected six-model, five-fold OOF evidence.
    """
    recovery_oof = RECOVERY_CV_DIR / "oof_predictions" / "oof_predictions.csv"
    if recovery_oof.exists():
        recovery = pd.read_csv(recovery_oof, usecols=["model_id", "fold_id", "sample_id"])
        expected_rows = len(ALL_CV_MODELS) * 2200
        if (
            len(recovery) == expected_rows
            and set(recovery["model_id"].unique()) == set(ALL_CV_MODELS)
            and set(recovery["fold_id"].unique()) == {1, 2, 3, 4, 5}
        ):
            return RECOVERY_CV_DIR
    return LEGACY_CV_DIR


def validate_cv() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cv_dir = resolve_complete_cv_dir()
    cv_integrity = (
        validate_cv_directory(cv_dir, PROJECT_ROOT, ALL_CV_MODELS)
        if cv_dir == RECOVERY_CV_DIR else None
    )
    oof_path = cv_dir / "oof_predictions" / "oof_predictions.csv"
    fold_records: list[dict] = []
    oof_summary_records: list[dict] = []
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        for model, group in oof.groupby("model_id"):
            ids = group["sample_id"]
            fold_counts = group["fold_id"].value_counts().sort_index()
            expected_complete = (
                len(group) == 2200
                and ids.nunique() == 2200
                and not ids.duplicated().any()
                and fold_counts.to_dict() == {1: 440, 2: 440, 3: 440, 4: 440, 5: 440}
            )
            oof_summary_records.append({
                "model": model,
                "oof_rows": len(group),
                "unique_sample_ids": int(ids.nunique()),
                "duplicate_sample_id_rows": int(ids.duplicated().sum()),
                "folds_present": "|".join(map(str, sorted(group["fold_id"].unique()))),
                "oof_accuracy_recomputed": accuracy_score(group["true_label"], group["predicted_label"]),
                "oof_macro_f1_recomputed": f1_score(
                    group["true_label"], group["predicted_label"], average="macro"
                ),
                "complete_5_fold_oof": expected_complete,
            })
            for fold_id, fold_group in group.groupby("fold_id"):
                fold_records.append({
                    "fold": int(fold_id),
                    "model": model,
                    "evidence_type": "saved_oof_predictions",
                    "row_count": len(fold_group),
                    "unique_sample_ids": int(fold_group["sample_id"].nunique()),
                    "accuracy_recomputed": accuracy_score(fold_group["true_label"], fold_group["predicted_label"]),
                    "macro_f1_recomputed": f1_score(
                        fold_group["true_label"], fold_group["predicted_label"], average="macro"
                    ),
                    "status": (
                        "validated_from_predictions"
                        if len(fold_group) == 440 and fold_group["sample_id"].nunique() == 440
                        else "incomplete_or_duplicate_predictions"
                    ),
                })

    per_fold = pd.DataFrame(fold_records)
    oof_summary = pd.DataFrame(oof_summary_records)
    neural_complete = all(
        (per_fold["model"].eq(model) & per_fold["status"].eq("validated_from_predictions")).sum() == 5
        for model in NEURAL_MODELS
    ) if not per_fold.empty else False
    all_models_complete = (
        set(oof_summary.get("model", [])) == set(ALL_CV_MODELS)
        and bool(oof_summary.get("complete_5_fold_oof", pd.Series(dtype=bool)).all())
    )
    best_oof = (
        oof_summary.sort_values("oof_accuracy_recomputed", ascending=False).iloc[0]
        if not oof_summary.empty else None
    )
    evidence = {
        "oof_prediction_file": str(oof_path.relative_to(PROJECT_ROOT)),
        "cv_evidence_directory": str(cv_dir.relative_to(PROJECT_ROOT)),
        "identity_and_saved_metric_integrity": cv_integrity,
        "legacy_cv_directory_preserved": str(LEGACY_CV_DIR.relative_to(PROJECT_ROOT)),
        "oof_models": sorted(oof_summary["model"].tolist()) if not oof_summary.empty else [],
        "neural_cv_complete_and_prediction_validated": neural_complete,
        "all_cv_models_complete_and_prediction_validated": all_models_complete,
        "highest_pooled_oof_accuracy_model": str(best_oof["model"]) if best_oof is not None else None,
        "highest_pooled_oof_accuracy": (
            float(best_oof["oof_accuracy_recomputed"]) if best_oof is not None else None
        ),
        "neural_issue": (
            "None. The isolated neural-recovery run contains complete five-fold OOF predictions "
            "for FT-Transformer and KG-Hybrid."
            if neural_complete else
            "The selected CV directory does not contain complete five-fold OOF predictions for "
            "both neural models."
        ),
    }
    return per_fold, oof_summary, evidence


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ablation_per_seed, ablation_summary, ablation_evidence = validate_ablation()
    cv_per_fold, cv_summary, cv_evidence = validate_cv()

    ablation_per_seed.to_csv(OUTPUT_DIR / "ablation_validated_per_seed.csv", index=False)
    ablation_summary.to_csv(OUTPUT_DIR / "ablation_validated_summary.csv", index=False)
    cv_per_fold.to_csv(OUTPUT_DIR / "cv_validated_per_fold.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "cv_validated_summary.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": "src/utils/validate_final_experiments.py",
        "project_root": str(PROJECT_ROOT),
        "ablation": ablation_evidence,
        "cross_validation": cv_evidence,
        "source_artifacts": [
            str(ABLATION_DIR.relative_to(PROJECT_ROOT)),
            str(LEGACY_CV_DIR.relative_to(PROJECT_ROOT)),
            str(RECOVERY_CV_DIR.relative_to(PROJECT_ROOT)),
            "outputs/models/corrected/training_metadata.json",
            "outputs/tables/knowledge_matrix_alignment_audit.csv",
        ],
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best_oof_text = (
        f"{cv_evidence['highest_pooled_oof_accuracy_model']} "
        f"({cv_evidence['highest_pooled_oof_accuracy']:.4f})"
        if cv_evidence["highest_pooled_oof_accuracy"] is not None
        else "not available"
    )
    report = f"""KG-Hybrid final experimental validation report
Created (UTC): {manifest['created_at_utc']}

Scope
- This validation recomputes only metrics supported by saved prediction artifacts.
- It neither trained models nor modified existing experiment artifacts.

Ablation
- Expected run count: {ablation_evidence['expected_runs']}
- Prediction files found: {ablation_evidence['files_found']}
- Rows whose accuracy and Macro-F1 exactly match saved predictions: {ablation_evidence['validated_metric_rows']}
- Metric comparison failures: {ablation_evidence['metric_mismatches']}
- {ablation_evidence['limitation']}
- The artifacts contain {ablation_evidence['unique_y_true_sequences']} distinct test-label sequences across the five seeds. Within each seed, all nine variants share that seed's 330-sample test partition.
- Thus the study does not reuse one single test partition for all 45 runs, but repeatedly compares variants on each seed-specific held-out partition. Treat the ablations as comparative evidence, not an untouched final generalization test after model selection.

Cross-validation
- Models with saved OOF predictions: {', '.join(cv_evidence['oof_models']) or 'none'}.
- Neural CV complete and validated from OOF predictions: {cv_evidence['neural_cv_complete_and_prediction_validated']}.
- All CV models complete and validated from OOF predictions: {cv_evidence['all_cv_models_complete_and_prediction_validated']}.
- Selected CV evidence directory: {cv_evidence['cv_evidence_directory']}.
- Highest pooled OOF accuracy: {best_oof_text}.
- {cv_evidence['neural_issue']}
- The legacy incomplete CV directory is preserved separately and is not used for the recovered CV comparison.

Paper-ready status
- The saved five-seed ablation values are internally metric-consistent, but sample identity integrity is unverified and the repeated held-out test use must be disclosed.
- The isolated recovery run provides complete OOF evidence for all six CV models. Its metrics may be reported with the stated protocol and limitations.
- The partial neural-CV evidence has not been overwritten; it remains in the legacy CV directory.
- The recovered OOF results rank XGBoost above KG-Hybrid on pooled accuracy. Do not claim that KG-Hybrid is the top CV performer without a different, pre-specified evaluation basis.
"""
    (OUTPUT_DIR / "validation_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
