"""Stage A Baselines: LightGBM, CatBoost, and DecisionTree (no max_depth=6 cap).

Runs under the exact same 5-fold StratifiedKFold protocol as run_cv.py and the
exact same 5-seed 70/15/15 protocol as run_knowledge_comparison.py, writing all
outputs to a new directory (outputs/stage_a_baselines/) without modifying any
historical evidence files.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation.run_five_seed_ablation import load_raw_data, prepare_seed_partitions
from src.experiments.calibration import compute_brier_score, compute_ece, compute_nll

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEEDS = [42, 123, 2024, 3407, 9999]


def _make_models(seed: int) -> dict[str, object]:
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=2,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=seed,
            thread_count=2,
            verbose=0,
        ),
        "DecisionTree_Unconstrained": DecisionTreeClassifier(
            max_depth=None,
            random_state=seed,
        ),
    }


def _eval_record(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "ece": float(compute_ece(y_true, probs)),
        "brier": float(compute_brier_score(y_true, probs)),
        "nll": float(compute_nll(y_true, probs)),
    }


def run_five_seed(df: pd.DataFrame, out_dir: Path) -> list[dict]:
    five_seed_dir = out_dir / "five_seed"
    five_seed_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for seed in SEEDS:
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = prepare_seed_partitions(df, seed)
        seed_dir = five_seed_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for name, clf in _make_models(seed).items():
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            probs = clf.predict_proba(X_test)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            np.save(seed_dir / f"{name}_probabilities.npy", probs)
            metrics = _eval_record(y_test, probs)
            rec = {
                "seed": seed,
                "model": name,
                "n_test": int(len(y_test)),
                **metrics,
                "training_seconds": float(train_sec),
                "test_batch_infer_ms": float(infer_ms),
            }
            records.append(rec)

    pd.DataFrame(records).to_csv(five_seed_dir / "per_seed_metrics.csv", index=False)
    return records


def run_five_fold_cv(df: pd.DataFrame, out_dir: Path) -> list[dict]:
    cv_dir = out_dir / "five_fold_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    X_df = df[FEATURES].copy()
    le = LabelEncoder()
    y_all = le.fit_transform(df["label"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records: list[dict] = []
    oof_probs: dict[str, np.ndarray] = {
        name: np.zeros((len(df), len(le.classes_)), dtype=np.float64)
        for name in _make_models(42)
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y_all), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_df.iloc[train_idx].values).astype(np.float32)
        X_val = scaler.transform(X_df.iloc[val_idx].values).astype(np.float32)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        for name, clf in _make_models(42).items():
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_sec = time.perf_counter() - t0

            probs = clf.predict_proba(X_val)
            oof_probs[name][val_idx] = probs
            metrics = _eval_record(y_val, probs)
            fold_records.append({
                "fold": fold_idx,
                "model": name,
                "n_val": int(len(y_val)),
                **metrics,
                "training_seconds": float(train_sec),
            })

    oof_summary = []
    for name, probs in oof_probs.items():
        np.save(cv_dir / f"{name}_oof_probabilities.npy", probs)
        metrics = _eval_record(y_all, probs)
        oof_summary.append({
            "model": name,
            "oof_rows": int(len(y_all)),
            **metrics,
        })

    pd.DataFrame(fold_records).to_csv(cv_dir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(oof_summary).to_csv(cv_dir / "cv_summary.csv", index=False)
    return oof_summary


def main() -> None:
    out_dir = ROOT / "outputs" / "stage_a_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()

    five_seed_records = run_five_seed(df, out_dir)
    cv_summary = run_five_fold_cv(df, out_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/raw/Crop_recommendation.csv",
        "n_rows": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "five_seed_protocol": {
            "seeds": SEEDS,
            "split": "70/15/15 stratified via prepare_seed_partitions",
        },
        "five_fold_cv_protocol": {
            "splitter": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "preprocessing": "StandardScaler fit strictly on fold-training split",
        },
        "tuning_budgets": {
            "LightGBM": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "note": "Matched tree count, depth, learning rate, and subsampling budget to XGBoost",
            },
            "CatBoost": {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.1,
                "note": "Matched tree count, depth, and learning rate budget to XGBoost",
            },
            "DecisionTree_Unconstrained": {
                "max_depth": None,
                "criterion": "gini",
                "note": "Unconstrained depth (replaces historical max_depth=6 cap)",
            },
        },
        "five_fold_cv_summary": cv_summary,
        "five_seed_records_count": len(five_seed_records),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Stage A baselines complete. CV Summary:")
    print(pd.DataFrame(cv_summary).to_string(index=False))


if __name__ == "__main__":
    main()
