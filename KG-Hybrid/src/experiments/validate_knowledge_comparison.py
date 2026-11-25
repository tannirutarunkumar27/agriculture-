"""Independently audit completed comparative predictions and saved provenance."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.experiments.calibration import _validate_inputs, compute_ece, compute_brier_score, compute_nll


def holm_adjust(pvalues):
    """Holm familywise adjustment; retain NaN for unavailable comparisons."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind="stable")]
    if len(order):
        adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return adjusted


def add_adjusted_comparisons(directory):
    path = Path(directory) / "paired_comparisons.csv"
    paired = pd.read_csv(path)
    for test in ("paired_t", "wilcoxon"):
        paired[f"{test}_holm_pvalue"] = holm_adjust(paired[f"{test}_pvalue"])
    paired["multiplicity_family"] = "All recorded model x metric comparisons, separately per test"
    paired["interpretation"] = "Five repeated splits overlap; exploratory paired tests, not independent field trials"
    paired.to_csv(path, index=False)
    return paired


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Comparison run is incomplete")
    raw = pd.read_csv(ROOT / "data/raw/Crop_recommendation.csv")
    features = manifest["features"]
    labels = LabelEncoder().fit_transform(raw.label)
    records = pd.read_csv(directory / "per_seed_metrics.csv")
    expected = {(seed, model) for seed in manifest["seeds"]
                for model in ["XGBoost", "FTTransformer", *manifest["variants"]]}
    if len(records) != len(expected) or set(zip(records.seed, records.model)) != expected:
        raise ValueError("Missing or duplicate model/seed results")
    checked = 0
    for seed in manifest["seeds"]:
        location = directory / f"seed_{seed}"
        parts = np.load(location / "partitions.npz")
        train, val, test = (parts[f"{name}_ids"] for name in ("train", "val", "test"))
        ids = np.concatenate((train, val, test))
        if len(ids) != len(raw) or not np.array_equal(np.sort(ids), np.arange(len(raw))):
            raise ValueError(f"Partitions overlap or omit raw samples for seed {seed}")
        from sklearn.model_selection import train_test_split
        expected_train, temporary = train_test_split(np.arange(len(raw)), test_size=.3, random_state=seed, stratify=labels)
        expected_val, expected_test = train_test_split(temporary, test_size=.5, random_state=seed, stratify=labels[temporary])
        for observed, expected_ids in zip((train, val, test), (expected_train, expected_val, expected_test)):
            np.testing.assert_array_equal(observed, expected_ids)
        # Establish that the saved scaler was fitted to training rows only.
        training_features = raw.iloc[train][features].to_numpy(dtype=float)
        np.testing.assert_allclose(parts["scaler_mean"], training_features.mean(0), atol=1e-10)
        np.testing.assert_allclose(parts["scaler_scale"], training_features.std(0), atol=1e-10)
        for record in records[records.seed == seed].to_dict("records"):
            name = record["model"]
            predictions = pd.read_csv(location / f"{name}_predictions.csv")
            probabilities = np.load(location / f"{name}_probabilities.npy")
            _validate_inputs(labels[test], probabilities)
            if probabilities.shape != (len(test), 22):
                raise ValueError(f"Invalid probability dimensions: {seed}/{name}")
            np.testing.assert_array_equal(predictions.sample_id, test)
            np.testing.assert_array_equal(predictions.y_true, labels[test])
            np.testing.assert_array_equal(predictions.y_pred, probabilities.argmax(1))
            np.testing.assert_allclose(predictions.confidence, probabilities.max(1), atol=1e-7)
            metrics = dict(accuracy=accuracy_score(labels[test], probabilities.argmax(1)),
                           macro_f1=f1_score(labels[test], probabilities.argmax(1), average="macro"),
                           ece=compute_ece(labels[test], probabilities),
                           brier=compute_brier_score(labels[test], probabilities),
                           nll=compute_nll(labels[test], probabilities))
            for metric, actual in metrics.items():
                if not np.isclose(record[metric], actual, atol=1e-7):
                    raise ValueError(f"Metric mismatch: {seed}/{name}/{metric}")
            if name != "XGBoost":
                state = torch.load(location / f"{name}.pth", map_location="cpu", weights_only=True)
                if "K" in name and name != "Hybrid_NoK":
                    prior = state["fusion.knowledge_prior"]
                    learned = state["fusion.knowledge_matrix"]
                    delta = (learned - prior).abs()
                    np.testing.assert_allclose(record["mean_abs_change"], delta.mean().item(), atol=1e-8)
                    np.testing.assert_allclose(record["max_abs_change"], delta.max().item(), atol=1e-8)
                    learned_csv = pd.read_csv(location / f"{name}_learned_matrix.csv", index_col=0)
                    np.testing.assert_allclose(learned_csv.to_numpy(), learned.numpy(), atol=1e-7)
                    if name in ("Hybrid_FixedK", "Hybrid_ShuffledK_Gate") and not torch.equal(prior, learned):
                        raise ValueError("A fixed prior was modified during training")
            checked += 1
    summary = records.groupby("model")[["accuracy", "macro_f1", "ece", "brier", "nll", "trainable_parameters"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.to_csv(directory / "summary_flat.csv")
    add_adjusted_comparisons(directory)
    result = {"status": "PASS", "validated_model_seed_results": checked,
              "validated_test_predictions": int(records.n_test.sum()),
              "raw_sample_ids_labels_and_shared_partitions": "validated",
              "training_only_scaler_statistics": "validated",
              "probabilities_metrics_and_prior_drift": "validated",
              "limitation": "Repeated splits overlap; source authority and field generalization are not established"}
    (directory / "validation_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/knowledge_comparison_review_fixes")
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
