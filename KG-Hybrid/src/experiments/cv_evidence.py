"""Validate OOF identity, fold isolation, labels, and saved metric agreement."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def validate_oof_frame(frame, raw_data, models):
    required = {"sample_id", "fold_id", "model_id", "true_label", "predicted_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF columns missing: {sorted(required - set(frame.columns))}")
    labels = LabelEncoder().fit_transform(raw_data["label"])
    expected_ids = set(range(len(raw_data)))
    expected_folds = np.empty(len(raw_data), dtype=int)
    for fold, (_, ids) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(raw_data, labels), 1):
        expected_folds[ids] = fold
    if set(frame.model_id.unique()) != set(models):
        raise ValueError("OOF model coverage differs from the expected protocol")
    for name in models:
        group = frame[frame.model_id == name]
        if len(group) != len(labels) or group.sample_id.duplicated().any() or set(group.sample_id) != expected_ids:
            raise ValueError(f"Incomplete or duplicate sample identity for {name}")
        for column in ("sample_id", "fold_id", "true_label", "predicted_label"):
            values = group[column].to_numpy()
            if not np.isfinite(values).all() or not np.all(values == np.floor(values)):
                raise ValueError(f"Non-integer or nonfinite {column} for {name}")
        ids = group.sample_id.to_numpy(dtype=int)
        if not np.array_equal(group.true_label, labels[ids]):
            raise ValueError(f"OOF labels do not match raw sample identities for {name}")
        if not np.array_equal(group.fold_id, expected_folds[ids]):
            raise ValueError(f"OOF fold assignment does not match the shared stratified protocol for {name}")
        if ((group.predicted_label < 0) | (group.predicted_label >= len(np.unique(labels)))).any():
            raise ValueError(f"Invalid predicted class for {name}")
    return {"sample_ids_match_raw_dataset": True, "true_labels_match_raw_dataset": True,
            "all_models_share_expected_fold_assignments": True, "n_models": len(models), "n_rows": len(frame)}


def validate_cv_directory(directory, root, models):
    directory, root = Path(directory), Path(root)
    frame = pd.read_csv(directory / "oof_predictions/oof_predictions.csv")
    raw = pd.read_csv(root / "data/raw/Crop_recommendation.csv")
    evidence = validate_oof_frame(frame, raw, models)
    for fold in range(1, 6):
        metrics = json.loads((directory / "per_fold" / f"fold_{fold}.json").read_text(encoding="utf-8"))
        for name in models:
            group = frame[(frame.model_id == name) & (frame.fold_id == fold)]
            recorded = metrics[name]
            for metric, recomputed in (
                ("accuracy", accuracy_score(group.true_label, group.predicted_label)),
                ("f1_macro", f1_score(group.true_label, group.predicted_label, average="macro")),
            ):
                if not np.isclose(recorded[metric], recomputed, atol=1e-6, rtol=0):
                    raise ValueError(f"Saved {metric} disagrees with OOF for {name}, fold {fold}")
            cm = confusion_matrix(group.true_label, group.predicted_label, labels=np.arange(raw.label.nunique()))
            if not np.array_equal(recorded["confusion_matrix"], cm):
                raise ValueError(f"Saved confusion matrix disagrees with OOF for {name}, fold {fold}")
    evidence["all_fold_metrics_match_saved_predictions"] = True
    return evidence
