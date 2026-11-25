"""Read-only scientific inventory of the frozen regression rows and partitions.

This report does not train models or reinterpret the continuous target as a
crop-class label. Output paths are versioned and must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from paper2.restart.data_pipeline import CATEGORICAL, FIXED, NUMERIC, SPECIES
from paper2.restart.verify_regression_data import verify


FEATURE_UNITS = {
    "soil_awc": "mm", "soil_clay": "fraction", "soil_pH": "unitless",
    "nitrogen_residual": "kg/ha", "N_amount": "kg/ha", "P_amount": "kg/ha",
    "irrigation_amount": "mm", "sowing_sin": "unitless", "sowing_cos": "unitless",
    "previous_crop": "category", "crop_season": "category", "pest_control": "category",
    **{f"contains_{name}": "binary" for name in SPECIES},
    "is_intercrop": "binary",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def client_profile(rows: pd.DataFrame, indices: list[int]) -> dict:
    subset = rows.loc[indices]
    result = {
        "n_systems": len(subset),
        "n_experiments": int(subset.experiment_id.nunique()),
        "sole_systems": int(subset.crop_type.eq("SC").sum()),
        "intercrop_systems": int(subset.crop_type.eq("IC").sum()),
        "target_median_t_ha": float(subset.target_seed_dm_t_ha.median()),
    }
    for name in SPECIES:
        result[f"contains_{name}_systems"] = int(subset[f"contains_{name}"].sum())
    return result


def collect(data: Path, root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, root = data.resolve(), root.resolve()
    verified = verify(data, root)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(data / "regression_rows.csv")
    features = list(NUMERIC + CATEGORICAL + FIXED)
    if set(features) != set(FEATURE_UNITS) or len(rows) != 607:
        raise ValueError("Unexpected frozen feature or row contract")
    if rows.row_id.duplicated().any() or rows.duplicated(["experiment_id", "management"]).any():
        raise ValueError("Duplicate system identity")
    if not rows.target_seed_dm_t_ha.notna().all():
        raise ValueError("Missing regression target")

    feature_table = pd.DataFrame([
        {"feature": name, "kind": "numeric" if name in NUMERIC else "categorical" if name in CATEGORICAL else "fixed_binary",
         "unit": FEATURE_UNITS[name], "missing_systems": int(rows[name].isna().sum())}
        for name in features
    ])
    groups = sorted(rows.geo_group.unique())
    group_table = pd.DataFrame([
        {"geo_group": group, **client_profile(rows, rows.index[rows.geo_group.eq(group)].tolist())}
        for group in groups
    ])
    client_records = []
    fold_counts = {"outer_refit": 0, "inner_selection": 0}
    for fold_id in sorted(manifest["assets"]):
        fold = json.loads((data / "folds" / fold_id / "partition.json").read_text(encoding="utf-8"))
        fold_counts[fold["kind"]] += 1
        if set(fold["clients"]) != set(groups) - {fold["test_group"]} - ({fold["validation_group"]} if fold["validation_group"] else set()):
            raise ValueError(f"Client roster does not match held-out groups: {fold_id}")
        for group, indices in sorted(fold["clients"].items()):
            client_records.append({
                "fold_id": fold_id, "kind": fold["kind"], "test_group": fold["test_group"],
                "validation_group": fold["validation_group"] or "", "client_group": group,
                **client_profile(rows, indices),
            })
    client_table = pd.DataFrame(client_records)
    if fold_counts != {"outer_refit": 8, "inner_selection": 56} or len(client_table) != 392:
        raise ValueError("Unexpected fold or client count")
    if int(group_table.n_systems.sum()) != len(rows) or int(group_table.n_experiments.sum()) != rows.experiment_id.nunique():
        raise ValueError("Group systems or experiments do not partition the source")
    if int(group_table.sole_systems.sum()) != 299 or int(group_table.intercrop_systems.sum()) != 308:
        raise ValueError("Unexpected recorded system composition")
    audit = {
        "status": "PASS_DATASET_CLIENT_AUDIT",
        "dataset": "Gaudio et al. cereal-legume field experiments, Zenodo 14160885, version 2.0.0",
        "source_manifest_sha256": sha256(data / "manifest.json"),
        "regression_rows_sha256": sha256(data / "regression_rows.csv"),
        "systems": len(rows), "experiments": int(rows.experiment_id.nunique()),
        "geographic_groups": len(groups), "conceptual_input_features": len(features),
        "crop_classes": None, "target": "continuous harvest seed dry matter, t/ha",
        "duplicate_row_ids": 0, "duplicate_experiment_management_keys": 0,
        "folds": fold_counts, "client_fold_records": len(client_table),
        "partition_verification": verified,
        "split_rule": "Leave one of eight geographic groups out; inner validation holds a second group out; remaining groups are simulated clients",
        "training_or_inference_performed": False,
        "caveats": [
            "Client species counts are multi-label system membership, not crop-class labels.",
            "The public study simulates clients centrally; it is not an operational federation.",
            "Recorded management makes this retrospective conditional regression, not pre-sowing recommendation.",
            "Outer-test outcomes have prior development exposure; these split checks cannot restore an untouched test set.",
        ],
    }
    return audit, feature_table, group_table, client_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    audit, features, groups, clients = collect(args.data, root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    features.to_csv(args.output / "dataset_features.csv", index=False)
    groups.to_csv(args.output / "geographic_groups.csv", index=False)
    clients.to_csv(args.output / "fold_clients.csv", index=False)
    print(json.dumps({"status": audit["status"], "systems": audit["systems"],
                      "groups": audit["geographic_groups"], "client_fold_records": audit["client_fold_records"]}))


if __name__ == "__main__":
    main()
