"""Recheck saved preparation identities, group isolation and train-only transforms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper2.restart.audit_cereal_legume import digest
from paper2.restart.data_pipeline import check_partition, fit_preprocessor, object_digest, transform, support_flags


def verify(folder, root):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sources"].items():
        if digest(root/"KG-Hybrid/data/raw"/name) != expected["sha256"]:
            raise ValueError("Source identity mismatch")
    for name, expected in manifest["code_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Preparation code identity mismatch")
    if digest(root/"paper2/results/restart_feasibility_v1/audit.json") != manifest["audit_sha256"]:
        raise ValueError("Audit changed")
    if digest(folder/"regression_rows.csv") != manifest["rows_sha256"]:
        raise ValueError("Row table changed")
    rows = pd.read_csv(folder/"regression_rows.csv")
    outer_seen, verified = [], 0
    for fold_id, hashes in manifest["assets"].items():
        path = folder/"folds"/fold_id
        if any(digest(path/name) != value for name, value in hashes.items()):
            raise ValueError("Partition/preprocessor/array changed")
        fold = json.loads((path/"partition.json").read_text(encoding="utf-8"))
        state = json.loads((path/"preprocessing.json").read_text(encoding="utf-8"))
        check_partition(rows, fold)
        clients = [i for ids in fold["clients"].values() for i in ids]
        if len(set(clients)) != len(clients) or sorted(clients) != sorted(fold["train"]):
            raise ValueError("Training client coverage mismatch")
        for group, ids in fold["clients"].items():
            if not ids or not rows.loc[ids, "geo_group"].eq(group).all():
                raise ValueError("Client group mismatch")
        if state["fit_row_ids"] != rows.loc[fold["train"], "row_id"].tolist():
            raise ValueError("Preprocessor not bound to training rows")
        base = {k: v for k, v in state.items() if k != "identity"}
        if object_digest(base) != state["identity"] or state["identity"] != fold["preprocessing_identity"]:
            raise ValueError("Preprocessing identity mismatch")
        fresh = fit_preprocessor(rows, fold["train"])
        if fresh["categorical"] != state["categorical"] or fresh["feature_names"] != state["feature_names"]:
            raise ValueError("Training vocabulary/schema differs")
        for name, values in state["numeric"].items():
            for key in ("median", "mean", "scale"):
                if not np.isclose(values[key], fresh["numeric"][name][key], rtol=1e-12, atol=1e-12):
                    raise ValueError("Training numeric statistics differ")
            if values["all_training_missing"] != fresh["numeric"][name]["all_training_missing"]:
                raise ValueError("Missing-feature policy differs")
        names = ["train", "validation"] if fold["kind"] == "inner_selection" else ["train", "test"]
        with np.load(path/"arrays.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {prefix+name for prefix in ("x_", "y_", "ids_") for name in names}:
                raise ValueError("Unexpected arrays or outer-test exposure in inner assets")
            for name in names:
                ids = fold[name]
                np.testing.assert_array_equal(arrays["ids_"+name], ids)
                np.testing.assert_allclose(arrays["x_"+name], transform(rows.loc[ids], state), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(arrays["y_"+name], rows.loc[ids, "target_seed_dm_t_ha"].to_numpy(np.float32), rtol=1e-6, atol=1e-6)
                if name != "train" and fold["support"][name] != support_flags(rows, fold["train"], ids):
                    raise ValueError("Support flags not derived from training taxonomy")
        if fold["kind"] == "outer_refit":
            outer_seen.extend(fold["test"])
        verified += 1
    if len(outer_seen) != len(rows) or len(set(outer_seen)) != len(rows):
        raise ValueError("Outer test folds must cover each row exactly once")
    return {"status": "PASS", "verified_folds": verified, "rows": len(rows),
            "outer_test_row_coverage_once": True, "training_performed": False,
            "manifest_sha256": digest(folder/"manifest.json"), "verifier_sha256": digest(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.data, Path(__file__).resolve().parents[2])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
