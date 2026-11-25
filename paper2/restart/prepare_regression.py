"""Create immutable new-dataset rows, nested split IDs and training-only transforms."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from paper2.restart.audit_cereal_legume import FILES, digest, harvest_components, system_outcomes
from paper2.restart.data_pipeline import (NUMERIC, CATEGORICAL, SPECIES, FIXED, build_rows, geographic_groups,
                           nested_partitions, fit_preprocessor, transform, support_flags)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    raw = root/"KG-Hybrid/data/raw"
    audit_path = root/"paper2/results/restart_feasibility_v1/audit.json"
    if output.exists():
        raise FileExistsError(output)
    if output == root or output in raw.parents or raw in output.parents:
        raise ValueError("Output must be separate from raw/root")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if any(digest(raw/name) != audit["sources"][name]["sha256"] for name in FILES):
        raise ValueError("Raw dataset changed from audited version")
    if digest(Path(__file__).with_name("audit_cereal_legume.py")) != audit["audit_code_sha256"]:
        raise ValueError("Audited extraction implementation changed")
    trials, management, traits = [pd.read_csv(raw/name) for name in FILES[:3]]
    components, harvesting = harvest_components(management, traits)
    rows = build_rows(system_outcomes(components), trials)
    mapping, pair_distances = geographic_groups(trials, radius_km=1.0)
    rows["geo_group"] = rows.site.map(mapping)
    partitions = nested_partitions(rows)
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output/"regression_rows.csv", index=False)
    pd.DataFrame(pair_distances).to_csv(output/"site_distances.csv", index=False)
    summary, assets = [], {}
    for fold in partitions:
        folder = output/"folds"/fold["id"]
        folder.mkdir(parents=True)
        state = fit_preprocessor(rows, fold["train"])
        # Inner assets intentionally never transform/save the outer test tensors.
        names = ["train", "validation"] if fold["kind"] == "inner_selection" else ["train", "test"]
        arrays = {}
        for name in names:
            ids = fold[name]
            arrays["x_"+name] = transform(rows.loc[ids], state)
            arrays["y_"+name] = rows.loc[ids, "target_seed_dm_t_ha"].to_numpy(np.float32)
            arrays["ids_"+name] = np.array(ids, dtype=np.int64)
        np.savez_compressed(folder/"arrays.npz", **arrays)
        fold["preprocessing_identity"] = state["identity"]
        fold["row_table_sha256"] = digest(output/"regression_rows.csv")
        fold["support"] = {name: support_flags(rows, fold["train"], fold[name]) for name in names if name != "train"}
        write_json(folder/"partition.json", fold)
        write_json(folder/"preprocessing.json", state)
        summary.append({"fold": fold["id"], "kind": fold["kind"], "test_group": fold["test_group"],
                        "validation_group": fold["validation_group"], "train_rows": len(fold["train"]),
                        "validation_rows": len(fold["validation"]), "test_rows": len(fold["test"]),
                        "training_clients": len(fold["clients"]), "feature_dimensions": len(state["feature_names"])})
        assets[fold["id"]] = {name: digest(folder/name) for name in ("arrays.npz", "partition.json", "preprocessing.json")}
    pd.DataFrame(summary).to_csv(output/"split_summary.csv", index=False)
    manifest = {"status": "PREPARED_REGRESSION_DATA_NO_MODEL_TRAINING", "rows": len(rows), "source_sites": int(rows.site.nunique()),
                "geographic_groups": int(rows.geo_group.nunique()), "outer_folds": sum(f["kind"] == "outer_refit" for f in partitions),
                "inner_folds": sum(f["kind"] == "inner_selection" for f in partitions), "site_mapping": mapping,
                "group_rule": "same source site OR any cross-site coordinates <=1km, transitive union",
                "group_review_limit": "coordinate-based conservative grouping, not independent field-boundary certification",
                "target": "sum harvest seed dry matter t/ha, retrospective conditional outcome",
                "numeric_features": list(NUMERIC), "categorical_features": list(CATEGORICAL), "public_taxonomy": list(SPECIES),
                "fixed_features": list(FIXED), "feature_timing": "initial context plus recorded management; NOT pre-sowing advice",
                "excluded": ["soil_sand and soil_silt: swapped metadata descriptions", "all realized climate", "actual harvest dates",
                             "measured crop traits", "site/country/coordinates/year/row IDs as predictors", "cultivar/density: unresolved controls"],
                "all_missing_numeric_policy": "training-only zero placeholder plus missingness flag and explicit all-training-missing state",
                "privacy": "NONE; centralized public training-only preprocessing reference",
                "training_enabled": False, "ranking_enabled": False, "knowledge_enabled": False,
                "harvest_extraction": harvesting, "audit_sha256": digest(audit_path), "sources": audit["sources"],
                "rows_sha256": digest(output/"regression_rows.csv"), "assets": assets,
                "code_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("data_pipeline.py"))},
                "runtime": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("pandas", "numpy", "openpyxl", "pypdf")}}}
    if any(digest(raw/name) != audit["sources"][name]["sha256"] for name in FILES):
        raise RuntimeError("Source changed during preparation")
    write_json(output/"manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "rows", "source_sites", "geographic_groups", "outer_folds", "inner_folds")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[2], args.output)
