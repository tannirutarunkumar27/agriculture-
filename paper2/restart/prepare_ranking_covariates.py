"""Recover cultivar/density/mixture covariates without enabling causal ranking."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2.restart.validation_search import ROOT, atomic_json, sha

RESULTS = ROOT / "paper2/results"
ROWS = RESULTS / "restart_regression_data_v1/regression_rows.csv"
MANAGEMENT = ROOT / "KG-Hybrid/data/raw/data_management.csv"
KEYS = ["experiment_id", "management"]
COMPONENT_FIELDS = ["species", "species_mix", "cultivar", "cultivar_mix", "density_level",
                    "density_relative", "density_factor", "interrow", "mixing_pattern", "mixture_design"]


def coverage(rows: pd.DataFrame, management: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required = set(KEYS + ["row_id", "crop_type"])
    if not required <= set(rows) or not set(KEYS + COMPONENT_FIELDS) <= set(management):
        raise ValueError("Required ranking-covariate columns are absent")
    if rows.duplicated(KEYS).any() or rows["row_id"].duplicated().any():
        raise ValueError("Regression system keys must be unique")
    if management.duplicated(KEYS + ["species"]).any():
        raise ValueError("Component species must be unique within a system")
    merged = management.merge(rows[KEYS + ["row_id", "crop_type"]].rename(columns={"crop_type": "system_crop_type"}),
                              on=KEYS, how="inner", validate="many_to_one")
    if "crop_type" in merged and not merged["crop_type"].astype(str).str.upper().eq(merged["system_crop_type"].astype(str).str.upper()).all():
        raise ValueError("Component and system crop types differ")
    matched = rows[KEYS].merge(management[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True)
    missing_systems = int((matched["_merge"] != "both").sum())
    component_complete = merged[["cultivar", "density_level", "density_relative", "density_factor"]].notna().all(axis=1)
    intercrop = merged["system_crop_type"].astype(str).str.upper().eq("IC")
    mixture_complete = merged.loc[intercrop, ["mixture_design", "mixing_pattern"]].notna().all(axis=1)
    report = {
        "status": "RANKING_COVARIATES_RECOVERED_DESIGN_REVIEW_REQUIRED",
        "systems": int(len(rows)), "component_rows": int(len(merged)),
        "unmatched_systems": missing_systems,
        "complete_component_cultivar_density_rows": int(component_complete.sum()),
        "incomplete_component_cultivar_density_rows": int((~component_complete).sum()),
        "intercrop_component_rows": int(intercrop.sum()),
        "complete_intercrop_design_rows": int(mixture_complete.sum()),
        "ranking_enabled": False,
        "remaining_gate": "Original experiment randomization, treatment comparability, references, and restricted utility protocol require independent review.",
    }
    if missing_systems or not component_complete.all() or not mixture_complete.all():
        report["status"] = "RANKING_COVARIATES_INCOMPLETE"
    merged["crop_type"] = merged["system_crop_type"]
    return report, merged[KEYS + ["row_id", "crop_type"] + COMPONENT_FIELDS].sort_values(KEYS + ["species"])


def prepare(output: Path, rows_path=ROWS, management_path=MANAGEMENT) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    rows_path, management_path = Path(rows_path).resolve(), Path(management_path).resolve()
    report, components = coverage(pd.read_csv(rows_path), pd.read_csv(management_path))
    output.mkdir(parents=True)
    components.to_csv(output / "component_covariates.csv", index=False)
    atomic_json(output / "coverage.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "ranking_enabled": False, "model_training": False,
        "rows_sha256": sha(rows_path), "management_sha256": sha(management_path),
        "components_sha256": sha(output / "component_covariates.csv"), "source_sha256": sha(Path(__file__)),
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(prepare(args.output))
