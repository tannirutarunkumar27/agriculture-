"""Record conservative prediction-time availability for Paper 2 features."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.validation_search import ROOT, atomic_json, read, sha


DATA = ROOT / "paper2/results/restart_regression_data_v1"


def registry(manifest):
    numeric = manifest["numeric_features"]
    categorical = manifest["categorical_features"]
    fixed = manifest["fixed_features"]
    rows = []
    initial = {"soil_awc", "soil_clay", "soil_pH", "nitrogen_residual"}
    management = {"N_amount", "P_amount", "irrigation_amount", "pest_control"}
    planned = {"previous_crop", "crop_season", "sowing_sin", "sowing_cos"}
    for name in numeric + categorical + fixed:
        if name in initial:
            timing, status, claim = "initial_context_candidate", "UNVERIFIED_AVAILABILITY", "retrospective_only"
        elif name in management:
            timing, status, claim = "recorded_applied_management", "RETROSPECTIVE_ONLY", "retrospective_only"
        elif name in planned:
            timing, status, claim = "planned_context_candidate", "UNVERIFIED_AVAILABILITY", "retrospective_only"
        elif name.startswith("contains_") or name == "is_intercrop":
            timing, status, claim = "specified_system_context", "CULTIVAR_DENSITY_UNRESOLVED", "retrospective_only"
        else:
            timing, status, claim = "unknown", "EXCLUDED_UNRESOLVED", "disabled"
        rows.append({"feature": name, "timing_class": timing, "status": status,
                     "allowed_claim": claim, "evidence_required": "documented availability at prediction cutoff"})
    return {"status": "FEATURE_TIMING_AUDIT_NO_MODEL_CHANGE", "target": manifest["target"],
            "feature_timing_statement": manifest["feature_timing"], "features": rows,
            "pre_sowing_recommendation_enabled": False,
            "unresolved": ["cultivar", "planting_density", "mixture_proportion", "fertilizer_timing", "irrigation_timing", "pest_control_timing"],
            "interpretation": "Feature availability is not inferred from column presence; current Paper 2 remains retrospective conditional regression."}


def audit(output, data=DATA):
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists() or ROOT / "paper2/results" not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    manifest = read(data / "manifest.json")
    result = registry(manifest); output.mkdir(parents=True)
    atomic_json(output / "registry.json", result)
    atomic_json(output / "manifest.json", {"status": result["status"], "data_manifest_sha256": sha(data / "manifest.json"),
                "registry_sha256": sha(output / "registry.json"), "model_training": False, "recommendation_enabled": False})
    return {"status": result["status"], "features": len(result["features"]), "pre_sowing_recommendation_enabled": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(audit(args.output))
