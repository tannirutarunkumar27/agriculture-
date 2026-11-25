"""Freeze RR07 development outer-evaluation planning after RR06 finalization."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


RESULTS = ROOT / "paper2/results"
SEEDS = (123, 2024, 3407, 9999, 2718)
TRAINING_SETTINGS = {"seed": 42, "batch_size": 32, "fedprox_mu": 0.01}
SEARCH_SETTINGS = {
    ("sgd", "constant", 32): {"learning_rate": 0.01, "weight_decay": 0.0, "local_steps": 4},
    ("adamw", "cosine", 64): {"learning_rate": 0.003, "weight_decay": 0.0001, "local_steps": 4},
}
RR07_PATHS = (
    ("centralized", "sample_weighted"),
    ("fedavg", "sample_weighted"),
    ("fedprox", "sample_weighted"),
    ("fedavg", "equal_client"),
)
RR06_DIAGNOSTIC_ONLY_PATHS = (("fednova_restricted", "sample_weighted"),)


def make_plan(finalization, data_manifest, seeds=SEEDS):
    if finalization.get("status") != "PASS_FINALIZED_INNER_SELECTION" or finalization.get("rr07_ready") is not True:
        raise ValueError("RR06 finalization is required")
    if tuple(seeds) != SEEDS or any(type(seed) is not int for seed in seeds):
        raise ValueError("RR07 seed set is frozen")
    selections = finalization["selection"]
    outer_groups = sorted({row["outer_fold"] for row in selections})
    if len(outer_groups) != 8:
        raise ValueError("Expected eight development outer groups")
    assets = data_manifest.get("assets", {})
    observed_paths = {(row["algorithm"], row["aggregation_objective"]) for row in selections}
    expected_paths = set(RR07_PATHS) | set(RR06_DIAGNOSTIC_ONLY_PATHS)
    if observed_paths != expected_paths:
        raise ValueError("RR06 selections do not match the prospectively fixed RR07 path set")
    plan = []
    for choice in sorted(selections, key=lambda row: (row["outer_fold"], row["algorithm"], row["aggregation_objective"])):
        if (choice["algorithm"], choice["aggregation_objective"]) not in RR07_PATHS:
            continue
        outer = choice["outer_fold"]
        if outer not in assets:
            raise ValueError("Missing outer refit fold")
        selected = choice["selected"]
        key = (selected["optimizer"], selected["schedule"], selected["hidden"])
        if key not in SEARCH_SETTINGS:
            raise ValueError("RR06 selection is not one of the frozen RR07 search configurations")
        for seed in seeds:
            plan.append({"id": f"{outer}_{choice['algorithm']}_{choice['aggregation_objective']}_{selected['optimizer']}_seed{seed}",
                         "fold": outer, "seed": seed, "algorithm": choice["algorithm"],
                         "aggregation_objective": choice["aggregation_objective"], "optimizer": selected["optimizer"],
                         "hidden": selected["hidden"], "schedule": selected["schedule"],
                         "rounds": choice["refit_rounds"], "variant": f"{choice['algorithm']}_{choice['aggregation_objective']}",
                         "outer_test_inference": False, **SEARCH_SETTINGS[key]})
    if len(plan) != 160 or len({row["id"] for row in plan}) != len(plan):
        raise ValueError("Unexpected RR07 plan")
    return plan


def freeze(output, finalization_path, data=RESULTS / "restart_public_data_v1"):
    output, finalization_path, data = Path(output).resolve(), Path(finalization_path).resolve(), Path(data).resolve()
    if RESULTS not in output.parents or output.exists():
        raise ValueError("Output must be a new Paper 2 results directory")
    finalization = read(finalization_path / "finalization.json")
    data_manifest = read(data / "manifest.json")
    plan = make_plan(finalization, data_manifest)
    output.mkdir(parents=True); (output / "source_snapshot").mkdir()
    shutil.copyfile(Path(__file__), output / "source_snapshot" / Path(__file__).name)
    atomic_json(output / "config.json", {"version": "rr07-development-outer-v2", "seeds": list(SEEDS),
                "training_settings": TRAINING_SETTINGS,
                "scope": "development_outer_evaluation_only", "outer_test_inference": False,
                "knowledge_enabled": False, "privacy_enabled": False, "ranking_enabled": False,
                "primary": "equal-geographic-group RMSE/MAE", "secondary": ["pooled RMSE/MAE", "support panels"],
                "rr07_paths": [list(path) for path in RR07_PATHS],
                "rr06_diagnostic_only_paths_excluded_before_results": [list(path) for path in RR06_DIAGNOSTIC_ONLY_PATHS],
                "limitations": ["outer groups have prior exposure", "not independent confirmation", "seeds are computational repeats",
                                "restricted FedNova remains an RR06 inner-only diagnostic and is excluded by the predeclared RR07 path set"]})
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "manifest.json", {"status": "FROZEN_RR07_PLAN_ONLY_NO_OUTER_INFERENCE", "planned_runs": len(plan),
                "runtime": runtime(), "data": str(data), "data_manifest_sha256": sha(data / "manifest.json"),
                "finalization_sha256": sha(finalization_path / "finalization.json"),
                "source_sha256": sha(Path(__file__)), "artifacts": {name: sha(output / name) for name in ("config.json", "plan.json", "source_snapshot/rr07_development_protocol.py")}})
    return {"status": "FROZEN_RR07_PLAN_ONLY_NO_OUTER_INFERENCE", "planned_runs": len(plan)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--finalization", type=Path, required=True)
    args = parser.parse_args(); print(freeze(args.output, args.finalization))
