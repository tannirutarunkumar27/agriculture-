"""Freeze a matched, inner-only ablation matrix before any new training.

This protocol is deliberately a plan, not a trainer.  It prohibits outer-test
arrays and records the factors that the later RR05/RR06 runner must hold fixed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


RESULTS = ROOT / "paper2/results"
DATA = RESULTS / "restart_public_data_v1"
VARIANTS = (
    ("centralized_sample_weighted", "centralized", "sample_weighted"),
    ("fedavg_sample_weighted", "fedavg", "sample_weighted"),
    ("fedprox_sample_weighted", "fedprox", "sample_weighted"),
    ("fednova_sample_weighted", "fednova_restricted", "sample_weighted"),
    ("fedavg_equal_client", "fedavg", "equal_client"),
)


def config():
    return {
        "version": "matched-ablation-inner-v1",
        "scope": "inner_validation_development_only",
        "target": "conditional_harvest_seed_dry_matter_t_ha",
        "seed": 42,
        "architecture": {"hidden": 32, "output": "scalar_regression"},
        "optimizer": {"name": "plain_sgd", "momentum": 0.0, "weight_decay": 0.0, "learning_rate": 0.01},
        "local_steps": 4,
        "batch_size": 32,
        "rounds": 240,
        "gradient_clip_norm": 1.0,
        "fedprox_mu": 0.01,
        "participation": 1.0,
        "selection": "equal_inner_geographic_group_mean_rmse_then_first_frozen_variant",
        "outer_test_inference": False,
        "knowledge_enabled": False,
        "privacy_enabled": False,
        "ranking_enabled": False,
        "limitations": [
            "all geographic groups have prior outer-test exposure",
            "this protocol is a development ablation, not independent confirmation",
            "equal-client and sample-weighted objectives are distinct estimands",
            "restricted FedNova applies only to plain SGD without momentum or weight decay",
        ],
    }


def make_plan(data_manifest):
    inner = sorted(name for name in data_manifest["assets"] if "_inner_" in name)
    if len(inner) != 56 or len(set(inner)) != 56:
        raise ValueError("Expected exactly 56 inner folds")
    plan = []
    for fold in inner:
        for name, algorithm, objective in VARIANTS:
            plan.append({"id": f"{fold}_{name}", "fold": fold, "variant": name,
                         "algorithm": algorithm, "aggregation_objective": objective,
                         "outer_test_inference": False})
    if len(plan) != 280 or len({row["id"] for row in plan}) != len(plan):
        raise ValueError("Unexpected ablation matrix")
    return plan


def _safe_output(output):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output == DATA or DATA in output.parents or output in DATA.parents:
        raise ValueError("Output must be a new separate Paper 2 results directory")
    if output.exists():
        raise FileExistsError(output)
    return output


def freeze(output, data=DATA):
    output, data = _safe_output(output), Path(data).resolve()
    manifest = read(data / "manifest.json")
    if manifest.get("status") != "PREPARED_NO_TRAINING" or manifest.get("folds") != 64:
        raise ValueError("Expected prepared fixed-public data")
    plan, settings = make_plan(manifest), config()
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "matched_ablation_protocol.py")
    atomic_json(output / "config.json", settings)
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "manifest.json", {
        "status": "FROZEN_MATCHED_ABLATION_PROTOCOL_NO_TRAINING",
        "planned_runs": len(plan), "data": str(data),
        "data_manifest_sha256": sha(data / "manifest.json"), "runtime": runtime(),
        "source_sha256": sha(__file__),
        "artifacts": {name: sha(output / name) for name in ("config.json", "plan.json", "matched_ablation_protocol.py")},
    })
    return {"status": "FROZEN_MATCHED_ABLATION_PROTOCOL_NO_TRAINING", "planned_runs": len(plan)}


def verify(package):
    package = Path(package).resolve()
    manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_MATCHED_ABLATION_PROTOCOL_NO_TRAINING" or manifest.get("runtime") != runtime():
        raise ValueError("Protocol runtime/status mismatch")
    data = Path(manifest["data"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data changed")
    if manifest["source_sha256"] != sha(__file__) or sha(package / "matched_ablation_protocol.py") != sha(__file__):
        raise ValueError("Protocol source changed")
    if any(sha(package / name) != expected for name, expected in manifest["artifacts"].items()):
        raise ValueError("Protocol artifact changed")
    if read(package / "config.json") != config() or read(package / "plan.json") != make_plan(read(data / "manifest.json")):
        raise ValueError("Protocol semantics changed")
    return {"status": "PASS_FROZEN_MATCHED_ABLATION_PROTOCOL", "planned_runs": 280, "training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.output) if args.action == "freeze" else verify(args.output))
