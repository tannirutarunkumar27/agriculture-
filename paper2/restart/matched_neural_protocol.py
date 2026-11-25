"""Freeze RR06's bounded, inner-only neural optimisation study.

This is deliberately a protocol freezer, not a trainer.  It keeps all outer
test arrays unavailable and makes AdamW comparisons only for algorithms whose
update definitions do not require the project's plain-SGD FedNova restriction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


RESULTS = ROOT / "paper2/results"
DATA = RESULTS / "restart_public_data_v1"
SGD_VARIANTS = (
    ("centralized_sample_weighted", "centralized", "sample_weighted"),
    ("fedavg_sample_weighted", "fedavg", "sample_weighted"),
    ("fedprox_sample_weighted", "fedprox", "sample_weighted"),
    ("fednova_sample_weighted", "fednova_restricted", "sample_weighted"),
    ("fedavg_equal_client", "fedavg", "equal_client"),
)
ADAMW_VARIANTS = tuple(row for row in SGD_VARIANTS if row[1] != "fednova_restricted")
SEARCH = (
    {"id": "sgd_constant_h32", "optimizer": "sgd", "schedule": "constant", "hidden": 32,
     "learning_rate": 0.01, "weight_decay": 0.0, "rounds": 240, "local_steps": 4},
    {"id": "adamw_cosine_h64", "optimizer": "adamw", "schedule": "cosine", "hidden": 64,
     "learning_rate": 0.003, "weight_decay": 0.0001, "rounds": 360, "local_steps": 4},
)


def config():
    return {
        "version": "matched-neural-inner-v1",
        "scope": "inner_validation_development_only",
        "target": "conditional_harvest_seed_dry_matter_t_ha",
        "seed": 42,
        "batch_size": 32,
        "gradient_clip_norm": 1.0,
        "fedprox_mu": 0.01,
        "participation": 1.0,
        "centralized_work_rule": "local_steps_times_number_of_training_clients",
        "centralized_example_rule": "match_sum_of_federated_fixed_step_example_exposure_per_fold",
        "optimizer_state_policy": {"centralized": "persistent_across_rounds", "federated": "reset_each_client_round"},
        "search": [dict(setting) for setting in SEARCH],
        "selection": "per_algorithm_objective_equal_inner_geographic_mean_best_rmse_then_frozen_search_order",
        "convergence": "record per-round validation RMSE, first best round, round-cap hits, and measured elapsed seconds",
        "outer_test_inference": False,
        "knowledge_enabled": False,
        "privacy_enabled": False,
        "ranking_enabled": False,
        "limitations": [
            "all geographic groups have prior outer-test exposure",
            "this is a development optimisation study, not independent confirmation",
            "equal-client and sample-weighted objectives are distinct estimands",
            "FedNova is restricted to plain SGD without momentum or weight decay and is therefore absent from AdamW candidates",
        ],
    }


def make_plan(data_manifest):
    folds = sorted(name for name in data_manifest["assets"] if "_inner_" in name)
    if len(folds) != 56 or len(set(folds)) != 56:
        raise ValueError("Expected exactly 56 inner folds")
    plan = []
    for fold in folds:
        for setting in SEARCH:
            variants = SGD_VARIANTS if setting["optimizer"] == "sgd" else ADAMW_VARIANTS
            for variant, algorithm, objective in variants:
                item = {**setting, "id": f"{fold}_{setting['id']}_{variant}", "fold": fold,
                        "variant": variant, "algorithm": algorithm,
                        "aggregation_objective": objective, "outer_test_inference": False}
                plan.append(item)
    if len(plan) != 504 or len({row["id"] for row in plan}) != len(plan):
        raise ValueError("Unexpected RR06 neural-search matrix")
    return plan


def _safe_output(output):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output.exists() or output == DATA or DATA in output.parents or output in DATA.parents:
        raise ValueError("Output must be a new separate Paper 2 results directory")
    return output


def freeze(output, data=DATA):
    output, data = _safe_output(output), Path(data).resolve()
    manifest = read(data / "manifest.json")
    if manifest.get("status") != "PREPARED_NO_TRAINING" or manifest.get("folds") != 64:
        raise ValueError("Expected prepared fixed-public data")
    plan, settings = make_plan(manifest), config()
    output.mkdir(parents=True)
    sources = (Path(__file__), ROOT / "paper2/restart/__init__.py", ROOT / "paper2/restart/matched_ablation_protocol.py",
               ROOT / "paper2/restart/validation_search.py")
    (output / "source_snapshot").mkdir()
    for source in sources:
        shutil.copyfile(source, output / "source_snapshot" / source.name)
    atomic_json(output / "config.json", settings)
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "manifest.json", {
        "status": "FROZEN_MATCHED_NEURAL_PROTOCOL_NO_TRAINING", "planned_runs": len(plan),
        "data": str(data), "data_manifest_sha256": sha(data / "manifest.json"), "runtime": runtime(),
        "source_sha256": {p.name: sha(p) for p in sources},
        "artifacts": {name: sha(output / name) for name in ("config.json", "plan.json")},
        "outer_test_inference": False,
    })
    return {"status": "FROZEN_MATCHED_NEURAL_PROTOCOL_NO_TRAINING", "planned_runs": len(plan)}


def verify(package, data_override=None):
    package = Path(package).resolve()
    manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_MATCHED_NEURAL_PROTOCOL_NO_TRAINING" or manifest.get("runtime") != runtime():
        raise ValueError("Protocol runtime/status mismatch")
    sources = (Path(__file__), ROOT / "paper2/restart/__init__.py", ROOT / "paper2/restart/matched_ablation_protocol.py",
               ROOT / "paper2/restart/validation_search.py")
    for source in sources:
        expected = manifest["source_sha256"].get(source.name)
        if expected != sha(source) or sha(package / "source_snapshot" / source.name) != expected:
            raise ValueError("Protocol source changed")
    data = Path(data_override).resolve() if data_override is not None else Path(manifest["data"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data changed")
    if any(sha(package / name) != digest for name, digest in manifest["artifacts"].items()):
        raise ValueError("Protocol artifact changed")
    if read(package / "config.json") != config() or read(package / "plan.json") != make_plan(read(data / "manifest.json")):
        raise ValueError("Protocol semantics changed")
    return {"status": "PASS_FROZEN_MATCHED_NEURAL_PROTOCOL", "planned_runs": 504, "training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.output) if args.action == "freeze" else verify(args.output))
