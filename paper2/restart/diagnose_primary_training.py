"""Read-only diagnosis of the frozen primary validation search.

This module deliberately reads only completed inner-validation receipts.  It
does not load checkpoints, data arrays, or outer-test predictions.  Its output
quantifies two limitations that must be reported before proposing a new,
matched follow-up: finite-round selection and the observed FedAvg--FedProx
difference under the already-frozen ``mu`` setting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from paper2.restart.validation_search import (
    ROOT, atomic_json, identity, read, sha, verify_completed, verify_frozen,
)


NEURAL = ("centralized", "fedavg", "fedprox", "fednova")


def _mean(values):
    if not values:
        raise ValueError("Cannot average an empty collection")
    return float(sum(values) / len(values))


def _validate_result(run, result, rounds):
    if result.get("id") != run["id"] or result.get("fold") != run["fold"]:
        raise ValueError("Completed result does not match frozen plan")
    if result.get("algorithm") != run["algorithm"]:
        raise ValueError("Completed algorithm does not match frozen plan")
    if run["algorithm"] not in NEURAL:
        return
    best_round = result.get("best_round")
    history = result.get("history")
    if not isinstance(best_round, int) or not 1 <= best_round <= rounds:
        raise ValueError("Invalid selected round")
    if not isinstance(history, list) or len(history) != rounds:
        raise ValueError("Incomplete neural training history")
    if [row.get("round") for row in history] != list(range(1, rounds + 1)):
        raise ValueError("Non-contiguous training history")


def analyze(frozen, study, output):
    """Audit frozen inner results and write one new, immutable diagnostic."""
    frozen, study, output = (Path(value).resolve() for value in (frozen, study, output))
    if output.exists():
        raise FileExistsError(output)
    manifest, config, plan = verify_frozen(frozen)
    binding = identity(manifest)
    binding_hash = sha(study / "binding.json")
    if read(study / "binding.json").get("identity") != binding:
        raise ValueError("Study binding mismatch")

    results = {}
    for run in plan:
        token = identity({"study": binding, "run": run})
        result = verify_completed(study / "runs" / run["id"], token)
        _validate_result(run, result, config["rounds"])
        results[run["id"]] = result
    if sha(study / "binding.json") != binding_hash:
        raise RuntimeError("Study binding changed during read-only diagnosis")

    neural_runs = [run for run in plan if run["algorithm"] in NEURAL]
    if not neural_runs:
        raise ValueError("No neural runs in frozen plan")
    per_algorithm = {}
    for algorithm in NEURAL:
        chosen = [results[run["id"]] for run in neural_runs if run["algorithm"] == algorithm]
        cap_hits = sum(result["best_round"] == config["rounds"] for result in chosen)
        per_algorithm[algorithm] = {
            "candidates": len(chosen),
            "selected_at_round_cap": cap_hits,
            "selected_at_round_cap_fraction": cap_hits / len(chosen),
            "mean_selected_round": _mean([result["best_round"] for result in chosen]),
        }

    pairs = []
    by_key = {(run["fold"], run.get("candidate"), run["algorithm"]): run for run in neural_runs}
    keys = sorted({(run["fold"], run.get("candidate")) for run in neural_runs})
    for fold, candidate in keys:
        avg = by_key.get((fold, candidate, "fedavg"))
        prox = by_key.get((fold, candidate, "fedprox"))
        if avg is None or prox is None:
            raise ValueError("FedAvg/FedProx pairing missing from frozen plan")
        avg_result, prox_result = results[avg["id"]], results[prox["id"]]
        pairs.append({
            "fold": fold,
            "candidate": candidate,
            "learning_rate": avg["learning_rate"],
            "fedavg_rmse": avg_result["validation"]["rmse"],
            "fedprox_rmse": prox_result["validation"]["rmse"],
            "fedprox_minus_fedavg_rmse": (
                prox_result["validation"]["rmse"] - avg_result["validation"]["rmse"]
            ),
            "fedavg_best_round": avg_result["best_round"],
            "fedprox_best_round": prox_result["best_round"],
        })
    deltas = [row["fedprox_minus_fedavg_rmse"] for row in pairs]
    report = {
        "status": "PASS_COMPLETE_READ_ONLY_PRIMARY_DIAGNOSIS",
        "scope": "Frozen primary inner-validation results only; no checkpoint/data/test-array loading or model inference.",
        "verified_completed_results": len(results),
        "neural_candidates": len(neural_runs),
        "round_cap": config["rounds"],
        "per_algorithm_round_selection": per_algorithm,
        "fedavg_fedprox_pairing": {
            "pairs": len(pairs),
            "mu": config["mu"],
            "mean_fedprox_minus_fedavg_rmse": _mean(deltas),
            "mean_absolute_rmse_difference": _mean([abs(value) for value in deltas]),
            "max_absolute_rmse_difference": max(abs(value) for value in deltas),
            "fedprox_lower_rmse_pairs": sum(value < 0 for value in deltas),
            "fedavg_lower_rmse_pairs": sum(value > 0 for value in deltas),
            "exact_ties": sum(value == 0 for value in deltas),
        },
        "pairs": pairs,
        "outer_test_inference": False,
        "interpretation": (
            "This is a descriptive diagnosis of a previously frozen, finite-budget search. "
            "Cap hits are evidence that the search does not certify convergence; paired "
            "RMSE differences do not establish a general FedProx effect or geographic inference."
        ),
        "input_hashes": {
            "frozen_manifest": sha(frozen / "manifest.json"),
            "study_binding": binding_hash,
            "selected_configurations": sha(study / "selected_configurations.json"),
            "diagnostic_source": sha(__file__),
        },
    }
    output.mkdir(parents=True)
    atomic_json(output / "diagnosis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=ROOT / "paper2/results/restart_validation_protocol_v1")
    parser.add_argument("--study", type=Path, default=ROOT / "paper2/results/restart_validation_search_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.frozen, args.study, args.output)
    print({key: report[key] for key in ("status", "verified_completed_results", "neural_candidates", "round_cap")})
