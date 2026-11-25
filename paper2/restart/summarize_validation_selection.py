"""Complete-search selection consistency and budget-boundary audit, no inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.restart.validation_search import (verify_frozen, read, identity,
    writer_lock, verify_completed, selections, sha)


def summarize(frozen, study, output):
    frozen, study, output = Path(frozen), Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, config, plan = verify_frozen(frozen)
    binding = identity(manifest)
    if read(study/"binding.json")["identity"] != binding:
        raise ValueError("Study binding mismatch")
    with writer_lock(study):
        results = [verify_completed(study/"runs"/run["id"], identity({"study": binding, "run": run})) for run in plan]
        chosen = selections(results, config, plan)
        if chosen != read(study/"selected_configurations.json"):
            raise ValueError("Saved configuration selection does not reproduce")
    neural = [result for result in results if result["algorithm"] in config["algorithms"]]
    report = {"status": "PASS_COMPLETE_SELECTION_AUDIT", "verified_results": len(results),
        "neural_candidates": len(neural), "selected_at_round_cap": sum(result["best_round"] == config["rounds"] for result in neural),
        "selected_common_lr_at_lowest_grid_value": sum(value["common_learning_rate"] == config["learning_rates"][0] for value in chosen.values()),
        "selected_common_lr_at_highest_grid_value": sum(value["common_learning_rate"] == config["learning_rates"][-1] for value in chosen.values()),
        "outer_refit_group_mapping": {outer: read(Path(manifest["data"])/"folds"/outer/"partition.json")["test_group"] for outer in chosen},
        "selected_configurations": chosen, "outer_test_inference": False,
        "limitations": "validation-selected development settings, not held-out scores or proof of convergence; boundary optima are search limits",
        "selected_configurations_sha256": sha(study/"selected_configurations.json"), "summarizer_sha256": sha(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    (output/"selection_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified_results", "neural_candidates", "selected_at_round_cap",
        "selected_common_lr_at_lowest_grid_value", "selected_common_lr_at_highest_grid_value", "outer_test_inference")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="paper2/results/restart_validation_protocol_v1")
    parser.add_argument("--study", default="paper2/results/restart_validation_search_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.frozen, args.study, args.output)
