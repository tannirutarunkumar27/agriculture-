"""Prospective outer study planning only: never loads test arrays or trains."""
from pathlib import Path
import argparse
import json
import shutil

from paper2.restart.validation_search import (
    ROOT, atomic_json, read, sha, runtime, verify_frozen, selections,
    verify_completed, identity, writer_lock,
)

SEEDS = [123, 2024, 3407, 9999, 2718]


def make_plan(selected, prepared, seeds):
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds) or 42 in seeds:
        raise ValueError("Distinct nonnegative study seeds excluding tuning seed required")
    plan = []
    for outer, choice in sorted(selected.items()):
        fold = outer
        if fold not in prepared["assets"]:
            raise ValueError("Missing outer refit fold")
        for seed in seeds:
            for algorithm, rounds in choice["outer_refit_rounds"].items():
                if type(rounds) is not int or not 1 <= rounds <= 80:
                    raise ValueError("Invalid selected round budget")
                plan.append({"id": f"{outer}_{algorithm}_seed{seed}", "fold": fold,
                             "seed": seed, "algorithm": algorithm, "rounds": rounds,
                             "learning_rate": choice["common_learning_rate"]})
            for name, reference in sorted(choice["references"].items()):
                plan.append({"id": f"{outer}_{name}_seed{seed}", "fold": fold,
                             "seed": seed, "algorithm": name,
                             "configuration": reference["configuration"]})
    if len({r["id"] for r in plan}) != len(plan):
        raise ValueError("Duplicate run identity")
    return plan


def freeze(output):
    output = Path(output).resolve()
    frozen = ROOT / "paper2/results/restart_validation_protocol_v1"
    search = ROOT / "paper2/results/restart_validation_search_v1"
    manifest, config, inner_plan = verify_frozen(frozen)
    data = Path(manifest["data"])
    prepared = read(data / "manifest.json")
    allowed = ROOT / "paper2/results"
    if output == allowed or allowed not in output.parents:
        raise ValueError("Output must be a new child of paper2/results")
    if output.exists():
        raise FileExistsError(output)
    with writer_lock(search):
        results = []
        binding = read(search / "binding.json")
        if binding["identity"] != identity(manifest):
            raise ValueError("Search binding changed")
        for run in inner_plan:
            run_identity = identity({"study": binding["identity"], "run": run})
            results.append(verify_completed(search / "runs" / run["id"], run_identity))
        selected = selections(results, config, inner_plan)
        if selected != read(search / "selected_configurations.json"):
            raise ValueError("Saved selections do not match verified results")
        plan = make_plan(selected, prepared, SEEDS)
        protocol = {"version": "restart-outer-study-v1", "seeds": SEEDS,
                    "task": "retrospective conditional harvest seed dry matter t/ha",
                    "neural_settings": {k: config[k] for k in ("hidden", "batch_size", "mu", "participation", "local_epochs", "optimizer")},
                    "checkpoint": "last fixed selected refit round; no outer-test selection",
                    "evaluation": "one model test inference per run, committed receipt/cache; ambiguous inference stops",
                    "primary": "equal-geographic-group mean RMSE per algorithm and seed",
                    "secondary": ["pooled RMSE", "MAE", "R2 null if undefined", "training-supported species/composition panels"],
                    "prediction_clipping": False, "knowledge_enabled": False,
                    "privacy_enabled": False, "ranking_enabled": False,
                    "limitations": ["286/672 inner neural candidates selected round cap", "4/8 outer groups selected highest grid LR", "seed123 diagnostic exposure; development dataset", "seeds are not independent regional replications"]}
        output.mkdir(parents=True)
        atomic_json(output / "config.json", protocol)
        atomic_json(output / "plan.json", plan)
        atomic_json(output / "selected.json", selected)
        shutil.copyfile(__file__, output / "outer_protocol.py")
        atomic_json(output / "manifest.json", {
            "status": "FROZEN_PLAN_ONLY_NO_OUTER_INFERENCE", "planned_runs": len(plan),
            "runtime": runtime(), "data": str(data),
            "data_manifest_sha256": sha(data / "manifest.json"),
            "search_manifest_sha256": sha(frozen / "manifest.json"),
            "selection_sha256": sha(search / "selected_configurations.json"),
            "artifact_sha256": {n: sha(output / n) for n in ("config.json", "plan.json", "selected.json", "outer_protocol.py")},
            "runner_status": "NOT_IMPLEMENTED; runner code needs separate prospective identity before execution"})
    return {"status": "FROZEN_PLAN_ONLY_NO_OUTER_INFERENCE", "planned_runs": len(plan)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))
