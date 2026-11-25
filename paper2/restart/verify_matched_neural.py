"""Independent receipt and work audit for partial or complete RR06 studies."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from paper2.restart import matched_neural_runner as runner
from paper2.restart.control_convergence import load_inner
from paper2.restart.matched_neural_training import fixed_step_examples
from paper2.restart.validation_search import ROOT, atomic_json, identity, read, sha


RESULTS = ROOT / "paper2/results"


def audit_history_work(run, history, groups, settings):
    expected_steps = run["local_steps"] * len(groups)
    expected_examples = sum(fixed_step_examples(len(ids), settings["batch_size"], run["local_steps"])
                            for ids in groups.values())
    if len(history) != run["rounds"]:
        raise ValueError("Completed history length differs")
    for row in history:
        if row["optimizer_steps"] != expected_steps or row["examples_processed"] != expected_examples:
            raise ValueError("Recorded matched work differs")
    return {"optimizer_steps_per_round": expected_steps, "examples_per_round": expected_examples}


def audit(package, study, output):
    package, study, output = Path(package).resolve(), Path(study).resolve(), Path(output).resolve()
    if RESULTS not in output.parents or output.exists() or output == study or study in output.parents or output in study.parents:
        raise ValueError("Audit output must be a new separate Paper 2 results directory")
    manifest, settings, plan, data = runner.verify_execution(package)
    if read(study / "binding.json") != {"identity": identity(manifest)}:
        raise ValueError("Study binding changed")
    runs_root = study / "runs"
    expected_ids = {run["id"] for run in plan}
    observed_ids = {folder.name for folder in runs_root.iterdir()} if runs_root.exists() else set()
    if not observed_ids <= expected_ids:
        raise ValueError("Unexpected RR06 run directory")
    results, work = {}, []
    for run in plan:
        folder = runs_root / run["id"]
        if not (folder / "completion.json").exists():
            continue
        token = identity({"binding": identity(manifest), "run": run})
        fold, arrays = load_inner(data, run["fold"])
        result = runner.completed(folder, run, token, arrays["x_train"].shape[1])
        saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        groups = runner._fold_groups(fold, len(arrays["y_train"]))
        measured = audit_history_work(run, saved["history"], groups, settings)
        attempts = read(folder / "attempts.json")
        if not attempts or attempts[-1]["status"] not in {"COMPLETE", "CACHE_VERIFIED_AFTER_INTERRUPTION"} or attempts[-1]["completion_sha256"] != sha(folder / "completion.json"):
            raise ValueError("Completion attempt journal differs")
        results[run["id"]] = result
        work.append({"id": run["id"], **measured, "best_round": result["best_round"],
                     "best_validation_rmse": result["best_validation_rmse"]})
    selection_path = study / "selection.json"
    if len(results) == len(plan):
        expected_selection = runner.select_complete(plan, results)
        if not selection_path.exists() or read(selection_path) != expected_selection:
            raise ValueError("Complete-study selection differs")
        status = "PASS_COMPLETE_MATCHED_NEURAL_STUDY"
    else:
        if selection_path.exists():
            raise ValueError("Partial study must not contain a selection")
        status = "PASS_PARTIAL_MATCHED_NEURAL_STUDY"
    progress = read(study / "progress.json")
    if progress["completed"] != len(results) or progress["planned"] != len(plan) or progress["selection_available"] != (len(results) == len(plan)) or progress["outer_test_inference"] is not False:
        raise ValueError("Study progress differs")
    device = progress.get("device")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Study device declaration is missing or invalid")
    output.mkdir(parents=True)
    report = {"status": status, "execution_manifest_sha256": sha(package / "manifest.json"),
              "study_binding_sha256": sha(study / "binding.json"), "verified": len(results),
              "pending": len(plan) - len(results), "selection_available": len(results) == len(plan),
              "outer_test_inference": False, "device": device,
              "hardware_qualification": "CPU_ONLY_NOT_GPU_EVIDENCE" if device == "cpu" else "CUDA_DECLARED_BY_EXECUTION",
              "work": work}
    atomic_json(output / "audit.json", report)
    atomic_json(output / "manifest.json", {"status": status, "audit_sha256": sha(output / "audit.json"),
                "source_sha256": sha(Path(__file__)), "study": str(study), "execution": str(package)})
    return {key: report[key] for key in ("status", "verified", "pending", "selection_available", "outer_test_inference", "device")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True); parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(audit(args.package, args.study, args.output))
