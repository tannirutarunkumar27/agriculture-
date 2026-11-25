"""Independently audit every completed RR07 outer-evaluation receipt."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.rr07_execution_gate import verify_execution
from paper2.restart.rr07_runner import load_training
from paper2.restart.rr07_training import verify_completed
from paper2.restart.validation_search import ROOT, atomic_json, identity, read, sha


RESULTS = ROOT / "paper2/results"


def audit(package, study, output):
    package, study, output = map(lambda value: Path(value).resolve(), (package, study, output))
    if output.exists() or RESULTS not in output.parents or output == study or study in output.parents or output in study.parents:
        raise ValueError("Audit output must be a new separate Paper 2 results directory")
    manifest, plan, data = verify_execution(package)
    if read(study / "binding.json") != {"identity": identity(manifest)}:
        raise ValueError("RR07 study binding changed")
    results = []
    for run in plan:
        folder = study / "runs" / run["id"]
        if not (folder / "completion.json").exists():
            raise ValueError("RR07 audit requires every planned completion receipt")
        _, train = load_training(data, run)
        token = identity({"binding": identity(manifest), "run": run})
        result = verify_completed(folder, run, token, train["x_train"].shape[1])
        attempts = read(folder / "attempts.json")
        if not attempts or attempts[-1].get("status") not in {"COMPLETE", "CACHE_VERIFIED_AFTER_INTERRUPTION"} or attempts[-1].get("completion_sha256") != sha(folder / "completion.json"):
            raise ValueError("RR07 attempt journal differs from completion receipt")
        results.append({"id": run["id"], "fold": run["fold"], "seed": run["seed"], "algorithm": run["algorithm"],
                        "aggregation_objective": run["aggregation_objective"], "test": result["test"]})
    progress = read(study / "progress.json")
    if progress.get("status") != "COMPLETE" or progress.get("completed") != len(plan) or progress.get("outer_test_inference") is not True:
        raise ValueError("RR07 progress is not complete")
    device = progress.get("device")
    if device not in {"cpu", "cuda"}:
        raise ValueError("RR07 device declaration is invalid")
    report = {"status": "PASS_COMPLETE_RR07_DEVELOPMENT_OUTER_STUDY", "verified": len(results), "planned": len(plan),
              "device": device, "hardware_qualification": "CPU_ONLY_NOT_GPU_EVIDENCE" if device == "cpu" else "CUDA_DECLARED_BY_EXECUTION",
              "outer_test_inference": True, "independent_confirmation": False,
              "interpretation": "Development outer evaluation only; every geographic group had prior development exposure.", "runs": results}
    output.mkdir(parents=True); atomic_json(output / "audit.json", report)
    atomic_json(output / "manifest.json", {"status": report["status"], "audit_sha256": sha(output / "audit.json"),
                "execution_manifest_sha256": sha(package / "manifest.json"), "study_binding_sha256": sha(study / "binding.json")})
    return {key: report[key] for key in ("status", "verified", "planned", "device", "outer_test_inference")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True); parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(audit(args.package, args.study, args.output))
