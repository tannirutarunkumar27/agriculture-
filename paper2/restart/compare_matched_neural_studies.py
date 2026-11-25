"""Compare independently audited CPU/GPU RR06 studies without merging them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

from paper2.restart.validation_search import ROOT, atomic_json, read, sha


RESULTS = ROOT / "paper2/results"


def load_audit(path):
    path = Path(path).resolve()
    report = read(path / "audit.json")
    if report.get("status") not in {"PASS_PARTIAL_MATCHED_NEURAL_STUDY", "PASS_COMPLETE_MATCHED_NEURAL_STUDY"}:
        raise ValueError("Study audit is not a passing RR06 audit")
    if report.get("outer_test_inference") is not False:
        raise ValueError("Study audit permits outer-test inference")
    rows = {row["id"]: row for row in report.get("work", [])}
    if len(rows) != report["verified"]:
        raise ValueError("Audit work rows are not unique/complete")
    return path, report, rows


def compare(cpu_audit, gpu_audit, output):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output.exists():
        raise ValueError("Comparison output must be a new Paper 2 results directory")
    cpu_path, cpu, cpu_rows = load_audit(cpu_audit)
    gpu_path, gpu, gpu_rows = load_audit(gpu_audit)
    overlap = sorted(set(cpu_rows) & set(gpu_rows))
    only_cpu, only_gpu = sorted(set(cpu_rows) - set(gpu_rows)), sorted(set(gpu_rows) - set(cpu_rows))
    paired = []
    for run_id in overlap:
        left, right = cpu_rows[run_id], gpu_rows[run_id]
        if left["optimizer_steps_per_round"] != right["optimizer_steps_per_round"] or left["examples_per_round"] != right["examples_per_round"]:
            raise ValueError(f"Work accounting differs for {run_id}")
        left_rmse, right_rmse = left["best_validation_rmse"], right["best_validation_rmse"]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (left_rmse, right_rmse)):
            raise ValueError(f"Non-finite paired metric for {run_id}")
        paired.append({"id": run_id, "cpu_best_validation_rmse": left_rmse,
                       "gpu_best_validation_rmse": right_rmse,
                       "gpu_minus_cpu_best_validation_rmse": right_rmse - left_rmse,
                       "cpu_best_round": left["best_round"], "gpu_best_round": right["best_round"]})
    report = {
        "status": "PASS_AUDITED_STUDY_COMPARISON",
        "cpu_audit_sha256": sha(cpu_path / "audit.json"),
        "gpu_audit_sha256": sha(gpu_path / "audit.json"),
        "cpu_execution_manifest_sha256": cpu["execution_manifest_sha256"],
        "gpu_execution_manifest_sha256": gpu["execution_manifest_sha256"],
        "execution_manifests_distinct": cpu["execution_manifest_sha256"] != gpu["execution_manifest_sha256"],
        "cpu_verified": cpu["verified"], "gpu_verified": gpu["verified"],
        "paired_runs": len(paired), "only_cpu_runs": len(only_cpu), "only_gpu_runs": len(only_gpu),
        "selection_available": False,
        "outer_test_inference": False,
        "interpretation": "Descriptive paired inner-validation comparison only; CPU and GPU studies are not merged and no outer/generalisation claim is made.",
        "paired": paired,
        "only_cpu": only_cpu,
        "only_gpu": only_gpu,
    }
    output.mkdir(parents=True)
    atomic_json(output / "comparison.json", report)
    atomic_json(output / "manifest.json", {"status": report["status"], "comparison_sha256": sha(output / "comparison.json"),
                "cpu_audit": str(cpu_path), "gpu_audit": str(gpu_path)})
    return {key: report[key] for key in ("status", "cpu_verified", "gpu_verified", "paired_runs", "only_cpu_runs", "only_gpu_runs", "execution_manifests_distinct")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--gpu-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.cpu_audit, args.gpu_audit, args.output), indent=2))
