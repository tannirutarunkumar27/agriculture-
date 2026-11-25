"""Freeze the RR07 outer-runner package without loading outer test arrays."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


RESULTS = ROOT / "paper2/results"
CODE = (
    Path(__file__),
    ROOT / "paper2/restart/rr07_development_protocol.py",
    ROOT / "paper2/restart/rr07_runner.py",
    ROOT / "paper2/restart/rr07_training.py",
    ROOT / "paper2/restart/matched_neural_training.py",
    ROOT / "paper2/restart/validation_search.py",
    ROOT / "paper2/federated.py",
)


def verify_plan(package):
    package = Path(package).resolve(); manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_RR07_PLAN_ONLY_NO_OUTER_INFERENCE":
        raise ValueError("RR07 plan is not frozen or has outer inference")
    if manifest.get("runtime") != runtime():
        raise ValueError("RR07 plan runtime changed")
    for name, expected in manifest.get("artifacts", {}).items():
        if sha(package / name) != expected:
            raise ValueError("RR07 plan artifact changed")
    data = Path(manifest["data"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("RR07 data binding changed")
    plan = read(package / "plan.json")
    if len(plan) != 160 or len({row["id"] for row in plan}) != 160 or any(row["outer_test_inference"] is not False for row in plan):
        raise ValueError("RR07 plan scope/identity changed")
    return manifest, plan, data


def freeze(output, plan_package):
    output, plan_package = Path(output).resolve(), Path(plan_package).resolve()
    if RESULTS not in output.parents or output.exists() or plan_package in output.parents or output in plan_package.parents:
        raise ValueError("Output must be a new separate Paper 2 results directory")
    plan_manifest, plan, data = verify_plan(plan_package)
    output.mkdir(parents=True); (output / "source_snapshot").mkdir()
    for source in CODE:
        shutil.copyfile(source, output / "source_snapshot" / source.name)
    atomic_json(output / "manifest.json", {"status": "FROZEN_RR07_EXECUTION_GATE_NO_OUTER_INFERENCE",
                "plan_package": str(plan_package), "plan_manifest_sha256": sha(plan_package / "manifest.json"),
                "data_manifest_sha256": sha(data / "manifest.json"), "runtime": runtime(),
                "planned_runs": len(plan), "outer_test_inference": False,
                "source_sha256": {source.name: sha(source) for source in CODE}})
    return {"status": "FROZEN_RR07_EXECUTION_GATE_NO_OUTER_INFERENCE", "planned_runs": len(plan)}


def verify_execution(package):
    """Verify a code-frozen execution package without loading outer arrays."""
    package = Path(package).resolve()
    manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_RR07_EXECUTION_GATE_NO_OUTER_INFERENCE":
        raise ValueError("RR07 execution gate is not frozen")
    if manifest.get("runtime") != runtime() or manifest.get("outer_test_inference") is not False:
        raise ValueError("RR07 execution runtime or scope changed")
    plan_package = Path(manifest["plan_package"]).resolve()
    _, plan, data = verify_plan(plan_package)
    if sha(plan_package / "manifest.json") != manifest.get("plan_manifest_sha256"):
        raise ValueError("RR07 execution plan binding changed")
    if sha(data / "manifest.json") != manifest.get("data_manifest_sha256"):
        raise ValueError("RR07 execution data binding changed")
    if set(manifest.get("source_sha256", {})) != {source.name for source in CODE}:
        raise ValueError("RR07 execution source inventory changed")
    for source in CODE:
        digest = manifest["source_sha256"][source.name]
        if sha(source) != digest or sha(package / "source_snapshot" / source.name) != digest:
            raise ValueError("RR07 execution source changed")
    if manifest.get("planned_runs") != len(plan):
        raise ValueError("RR07 execution plan count changed")
    return manifest, plan, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(); print(freeze(args.output, args.plan))
