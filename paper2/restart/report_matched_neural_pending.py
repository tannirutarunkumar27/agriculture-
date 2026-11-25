"""Report the exact RR06 runs still needed before selection is allowed."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from paper2.restart.validation_search import ROOT, atomic_json, read, sha

RESULTS = ROOT / "paper2/results"


def report(package, study, audit, output):
    package, study, audit, output = (Path(p).resolve() for p in (package, study, audit, output))
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    package_manifest = read(package / "manifest.json")
    protocol = Path(package_manifest["protocol"])
    plan = read(protocol / "plan.json")
    audit_data = read(audit / "audit.json")
    verified = {row["id"] for row in audit_data.get("work", [])}
    planned = {row["id"] for row in plan}
    if not verified <= planned:
        raise ValueError("Audit contains an unexpected run")
    pending = [row for row in plan if row["id"] not in verified]
    by_outer = Counter(row["fold"] for row in pending)
    by_algorithm = Counter(row["algorithm"] for row in pending)
    by_objective = Counter(row["aggregation_objective"] for row in pending)
    result = {
        "status": "RR06_PENDING_RUNS_REPORTED",
        "planned": len(plan), "verified": len(verified), "pending": len(pending),
        "selection_allowed": len(pending) == 0,
        "outer_test_inference": False,
        "pending_run_ids": [row["id"] for row in pending],
        "pending_by_outer_fold": dict(sorted(by_outer.items())),
        "pending_by_algorithm": dict(sorted(by_algorithm.items())),
        "pending_by_aggregation_objective": dict(sorted(by_objective.items())),
    }
    output.mkdir(parents=True)
    atomic_json(output / "pending.json", result)
    atomic_json(output / "manifest.json", {
        "status": result["status"], "package_sha256": sha(package / "manifest.json"),
        "audit_sha256": sha(audit / "audit.json"), "study": str(study),
        "source_sha256": sha(Path(__file__)), "model_training": False,
    })
    return {key: result[key] for key in ("status", "planned", "verified", "pending", "selection_allowed")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.package, args.study, args.audit, args.output))
