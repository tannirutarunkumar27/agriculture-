"""Independent finalization gate for complete RR06 inner selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2.restart import matched_neural_runner as runner
from paper2.restart.compare_matched_neural_studies import load_audit
from paper2.restart.validation_search import atomic_json, identity, read, sha


def finalize(package, study, audit, output):
    package, study, audit, output = map(lambda value: Path(value).resolve(), (package, study, audit, output))
    if output.exists():
        raise ValueError("Selection-finalization output must be new")
    manifest, settings, plan, _ = runner.verify_execution(package)
    audit_path, audit_report, rows = load_audit(audit)
    if audit_report["status"] != "PASS_COMPLETE_MATCHED_NEURAL_STUDY" or audit_report["verified"] != len(plan):
        raise ValueError("Complete 504-candidate audit is required")
    saved_progress = read(study / "progress.json")
    if saved_progress.get("selection_available") is not True or saved_progress.get("completed") != len(plan):
        raise ValueError("Study progress is not complete")
    saved = read(study / "selection.json")
    # Reconstruct the minimal result map needed by the frozen selection rule.
    results = {}
    for run in plan:
        row = rows.get(run["id"])
        if row is None:
            raise ValueError("Audit is missing a planned run")
        results[run["id"]] = {"run": run, "best_validation_rmse": row["best_validation_rmse"],
                               "best_round": row["best_round"], "outer_test_inference": False}
    expected = runner.select_complete(plan, results)
    if saved != expected:
        raise ValueError("Saved selection differs from independently recomputed selection")
    report = {"status": "PASS_FINALIZED_INNER_SELECTION", "planned": len(plan), "verified": len(rows),
              "execution_manifest_sha256": sha(package / "manifest.json"),
              "audit_sha256": sha(audit_path / "audit.json"), "selection_sha256": sha(study / "selection.json"),
              "selection_identity": identity(saved), "outer_test_inference": False,
              "rr07_ready": True, "interpretation": "Development selection only; all geographic groups have prior outer exposure."}
    report["selection"] = saved["selections"]
    output.mkdir(parents=True)
    atomic_json(output / "finalization.json", report)
    atomic_json(output / "manifest.json", {"status": report["status"], "finalization_sha256": sha(output / "finalization.json"),
                "execution": str(package), "study": str(study), "audit": str(audit)})
    return {key: report[key] for key in ("status", "planned", "verified", "rr07_ready", "outer_test_inference")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True); parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.package, args.study, args.audit, args.output), indent=2))
