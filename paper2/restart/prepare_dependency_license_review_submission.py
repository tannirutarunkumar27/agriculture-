"""Prepare an evidence-bound license-review handoff from a clean Linux inventory.

It records observed package metadata and hashes, but it cannot grant legal
rights or alter the release registry. Those require an independent reviewer
and an explicit owner decision.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper2.restart.paper2_completion_audit import RESULTS, atomic_json, latest_result_file, read, sha


DECISION_FIELDS = (
    "package", "locked_version", "installed_version_observed", "license_expression_observed",
    "license_field_observed", "license_classifiers_observed", "license_file_sha256",
    "review_decision", "review_rationale",
)


def prepare(output: Path, inventory_path: Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    inventory_path = Path(inventory_path or latest_result_file(
        "restart_dependency_license_inventory", "license_inventory.json"
    )).resolve()
    report = read(inventory_path)
    if (report.get("environment_scope") != "clean_wsl_linux" or report.get("missing_from_environment") != 0
            or report.get("locked_version_mismatches") or report.get("environment_matches_lock") is not True):
        raise ValueError("A complete clean-Linux inventory with exact locked versions is required")
    csv_path = inventory_path.parent / "dependency_license_inventory.csv"
    if not csv_path.is_file():
        raise ValueError("Dependency inventory CSV is required")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != report.get("inventory_package_count"):
        raise ValueError("Inventory CSV count differs from its report")
    output.mkdir(parents=True)
    decisions = []
    for row in rows:
        decisions.append({
            "package": row.get("package", ""), "locked_version": row.get("locked_version", ""),
            "installed_version_observed": row.get("installed_version_observed", ""),
            "license_expression_observed": row.get("license_expression_observed", ""),
            "license_field_observed": row.get("license_field_observed", ""),
            "license_classifiers_observed": row.get("license_classifiers_observed", ""),
            "license_file_sha256": row.get("license_file_sha256", ""),
            "review_decision": "", "review_rationale": "",
        })
    with (output / "dependency_license_decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS); writer.writeheader(); writer.writerows(decisions)
    (output / "SUBMISSION_INSTRUCTIONS.md").write_text(
        "# Dependency-license review submission\n\n"
        "Review every package against its pinned version and observed license/notice evidence. "
        "For each row enter one of: `acceptable`, `requires-remediation`, or `not-used`, and provide a rationale. "
        "This review does not decide dataset redistribution, project-code licensing, journal policy, or legal ownership. "
        "Return the completed CSV and a completed independent-review attestation.\n",
        encoding="utf-8",
    )
    packet = {
        "status": "DEPENDENCY_LICENSE_REVIEW_SUBMISSION_KIT_READY_PENDING_INDEPENDENT_REVIEW",
        "packages": len(decisions), "inventory_sha256": sha(inventory_path),
        "clean_linux_lock_matched": True, "license_inventory_complete": False,
        "release_allowed": False,
        "interpretation": "Observed license evidence is ready for independent review; no distribution right is granted.",
    }
    atomic_json(output / "license_review_packet.json", packet)
    atomic_json(output / "independent_review.template.json", {
        "schema_version": 1, "reviewer": "", "reviewer_affiliation": "", "review_date": "",
        "independent": "", "conflict_of_interest_disclosed": "", "decision": "",
        "rationale": "", "packet_sha256": sha(output / "license_review_packet.json"),
        "required_decision": "approve-for-project-license-record, reject, or request-changes; no dataset redistribution approval option",
    })
    atomic_json(output / "manifest.json", {
        "status": packet["status"], "inventory_sha256": sha(inventory_path),
        "inventory_csv_sha256": sha(csv_path), "decision_template_sha256": sha(output / "dependency_license_decisions.csv"),
        "packet_sha256": sha(output / "license_review_packet.json"), "release_allowed": False,
    })
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    print(prepare(args.output, args.inventory))
