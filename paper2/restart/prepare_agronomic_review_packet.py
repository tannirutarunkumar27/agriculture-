"""Prepare a source-bound packet for independent agronomic-rule review.

This module deliberately does not approve, reject, or activate a rule.  It
turns the frozen candidate registry into a compact reviewer packet so an
independent agronomist can make auditable include/exclude decisions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper2.restart.paper2_completion_audit import RESULTS, atomic_json, read, sha
from paper2.restart.validate_agronomic_rules import REGISTRY, validate_registry


PACKET_FIELDS = (
    "rule_id", "crop_taxon", "source_crop_name", "source_url_or_doi", "source_locator",
    "optimal_ph_min", "optimal_ph_max", "original_unit", "dataset_unit", "transformation",
    "region_scope", "cultivar_stage_scope", "water_regime_scope", "confidence",
    "required_reviewer_decision", "required_checks",
)


def packet_rows(registry: dict) -> list[dict]:
    report = validate_registry(registry)
    if report["status"] != "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW":
        raise ValueError("Only schema-valid, still-pending candidate rules may be packaged for review")
    rows = []
    for rule in sorted(registry["rules"], key=lambda value: value["rule_id"]):
        rows.append({
            "rule_id": rule["rule_id"], "crop_taxon": rule["crop_taxon"],
            "source_crop_name": rule["source_crop_name"], "source_url_or_doi": rule["source_url_or_doi"],
            "source_locator": rule["source_locator"], "optimal_ph_min": rule["original_min"],
            "optimal_ph_max": rule["original_max"], "original_unit": rule["original_unit"],
            "dataset_unit": rule["dataset_unit"], "transformation": rule["transformation"],
            "region_scope": rule["region"], "cultivar_stage_scope": rule["variety_or_stage"],
            "water_regime_scope": rule["water_regime"], "confidence": rule["confidence"],
            "required_reviewer_decision": "include or exclude; no default inclusion",
            "required_checks": (
                "confirm source record; taxonomy; pH unit; geographic applicability; "
                "cultivar/stage limitation; water-regime limitation; and soft-prior use"
            ),
        })
    return rows


def prepare(output: Path, registry_path: Path = REGISTRY) -> dict:
    output, registry_path = Path(output).resolve(), Path(registry_path).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    registry = read(registry_path)
    rows = packet_rows(registry)
    output.mkdir(parents=True)
    with (output / "agronomic_review_packet.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (output / "review_instructions.md").write_text(
        "# Independent agronomic review instructions\n\n"
        "This packet is a review aid only. It does not authorize agronomic guidance, ranking, or publication claims. "
        "For each row, record an include/exclude decision in the domain-review template and complete every listed check. "
        "An include decision requires a conflict-disclosed independent reviewer and a separate frozen guidance study.\n",
        encoding="utf-8",
    )
    report = {
        "status": "AGRONOMIC_REVIEW_PACKET_READY_PENDING_INDEPENDENT_REVIEW",
        "rules": len(rows), "guidance_enabled": False, "model_training": False,
        "interpretation": "Source-bound packet prepared; no rule has been independently approved or applied.",
    }
    atomic_json(output / "review_packet.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "registry_sha256": sha(registry_path),
        "packet_sha256": sha(output / "agronomic_review_packet.csv"),
        "instructions_sha256": sha(output / "review_instructions.md"),
        "report_sha256": sha(output / "review_packet.json"), "guidance_enabled": False,
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    print(prepare(args.output, args.registry))
