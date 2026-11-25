"""Validate the regression-focused Paper 2 closest-work registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_STUDY_FIELDS = {
    "id",
    "citation",
    "year",
    "source_url",
    "task",
    "data_and_unit",
    "federation",
    "validation",
    "privacy_scope",
    "comparability_limits",
    "manuscript_role",
    "numerical_comparison_allowed",
}

REQUIRED_BOUNDARY_PHRASES = (
    "federated crop or yield prediction is novel",
    "numerical superiority across papers",
    "formal privacy guarantee",
    "best-crop recommendation",
)


def _valid_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate_registry(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    active_task = str(payload.get("active_task", "")).lower()
    if "regression" not in active_task or "cereal-legume" not in active_task:
        errors.append("active_task must identify cereal-legume regression")

    boundaries = payload.get("claim_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        errors.append("claim_boundaries must be a non-empty list")
        boundary_text = ""
    else:
        boundary_text = " ".join(str(item).lower() for item in boundaries)
    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase not in boundary_text:
            errors.append(f"missing required claim boundary: {phrase}")

    studies = payload.get("closest_studies")
    if not isinstance(studies, list) or len(studies) < 7:
        errors.append("at least seven closest studies are required")
        studies = studies if isinstance(studies, list) else []

    seen_ids: set[str] = set()
    for index, study in enumerate(studies):
        label = f"closest_studies[{index}]"
        if not isinstance(study, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = sorted(REQUIRED_STUDY_FIELDS - set(study))
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
            continue
        study_id = study["id"]
        if not isinstance(study_id, str) or not study_id.strip():
            errors.append(f"{label}.id must be non-empty")
        elif study_id in seen_ids:
            errors.append(f"duplicate study id: {study_id}")
        else:
            seen_ids.add(study_id)
        if not _valid_https_url(study["source_url"]):
            errors.append(f"{label}.source_url must be an HTTPS primary-source URL")
        if study["numerical_comparison_allowed"] is not False:
            errors.append(f"{label} must prohibit cross-paper numerical comparison")
        for field in (
            "citation",
            "task",
            "data_and_unit",
            "federation",
            "validation",
            "privacy_scope",
            "comparability_limits",
            "manuscript_role",
        ):
            if not isinstance(study[field], str) or not study[field].strip():
                errors.append(f"{label}.{field} must be non-empty")

    algorithms = payload.get("algorithm_sources")
    names = {
        item.get("name")
        for item in algorithms
        if isinstance(item, dict) and _valid_https_url(item.get("source_url"))
    } if isinstance(algorithms, list) else set()
    for required in ("FedAvg", "FedProx", "FedNova"):
        if required not in names:
            errors.append(f"missing valid primary algorithm source: {required}")

    positioning = payload.get("positioning_decision")
    if not isinstance(positioning, dict):
        errors.append("positioning_decision must be an object")
    else:
        for field in ("closest_contexts", "defensible_paper2_focus", "not_established"):
            if not positioning.get(field):
                errors.append(f"positioning_decision.{field} must be non-empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_registry(payload)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "registry": str(args.registry),
        "review_id": payload.get("review_id"),
        "study_count": len(payload.get("closest_studies", [])),
        "algorithm_source_count": len(payload.get("algorithm_sources", [])),
        "errors": errors,
        "checks": {
            "minimum_study_count": 7,
            "cross_paper_numeric_comparison_prohibited": not any(
                item.get("numerical_comparison_allowed") is not False
                for item in payload.get("closest_studies", [])
                if isinstance(item, dict)
            ),
            "required_algorithm_sources": ["FedAvg", "FedProx", "FedNova"],
            "claim_boundaries_present": len(payload.get("claim_boundaries", [])),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "verification.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
