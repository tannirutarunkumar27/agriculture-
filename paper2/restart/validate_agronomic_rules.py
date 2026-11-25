"""Validate independent agronomic-rule evidence; empty or weak banks fail closed."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from paper2.restart.validation_search import ROOT, atomic_json, read, sha

RESULTS = ROOT / "paper2/results"
# v1 is an intentionally empty legacy Paper 1 bank. Paper 2 validation must
# use the current candidate registry and still fail closed at domain review.
REGISTRY = ROOT / "paper2/configs/knowledge_registry_v2_candidates.json"
REQUIRED = {
    "rule_id", "crop_taxon", "source_crop_name", "feature", "source_url_or_doi", "source_locator",
    "original_min", "original_max", "original_unit", "dataset_unit", "transformation",
    "region", "variety_or_stage", "water_regime", "confidence", "approval_rationale",
    "independent_of_dataset", "checked_date", "domain_review_status",
}


def _valid_fao_source(rule: dict) -> bool:
    parsed = urlparse(rule["source_url_or_doi"])
    identifiers = parse_qs(parsed.query).get("id", [])
    return (parsed.scheme == "https" and parsed.hostname == "ecocrop.apps.fao.org"
            and len(identifiers) == 1 and identifiers[0].isdigit()
            and f"code {identifiers[0]}" in rule["source_locator"]
            and "Soil PH" in rule["source_locator"] and "Optimal Min/Max" in rule["source_locator"])


def validate_registry(registry: dict) -> dict:
    if registry.get("schema_version") != 2:
        raise ValueError("Candidate registry schema_version must be 2")
    if registry.get("status") != "candidate_rules_pending_independent_domain_review":
        raise ValueError("Candidate registry must remain pending independent domain review")
    rules = registry.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Registry rules must be a list")
    accepted, approved, errors, identifiers = [], [], [], set()
    for index, rule in enumerate(rules):
        missing = sorted(REQUIRED - set(rule)) if isinstance(rule, dict) else sorted(REQUIRED)
        row_errors = []
        if missing:
            row_errors.append("missing:" + ",".join(missing))
        if isinstance(rule, dict) and not missing:
            if rule["rule_id"] in identifiers: row_errors.append("duplicate_rule_id")
            identifiers.add(rule["rule_id"])
            if rule["independent_of_dataset"] is not True: row_errors.append("not_independent")
            if rule["domain_review_status"] not in {"pending", "approved", "rejected"}:
                row_errors.append("invalid_domain_review_status")
            if rule["domain_review_status"] == "approved":
                if not isinstance(rule.get("domain_reviewer"), str) or not rule["domain_reviewer"].strip():
                    row_errors.append("blank:domain_reviewer")
                if not isinstance(rule.get("domain_review_date"), str) or not rule["domain_review_date"].strip():
                    row_errors.append("blank:domain_review_date")
                else:
                    try: date.fromisoformat(rule["domain_review_date"])
                    except ValueError: row_errors.append("invalid:domain_review_date")
            if (not isinstance(rule["original_min"], (int, float)) or isinstance(rule["original_min"], bool)
                    or not isinstance(rule["original_max"], (int, float)) or isinstance(rule["original_max"], bool)
                    or not 0 <= rule["original_min"] < rule["original_max"] <= 14):
                row_errors.append("invalid_numeric_bounds")
            if rule["feature"] != "soil_pH": row_errors.append("unsupported_feature")
            if rule["original_unit"] != "pH scale (-)" or rule["dataset_unit"] != "-": row_errors.append("unit_mismatch")
            if not rule["transformation"].lower().startswith("identity;"): row_errors.append("nonidentity_transformation")
            if not _valid_fao_source(rule): row_errors.append("invalid_fao_source_locator")
            try: date.fromisoformat(rule["checked_date"])
            except ValueError: row_errors.append("invalid_checked_date")
            for key in REQUIRED - {"original_min", "original_max", "independent_of_dataset"}:
                if not isinstance(rule[key], str) or not rule[key].strip(): row_errors.append(f"blank:{key}")
        if row_errors: errors.append({"index": index, "errors": row_errors})
        else:
            accepted.append(rule["rule_id"])
            if rule["domain_review_status"] == "approved": approved.append(rule["rule_id"])
    enabled = bool(rules) and not errors
    return {
        "status": "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW" if enabled else "AGRONOMIC_RULES_BLOCKED",
        "rules_submitted": len(rules), "rules_schema_valid": len(accepted),
        "rules_domain_approved_declared": len(approved), "errors": errors,
        "guidance_enabled": False,
        "interpretation": "Schema validity and approval declarations do not replace an independent agronomic/domain audit.",
    }


def audit(output: Path, registry_path=REGISTRY) -> dict:
    output = Path(output).resolve(); registry_path = Path(registry_path).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    report = validate_registry(read(registry_path))
    output.mkdir(parents=True)
    atomic_json(output / "rule_validation.json", report)
    atomic_json(output / "manifest.json", {"status": report["status"], "guidance_enabled": False,
                "registry_sha256": sha(registry_path), "source_sha256": sha(Path(__file__)), "model_training": False})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(); print(audit(args.output, args.registry))
