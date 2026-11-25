"""Validate candidate publication links for source experiments with blank references.

The gate is deliberately separate from the frozen Zenodo references workbook.  It
can document plausible links, but it cannot make a link usable unless an exact
site/year/design match and an independent review are both recorded.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from paper2.restart.review_ranking_experiments import RESULTS, atomic_json, sha

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "paper2/configs/ranking_reference_candidates_v1.json"
EXPERIMENT_REVIEW = RESULTS / "restart_ranking_experiment_review_v2/experiment_review.csv"

REQUIRED = {
    "experiment_id", "candidate_source", "doi_or_url", "source_type",
    "exact_site_year_link", "coverage", "evidence_locator", "evidence_summary",
    "confidence", "independent_review_status", "reviewer", "review_date",
    "review_rationale", "usable_as_original_reference",
}
LINK_STATES = {"confirmed", "unconfirmed", "mismatch"}
COVERAGE_STATES = {"full", "partial", "none"}
CONFIDENCE_STATES = {"high", "medium", "low"}
REVIEW_STATES = {"pending", "approved", "rejected"}
ORIGINAL_SOURCE_TYPES = {
    "peer_reviewed_original_article", "doctoral_thesis", "authoritative_experiment_report",
}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _valid_url(value: object) -> bool:
    parsed = urlparse(_text(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _valid_iso_date(value: object) -> bool:
    try:
        date.fromisoformat(_text(value))
        return True
    except ValueError:
        return False


def evaluate(registry: dict, source_missing_experiments: set[str]) -> tuple[dict, pd.DataFrame]:
    """Return a fail-closed report and row audit for a candidate registry."""
    if registry.get("schema_version") != "1.0":
        raise ValueError("Reference candidate registry schema_version must be 1.0")
    if not _valid_iso_date(registry.get("checked_date")):
        raise ValueError("Reference candidate registry requires an ISO checked_date")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Registry candidates must be a non-empty list")
    rows = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be an object")
        missing = sorted(REQUIRED - set(candidate))
        errors = [f"missing_{field}" for field in missing]
        experiment_id = _text(candidate.get("experiment_id"))
        link = _text(candidate.get("exact_site_year_link")).lower()
        coverage = _text(candidate.get("coverage")).lower()
        confidence = _text(candidate.get("confidence")).lower()
        review = _text(candidate.get("independent_review_status")).lower()
        source_type = _text(candidate.get("source_type")).lower()
        declared_usable = candidate.get("usable_as_original_reference") is True

        for field in ("experiment_id", "candidate_source", "source_type", "evidence_summary"):
            if not _text(candidate.get(field)):
                errors.append(f"blank_{field}")
        if experiment_id not in source_missing_experiments:
            errors.append("experiment_not_missing_from_frozen_source_registry")
        if not _valid_url(candidate.get("doi_or_url")):
            errors.append("invalid_candidate_url")
        if link not in LINK_STATES:
            errors.append("invalid_exact_site_year_link")
        if coverage not in COVERAGE_STATES:
            errors.append("invalid_coverage")
        if confidence not in CONFIDENCE_STATES:
            errors.append("invalid_confidence")
        if review not in REVIEW_STATES:
            errors.append("invalid_independent_review_status")
        if link == "confirmed" and not _text(candidate.get("evidence_locator")):
            errors.append("confirmed_link_requires_evidence_locator")
        if review in {"approved", "rejected"}:
            for field in ("reviewer", "reviewer_affiliation", "review_rationale"):
                if not _text(candidate.get(field)):
                    errors.append(f"completed_review_requires_{field}")
            for field in ("review_date", "source_check_date"):
                if not _valid_iso_date(candidate.get(field)):
                    errors.append(f"completed_review_requires_iso_{field}")
            if candidate.get("reviewer_independent") is not True:
                errors.append("reviewer_independence_not_confirmed")
            if candidate.get("conflict_of_interest_disclosed") is not True:
                errors.append("conflict_of_interest_not_disclosed")

        computed_usable = (
            not errors
            and link == "confirmed"
            and coverage == "full"
            and review == "approved"
            and source_type in ORIGINAL_SOURCE_TYPES
        )
        if declared_usable != computed_usable:
            errors.append("declared_usability_disagrees_with_fail_closed_rule")
            computed_usable = False

        if errors:
            status = "INVALID_CANDIDATE_RECORD"
        elif computed_usable:
            status = "APPROVED_EXACT_ORIGINAL_REFERENCE"
        elif link == "mismatch" or review == "rejected":
            status = "REJECTED_AS_ORIGINAL_REFERENCE"
        elif link == "confirmed":
            status = "EXACT_MATCH_PENDING_INDEPENDENT_REVIEW"
        else:
            status = "UNCONFIRMED_CANDIDATE"
        rows.append({
            "experiment_id": experiment_id,
            "candidate_source": _text(candidate.get("candidate_source")),
            "doi_or_url": _text(candidate.get("doi_or_url")),
            "source_type": source_type,
            "exact_site_year_link": link,
            "coverage": coverage,
            "confidence": confidence,
            "independent_review_status": review,
            "computed_usable": computed_usable,
            "gate_status": status,
            "errors": ";".join(sorted(set(errors))),
        })

    audited = pd.DataFrame(rows).sort_values(["experiment_id", "doi_or_url"]).reset_index(drop=True)
    covered = set(audited.experiment_id)
    missing_candidate_records = sorted(source_missing_experiments - covered)
    duplicate_experiments = sorted(audited.loc[audited.experiment_id.duplicated(False), "experiment_id"].unique())
    usable_experiments = set(audited.loc[audited.computed_usable, "experiment_id"])
    unresolved = sorted(source_missing_experiments - usable_experiments)
    invalid = int(audited.gate_status.eq("INVALID_CANDIDATE_RECORD").sum())
    enabled = not invalid and not missing_candidate_records and not unresolved and not duplicate_experiments
    status = (
        "RANKING_REFERENCE_SUPPLEMENT_ELIGIBLE"
        if enabled
        else "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_INVALID_RECORDS"
        if invalid or missing_candidate_records or duplicate_experiments
        else "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_UNRESOLVED"
    )
    report = {
        "status": status,
        "source_missing_experiments": len(source_missing_experiments),
        "candidate_records": int(len(audited)),
        "experiments_with_candidates": len(covered & source_missing_experiments),
        "experiments_without_candidates": missing_candidate_records,
        "duplicate_experiment_candidate_records": duplicate_experiments,
        "exact_matches": int(audited.exact_site_year_link.eq("confirmed").sum()),
        "exact_matches_pending_independent_review": int(
            audited.gate_status.eq("EXACT_MATCH_PENDING_INDEPENDENT_REVIEW").sum()
        ),
        "approved_usable_reference_supplements": len(usable_experiments),
        "unresolved_experiments": unresolved,
        "invalid_candidate_records": invalid,
        "ranking_reference_supplement_enabled": enabled,
        "source_workbook_modified": False,
        "ranking_enabled": False,
        "model_training": False,
        "model_inference": False,
    }
    return report, audited


def audit(output: Path, registry_path: Path = REGISTRY,
          experiment_review_path: Path = EXPERIMENT_REVIEW) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    registry_path = Path(registry_path).resolve()
    experiment_review_path = Path(experiment_review_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments = pd.read_csv(experiment_review_path, keep_default_na=False)
    if not {"experiment_id", "reference_present"} <= set(experiments):
        raise ValueError("Experiment review schema is incomplete")
    source_missing = set(experiments.loc[
        experiments.reference_present.astype(str).str.lower().eq("false"), "experiment_id"
    ].astype(str))
    report, audited = evaluate(registry, source_missing)
    output.mkdir(parents=True)
    audited.to_csv(output / "candidate_reference_audit.csv", index=False)
    template = audited[["experiment_id", "candidate_source", "doi_or_url", "exact_site_year_link", "coverage"]].copy()
    template["independent_decision"] = ""
    template["reviewer"] = ""
    template["reviewer_affiliation"] = ""
    template["review_date"] = ""
    template["source_check_date"] = ""
    template["reviewer_independent"] = ""
    template["conflict_of_interest_disclosed"] = ""
    template["evidence_locator_verified"] = ""
    template["exact_site_year_design_match_verified"] = ""
    template["original_source_type_verified"] = ""
    template["rationale"] = ""
    template.to_csv(output / "independent_review_template.csv", index=False)
    atomic_json(output / "reference_gate.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"],
        "ranking_reference_supplement_enabled": report["ranking_reference_supplement_enabled"],
        "registry_sha256": sha(registry_path),
        "experiment_review_sha256": sha(experiment_review_path),
        "candidate_audit_sha256": sha(output / "candidate_reference_audit.csv"),
        "review_template_sha256": sha(output / "independent_review_template.csv"),
        "gate_sha256": sha(output / "reference_gate.json"),
        "code_sha256": sha(Path(__file__)),
        "model_training": False,
        "model_inference": False,
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--experiment-review", type=Path, default=EXPERIMENT_REVIEW)
    args = parser.parse_args()
    print(audit(args.output, args.registry, args.experiment_review))
