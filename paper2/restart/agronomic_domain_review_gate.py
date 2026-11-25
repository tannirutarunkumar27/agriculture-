"""Create and audit independent-domain-review declarations for agronomic rules."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from paper2.restart.paper2_completion_audit import read
from paper2.restart.review_ranking_experiments import RESULTS, ROOT, atomic_json, sha

REGISTRY = ROOT / "paper2/configs/knowledge_registry_v2_candidates.json"
REVIEW_FIELDS = [
    "source_record_confirmed",
    "taxonomy_mapping_approved",
    "unit_transformation_approved",
    "geographic_scope_approved",
    "cultivar_stage_scope_approved",
    "water_regime_scope_approved",
    "soft_prior_use_approved",
]
INDEPENDENCE_FIELDS = ["reviewer_independent", "conflict_of_interest_disclosed"]


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _true(value: object) -> bool:
    return _text(value).lower() in {"true", "yes", "1"}


def make_template(registry: dict) -> pd.DataFrame:
    rules = registry.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("A nonempty candidate rule registry is required")
    rows = []
    for rule in rules:
        if not isinstance(rule, dict) or not _text(rule.get("rule_id")):
            raise ValueError("Every candidate rule needs a rule_id")
        row = {
            "rule_id": rule["rule_id"], "crop_taxon": rule.get("crop_taxon", ""),
            "feature": rule.get("feature", ""), "reviewer": "", "reviewer_affiliation": "",
            "review_date": "", "source_check_date": "",
            "decision": "", "rationale": "",
        }
        row.update({field: "" for field in INDEPENDENCE_FIELDS})
        row.update({field: "" for field in REVIEW_FIELDS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.rule_id.duplicated().any():
        raise ValueError("Candidate rule identifiers must be unique")
    return frame.sort_values("rule_id").reset_index(drop=True)


def evaluate(registry: dict, reviews: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    template = make_template(registry)
    required = set(template.columns)
    if not required <= set(reviews):
        raise ValueError("Domain review schema is incomplete")
    if reviews.rule_id.isna().any() or reviews.rule_id.duplicated().any():
        raise ValueError("Domain review rule identifiers must be unique")
    if set(reviews.rule_id) != set(template.rule_id):
        raise ValueError("Review file must contain every candidate rule exactly once")
    joined = template[["rule_id", "crop_taxon", "feature"]].merge(
        reviews, on="rule_id", suffixes=("_registry", "_review"), validate="one_to_one"
    )
    if not joined.crop_taxon_registry.astype(str).eq(joined.crop_taxon_review.astype(str)).all():
        raise ValueError("Review crop taxon differs from registry")
    if not joined.feature_registry.astype(str).eq(joined.feature_review.astype(str)).all():
        raise ValueError("Review feature differs from registry")

    rows = []
    for row in joined.itertuples(index=False):
        decision = _text(row.decision).lower()
        errors = []
        if decision == "":
            status = "PENDING_REVIEW"
        elif decision not in {"include", "exclude"}:
            status = "INVALID_REVIEW"
            errors.append("decision_must_be_include_or_exclude")
        else:
            for field in ("reviewer", "reviewer_affiliation", "review_date", "source_check_date", "rationale"):
                if not _text(getattr(row, field)):
                    errors.append(f"blank_{field}")
            for field in ("review_date", "source_check_date"):
                try:
                    date.fromisoformat(_text(getattr(row, field)))
                except ValueError:
                    errors.append(f"{field}_must_be_iso_yyyy_mm_dd")
            for field in INDEPENDENCE_FIELDS:
                if not _true(getattr(row, field)):
                    errors.append(f"{field}_not_true")
            if decision == "include":
                for field in REVIEW_FIELDS:
                    if not _true(getattr(row, field)):
                        errors.append(f"{field}_not_true")
            status = "DECLARED_INCLUDE" if decision == "include" and not errors else (
                "DECLARED_EXCLUDE" if decision == "exclude" and not errors else "INVALID_REVIEW"
            )
        rows.append({"rule_id": row.rule_id, "crop_taxon": row.crop_taxon_registry,
                     "feature": row.feature_registry, "decision": decision,
                     "gate_status": status, "errors": ";".join(errors)})
    audited = pd.DataFrame(rows).sort_values("rule_id").reset_index(drop=True)
    pending = int(audited.gate_status.eq("PENDING_REVIEW").sum())
    invalid = int(audited.gate_status.eq("INVALID_REVIEW").sum())
    included = int(audited.gate_status.eq("DECLARED_INCLUDE").sum())
    excluded = int(audited.gate_status.eq("DECLARED_EXCLUDE").sum())
    declarations_complete = pending == 0 and invalid == 0 and included > 0 and included + excluded == len(audited)
    status = "AGRONOMIC_DECLARATIONS_COMPLETE_INDEPENDENT_AUDIT_REQUIRED" if declarations_complete else (
        "AGRONOMIC_DOMAIN_REVIEW_BLOCKED_INVALID" if invalid else
        "AGRONOMIC_DOMAIN_REVIEW_PENDING" if pending else
        "AGRONOMIC_DOMAIN_REVIEW_BLOCKED_NO_INCLUDED_RULES"
    )
    return {
        "status": status, "rules_total": int(len(audited)), "rules_pending": pending,
        "rules_invalid": invalid, "rules_declared_include": included,
        "rules_declared_exclude": excluded, "declarations_complete": declarations_complete,
        "guidance_enabled": False, "model_training": False,
        "interpretation": "Conflict-disclosed independent reviewer declarations require a separate audit and frozen guidance study before use.",
    }, audited


def audit(output: Path, registry_path: Path = REGISTRY, reviews_path: Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    registry_path = Path(registry_path).resolve()
    registry = read(registry_path)
    template = make_template(registry)
    reviews = pd.read_csv(reviews_path, keep_default_na=False) if reviews_path else template
    report, audited = evaluate(registry, reviews)
    output.mkdir(parents=True)
    template.to_csv(output / "domain_review_template.csv", index=False)
    audited.to_csv(output / "rule_decisions.csv", index=False)
    atomic_json(output / "domain_review_gate.json", report)
    manifest = {
        "status": report["status"], "guidance_enabled": False, "model_training": False,
        "registry_sha256": sha(registry_path), "code_sha256": sha(Path(__file__)),
        "review_input_sha256": sha(Path(reviews_path).resolve()) if reviews_path else sha(output / "domain_review_template.csv"),
        "template_sha256": sha(output / "domain_review_template.csv"),
        "decisions_sha256": sha(output / "rule_decisions.csv"),
        "gate_sha256": sha(output / "domain_review_gate.json"),
    }
    atomic_json(output / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    print(audit(args.output, args.registry, args.reviews))
