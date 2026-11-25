"""Fail-closed activation gate for the restricted observed-set ranking study."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from paper2.restart.review_ranking_experiments import APPROVAL_FIELDS, RESULTS, atomic_json, sha
from paper2.restart.validation_search import read

REVIEW = RESULTS / "restart_ranking_experiment_review_v2"
REQUIRED_TEXT = ["reviewer", "reviewer_affiliation", "review_date", "source_check_date", "source_locator", "rationale"]
INDEPENDENCE_FIELDS = ["reviewer_independent", "conflict_of_interest_disclosed"]


def validate_protocol(protocol: dict) -> None:
    expected_approvals = set(APPROVAL_FIELDS)
    prohibited = {
        "causal treatment effect", "universal best crop", "unobserved counterfactual performance",
        "profitability", "nutritional benefit", "prospective field utility",
    }
    utility = protocol.get("utility", {})
    if (protocol.get("version") != "restricted-observed-ranking-review-v1"
            or protocol.get("status") != "FROZEN_REVIEW_CONTRACT_NOT_APPROVED_FOR_RANKING"
            or protocol.get("candidate_scope") != "only packages actually tested in the source experiment"
            or utility.get("measure") != "harvest seed dry matter" or utility.get("unit") != "t/ha"
            or utility.get("economic") is not False or utility.get("nutritional") is not False
            or set(protocol.get("required_block_approvals", [])) != expected_approvals
            or not prohibited <= set(protocol.get("prohibited_claims", []))):
        raise ValueError("Ranking protocol differs from the frozen restricted observed-set contract")


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _true(value: object) -> bool:
    return _text(value).lower() in {"true", "yes", "1"}


def evaluate(blocks: pd.DataFrame, approvals: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required_blocks = {"block_id", "experiment_id", "reference_present"}
    required_approvals = {"block_id", "experiment_id", "decision", *REQUIRED_TEXT,
                          *INDEPENDENCE_FIELDS, *APPROVAL_FIELDS}
    if not required_blocks <= set(blocks):
        raise ValueError("Block review schema is incomplete")
    if not required_approvals <= set(approvals):
        raise ValueError("Approval schema is incomplete")
    if blocks.block_id.isna().any() or blocks.block_id.duplicated().any():
        raise ValueError("Block review identities must be unique")
    if approvals.block_id.isna().any() or approvals.block_id.duplicated().any():
        raise ValueError("Approval identities must be unique")
    if set(blocks.block_id) != set(approvals.block_id):
        raise ValueError("Approval file must contain every reviewed block exactly once")

    joined = blocks[["block_id", "experiment_id", "reference_present"]].merge(
        approvals, on="block_id", suffixes=("_review", "_approval"), validate="one_to_one"
    )
    if not joined.experiment_id_review.astype(str).eq(joined.experiment_id_approval.astype(str)).all():
        raise ValueError("Approval experiment identifiers differ from the reviewed blocks")

    decisions = []
    for row in joined.itertuples(index=False):
        decision = _text(row.decision).lower()
        errors = []
        if decision == "":
            status = "PENDING_REVIEW"
        elif decision not in {"include", "exclude"}:
            status = "INVALID_REVIEW"
            errors.append("decision_must_be_include_or_exclude")
        else:
            for field in REQUIRED_TEXT:
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
                if not _true(row.reference_present):
                    errors.append("publication_reference_missing")
                for field in APPROVAL_FIELDS:
                    if not _true(getattr(row, field)):
                        errors.append(f"{field}_not_true")
            status = "APPROVED_INCLUDE" if decision == "include" and not errors else (
                "APPROVED_EXCLUDE" if decision == "exclude" and not errors else "INVALID_REVIEW"
            )
        decisions.append({
            "block_id": row.block_id,
            "experiment_id": row.experiment_id_review,
            "decision": decision,
            "gate_status": status,
            "errors": ";".join(errors),
        })
    audited = pd.DataFrame(decisions).sort_values("block_id").reset_index(drop=True)
    pending = int(audited.gate_status.eq("PENDING_REVIEW").sum())
    invalid = int(audited.gate_status.eq("INVALID_REVIEW").sum())
    included = int(audited.gate_status.eq("APPROVED_INCLUDE").sum())
    excluded = int(audited.gate_status.eq("APPROVED_EXCLUDE").sum())
    enabled = pending == 0 and invalid == 0 and included > 0 and included + excluded == len(audited)
    status = "RANKING_APPROVED_FOR_RESTRICTED_OBSERVED_SET" if enabled else (
        "RANKING_APPROVAL_BLOCKED_INVALID_REVIEW" if invalid else
        "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW" if pending else
        "RANKING_APPROVAL_BLOCKED_NO_INCLUDED_BLOCKS"
    )
    report = {
        "status": status,
        "blocks_total": int(len(audited)),
        "blocks_pending": pending,
        "blocks_invalid": invalid,
        "blocks_approved_include": included,
        "blocks_approved_exclude": excluded,
        "ranking_enabled": enabled,
        "model_training": False,
        "scope_if_enabled": "restricted observed management packages in approved included blocks only",
    }
    return report, audited


def audit(
    output: Path,
    blocks_path: Path = REVIEW / "block_review.csv",
    approvals_path: Path = REVIEW / "approval_template.csv",
    protocol_path: Path = REVIEW / "ranking_protocol.json",
) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    blocks_path, approvals_path, protocol_path = map(lambda value: Path(value).resolve(),
                                                      (blocks_path, approvals_path, protocol_path))
    validate_protocol(read(protocol_path))
    report, decisions = evaluate(pd.read_csv(blocks_path), pd.read_csv(approvals_path, keep_default_na=False))
    output.mkdir(parents=True)
    decisions.to_csv(output / "block_decisions.csv", index=False)
    atomic_json(output / "ranking_gate.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "ranking_enabled": report["ranking_enabled"], "model_training": False,
        "block_review_sha256": sha(blocks_path), "approval_file_sha256": sha(approvals_path),
        "protocol_sha256": sha(protocol_path), "code_sha256": sha(Path(__file__)),
        "decisions_sha256": sha(output / "block_decisions.csv"),
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, default=REVIEW / "block_review.csv")
    parser.add_argument("--approvals", type=Path, default=REVIEW / "approval_template.csv")
    parser.add_argument("--protocol", type=Path, default=REVIEW / "ranking_protocol.json")
    args = parser.parse_args()
    print(audit(args.output, args.blocks, args.approvals, args.protocol))
