"""Join reference and treatment-block evidence for independent ranking review.

The packet is an administrative/research aid only. It never grants a source
approval, a block approval, or permission to execute ranking.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2.restart.paper2_completion_audit import RESULTS, atomic_json, latest_result_file, read, sha


def as_bool(values: pd.Series) -> pd.Series:
    """Parse review booleans strictly; non-boolean text must never become true."""
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        raise ValueError(f"Invalid boolean review values: {sorted(normalized[invalid].unique())}")
    return normalized.isin({"true", "1", "yes"})


def prepare(output: Path, reference_audit: Path | None = None, review_package: Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    reference_audit = Path(reference_audit or latest_result_file(
        "restart_ranking_reference_gate", "candidate_reference_audit.csv"
    )).resolve()
    review_package = Path(review_package or latest_result_file(
        "restart_ranking_experiment_review", "summary.json"
    ).parent).resolve()
    references = pd.read_csv(reference_audit, keep_default_na=False)
    blocks = pd.read_csv(review_package / "block_review.csv", keep_default_na=False)
    approvals = pd.read_csv(review_package / "approval_template.csv", keep_default_na=False)
    protocol = read(review_package / "ranking_protocol.json")
    required_reference = {"experiment_id", "gate_status", "computed_usable", "doi_or_url"}
    required_blocks = {"block_id", "experiment_id", "reference_present", "blocking_reasons"}
    if not required_reference <= set(references) or not required_blocks <= set(blocks):
        raise ValueError("Reference or block review schema is incomplete")
    if set(blocks.block_id) != set(approvals.block_id):
        raise ValueError("Every candidate block must have exactly one approval row")
    # CSV readers may preserve review fields as strings; normalize before any
    # eligibility or required-action decision so "False" cannot be truthy.
    references["computed_usable"] = as_bool(references["computed_usable"])
    blocks["reference_present"] = as_bool(blocks["reference_present"])
    reference_summary = references[["experiment_id", "doi_or_url", "gate_status", "computed_usable"]].copy()
    reference_summary.columns = ["experiment_id", "candidate_doi_or_url", "reference_gate_status", "reference_usable"]
    worklist = blocks.merge(reference_summary, on="experiment_id", how="left", validate="many_to_one")
    worklist["reference_gate_status"] = worklist["reference_gate_status"].fillna("FROZEN_REFERENCE_PRESENT_NO_SUPPLEMENT_NEEDED")
    worklist["reference_usable"] = as_bool(worklist["reference_usable"].fillna(False))
    worklist["required_next_action"] = worklist.apply(
        lambda row: "independent source-reference review before block approval"
        if not bool(row.reference_present) else "independent block-design review", axis=1
    )
    output.mkdir(parents=True)
    references.to_csv(output / "reference_review_worklist.csv", index=False)
    worklist.to_csv(output / "block_reference_worklist.csv", index=False)
    approvals.to_csv(output / "block_approval_template.csv", index=False)
    (output / "review_sequence.md").write_text(
        "# Ranking review sequence\n\n"
        "1. Resolve every missing original-publication reference using the reference worklist. "
        "A candidate is not usable unless it has an exact site/year/design link and independent approval.\n"
        "2. For each candidate block, document original randomization, treatment and harvest comparability, "
        "replicate/tie handling, and restricted-utility approval in the block template.\n"
        "3. Keep every decision as include or exclude. Do not score, rank, or train until both gates pass.\n\n"
        f"Frozen restricted protocol: `{protocol.get('status', '')}`.\n",
        encoding="utf-8",
    )
    report = {
        "status": "RANKING_COMBINED_REVIEW_PACKET_READY_PENDING_INDEPENDENT_REVIEW",
        "unresolved_reference_candidates": int((~as_bool(references["computed_usable"])).sum()),
        "candidate_blocks": int(len(blocks)),
        "blocks_with_missing_frozen_reference": int((~as_bool(blocks["reference_present"])).sum()),
        "ranking_enabled": False, "model_training": False,
        "interpretation": "Combined review worklist only; source and block approvals remain external and ranking is disabled.",
    }
    atomic_json(output / "review_packet.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "reference_audit_sha256": sha(reference_audit),
        "block_review_sha256": sha(review_package / "block_review.csv"),
        "approval_template_sha256": sha(review_package / "approval_template.csv"),
        "protocol_sha256": sha(review_package / "ranking_protocol.json"),
        "packet_sha256": sha(output / "review_packet.json"), "ranking_enabled": False,
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path)
    parser.add_argument("--review-package", type=Path)
    args = parser.parse_args()
    print(prepare(args.output, args.reference_audit, args.review_package))
