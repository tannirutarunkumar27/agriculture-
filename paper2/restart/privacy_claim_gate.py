"""Fail-closed gate for privacy claims in Paper 2 artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.paper2_completion_audit import latest_result_file, read
from paper2.restart.validation_search import ROOT, atomic_json, sha

RESULTS = ROOT / "paper2/results"


def evaluate(*, dp_training: bool, secure_aggregation: bool, threat_model: bool,
             release_audit: bool, cryptographic_review: bool,
             known_security_issues_resolved: bool) -> dict:
    checks = {
        "dp_training": bool(dp_training),
        "secure_aggregation": bool(secure_aggregation),
        "threat_model": bool(threat_model),
        "release_audit": bool(release_audit),
        "cryptographic_review": bool(cryptographic_review),
        "known_security_issues_resolved": bool(known_security_issues_resolved),
    }
    eligible = all(checks.values())
    return {
        "status": "PRIVACY_CLAIM_ELIGIBLE_PENDING_REVIEW" if eligible else "PRIVACY_CLAIM_BLOCKED",
        "privacy_claim_enabled": eligible,
        "checks": checks,
        "interpretation": (
            "All required evidence flags are present; independent review is still required."
            if eligible else
            "Current evidence is simulation/foundation-only; do not claim end-to-end application DP or production-secure aggregation."
        ),
    }


def audit(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    pilot_path = latest_result_file("restart_dp_secagg_pilot_wsl_cpu_audit", "audit.json")
    study_path = latest_result_file("restart_dp_secagg_study_wsl_cpu_audit", "audit.json")
    security_path = ROOT / "paper2/configs/upstream_security_registry_v1.json"
    pilot, study, security = read(pilot_path), read(study_path), read(security_path)
    pilot_complete = (
        pilot.get("status") == "PASS_EXECUTED_DP_SECAGG_PILOT_NOT_CERTIFIED"
        and pilot.get("rounds") == 1
        and pilot.get("semantic_log_audit", {}).get("passed") is True
    )
    study_complete = (
        study.get("status") == "PASS_EXECUTED_DP_SECAGG_STUDY_NOT_CERTIFIED"
        and study.get("rounds") == 20
        and study.get("semantic_log_audit", {}).get("passed") is True
    )
    result = evaluate(
        dp_training=pilot_complete and study_complete,
        secure_aggregation=pilot_complete and study_complete,
        threat_model=True,
        release_audit=study_complete,
        cryptographic_review=security.get("independent_cryptographic_review_complete") is True,
        known_security_issues_resolved=security.get("all_known_security_issues_resolved") is True,
    )
    result["evidence_scope"] = (
        "Audited seven-logical-client WSL simulation with one- and 20-round central-DP plus SecAgg+ execution; "
        "not an authenticated multi-host deployment or production cryptographic certification."
    )
    output.mkdir(parents=True)
    atomic_json(output / "privacy_gate.json", result)
    atomic_json(output / "manifest.json", {
        "status": result["status"], "privacy_claim_enabled": result["privacy_claim_enabled"],
        "source_sha256": sha(Path(__file__)), "training_performed_by_gate": False,
        "evidence": {
            "pilot_audit": str(pilot_path.relative_to(ROOT)),
            "study_audit": str(study_path.relative_to(ROOT)),
            "security_registry": str(security_path.relative_to(ROOT)),
        },
        "evidence_sha256": {
            "pilot_audit": sha(pilot_path),
            "study_audit": sha(study_path),
            "security_registry": sha(security_path),
        },
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.output))
