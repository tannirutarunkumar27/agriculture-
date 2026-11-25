"""Build a recomputed, source-bound handoff packet for DP/SecAgg review.

This creates review evidence only. It never turns a conditional central-DP
calculation into an end-to-end privacy, security, or deployment certificate.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from paper2.restart.paper2_completion_audit import RESULTS, atomic_json, latest_result_file, read, sha
from paper2.restart.privacy_accounting import gaussian_budget, replacement_sensitivity


def _close(left: object, right: object) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-15
    )


def _audit_path(protocol: Path) -> Path:
    rounds = read(protocol / "run_config.json").get("num-server-rounds")
    prefix = "restart_dp_secagg_pilot_wsl_cpu_audit" if rounds == 1 else "restart_dp_secagg_study_wsl_cpu_audit"
    return latest_result_file(prefix, "audit.json")


def prepare(output: Path, protocol: Path, execution_audit: Path | None = None) -> dict:
    output, protocol = Path(output).resolve(), Path(protocol).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    contract = read(protocol / "privacy_contract.json")
    run = read(protocol / "run_config.json")
    policy = read(protocol / "release_policy.json") if (protocol / "release_policy.json").is_file() else {}
    required = {"participants", "clip_norm", "noise_multiplier", "delta", "adaptive_releases", "flower_noise_std", "conditional_budget"}
    if not required <= set(contract):
        raise ValueError("Frozen privacy contract is incomplete")
    participants, rounds = contract["participants"], contract["adaptive_releases"]
    if type(participants) is not int or type(rounds) is not int or participants < 1 or rounds < 1:
        raise ValueError("Frozen participant/release counts are invalid")
    if run.get("num-server-rounds") != rounds or run.get("central-dp") is not True or run.get("differential-privacy-enabled") is not True:
        raise ValueError("Run configuration differs from the frozen DP contract")
    sensitivity = replacement_sensitivity(contract["clip_norm"], participants)
    expected_std = float(contract["noise_multiplier"]) * float(contract["clip_norm"]) / participants
    recomputed = gaussian_budget([{"sensitivity": sensitivity, "noise_std": expected_std, "count": rounds}], contract["delta"])
    reported = contract["conditional_budget"]
    if (not _close(contract["flower_noise_std"], expected_std)
            or any(not _close(reported.get(key), recomputed[key]) for key in ("rho", "epsilon_upper_bound", "delta"))
            or reported.get("releases") != rounds):
        raise ValueError("Frozen conditional budget does not recompute from its declared mechanism")
    if policy:
        required_policy = {"adaptive_model_releases", "client_metrics_released", "private_validation_released",
                           "intermediate_checkpoints_released", "noise_seeds_released", "raw_or_individual_updates_released"}
        if not required_policy <= set(policy) or policy["adaptive_model_releases"] != rounds:
            raise ValueError("Release policy is incomplete or has a mismatched release count")
        if any(policy[name] is not False for name in required_policy - {"adaptive_model_releases"}):
            raise ValueError("Private or intermediate release is not permitted in this review packet")
    audit_path = Path(execution_audit).resolve() if execution_audit else _audit_path(protocol)
    audit = read(audit_path)
    if (not str(audit.get("status", "")).startswith("PASS_EXECUTED_DP_SECAGG_")
            or audit.get("rounds") != rounds or audit.get("semantic_log_audit", {}).get("passed") is not True):
        raise ValueError("Execution audit is not a successful semantic DP/SecAgg receipt")
    output.mkdir(parents=True)
    packet = {
        "status": "DP_SECAGG_PRIVACY_REVIEW_PACKET_READY_PENDING_INDEPENDENT_REVIEW",
        "mechanism": contract.get("mechanism"), "participants": participants, "rounds": rounds,
        "central_dp": True, "trusted_server_required": contract.get("trusted_server_required"),
        "individual_updates_hidden_from_server": contract.get("individual_updates_hidden_from_server"),
        "sensitivity_replacement": sensitivity, "noise_std": expected_std,
        "conditional_zcdp_budget": recomputed, "release_policy": policy,
        "execution_audit_status": audit["status"],
        "privacy_claim_enabled": False, "security_claim_enabled": False, "deployment_claim_enabled": False,
        "known_security_issues_resolved": contract.get("known_security_issues_resolved"),
        "independent_cryptographic_review_complete": contract.get("independent_cryptographic_review_complete"),
        "interpretation": "Recomputed conditional accounting and logical-client execution evidence only; no end-to-end application privacy, cryptographic-security, or farm-deployment claim.",
    }
    atomic_json(output / "privacy_review_packet.json", packet)
    atomic_json(output / "independent_review.template.json", {
        "schema_version": 1, "reviewer": "", "reviewer_affiliation": "", "review_date": "",
        "independent": "", "conflict_of_interest_disclosed": "", "decision": "",
        "rationale": "", "packet_sha256": sha(output / "privacy_review_packet.json"),
        "required_decision": "approve-for-bounded-study, reject, or request-changes; no production certification option",
    })
    atomic_json(output / "manifest.json", {
        "status": packet["status"], "protocol_privacy_contract_sha256": sha(protocol / "privacy_contract.json"),
        "run_config_sha256": sha(protocol / "run_config.json"),
        "release_policy_sha256": sha(protocol / "release_policy.json") if policy else None,
        "execution_audit_sha256": sha(audit_path), "packet_sha256": sha(output / "privacy_review_packet.json"),
        "privacy_claim_enabled": False,
    })
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-audit", type=Path)
    args = parser.parse_args()
    print(prepare(args.output, args.protocol, args.execution_audit))
