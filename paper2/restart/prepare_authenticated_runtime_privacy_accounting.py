"""Bind the completed local authenticated DP/SecAgg run to its conditional ledger.

This is an evidence-accounting packet, not a privacy certification.  In
particular, a loopback deployment, a trusted server, public-prepared data, and
the absence of an independent security review prevent an end-to-end DP claim.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, read, sha
from paper2.restart.prepare_authenticated_dp_secagg_adapter import validate as validate_adapter
from paper2.restart.privacy_accounting import gaussian_budget, replacement_sensitivity


PROTOCOL = RESULTS / "restart_dp_secagg_study_protocol_v12"
ADAPTER = RESULTS / "restart_authenticated_dp_secagg_adapter_v2/adapter_binding.json"
RUNTIME_AUDIT = RESULTS / "restart_local_authenticated_dp_secagg_deployment_audit_v2/audit.json"
RUNTIME_RECEIPT = RESULTS / "restart_local_authenticated_dp_secagg_deployment_v4/receipt.json"


def close(left: object, right: object) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-15
    )


def evaluate(protocol: dict, adapter: dict, runtime_audit: dict, receipt: dict) -> dict:
    """Validate only facts needed to construct a conditional accounting packet."""
    contract, run, policy = protocol["contract"], protocol["run"], protocol["policy"]
    rounds, participants = contract.get("adaptive_releases"), contract.get("participants")
    errors: list[str] = []
    if (rounds != 20 or participants != 7 or run.get("num-server-rounds") != rounds
            or run.get("central-dp") is not True or run.get("differential-privacy-enabled") is not True):
        errors.append("frozen_dp_contract")
    if adapter.get("status") != "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING":
        errors.append("adapter_status")
    if runtime_audit.get("status") != "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_SEMANTIC_AUDIT_NOT_CERTIFIED":
        errors.append("runtime_audit_status")
    findings = runtime_audit.get("findings", {})
    evidence = findings.get("protocol_evidence", {})
    if (findings.get("semantic_passed") is not True
            or evidence.get("all_seven_nodes_reached_every_requested_round") is not True
            or evidence.get("rounds_requested") != rounds):
        errors.append("runtime_round_evidence")
    if (receipt.get("status") != "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_RUNTIME_NOT_CERTIFIED"
            or receipt.get("rounds_requested") != rounds
            or receipt.get("online_nodes_before_run") != participants
            or receipt.get("unique_node_ids_before_run") != participants
            or receipt.get("tls_enabled") is not True
            or receipt.get("cli_managed_supernode_auth") is not True
            or receipt.get("clientapp_isolation") != "subprocess"):
        errors.append("runtime_receipt")
    required_policy = {
        "adaptive_model_releases", "client_metrics_released", "private_validation_released",
        "intermediate_checkpoints_released", "noise_seeds_released", "raw_or_individual_updates_released",
    }
    if (not required_policy <= set(policy) or policy.get("adaptive_model_releases") != rounds
            or any(policy.get(key) is not False for key in required_policy - {"adaptive_model_releases"})):
        errors.append("release_policy")
    return {"passed": not errors, "errors": errors}


def prepare(output: Path, protocol_dir: Path = PROTOCOL, adapter_path: Path = ADAPTER,
            runtime_audit_path: Path = RUNTIME_AUDIT, runtime_receipt_path: Path = RUNTIME_RECEIPT) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new directory under paper2/results")
    protocol_dir, adapter_path = Path(protocol_dir).resolve(), Path(adapter_path).resolve()
    runtime_audit_path, runtime_receipt_path = Path(runtime_audit_path).resolve(), Path(runtime_receipt_path).resolve()
    contract = read(protocol_dir / "privacy_contract.json")
    run = read(protocol_dir / "run_config.json")
    policy = read(protocol_dir / "release_policy.json")
    adapter, runtime_audit, receipt = read(adapter_path), read(runtime_audit_path), read(runtime_receipt_path)
    current_adapter = validate_adapter()
    if current_adapter["app_sources_sha256"] != adapter.get("app_sources_sha256"):
        raise ValueError("Deployment source no longer matches the immutable adapter binding")
    if adapter.get("frozen_study_manifest_sha256") != sha(protocol_dir / "manifest.json"):
        raise ValueError("Adapter is not bound to the frozen DP study")
    if runtime_audit.get("source_receipt_sha256") != sha(runtime_receipt_path):
        raise ValueError("Runtime audit is not bound to the supplied raw receipt")
    records = {"contract": contract, "run": run, "policy": policy}
    checks = evaluate(records, adapter, runtime_audit, receipt)
    if not checks["passed"]:
        raise ValueError(f"Runtime privacy-accounting inputs failed validation: {checks['errors']}")
    sensitivity = replacement_sensitivity(contract["clip_norm"], contract["participants"])
    noise_std = contract["noise_multiplier"] * contract["clip_norm"] / contract["participants"]
    budget = gaussian_budget([{"sensitivity": sensitivity, "noise_std": noise_std, "count": contract["adaptive_releases"]}], contract["delta"])
    reported = contract["conditional_budget"]
    if (not close(reported.get("rho"), budget["rho"])
            or not close(reported.get("epsilon_upper_bound"), budget["epsilon_upper_bound"])
            or reported.get("releases") != budget["releases"]):
        raise ValueError("Frozen conditional Gaussian accounting does not recompute")
    output.mkdir(parents=True)
    packet = {
        "status": "LOCAL_AUTHENTICATED_RUNTIME_PRIVACY_ACCOUNTING_READY_PENDING_INDEPENDENT_REVIEW",
        "execution_scope": "single_laptop_wsl_loopback",
        "protocol": "Flower 1.42 SecAgg+ with central DP and client-side fixed clipping",
        "participants": contract["participants"], "rounds": contract["adaptive_releases"],
        "tls_enabled": True, "supernode_authentication_enabled": True,
        "conditional_zcdp_budget": budget,
        "release_accounting": {
            "adaptive_model_releases": policy["adaptive_model_releases"],
            "client_metrics_released": False, "private_validation_released": False,
            "intermediate_checkpoints_released": False, "noise_seeds_released": False,
            "raw_or_individual_updates_released": False,
            "stored_runtime_evidence": "sanitized operational logs and hash-bound receipts only",
        },
        "bound_evidence": {
            "frozen_protocol_manifest_sha256": sha(protocol_dir / "manifest.json"),
            "adapter_binding_sha256": sha(adapter_path),
            "runtime_receipt_sha256": sha(runtime_receipt_path),
            "runtime_audit_sha256": sha(runtime_audit_path),
        },
        "privacy_claim_enabled": False, "security_claim_enabled": False, "external_deployment_claim_enabled": False,
        "excluded_or_unverified_components": [
            "independent cryptographic/security review", "real multi-host deployment",
            "end-to-end preprocessing and data-governance accounting", "hyperparameter-selection accounting",
            "trusted-server threat-model assessment", "farm-level/private-data validation",
        ],
        "interpretation": "This packet verifies a completed local authenticated runtime against a conditional Gaussian ledger. It is not an end-to-end differential-privacy guarantee, cryptographic-security certificate, or external deployment claim.",
    }
    atomic_json(output / "runtime_privacy_accounting.json", packet)
    atomic_json(output / "independent_review.template.json", {
        "schema_version": 1, "reviewer": "", "reviewer_affiliation": "", "review_date": "",
        "independent": "", "conflict_of_interest_disclosed": "", "decision": "", "rationale": "",
        "packet_sha256": sha(output / "runtime_privacy_accounting.json"),
        "required_decision": "approve-for-bounded-local-study, reject, or request-changes; no production-certification option",
    })
    atomic_json(output / "manifest.json", {
        "status": packet["status"], "packet_sha256": sha(output / "runtime_privacy_accounting.json"),
        "privacy_claim_enabled": False, "security_claim_enabled": False,
    })
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--runtime-audit", type=Path, default=RUNTIME_AUDIT)
    parser.add_argument("--runtime-receipt", type=Path, default=RUNTIME_RECEIPT)
    args = parser.parse_args()
    print(prepare(args.output, args.protocol_dir, args.adapter, args.runtime_audit, args.runtime_receipt))
