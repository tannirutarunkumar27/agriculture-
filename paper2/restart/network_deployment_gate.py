"""Fail-closed evidence gate for an authenticated multi-node Flower deployment."""
from __future__ import annotations

import argparse
import ipaddress
import re
from pathlib import Path

from paper2.restart.validation_search import ROOT, atomic_json, read, sha

RESULTS = ROOT / "paper2/results"
FLOWER_VERSION = "1.42.0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RECEIPTS = ("tls_handshake", "supernode_registration", "online_nodes", "secagg_run", "dp_audit", "result_retrieval")


def _valid_remote_endpoint(address: object) -> bool:
    value = str(address or "").strip()
    try:
        if value.startswith("["):
            close = value.index("]")
            host, port_text = value[1:close], value[close + 2:]
        else:
            host, port_text = value.rsplit(":", 1)
        port = int(port_text)
        if not 1 <= port <= 65535:
            return False
    except (ValueError, IndexError):
        return False
    lowered = host.lower().rstrip(".")
    try:
        ip = ipaddress.ip_address(lowered)
        # Reject private, loopback, link-local, documentation, multicast, and
        # otherwise reserved addresses. A static plan cannot prove reachability,
        # but it must not accept an address that is categorically non-deployable.
        return ip.is_global
    except ValueError:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", lowered):
            return False
        return lowered != "localhost" and not lowered.endswith((".example", ".invalid", ".test", ".local"))


def evaluate(evidence: dict | None) -> dict:
    if evidence is None:
        return {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False,
                "missing": ["remote Linux hosts", "TLS receipts", "registered SuperNode identities", "online-node receipt", "audited run receipt"]}
    if not isinstance(evidence, dict):
        return {"status": "FAIL_NETWORK_DEPLOYMENT_GATE", "deployment_verified": False,
                "errors": ["evidence_object"],
                "qualification": "Malformed evidence cannot establish a deployment claim."}
    errors = []
    if evidence.get("schema_version") != 1: errors.append("schema_version")
    if evidence.get("deployment_mode") != "authenticated_multi_process_network": errors.append("deployment_mode")
    required_true = ("tls_enabled", "runtime_api_tls", "cli_managed_supernode_auth", "all_nodes_online",
                     "secagg_completed", "dp_execution_audit_passed", "distinct_processes")
    for key in required_true:
        if evidence.get(key) is not True: errors.append(key)
    if evidence.get("flower_version") != FLOWER_VERSION: errors.append("flower_version")
    nodes = evidence.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 7: errors.append("seven_nodes")
    elif any(not isinstance(node, dict) for node in nodes): errors.append("node_schema")
    else:
        ids = [node.get("node_id") for node in nodes]; addresses = [node.get("address") for node in nodes]
        fingerprints = [node.get("public_key_fingerprint") for node in nodes]
        tls_fingerprints = [node.get("tls_peer_certificate_sha256") for node in nodes]
        processes = [node.get("process_instance_id") for node in nodes]
        hosts = [node.get("host_instance_id") for node in nodes]
        if len(set(ids)) != 7 or any(not value for value in ids): errors.append("unique_node_ids")
        if len(set(addresses)) != 7 or any(not _valid_remote_endpoint(value) for value in addresses): errors.append("unique_remote_endpoints")
        if len(set(fingerprints)) != 7 or any(not HEX64.fullmatch(str(value or "")) for value in fingerprints): errors.append("unique_public_keys")
        if len(set(tls_fingerprints)) != 7 or any(not HEX64.fullmatch(str(value or "")) for value in tls_fingerprints): errors.append("unique_tls_peers")
        if len(set(processes)) != 7 or any(not value for value in processes): errors.append("unique_process_instances")
        if any(not value for value in hosts): errors.append("host_instance_identity")
        if any(node.get("platform_system") != "Linux" for node in nodes): errors.append("linux_nodes")
    if evidence.get("private_keys_exported") is not False: errors.append("private_key_policy")
    if (not HEX64.fullmatch(str(evidence.get("tls_ca_fingerprint", ""))) or not evidence.get("run_id")
            or not HEX64.fullmatch(str(evidence.get("completion_sha256", "")))):
        errors.append("receipt_identity")
    receipts = evidence.get("receipt_sha256", {})
    if not isinstance(receipts, dict) or any(not HEX64.fullmatch(str(receipts.get(name, ""))) for name in RECEIPTS):
        errors.append("receipt_hashes")
    review = evidence.get("independent_evidence_review", {})
    if not isinstance(review, dict):
        review = {}
    if (review.get("status") != "APPROVED" or not review.get("reviewer")
            or not review.get("reviewed_at_utc") or not HEX64.fullmatch(str(review.get("review_record_sha256", "")))):
        errors.append("independent_evidence_review")
    return {"status": "PASS_AUTHENTICATED_NETWORK_DEPLOYMENT_NOT_SECURITY_CERTIFICATION" if not errors else "FAIL_NETWORK_DEPLOYMENT_GATE",
            "deployment_verified": not errors, "errors": sorted(set(errors)),
            "qualification": "Operational evidence only; not an independent penetration test, farm deployment, or application privacy certification."}


def audit(output: Path, evidence_path: Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    evidence = read(evidence_path) if evidence_path else None
    report = evaluate(evidence); output.mkdir(parents=True)
    atomic_json(output / "deployment_gate.json", report)
    atomic_json(output / "contract.json", {"flower_version": FLOWER_VERSION, "required_nodes": 7,
                "schema_version": 1, "deployment_mode": "authenticated_multi_process_network",
                "transport": "TLS", "supernode_auth": "CLI-managed public-key allowlist",
                "runtime_api_tls": True, "private_key_storage": "external; never repository or released transcript",
                "secagg_threshold": 5, "insecure_mode_allowed": False,
                "required_receipts": list(RECEIPTS), "independent_evidence_review": True})
    atomic_json(output / "manifest.json", {"status": report["status"], "source_sha256": sha(Path(__file__)),
                "evidence_sha256": sha(evidence_path) if evidence_path else None, "model_training": False})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path); args = parser.parse_args(); print(audit(args.output, args.evidence))
