"""Collect fail-closed authenticated-network evidence from a real Linux deployment.

This tool does not start services, manufacture network evidence, or handle
private keys.  An operator supplies machine-generated JSON receipts from an
already completed deployment.  The collector binds their hashes and the
runtime node inventory to a frozen deployment plan, then invokes the existing
independent-review gate.  A successful collection is still only a candidate
until the external reviewer record is supplied and approved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from paper2.restart.network_deployment_gate import RECEIPTS, RESULTS, audit, evaluate
from paper2.restart.validation_search import atomic_json, read, sha

REQUIRED_RECEIPT_FIELDS = {"schema_version", "receipt_type", "run_id", "captured_at_utc", "payload"}
NODE_FIELDS = {"node_id", "address", "public_key_fingerprint", "tls_peer_certificate_sha256",
               "process_instance_id", "host_instance_id", "platform_system"}
SUMMARY_FIELDS = {"schema_version", "run_id", "deployment_mode", "flower_version", "tls_enabled",
                  "runtime_api_tls", "cli_managed_supernode_auth", "all_nodes_online", "distinct_processes",
                  "secagg_completed", "dp_execution_audit_passed", "private_keys_exported", "tls_ca_fingerprint"}
FORBIDDEN_SECRET_MARKERS = ("private_key", "privatekey", "secret", "password", "passphrase", "token")
ALLOWED_POLICY_KEYS = {"private_keys_exported"}


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any((str(key).lower() not in ALLOWED_POLICY_KEYS
                    and any(marker in str(key).lower() for marker in FORBIDDEN_SECRET_MARKERS))
                   or _contains_secret_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _validate_plan(plan: dict) -> tuple[dict, int]:
    if not isinstance(plan, dict) or plan.get("status") != "AUTHENTICATED_NETWORK_DEPLOYMENT_PLAN_NOT_EXECUTED":
        raise ValueError("A frozen authenticated-network deployment plan is required")
    nodes = plan.get("supernodes")
    if not isinstance(nodes, list) or len(nodes) != 7:
        raise ValueError("Frozen plan must contain exactly seven SuperNodes")
    expected = {}
    for node in nodes:
        if not isinstance(node, dict) or not {"node_id", "address", "public_key_fingerprint"} <= set(node):
            raise ValueError("Frozen plan node schema is incomplete")
        expected[node["node_id"]] = (node["address"], node["public_key_fingerprint"])
    if len(expected) != 7:
        raise ValueError("Frozen plan has duplicate SuperNode identities")
    return expected, int(plan["secagg_threshold"])


def _validate_receipt(
    path: Path,
    receipt_type: str,
    run_id: str,
    *,
    summary: dict,
    nodes: dict,
    completion_sha256: str,
    secagg_threshold: int,
) -> None:
    value = read(path)
    if not isinstance(value, dict) or set(value) != REQUIRED_RECEIPT_FIELDS:
        raise ValueError(f"Invalid {receipt_type} receipt schema")
    if value["schema_version"] != 1 or value["receipt_type"] != receipt_type or value["run_id"] != run_id:
        raise ValueError(f"Receipt binding failed for {receipt_type}")
    if not isinstance(value["captured_at_utc"], str) or not value["captured_at_utc"].endswith("Z"):
        raise ValueError(f"Receipt timestamp is invalid for {receipt_type}")
    if _contains_secret_key(value):
        raise ValueError(f"Receipt {receipt_type} contains prohibited secret-like key material")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ValueError(f"Receipt payload is invalid for {receipt_type}")
    node_ids = set(nodes)
    if receipt_type == "tls_handshake":
        if set(payload) != {"tls_ca_fingerprint", "node_tls_peer_certificate_sha256"}:
            raise ValueError("TLS receipt payload schema is invalid")
        peers = payload["node_tls_peer_certificate_sha256"]
        if (payload["tls_ca_fingerprint"] != summary["tls_ca_fingerprint"] or not isinstance(peers, dict)
                or set(peers) != node_ids
                or any(peers[node_id] != nodes[node_id]["tls_peer_certificate_sha256"] for node_id in node_ids)):
            raise ValueError("TLS receipt is not bound to the runtime node inventory")
    elif receipt_type == "supernode_registration":
        if set(payload) != {"registered_node_ids", "public_key_fingerprints"}:
            raise ValueError("SuperNode registration receipt payload schema is invalid")
        fingerprints = payload["public_key_fingerprints"]
        if (not isinstance(payload["registered_node_ids"], list) or set(payload["registered_node_ids"]) != node_ids
                or len(payload["registered_node_ids"]) != len(node_ids) or not isinstance(fingerprints, dict)
                or set(fingerprints) != node_ids
                or any(fingerprints[node_id] != nodes[node_id]["public_key_fingerprint"] for node_id in node_ids)):
            raise ValueError("SuperNode registration receipt is not bound to the runtime node inventory")
    elif receipt_type == "online_nodes":
        if set(payload) != {"online_node_ids", "process_instance_ids"}:
            raise ValueError("Online-node receipt payload schema is invalid")
        processes = payload["process_instance_ids"]
        if (not isinstance(payload["online_node_ids"], list) or set(payload["online_node_ids"]) != node_ids
                or len(payload["online_node_ids"]) != len(node_ids) or not isinstance(processes, dict)
                or set(processes) != node_ids
                or any(processes[node_id] != nodes[node_id]["process_instance_id"] for node_id in node_ids)):
            raise ValueError("Online-node receipt is not bound to the runtime process inventory")
    elif receipt_type == "secagg_run":
        if (set(payload) != {"threshold", "completed", "round_count"}
                or payload["threshold"] != secagg_threshold or payload["completed"] is not True
                or not isinstance(payload["round_count"], int) or isinstance(payload["round_count"], bool)
                or payload["round_count"] < 1):
            raise ValueError("SecAgg receipt does not satisfy the frozen threshold/run contract")
    elif receipt_type == "dp_audit":
        if (set(payload) != {"audit_status", "rounds", "privacy_accountant_config_sha256"}
                or payload["audit_status"] != "PASS" or not isinstance(payload["rounds"], int)
                or isinstance(payload["rounds"], bool) or payload["rounds"] < 1
                or not isinstance(payload["privacy_accountant_config_sha256"], str)
                or len(payload["privacy_accountant_config_sha256"]) != 64):
            raise ValueError("DP audit receipt payload is invalid")
    elif receipt_type == "result_retrieval":
        if set(payload) != {"completion_sha256", "result_manifest_sha256", "retrieved_at_utc"}:
            raise ValueError("Result-retrieval receipt payload schema is invalid")
        if (payload["completion_sha256"] != completion_sha256
                or not isinstance(payload["result_manifest_sha256"], str)
                or len(payload["result_manifest_sha256"]) != 64
                or not isinstance(payload["retrieved_at_utc"], str)
                or not payload["retrieved_at_utc"].endswith("Z")):
            raise ValueError("Result-retrieval receipt is not bound to the completion receipt")


def _pending_review() -> dict:
    return {"status": "PENDING", "reviewer": "", "reviewed_at_utc": "", "review_record_sha256": ""}


def collect(plan_path: Path, receipts_root: Path, output: Path, review_path: Path | None = None) -> dict:
    """Create a candidate gate input from a frozen plan and actual receipts."""
    plan_path, receipts_root, output = Path(plan_path).resolve(), Path(receipts_root).resolve(), Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    expected_nodes, secagg_threshold = _validate_plan(read(plan_path))
    summary_path, nodes_path, completion_path = (receipts_root / name for name in
                                                  ("deployment_summary.json", "node_inventory.json", "completion.json"))
    if not all(path.is_file() for path in (summary_path, nodes_path, completion_path)):
        raise ValueError("Deployment summary, node inventory, and completion receipt are required")
    summary = read(summary_path)
    if not isinstance(summary, dict) or set(summary) != SUMMARY_FIELDS:
        raise ValueError("Deployment summary schema is invalid")
    if summary["schema_version"] != 1 or summary["deployment_mode"] != "authenticated_multi_process_network":
        raise ValueError("Deployment summary is not an authenticated multi-process execution")
    run_id = summary["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Deployment summary has no run identifier")
    if _contains_secret_key(summary):
        raise ValueError("Deployment summary contains prohibited secret-like key material")
    inventory = read(nodes_path)
    if not isinstance(inventory, dict) or set(inventory) != {"schema_version", "run_id", "nodes"}:
        raise ValueError("Node inventory schema is invalid")
    if inventory["schema_version"] != 1 or inventory["run_id"] != run_id or not isinstance(inventory["nodes"], list):
        raise ValueError("Node inventory is not bound to the deployment run")
    actual_nodes = inventory["nodes"]
    if len(actual_nodes) != 7 or any(not isinstance(node, dict) or set(node) != NODE_FIELDS for node in actual_nodes):
        raise ValueError("Node inventory must contain exactly seven complete node records")
    actual_map = {node["node_id"]: (node["address"], node["public_key_fingerprint"]) for node in actual_nodes}
    if actual_map != expected_nodes:
        raise ValueError("Runtime node identities, endpoints, or public keys differ from the frozen plan")
    if _contains_secret_key(inventory):
        raise ValueError("Node inventory contains prohibited secret-like key material")
    completion = read(completion_path)
    if not isinstance(completion, dict) or completion.get("schema_version") != 1 or completion.get("run_id") != run_id:
        raise ValueError("Completion receipt is not bound to the deployment run")
    if _contains_secret_key(completion):
        raise ValueError("Completion receipt contains prohibited secret-like key material")
    runtime_nodes = {node["node_id"]: node for node in actual_nodes}
    receipt_paths = {name: receipts_root / f"{name}.json" for name in RECEIPTS}
    if any(not path.is_file() for path in receipt_paths.values()):
        raise ValueError("All six authenticated-network receipt files are required")
    for name, path in receipt_paths.items():
        _validate_receipt(path, name, run_id, summary=summary, nodes=runtime_nodes,
                          completion_sha256=sha(completion_path), secagg_threshold=secagg_threshold)
    review = _pending_review()
    if review_path is not None:
        candidate = read(Path(review_path).resolve())
        if not isinstance(candidate, dict) or set(candidate) != {"status", "reviewer", "reviewed_at_utc"}:
            raise ValueError("Independent review record schema is invalid")
        if _contains_secret_key(candidate):
            raise ValueError("Independent review record contains prohibited secret-like key material")
        review = {**candidate, "review_record_sha256": sha(Path(review_path).resolve())}
    evidence = {**summary, "nodes": actual_nodes, "completion_sha256": sha(completion_path),
                "receipt_sha256": {name: sha(path) for name, path in receipt_paths.items()},
                "independent_evidence_review": review}
    output.mkdir(parents=True)
    atomic_json(output / "network_deployment_evidence.json", evidence)
    atomic_json(output / "collection_manifest.json", {
        "status": "NETWORK_EVIDENCE_CANDIDATE_COLLECTED_NOT_SECURITY_CERTIFICATION",
        "collected_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan_sha256": sha(plan_path), "receipts_root": str(receipts_root),
        "review_supplied": review_path is not None, "private_keys_exported": False,
        "collection_verdict": evaluate(evidence),
    })
    audit_dir = output / "network_deployment_gate_audit"
    gate = audit(audit_dir, output / "network_deployment_evidence.json")
    return {"status": "NETWORK_EVIDENCE_CANDIDATE_COLLECTED_NOT_SECURITY_CERTIFICATION",
            "deployment_verified": gate["deployment_verified"], "gate_status": gate["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    print(collect(args.plan, args.receipts_root, args.output, args.review))
