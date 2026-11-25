"""Compile a credential-free, authenticated seven-SuperNode Flower deployment plan.

This module never generates, stores, or transfers private keys.  It produces an
immutable deployment plan which an operator executes only after supplying TLS
and SuperNode authentication material through an external secret manager.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.network_deployment_gate import FLOWER_VERSION, RESULTS, _valid_remote_endpoint
from paper2.restart.validation_search import atomic_json, identity, read, sha


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    """Return a Flower host/port pair from a validated endpoint."""
    if endpoint.startswith("["):
        close = endpoint.index("]")
        return endpoint[1:close], endpoint[close + 2:]
    return endpoint.rsplit(":", 1)


def receipt_contract() -> dict:
    """Public, non-secret receipt schema required by the collector."""
    return {
        "schema_version": 1,
        "common": {"schema_version": 1, "run_id": "same nonempty deployment run identifier",
                   "captured_at_utc": "ISO-8601 UTC ending Z"},
        "tls_handshake": ["tls_ca_fingerprint", "node_tls_peer_certificate_sha256"],
        "supernode_registration": ["registered_node_ids", "public_key_fingerprints"],
        "online_nodes": ["online_node_ids", "process_instance_ids"],
        "secagg_run": ["threshold", "completed", "round_count"],
        "dp_audit": ["audit_status", "rounds", "privacy_accountant_config_sha256"],
        "result_retrieval": ["completion_sha256", "result_manifest_sha256", "retrieved_at_utc"],
        "secret_policy": "Never include private keys, tokens, passwords, or passphrases in receipts.",
        "binding_policy": "Every node identifier, certificate fingerprint, public-key fingerprint, and process identity must match the frozen plan and runtime inventory.",
    }


def validate(config):
    required = {"schema_version", "flower_version", "superlink", "nodes", "secagg_threshold", "private_keys_exported"}
    if set(config) != required or config["schema_version"] != 1 or config["flower_version"] != FLOWER_VERSION:
        raise ValueError("Unexpected authenticated-network configuration")
    if config["private_keys_exported"] is not False or config["secagg_threshold"] != 5:
        raise ValueError("Network security policy differs from the frozen protocol")
    superlink = config["superlink"]
    if not isinstance(superlink, dict) or set(superlink) != {"fleet_address", "control_address", "runtime_address", "host"}:
        raise ValueError("Invalid SuperLink endpoint configuration")
    if not all(_valid_remote_endpoint(superlink[key]) for key in ("fleet_address", "control_address", "runtime_address")):
        raise ValueError("SuperLink endpoints must be real remote endpoints")
    if _split_endpoint(superlink["runtime_address"])[0] != superlink["host"]:
        raise ValueError("SuperLink host must match the frozen Runtime API endpoint")
    nodes = config["nodes"]
    if not isinstance(nodes, list) or len(nodes) != 7 or any(not isinstance(node, dict) for node in nodes):
        raise ValueError("Exactly seven SuperNode configurations are required")
    required_node = {"node_id", "address", "ssh_host", "remote_root", "partition_id", "public_key_fingerprint"}
    if any(set(node) != required_node for node in nodes):
        raise ValueError("Unexpected SuperNode configuration schema")
    if {node["partition_id"] for node in nodes} != set(range(7)):
        raise ValueError("SuperNode partitions must be exactly 0 through 6")
    if len({node["node_id"] for node in nodes}) != 7 or len({node["address"] for node in nodes}) != 7:
        raise ValueError("SuperNode identities and addresses must be unique")
    if any(not _valid_remote_endpoint(node["address"]) or not isinstance(node["ssh_host"], str) or "@" not in node["ssh_host"]
           or not isinstance(node["remote_root"], str) or not node["remote_root"].startswith("/") or ".." in node["remote_root"].split("/")
           or not isinstance(node["public_key_fingerprint"], str) or len(node["public_key_fingerprint"]) != 64
           for node in nodes):
        raise ValueError("Invalid SuperNode endpoint, host, root, or public-key fingerprint")
    if len({node["public_key_fingerprint"] for node in nodes}) != 7:
        raise ValueError("SuperNode public keys must be unique")
    return config


def compile_plan(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    config = validate(read(config_path)); output.mkdir(parents=True)
    link = config["superlink"]
    runtime_host, runtime_port = _split_endpoint(link["runtime_address"])
    # Flower 1.42 accepts the legacy Fleet/Control address options.  It does
    # not accept --fleet-address, --control-address, or --runtime-address;
    # Runtime uses --host/--port. TLS path placeholders are intentionally not
    # concrete secret paths and must be injected by the operator.
    superlink = ["flower-superlink", "--host", runtime_host, "--port", runtime_port,
                 "--fleet-api-address", link["fleet_address"], "--control-api-address", link["control_address"],
                 "--enable-supernode-auth",
                 "# Inject: --ssl-ca-certfile <CA> --ssl-certfile <CERT> --ssl-keyfile <KEY>"]
    nodes = []
    for node in sorted(config["nodes"], key=lambda value: value["partition_id"]):
        node_host, node_port = _split_endpoint(node["address"])
        nodes.append({"node_id": node["node_id"], "address": node["address"], "ssh_host": node["ssh_host"], "remote_root": node["remote_root"],
                      "public_key_fingerprint": node["public_key_fingerprint"],
                      "command": ["flower-supernode", "--root-certificates", "<CA>", "--auth-supernode-private-key", "<NODE_PRIVATE_KEY>",
                                  "--superlink", link["fleet_address"], "--host", node_host,
                                  "--port", node_port, "--node-config",
                                  f"partition-id={node['partition_id']} num-partitions=7",
                                  "# Inject Runtime API TLS: --ssl-ca-certfile <CA> --ssl-certfile <CERT> --ssl-keyfile <KEY>"]})
    plan = {"status": "AUTHENTICATED_NETWORK_DEPLOYMENT_PLAN_NOT_EXECUTED", "flower_version": FLOWER_VERSION,
            "deployment_mode": "authenticated_multi_process_network", "superlink_command": superlink,
            "supernodes": nodes, "secagg_threshold": 5, "private_keys_exported": False,
            "operator_gates": ["issue TLS material with SANs for every endpoint", "configure a Flower CLI profile for the Control API plus the CA root certificate",
                               "register each public key through Flower CLI",
                               "start seven separate SuperNode processes", "collect actual TLS/registration/online/SecAgg/DP/retrieval receipts",
                               "run collect_authenticated_network_evidence against the frozen plan",
                               "obtain independent evidence review before passing network_deployment_gate"]}
    atomic_json(output / "deployment_plan.json", plan)
    atomic_json(output / "receipt_contract.json", receipt_contract())
    atomic_json(output / "manifest.json", {"status": plan["status"], "config_sha256": sha(config_path),
                "plan_sha256": sha(output / "deployment_plan.json"), "receipt_contract_sha256": sha(output / "receipt_contract.json"),
                "plan_identity": identity(plan), "private_keys_exported": False})
    return {"status": plan["status"], "planned_supernodes": len(nodes)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(compile_plan(args.config, args.output))
