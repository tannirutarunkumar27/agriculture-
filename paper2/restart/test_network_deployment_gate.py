import unittest

from paper2.restart.network_deployment_gate import evaluate


def evidence():
    return {"schema_version": 1, "deployment_mode": "authenticated_multi_process_network",
            "tls_enabled": True, "runtime_api_tls": True, "cli_managed_supernode_auth": True,
            "all_nodes_online": True, "secagg_completed": True, "dp_execution_audit_passed": True,
            "distinct_processes": True, "flower_version": "1.42.0", "private_keys_exported": False,
            "tls_ca_fingerprint": "a" * 64, "run_id": "run", "completion_sha256": "b" * 64,
            "receipt_sha256": {name: "c" * 64 for name in
                ("tls_handshake", "supernode_registration", "online_nodes", "secagg_run", "dp_audit", "result_retrieval")},
            "independent_evidence_review": {"status": "APPROVED", "reviewer": "reviewer-1",
                "reviewed_at_utc": "2026-10-27T00:00:00Z", "review_record_sha256": "d" * 64},
            "nodes": [{"node_id": f"n{i}", "address": f"node-{i}.network.example.org:9094",
                       "public_key_fingerprint": f"{i + 1:064x}",
                       "tls_peer_certificate_sha256": f"{i + 11:064x}",
                       "process_instance_id": f"process-{i}", "host_instance_id": f"host-{i // 2}",
                       "platform_system": "Linux"} for i in range(7)]}


class NetworkDeploymentGateTests(unittest.TestCase):
    def test_complete_remote_evidence_passes_with_qualification(self):
        result = evaluate(evidence())
        self.assertTrue(result["deployment_verified"])
        self.assertIn("NOT_SECURITY_CERTIFICATION", result["status"])

    def test_loopback_and_missing_auth_fail(self):
        value = evidence(); value["cli_managed_supernode_auth"] = False; value["nodes"][0]["address"] = "127.0.0.1:9094"
        result = evaluate(value)
        self.assertFalse(result["deployment_verified"])
        self.assertIn("unique_remote_endpoints", result["errors"])

    def test_private_and_documentation_addresses_fail(self):
        for address in ("10.0.0.1:9094", "203.0.113.20:9094"):
            value = evidence(); value["nodes"][0]["address"] = address
            self.assertIn("unique_remote_endpoints", evaluate(value)["errors"])

    def test_same_process_cannot_masquerade_as_seven_clients(self):
        value = evidence()
        for node in value["nodes"]:
            node["process_instance_id"] = "one-process"
        result = evaluate(value)
        self.assertIn("unique_process_instances", result["errors"])

    def test_placeholder_fingerprints_are_rejected(self):
        value = evidence(); value["tls_ca_fingerprint"] = "ca"; value["completion_sha256"] = "hash"
        result = evaluate(value)
        self.assertIn("receipt_identity", result["errors"])

    def test_absent_evidence_is_blocked(self):
        self.assertEqual(evaluate(None)["status"], "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE")

    def test_malformed_external_evidence_fails_closed(self):
        self.assertEqual(evaluate([])["errors"], ["evidence_object"])
        value = evidence(); value["nodes"][0] = "not-a-node"; value["independent_evidence_review"] = []
        result = evaluate(value)
        self.assertFalse(result["deployment_verified"])
        self.assertIn("node_schema", result["errors"])
        self.assertIn("independent_evidence_review", result["errors"])


if __name__ == "__main__":
    unittest.main()
