import json
from pathlib import Path
import shutil
import tempfile
import unittest

from paper2.restart.authenticated_network_runner import compile_plan
from paper2.restart.collect_authenticated_network_evidence import collect
from paper2.restart.validation_search import ROOT


class AuthenticatedNetworkEvidenceCollectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT / "paper2/results", prefix="network-collector-test-"))
        self.config_path = self.root / "config.json"
        config = {
            "schema_version": 1, "flower_version": "1.42.0", "secagg_threshold": 5, "private_keys_exported": False,
            "superlink": {"fleet_address": "superlink.network.example.org:9092", "control_address": "superlink.network.example.org:9093",
                          "runtime_address": "superlink.network.example.org:9094", "host": "superlink.network.example.org"},
            "nodes": [{"node_id": f"node-{index}", "address": f"node-{index}.network.example.org:9094",
                       "ssh_host": f"ops@node-{index}.network.example.org", "remote_root": "/srv/paper2",
                       "partition_id": index, "public_key_fingerprint": f"{index + 1:064x}"} for index in range(7)],
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.plan_dir = self.root / "plan"
        compile_plan(self.config_path, self.plan_dir)
        self.receipts = self.root / "receipts"; self.receipts.mkdir()
        self.run_id = "network-run-1"
        summary = {"schema_version": 1, "run_id": self.run_id,
                   "deployment_mode": "authenticated_multi_process_network", "flower_version": "1.42.0",
                   "tls_enabled": True, "runtime_api_tls": True, "cli_managed_supernode_auth": True,
                   "all_nodes_online": True, "distinct_processes": True, "secagg_completed": True,
                   "dp_execution_audit_passed": True, "private_keys_exported": False,
                   "tls_ca_fingerprint": "a" * 64}
        self.write("deployment_summary.json", summary)
        nodes = [{"node_id": f"node-{index}", "address": f"node-{index}.network.example.org:9094",
                  "public_key_fingerprint": f"{index + 1:064x}", "tls_peer_certificate_sha256": f"{index + 11:064x}",
                  "process_instance_id": f"process-{index}", "host_instance_id": f"host-{index // 2}",
                  "platform_system": "Linux"} for index in range(7)]
        self.write("node_inventory.json", {"schema_version": 1, "run_id": self.run_id, "nodes": nodes})
        self.write("completion.json", {"schema_version": 1, "run_id": self.run_id, "complete": True})
        node_ids = [node["node_id"] for node in nodes]
        fingerprints = {node["node_id"]: node["public_key_fingerprint"] for node in nodes}
        certificates = {node["node_id"]: node["tls_peer_certificate_sha256"] for node in nodes}
        processes = {node["node_id"]: node["process_instance_id"] for node in nodes}
        payloads = {
            "tls_handshake": {"tls_ca_fingerprint": "a" * 64, "node_tls_peer_certificate_sha256": certificates},
            "supernode_registration": {"registered_node_ids": node_ids, "public_key_fingerprints": fingerprints},
            "online_nodes": {"online_node_ids": node_ids, "process_instance_ids": processes},
            "secagg_run": {"threshold": 5, "completed": True, "round_count": 20},
            "dp_audit": {"audit_status": "PASS", "rounds": 20, "privacy_accountant_config_sha256": "d" * 64},
            "result_retrieval": {"completion_sha256": self.sha_of("completion.json"), "result_manifest_sha256": "e" * 64,
                                 "retrieved_at_utc": "2026-10-28T00:01:00Z"},
        }
        for name, payload in payloads.items():
            self.write(f"{name}.json", {"schema_version": 1, "receipt_type": name, "run_id": self.run_id,
                                         "captured_at_utc": "2026-10-28T00:00:00Z", "payload": payload})

    def tearDown(self):
        shutil.rmtree(self.root)

    def write(self, name, value):
        (self.receipts / name).write_text(json.dumps(value), encoding="utf-8")

    def sha_of(self, name):
        import hashlib
        return hashlib.sha256((self.receipts / name).read_bytes()).hexdigest()

    def test_pending_review_collects_but_cannot_verify_deployment(self):
        result = collect(self.plan_dir / "deployment_plan.json", self.receipts, self.root / "output")
        self.assertFalse(result["deployment_verified"])
        evidence = json.loads((self.root / "output/network_deployment_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["independent_evidence_review"]["status"], "PENDING")

    def test_approved_review_passes_the_operational_gate(self):
        review = self.root / "review.json"
        review.write_text(json.dumps({"status": "APPROVED", "reviewer": "independent-reviewer",
                                      "reviewed_at_utc": "2026-10-28T01:00:00Z"}), encoding="utf-8")
        result = collect(self.plan_dir / "deployment_plan.json", self.receipts, self.root / "approved", review)
        self.assertTrue(result["deployment_verified"])
        self.assertIn("NOT_SECURITY_CERTIFICATION", result["gate_status"])

    def test_rejects_secret_like_receipt_key_and_plan_mismatch(self):
        path = self.receipts / "dp_audit.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["payload"] = {"private_key": "x"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "secret-like"):
            collect(self.plan_dir / "deployment_plan.json", self.receipts, self.root / "bad-secret")
        payload["payload"] = {"event": "dp_audit"}; path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "DP audit receipt payload"):
            collect(self.plan_dir / "deployment_plan.json", self.receipts, self.root / "bad-payload")
        payload["payload"] = {"audit_status": "PASS", "rounds": 20, "privacy_accountant_config_sha256": "d" * 64}
        path.write_text(json.dumps(payload), encoding="utf-8")
        inventory = json.loads((self.receipts / "node_inventory.json").read_text(encoding="utf-8")); inventory["nodes"][0]["address"] = "changed.network.example.org:9094"
        (self.receipts / "node_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            collect(self.plan_dir / "deployment_plan.json", self.receipts, self.root / "bad-plan")


if __name__ == "__main__":
    unittest.main()
