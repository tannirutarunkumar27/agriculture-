import unittest

from paper2.restart.authenticated_network_runner import compile_plan, validate
from paper2.restart.validation_search import ROOT
import json
import shutil
import tempfile
from pathlib import Path


def config():
    return {"schema_version": 1, "flower_version": "1.42.0", "secagg_threshold": 5, "private_keys_exported": False,
            "superlink": {"fleet_address": "superlink.network.example.org:9092", "control_address": "superlink.network.example.org:9093", "runtime_address": "superlink.network.example.org:9094", "host": "superlink.network.example.org"},
            "nodes": [{"node_id": f"node-{index}", "address": f"node-{index}.network.example.org:9094", "ssh_host": f"ops@node-{index}.network.example.org",
                       "remote_root": "/srv/paper2", "partition_id": index, "public_key_fingerprint": f"{index + 1:064x}"} for index in range(7)]}


class AuthenticatedNetworkRunnerTests(unittest.TestCase):
    def test_valid_seven_node_plan(self):
        self.assertEqual(len(validate(config())["nodes"]), 7)

    def test_private_key_export_or_duplicate_node_is_rejected(self):
        value = config(); value["private_keys_exported"] = True
        with self.assertRaisesRegex(ValueError, "security policy"):
            validate(value)
        value = config(); value["nodes"][1]["node_id"] = value["nodes"][0]["node_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate(value)

    def test_documentation_address_is_rejected(self):
        value = config(); value["nodes"][0]["address"] = "203.0.113.20:9094"
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate(value)

    def test_runtime_host_must_match_and_plan_uses_flower_142_flags(self):
        value = config(); value["superlink"]["host"] = "other.network.example.org"
        with self.assertRaisesRegex(ValueError, "host"):
            validate(value)
        root = Path(tempfile.mkdtemp(dir=ROOT / "paper2/results", prefix="network-plan-test-"))
        try:
            config_path = root / "config.json"; config_path.write_text(json.dumps(config()), encoding="utf-8")
            compile_plan(config_path, root / "plan")
            plan = json.loads((root / "plan/deployment_plan.json").read_text(encoding="utf-8"))
            command = plan["superlink_command"]
            self.assertIn("--fleet-api-address", command)
            self.assertIn("--control-api-address", command)
            self.assertNotIn("--fleet-address", command)
            self.assertNotIn("--runtime-address", command)
            self.assertIn("--root-certificates", plan["supernodes"][0]["command"])
            contract = json.loads((root / "plan/receipt_contract.json").read_text(encoding="utf-8"))
            self.assertIn("completion_sha256", contract["result_retrieval"])
            self.assertIn("Never include private keys", contract["secret_policy"])
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
