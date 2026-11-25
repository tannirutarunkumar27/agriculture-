import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper2/restart/run_local_wsl_authenticated_smoke.sh"


class LocalWslSmokeScriptTests(unittest.TestCase):
    def test_marks_loopback_smoke_as_non_deployment_and_uses_seven_nodes(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PASS_LOCAL_WSL_LOOPBACK_TLS_AUTH_SMOKE_NOT_DISTRIBUTED_DEPLOYMENT", script)
        self.assertIn("network_deployment_gate_passed\": False", script)
        self.assertIn("for index in $(seq 0 6)", script)
        self.assertIn("--enable-supernode-auth", script)
        self.assertIn("--root-certificates", script)
        self.assertIn("--auth-supernode-private-key", script)
        self.assertIn("PAPER2_FLOWER_VENV", script)
        self.assertIn("paper2-clean-v1", script)
        self.assertIn("Expected frozen Flower 1.42.0", script)
        self.assertIn("Expected exactly seven distinct online SuperNodes", script)
        self.assertIn("unique_node_ids_observed", script)
        self.assertIn("rm -rf \"$WORK\"", script)
        self.assertNotIn("--insecure", script)


if __name__ == "__main__":
    unittest.main()
