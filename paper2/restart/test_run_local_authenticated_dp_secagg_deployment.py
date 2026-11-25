import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper2/restart/run_local_authenticated_dp_secagg_deployment.sh"


class LocalAuthenticatedDeploymentScriptTests(unittest.TestCase):
    def test_is_bound_to_adapter_and_never_mislabels_local_execution(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("prepare_authenticated_dp_secagg_adapter import validate", source)
        self.assertIn('report["app_sources_sha256"] == binding["app_sources_sha256"]', source)
        self.assertIn("timeout 1800 flwr run", source)
        self.assertIn("rounds_requested\": 20", source)
        self.assertIn("single_laptop_wsl_loopback", source)
        self.assertIn("network_deployment_gate_passed\": False", source)
        self.assertIn("privacy_claim_allowed\": False", source)
        self.assertIn("private_keys_retained_after_run\": False", source)
        self.assertIn("record_interruption", source)
        self.assertIn("semantic_log_audit", source)
        self.assertIn("RuntimeDependencyInstallationError", source)
        self.assertIn("run_semantic_status.txt", source)
        self.assertIn("--isolation subprocess", source)
        self.assertNotIn("--isolation process", source)
        self.assertIn("--enable-supernode-auth >", source)
        self.assertNotIn("--disable-runtime-dependency-installation", source)
        self.assertIn("rm -rf \"$WORK\"", source)
        self.assertNotIn("--insecure", source)


if __name__ == "__main__":
    unittest.main()
