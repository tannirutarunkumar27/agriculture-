import unittest
from unittest.mock import patch

from paper2.restart.local_linux_preflight import REQUIRED_DEPENDENCIES, evaluate


def valid_snapshot():
    return {
        "platform_system": "Linux",
        "kernel_release": "6.18.0-microsoft-standard-WSL2",
        "python": "3.14.4",
        "python_tuple": [3, 14, 4],
        "cpu_count": 12,
        "memory_bytes": 8 * 1024**3,
        "disk_free_bytes": 100 * 1024**3,
        "dependencies": dict(REQUIRED_DEPENDENCIES),
        "torch_cuda_available": False,
        "data_manifest_exists": True,
        "security_registry_exists": True,
    }


class LocalLinuxPreflightTests(unittest.TestCase):
    def test_valid_wsl_cpu_runtime_passes(self):
        with patch("paper2.restart.local_linux_preflight.read", return_value={
                 "production_security_claim_allowed": False,
                 "all_known_security_issues_resolved": False,
             }):
            result = evaluate(valid_snapshot())
        self.assertEqual(result["status"], "PASS_LOCAL_LINUX_CPU_PREFLIGHT")
        self.assertEqual(result["environment_kind"], "WSL2")

    def test_missing_dependency_fails_closed(self):
        snapshot = valid_snapshot()
        snapshot["dependencies"]["flwr"] = None
        with patch("paper2.restart.local_linux_preflight.read", return_value={
                 "production_security_claim_allowed": False,
                 "all_known_security_issues_resolved": False,
             }):
            result = evaluate(snapshot)
        self.assertEqual(result["status"], "BLOCKED_LOCAL_LINUX_CPU_PREFLIGHT")
        self.assertFalse(result["checks"]["dependencies_exact"])

    def test_native_windows_is_rejected(self):
        snapshot = valid_snapshot()
        snapshot["platform_system"] = "Windows"
        snapshot["memory_bytes"] = 0
        with patch("paper2.restart.local_linux_preflight.read", return_value={
                 "production_security_claim_allowed": False,
                 "all_known_security_issues_resolved": False,
             }):
            result = evaluate(snapshot)
        self.assertFalse(result["checks"]["linux_runtime"])
        self.assertEqual(result["environment_kind"], "unsupported")


if __name__ == "__main__":
    unittest.main()
