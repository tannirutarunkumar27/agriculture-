import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RemoteRr06LauncherTests(unittest.TestCase):
    def test_launcher_is_authenticated_fail_closed_and_retrieves_for_audit(self):
        text = (ROOT / "paper2/remote_gpu/run_remote_rr06.ps1").read_text(encoding="utf-8")
        for required in (
            "[Parameter(Mandatory=$true)][string]$SshKey",
            "StrictHostKeyChecking=yes",
            "BatchMode=yes",
            "IdentitiesOnly=yes",
            "verify_remote_bundle",
            "verify_matched_neural",
            "Retrieving immutable RR06 results",
            "$Cap -ne 504",
        ):
            self.assertIn(required, text)

    def test_preflight_covers_runtime_and_gpu_identity(self):
        text = (ROOT / "paper2/remote_gpu/remote_preflight.sh").read_text(encoding="utf-8")
        for package in ("torch", "numpy", "pandas", "scikit-learn", "xgboost", "ray", "flwr", "cryptography"):
            self.assertIn(package, text)
        self.assertIn("nvidia-smi", text)
        self.assertIn("uuid", text)

    def test_privacy_launchers_are_authenticated_and_independently_audited(self):
        for name, package_pattern in (
            ("run_remote_dp_secagg.ps1", "restart_dp_secagg_pilot_protocol_v"),
            ("run_remote_dp_secagg_study.ps1", "restart_dp_secagg_study_protocol_v"),
        ):
            text = (ROOT / "paper2/remote_gpu" / name).read_text(encoding="utf-8")
            for required in (
                "[Parameter(Mandatory=$true)][string]$SshKey",
                "StrictHostKeyChecking=yes",
                "BatchMode=yes",
                "IdentitiesOnly=yes",
                "verify_remote_bundle",
                "verify_dp_secagg_execution",
                "ConvertFrom-Json",
                package_pattern,
            ):
                self.assertIn(required, text, f"{name} missing {required}")


if __name__ == "__main__":
    unittest.main()
