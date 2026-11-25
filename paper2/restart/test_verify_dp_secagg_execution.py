import tempfile
import unittest
from pathlib import Path

from paper2.restart.validation_search import atomic_json, sha
from paper2.restart.verify_dp_secagg_execution import inspect_execution
from paper2.restart.run_dp_secagg_pilot import semantic_log_audit


class VerifyDpSecAggExecutionTests(unittest.TestCase):
    def test_valid_receipts_pass_with_noncertification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); execution = root / "execution"; protocol = root / "protocol"
            execution.mkdir(); protocol.mkdir()
            atomic_json(protocol / "manifest.json", {"status": "frozen"})
            atomic_json(protocol / "privacy_contract.json", {"secure_aggregation_requested": True,
                        "differential_privacy_enabled": True, "adaptive_releases": 1,
                        "conditional_budget": {"epsilon_upper_bound": 1.0, "releases": 1}})
            atomic_json(protocol / "run_config.json", {"differential-privacy-enabled": True,
                        "central-dp": True, "num-server-rounds": 1, "outer-test-evaluation": False})
            stdout = "Secure aggregation commencing.\nSecure aggregation completed."
            (execution / "flower_stdout.log").write_text(stdout, encoding="utf-8")
            (execution / "flower_stderr.log").write_text("", encoding="utf-8")
            atomic_json(execution / "manifest.json", {"protocol_manifest_sha256": sha(protocol / "manifest.json"), "raw_updates_exported": False})
            atomic_json(execution / "completion.json", {"returncode": 0, "status": "COMPLETE", "raw_updates_exported": False,
                        "semantic_log_audit": semantic_log_audit(stdout, "", 1),
                        "log_sha256": {name: sha(execution / name) for name in ("flower_stdout.log", "flower_stderr.log")}})
            result = inspect_execution(execution, protocol)
            self.assertEqual(result["status"], "PASS_EXECUTED_DP_SECAGG_PILOT_NOT_CERTIFIED")
            self.assertFalse(result["application_privacy_certified"])

    def test_failed_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); execution = root / "execution"; protocol = root / "protocol"
            execution.mkdir(); protocol.mkdir()
            atomic_json(protocol / "manifest.json", {})
            atomic_json(protocol / "privacy_contract.json", {"secure_aggregation_requested": True,
                        "differential_privacy_enabled": True, "adaptive_releases": 1,
                        "conditional_budget": {"releases": 1}})
            atomic_json(protocol / "run_config.json", {"differential-privacy-enabled": True,
                        "central-dp": True, "num-server-rounds": 1, "outer-test-evaluation": False})
            atomic_json(execution / "manifest.json", {"protocol_manifest_sha256": sha(protocol / "manifest.json"), "raw_updates_exported": False})
            atomic_json(execution / "completion.json", {"returncode": 1, "status": "FAILED", "raw_updates_exported": False, "log_sha256": {}})
            with self.assertRaisesRegex(ValueError, "execution_failed"):
                inspect_execution(execution, protocol)

    def test_release_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); execution = root / "execution"; protocol = root / "protocol"
            execution.mkdir(); protocol.mkdir()
            atomic_json(protocol / "manifest.json", {})
            atomic_json(protocol / "privacy_contract.json", {"secure_aggregation_requested": True,
                        "differential_privacy_enabled": True, "adaptive_releases": 1,
                        "conditional_budget": {"releases": 1}})
            atomic_json(protocol / "run_config.json", {"differential-privacy-enabled": True,
                        "central-dp": True, "num-server-rounds": 20, "outer-test-evaluation": False})
            for name in ("flower_stdout.log", "flower_stderr.log"):
                (execution / name).write_text("", encoding="utf-8")
            atomic_json(execution / "manifest.json", {"protocol_manifest_sha256": sha(protocol / "manifest.json"), "raw_updates_exported": False})
            atomic_json(execution / "completion.json", {"returncode": 0, "status": "COMPLETE", "raw_updates_exported": False,
                        "log_sha256": {name: sha(execution / name) for name in ("flower_stdout.log", "flower_stderr.log")}})
            with self.assertRaisesRegex(ValueError, "privacy_release_count"):
                inspect_execution(execution, protocol)


if __name__ == "__main__":
    unittest.main()
