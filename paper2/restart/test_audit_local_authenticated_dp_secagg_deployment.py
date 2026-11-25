import unittest

from paper2.restart.audit_local_authenticated_dp_secagg_deployment import audit_receipt


class LocalAuthenticatedDeploymentAuditTests(unittest.TestCase):
    def test_cli_success_does_not_override_runtime_failure(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            stdout = logs / "run_stdout.log"
            stdout.write_text("ERROR : ServerApp raised an exception\nERROR : Exit Code: 608\n")
            receipt = {"run_returncode": 0, "log_sha256": {"run_stdout.log": __import__("hashlib").sha256(stdout.read_bytes()).hexdigest()}}
            result = audit_receipt(receipt, logs)
        self.assertFalse(result["semantic_passed"])
        self.assertEqual(result["matched_failure_signatures"], ["serverapp_exception", "flower_nonzero_exit"])
        self.assertFalse(result["cli_return_code_is_sufficient"])

    def test_requires_all_nodes_to_reach_all_frozen_rounds(self):
        import hashlib
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            for index in range(7):
                (logs / f"supernode-{index}.log").write_text(
                    "[RUN 1, ROUND 1]\n[RUN 1, ROUND 2]\n", encoding="utf-8"
                )
            expected = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in logs.iterdir()
            }
            receipt = {
                "run_returncode": 0, "rounds_requested": 2,
                "online_nodes_before_run": 7, "unique_node_ids_before_run": 7,
                "clientapp_isolation": "subprocess", "log_sha256": expected,
            }
            result = audit_receipt(receipt, logs)
        self.assertTrue(result["semantic_passed"])
        self.assertTrue(result["protocol_evidence"]["all_seven_nodes_reached_every_requested_round"])


if __name__ == "__main__":
    unittest.main()
