import unittest

from paper2.restart.prepare_authenticated_runtime_privacy_accounting import evaluate


class RuntimePrivacyAccountingTests(unittest.TestCase):
    def test_requires_runtime_evidence_and_release_policy(self):
        protocol = {
            "contract": {"adaptive_releases": 20, "participants": 7},
            "run": {"num-server-rounds": 20, "central-dp": True, "differential-privacy-enabled": True},
            "policy": {"adaptive_model_releases": 20, "client_metrics_released": False,
                       "private_validation_released": False, "intermediate_checkpoints_released": False,
                       "noise_seeds_released": False, "raw_or_individual_updates_released": False},
        }
        adapter = {"status": "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING"}
        audit = {"status": "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_SEMANTIC_AUDIT_NOT_CERTIFIED",
                 "findings": {"semantic_passed": True, "protocol_evidence": {
                     "all_seven_nodes_reached_every_requested_round": True, "rounds_requested": 20}}}
        receipt = {"status": "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_RUNTIME_NOT_CERTIFIED",
                   "rounds_requested": 20, "online_nodes_before_run": 7, "unique_node_ids_before_run": 7,
                   "tls_enabled": True, "cli_managed_supernode_auth": True, "clientapp_isolation": "subprocess"}
        self.assertTrue(evaluate(protocol, adapter, audit, receipt)["passed"])
        receipt["clientapp_isolation"] = "process"
        self.assertIn("runtime_receipt", evaluate(protocol, adapter, audit, receipt)["errors"])


if __name__ == "__main__":
    unittest.main()
