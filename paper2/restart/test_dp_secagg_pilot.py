"""Contract tests for DP+SecAgg+ pilot configuration, no data or network use."""
import unittest
from paper2.restart import freeze_dp_secagg_pilot as pilot


class DpSecAggPilotTests(unittest.TestCase):
    def test_contract_uses_one_positive_gaussian_event(self):
        settings = pilot.SecAggPilotConfig()
        sensitivity = pilot.replacement_sensitivity(pilot.CLIP_NORM, settings.participants)
        noise_std = pilot.flower_noise_std(pilot.CLIP_NORM, pilot.NOISE_MULTIPLIER, settings.participants)
        budget = pilot.gaussian_budget([{"sensitivity": sensitivity, "noise_std": noise_std, "count": 1}], pilot.DELTA)
        self.assertEqual(budget["releases"], 1)
        self.assertAlmostEqual(budget["epsilon_upper_bound"], 1.2308814780470203)

    def test_contract_stays_central_not_local_or_certified(self):
        self.assertEqual(pilot.CLIP_NORM, 1.0)
        self.assertEqual(pilot.NOISE_MULTIPLIER, 8.0)

    def test_run_config_explicitly_enables_dp(self):
        run_config = pilot.dp_payload()["run_config.json"]
        self.assertTrue(run_config["differential-privacy-enabled"])
        self.assertTrue(run_config["central-dp"])
        self.assertFalse(run_config["outer-test-evaluation"])
        self.assertNotIn("differential_privacy_enabled", run_config)

    def test_known_upstream_security_issue_blocks_production_claim(self):
        payload = pilot.dp_payload()
        scope = payload["security_scope.json"]
        contract = payload["privacy_contract.json"]
        self.assertFalse(scope["production_security_claim_allowed"])
        self.assertFalse(contract["known_security_issues_resolved"])
        self.assertFalse(contract["independent_cryptographic_review_complete"])

    def test_execution_profile_is_seven_client_cpu_only_and_isolated(self):
        profile = pilot.dp_payload()["execution_profile.json"]
        self.assertEqual(profile["num_supernodes"], 7)
        self.assertEqual(profile["client_resources_num_gpus"], 0.0)
        self.assertFalse(profile["runtime_dependency_installation"])
        self.assertEqual(profile["managed_superlink_http_port"], 39181)

    def test_multiround_accounting_composes_every_release(self):
        contract = pilot.dp_payload(rounds=20)["privacy_contract.json"]
        self.assertEqual(contract["conditional_budget"]["releases"], 20)
        self.assertAlmostEqual(contract["conditional_budget"]["epsilon_upper_bound"], 5.989915065723368)


if __name__ == "__main__":
    unittest.main()
