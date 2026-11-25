import unittest
from unittest.mock import patch

from paper2.restart import freeze_dp_secagg_study as study
from paper2.restart import run_dp_secagg_study as runner
from paper2.restart.test_run_dp_secagg_pilot import PROFILE


class DpSecAggStudyTests(unittest.TestCase):
    def test_twenty_releases_are_composed(self):
        values = study.study_payload()
        self.assertEqual(values["run_config.json"]["num-server-rounds"], 20)
        self.assertEqual(values["privacy_contract.json"]["conditional_budget"]["releases"], 20)
        self.assertAlmostEqual(values["privacy_contract.json"]["conditional_budget"]["epsilon_upper_bound"], 5.989915065723368)
        self.assertFalse(values["release_policy.json"]["private_validation_released"])

    def test_launcher_defaults_to_dry_run(self):
        with patch.object(runner, "verify", return_value={"status": "PASS"}), \
             patch.object(runner, "execution_profile", return_value=PROFILE), \
             patch.object(runner, "command", return_value=["flwr"]):
            result = runner.launch("package", "output")
        self.assertEqual(result["status"], "DRY_RUN_NO_TRAINING")

    def test_native_windows_execution_is_blocked_before_output(self):
        with patch.object(runner, "verify", return_value={"status": "PASS"}), \
             patch.object(runner, "execution_profile", return_value=PROFILE), \
             patch.object(runner, "command", return_value=["flwr"]), \
             patch.object(runner.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "WSL2/Linux"):
                runner.launch("package", "output", execute=True)


if __name__ == "__main__":
    unittest.main()
