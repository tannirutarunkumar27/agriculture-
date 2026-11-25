"""Compatibility tests for the pinned, vetted SecAgg+ integration gate."""
import unittest

from paper2.restart import flower_secagg as secagg


class FlowerSecAggTests(unittest.TestCase):
    def test_pinned_workflow_constructs(self):
        workflow = secagg.build_workflow()
        self.assertEqual(workflow.num_shares, 7)
        self.assertEqual(workflow.reconstruction_threshold, 5)
        self.assertEqual(workflow.max_weight, 1.0)

    def test_invalid_rosters_and_weights_are_rejected(self):
        for config in (
            secagg.SecAggPilotConfig(participants=2),
            secagg.SecAggPilotConfig(num_shares=8),
            secagg.SecAggPilotConfig(reconstruction_threshold=7),
            secagg.SecAggPilotConfig(max_weight=2.0),
            secagg.SecAggPilotConfig(modulus_range=100),
        ):
            with self.assertRaises(ValueError):
                secagg.validate(config)

    def test_record_has_no_privacy_or_deployment_claim(self):
        record = secagg.protocol_record()
        self.assertTrue(record["secure_aggregation_requested"])
        self.assertFalse(record["differential_privacy_enabled"])
        self.assertFalse(record["production_deployment"])


if __name__ == "__main__":
    unittest.main()
