"""No-training tests for the SecAgg+ pilot freeze contract."""
import unittest
from unittest.mock import patch

from paper2.restart import freeze_secagg_pilot as pilot


class SecAggPilotFreezeTests(unittest.TestCase):
    def test_payload_is_no_evaluation_and_has_complete_roster(self):
        manifest = {"status": "PREPARED_NO_TRAINING", "assets": {"outer_0": {}}}
        partition = {"kind": "outer_refit", "clients": {f"c{i}": [i] for i in range(7)}}
        with patch.object(pilot, "read", side_effect=[manifest, partition]):
            values = pilot.payload("fixture", "outer_0")
        self.assertFalse(values["run_config.json"]["outer_test_evaluation"])
        self.assertEqual(len(values["roster.json"]["clients"]), 7)
        self.assertFalse(values["security_scope.json"]["differential_privacy_enabled"])

    def test_incomplete_roster_is_rejected(self):
        manifest = {"status": "PREPARED_NO_TRAINING", "assets": {"outer_0": {}}}
        partition = {"kind": "outer_refit", "clients": {f"c{i}": [i] for i in range(6)}}
        with patch.object(pilot, "read", side_effect=[manifest, partition]):
            with self.assertRaises(ValueError): pilot.payload("fixture", "outer_0")


if __name__ == "__main__":
    unittest.main()
