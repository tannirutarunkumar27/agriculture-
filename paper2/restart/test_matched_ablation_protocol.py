"""Synthetic tests for the no-training matched-ablation freeze."""
import copy
import tempfile
import unittest
from pathlib import Path

from paper2.restart import matched_ablation_protocol as protocol


def manifest():
    return {"assets": {f"outer_{outer}_inner_{inner}": {} for outer in range(8) for inner in range(8) if inner != outer}}


class MatchedAblationProtocolTests(unittest.TestCase):
    def test_matrix_is_complete_and_inner_only(self):
        plan = protocol.make_plan(manifest())
        self.assertEqual(len(plan), 280)
        self.assertEqual(len({row["id"] for row in plan}), 280)
        self.assertTrue(all(row["outer_test_inference"] is False for row in plan))
        self.assertEqual({row["variant"] for row in plan}, {row[0] for row in protocol.VARIANTS})

    def test_missing_inner_fold_is_rejected(self):
        bad = manifest()
        bad["assets"].pop(next(iter(bad["assets"])))
        with self.assertRaises(ValueError):
            protocol.make_plan(bad)

    def test_config_keeps_disallowed_claims_off(self):
        settings = protocol.config()
        self.assertFalse(settings["knowledge_enabled"])
        self.assertFalse(settings["privacy_enabled"])
        self.assertFalse(settings["outer_test_inference"])


if __name__ == "__main__":
    unittest.main()
