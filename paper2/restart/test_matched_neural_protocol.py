"""Synthetic guard tests for the RR06 no-training neural protocol."""
import unittest

from paper2.restart import matched_neural_protocol as protocol


def manifest():
    return {"assets": {f"outer_{outer}_inner_{inner}": {} for outer in range(8) for inner in range(8) if inner != outer}}


class MatchedNeuralProtocolTests(unittest.TestCase):
    def test_matrix_is_complete_and_inner_only(self):
        plan = protocol.make_plan(manifest())
        self.assertEqual(len(plan), 504)
        self.assertEqual(len({row["id"] for row in plan}), 504)
        self.assertTrue(all(row["outer_test_inference"] is False for row in plan))

    def test_adamw_does_not_pair_with_restricted_fednova(self):
        plan = protocol.make_plan(manifest())
        self.assertFalse(any(row["optimizer"] == "adamw" and row["algorithm"] == "fednova_restricted" for row in plan))
        self.assertTrue(any(row["optimizer"] == "sgd" and row["algorithm"] == "fednova_restricted" for row in plan))

    def test_config_keeps_disallowed_claims_off(self):
        settings = protocol.config()
        self.assertFalse(settings["knowledge_enabled"])
        self.assertFalse(settings["privacy_enabled"])
        self.assertFalse(settings["outer_test_inference"])
        self.assertEqual(settings["centralized_work_rule"], "local_steps_times_number_of_training_clients")
        self.assertEqual(settings["centralized_example_rule"], "match_sum_of_federated_fixed_step_example_exposure_per_fold")
        self.assertEqual(settings["optimizer_state_policy"]["centralized"], "persistent_across_rounds")


if __name__ == "__main__":
    unittest.main()
