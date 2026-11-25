"""Synthetic work-accounting tests for the independent RR06 auditor."""
import unittest

from paper2.restart.verify_matched_neural import audit_history_work


class VerifyMatchedNeuralTests(unittest.TestCase):
    def test_exact_step_and_example_accounting(self):
        run = {"local_steps": 4, "rounds": 2}
        groups = {"a": list(range(5)), "b": list(range(5, 7))}
        settings = {"batch_size": 3}
        history = [{"optimizer_steps": 8, "examples_processed": 18} for _ in range(2)]
        self.assertEqual(audit_history_work(run, history, groups, settings),
                         {"optimizer_steps_per_round": 8, "examples_per_round": 18})
        history[1]["examples_processed"] = 17
        with self.assertRaisesRegex(ValueError, "work"):
            audit_history_work(run, history, groups, settings)

    def test_incomplete_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "length"):
            audit_history_work({"local_steps": 4, "rounds": 2},
                               [{"optimizer_steps": 4, "examples_processed": 4}], {"a": [0]}, {"batch_size": 1})


if __name__ == "__main__":
    unittest.main()
