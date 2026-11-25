"""RR07 planning safeguards; no outer arrays or inference."""
import unittest

from paper2.restart.rr07_development_protocol import make_plan


class Rr07ProtocolTests(unittest.TestCase):
    def test_requires_finalization_and_plans_160_runs(self):
        selections = []
        for outer in range(8):
            for algorithm, objective in (("centralized", "sample_weighted"), ("fedavg", "sample_weighted"),
                                         ("fedprox", "sample_weighted"), ("fedavg", "equal_client"),
                                         ("fednova_restricted", "sample_weighted")):
                selections.append({"outer_fold": f"outer_{outer}", "algorithm": algorithm,
                    "aggregation_objective": objective, "selected": {"optimizer": "sgd", "hidden": 32, "schedule": "constant"},
                    "refit_rounds": 12})
        plan = make_plan({"status": "PASS_FINALIZED_INNER_SELECTION", "rr07_ready": True, "selection": selections},
                         {"assets": {f"outer_{i}": {} for i in range(8)}})
        self.assertEqual(len(plan), 160)
        self.assertFalse(any(row["outer_test_inference"] for row in plan))
        self.assertTrue(all(row["local_steps"] == 4 and row["rounds"] == 12 for row in plan))
        self.assertTrue(all(row["learning_rate"] == .01 and row["weight_decay"] == 0. for row in plan))
        self.assertFalse(any(row["algorithm"] == "fednova_restricted" for row in plan))

    def test_partial_finalization_rejected(self):
        with self.assertRaisesRegex(ValueError, "finalization"):
            make_plan({"status": "PASS_PARTIAL_MATCHED_NEURAL_STUDY", "rr07_ready": False, "selection": []}, {"assets": {}})


if __name__ == "__main__":
    unittest.main()
