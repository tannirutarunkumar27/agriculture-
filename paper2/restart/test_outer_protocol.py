"""Synthetic protocol planning safeguards; no real test inference."""
import unittest
from paper2.restart.outer_protocol import make_plan


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.selected = {"outer_0": {"common_learning_rate": .001,
            "outer_refit_rounds": {"centralized": 9, "fedavg": 80, "fedprox": 80, "fednova": 20},
            "references": {name: {"configuration": {"model": name}} for name in
                           ("ridge", "training_mean", "random_forest", "xgboost")}}}
        self.prepared = {"assets": {"outer_0": {}}}

    def test_counts_and_unique_ids(self):
        plan = make_plan(self.selected, self.prepared, [123, 2024])
        self.assertEqual(len(plan), 16)
        self.assertEqual(len({r["id"] for r in plan}), 16)

    def test_preserves_selected_budgets(self):
        plan = make_plan(self.selected, self.prepared, [123])
        self.assertEqual(plan[0]["rounds"], 9)
        self.assertEqual(plan[0]["learning_rate"], .001)
        self.assertTrue(all(r["fold"] == "outer_0" for r in plan))

    def test_bad_seeds(self):
        for seeds in ([], [42], [123, 123], [True], [-1]):
            with self.assertRaises(ValueError):
                make_plan(self.selected, self.prepared, seeds)

    def test_missing_fold(self):
        with self.assertRaises(ValueError):
            make_plan(self.selected, {"assets": {}}, [123])

    def test_invalid_rounds(self):
        for rounds in (0, 81, True):
            self.selected["outer_0"]["outer_refit_rounds"]["centralized"] = rounds
            with self.assertRaises(ValueError):
                make_plan(self.selected, self.prepared, [123])


if __name__ == "__main__":
    unittest.main()
