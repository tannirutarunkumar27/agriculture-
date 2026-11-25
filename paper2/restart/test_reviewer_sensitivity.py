import math
import json
import unittest
from pathlib import Path

from paper2.restart.reviewer_sensitivity import aggregate, analyze, index_all_panel


def rows():
    result = []
    for algorithm, values in (("xgboost", (1.0, 2.0)), ("fedavg", (2.0, 4.0))):
        for seed in (1, 2):
            for group, n, rmse in (("a", 4, values[0]), ("b", 16, values[1])):
                result.append({"algorithm": algorithm, "seed": seed, "group": group,
                               "panel": "all", "metrics": {"samples": n, "rmse": rmse}})
    return result


class ReviewerSensitivityTests(unittest.TestCase):
    def test_archived_primary_metric_reproduced(self):
        source = Path(__file__).resolve().parents[1] / "results/restart_outer_analysis_v1/analysis.json"
        archived = json.loads(source.read_text(encoding="utf-8"))
        result = analyze(archived["per_region"])
        all_equal = next(x for x in result["scenarios"]
                         if x["scenario"] == "all_groups" and x["weighting"] == "equal_group")
        all_pooled = next(x for x in result["scenarios"]
                          if x["scenario"] == "all_groups" and x["weighting"] == "pooled")
        self.assertEqual(sum(result["group_sizes"].values()), 607)
        self.assertAlmostEqual(all_equal["methods"]["xgboost"]["mean"], 1.2461, places=4)
        self.assertAlmostEqual(all_pooled["methods"]["xgboost"]["mean"], 1.7335, places=4)

    def test_pooled_uses_squared_error_not_average_rmse(self):
        indexed, _, _, groups = index_all_panel(rows())
        self.assertAlmostEqual(aggregate(indexed, "xgboost", 1, groups, "pooled"),
                               math.sqrt((4 + 16 * 4) / 20))
        self.assertEqual(aggregate(indexed, "xgboost", 1, groups, "equal_group"), 1.5)

    def test_pairing_and_tiny_group_exclusion(self):
        result = analyze(rows())
        self.assertEqual(result["group_sizes"], {"a": 4, "b": 16})
        entry = next(x for x in result["scenarios"] if x["scenario"] == "exclude_n_le_6"
                     and x["weighting"] == "equal_group")
        self.assertEqual(entry["groups"], ["b"])
        self.assertEqual(entry["methods"]["xgboost"]["mean"], 2.0)
        diff = next(x for x in result["paired_group_rmse_differences"]
                    if x["group"] == "a" and x["method"] == "fedavg")
        self.assertEqual(diff["rmse_difference_mean"], 1.0)

    def test_missing_duplicate_and_invalid_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete paired grid"):
            analyze(rows()[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate row"):
            analyze(rows() + rows()[:1])
        invalid = rows()
        invalid[0]["metrics"]["rmse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze(invalid)


if __name__ == "__main__":
    unittest.main()
