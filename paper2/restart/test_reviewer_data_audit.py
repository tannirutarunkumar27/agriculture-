import unittest
import csv
from pathlib import Path

from paper2.restart.reviewer_data_audit import analyze, category_bin


class ReviewerDataAuditTests(unittest.TestCase):
    def test_public_rows_provenance_counts(self):
        source = Path(__file__).resolve().parents[1] / "results/restart_public_data_v1/regression_rows.csv"
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            report = analyze(list(csv.DictReader(handle)))
        self.assertEqual(report["n_systems"], 607)
        self.assertEqual(report["n_experiments"], 37)
        self.assertEqual(report["groups"]["Auzeville"]["systems"], 354)
        self.assertEqual(report["category_hash"]["fields"]["previous_crop"]["distinct_colliding_pairs"], 9)

    def test_clustering_and_hash_collisions(self):
        rows = [
            {"row_id": "a", "experiment_id": "e1", "geo_group": "g1",
             "previous_crop": "p", "crop_season": "s", "pest_control": ""},
            {"row_id": "b", "experiment_id": "e1", "geo_group": "g1",
             "previous_crop": "p", "crop_season": "s", "pest_control": ""},
            {"row_id": "c", "experiment_id": "e2", "geo_group": "g2",
             "previous_crop": "q", "crop_season": "s", "pest_control": "x"},
        ]
        result = analyze(rows)
        self.assertEqual(result["n_experiments"], 2)
        self.assertEqual(result["n_experiments_with_multiple_systems"], 1)
        self.assertEqual(result["groups"]["g1"]["experiments"], 1)
        self.assertEqual(result["category_hash"]["fields"]["pest_control"]["missing_rows"], 2)
        self.assertEqual(category_bin("x", "y"), category_bin("x", "y"))
        rows[1]["geo_group"] = "g2"
        with self.assertRaisesRegex(ValueError, "crosses"):
            analyze(rows)


if __name__ == "__main__":
    unittest.main()
