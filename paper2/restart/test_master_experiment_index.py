import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper2.restart.master_experiment_index import _best_validation, record, verify_run


class MasterExperimentIndexTests(unittest.TestCase):
    def test_receipt_verifies_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = {"identity": "abc", "run": {"id": "run"}}
            source = json.dumps(result).encode()
            (root / "result.json").write_bytes(source)
            receipt = {"identity": "abc", "assets": {
                "result.json": hashlib.sha256(source).hexdigest()}}
            (root / "completion.json").write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(verify_run(root)[0], result)
            (root / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_run(root)

    def test_best_validation_uses_minimum_not_final(self):
        metric, round_number = _best_validation([
            {"round": 1, "validation": {"rmse": 1.0}},
            {"round": 2, "validation": {"rmse": 2.0}},
        ])
        self.assertEqual(round_number, 1)
        self.assertEqual(metric["rmse"], 1.0)

    def test_outer_split_keeps_classification_fields_blank(self):
        result = {"identity": "abc", "run": {
            "id": "outer_0_fedavg_seed1", "fold": "outer_0", "seed": 1,
            "algorithm": "fedavg", "rounds": 10, "learning_rate": .01},
            "test": {"samples": 2, "rmse": 1.0, "mae": .8}}
        partitions = {"outer_0": {"test_group": "A", "validation_group": None,
                                  "test": [0, 1], "validation": [], "clients": {"B": [2]}}}
        row = record("restart_outer_study_v1", result, Path("outer_0_fedavg_seed1"),
                     "hash", "receipt", partitions, {})
        self.assertEqual(row["evaluation_split"], "outer_test")
        self.assertEqual(row["client_count"], 1)
        self.assertEqual(row["accuracy"], "")
        self.assertEqual(row["communication_bytes_measured"], "")
        result["test"]["samples"] = 3
        with self.assertRaisesRegex(ValueError, "group size mismatch"):
            record("restart_outer_study_v1", result, Path("outer_0_fedavg_seed1"),
                   "hash", "receipt", partitions, {})


if __name__ == "__main__":
    unittest.main()
