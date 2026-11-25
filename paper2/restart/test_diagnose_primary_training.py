"""Synthetic tests for the read-only primary-search diagnostic."""
import tempfile
import unittest
from pathlib import Path

from paper2.restart import diagnose_primary_training as diagnostic


class PrimaryDiagnosticTests(unittest.TestCase):
    def test_mean_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            diagnostic._mean([])

    def test_validate_result_rejects_incomplete_history(self):
        run = {"id": "r", "fold": "f", "algorithm": "fedavg"}
        result = {"id": "r", "fold": "f", "algorithm": "fedavg", "best_round": 2,
                  "history": [{"round": 1}]}
        with self.assertRaises(ValueError):
            diagnostic._validate_result(run, result, 2)

    def test_validate_result_accepts_complete_neural_history(self):
        run = {"id": "r", "fold": "f", "algorithm": "fedprox"}
        result = {"id": "r", "fold": "f", "algorithm": "fedprox", "best_round": 2,
                  "history": [{"round": 1}, {"round": 2}]}
        diagnostic._validate_result(run, result, 2)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(FileExistsError):
                diagnostic.analyze(output / "frozen", output / "study", output)


if __name__ == "__main__":
    unittest.main()
