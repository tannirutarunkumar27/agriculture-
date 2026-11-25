"""Tests for provenance-safe CPU/GPU study comparison."""
import json
import tempfile
import unittest
from pathlib import Path

from paper2.restart import compare_matched_neural_studies as comparator
from paper2.restart.validation_search import ROOT, atomic_json


class CompareMatchedNeuralTests(unittest.TestCase):
    def test_comparison_keeps_distinct_execution_manifests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            root = Path(temp); cpu = root / "cpu"; gpu = root / "gpu"; output = root / "comparison"
            cpu.mkdir(); gpu.mkdir()
            base = {"id": "r1", "optimizer_steps_per_round": 24, "examples_per_round": 384,
                    "best_validation_rmse": 1.0, "best_round": 2}
            atomic_json(cpu / "audit.json", {"status": "PASS_PARTIAL_MATCHED_NEURAL_STUDY", "verified": 1,
                        "outer_test_inference": False, "execution_manifest_sha256": "cpu", "work": [base]})
            gpu_row = dict(base, best_validation_rmse=1.1, best_round=3)
            atomic_json(gpu / "audit.json", {"status": "PASS_COMPLETE_MATCHED_NEURAL_STUDY", "verified": 1,
                        "outer_test_inference": False, "execution_manifest_sha256": "gpu", "work": [gpu_row]})
            report = comparator.compare(cpu, gpu, output)
            self.assertEqual(report["paired_runs"], 1)
            self.assertTrue(report["execution_manifests_distinct"])
            saved = json.loads((output / "comparison.json").read_text())
            self.assertFalse(saved["selection_available"])

    def test_work_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            root = Path(temp); cpu = root / "cpu"; gpu = root / "gpu"; cpu.mkdir(); gpu.mkdir()
            common = {"status": "PASS_PARTIAL_MATCHED_NEURAL_STUDY", "verified": 1,
                      "outer_test_inference": False, "execution_manifest_sha256": "x"}
            atomic_json(cpu / "audit.json", dict(common, work=[{"id": "r", "optimizer_steps_per_round": 24, "examples_per_round": 384,
                "best_validation_rmse": 1., "best_round": 1}]))
            atomic_json(gpu / "audit.json", dict(common, work=[{"id": "r", "optimizer_steps_per_round": 24, "examples_per_round": 400,
                "best_validation_rmse": 1., "best_round": 1}]))
            with self.assertRaisesRegex(ValueError, "Work"):
                comparator.compare(cpu, gpu, root / "out")


if __name__ == "__main__":
    unittest.main()
