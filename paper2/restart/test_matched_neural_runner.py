"""Synthetic resume and selection tests for the RR06 runner."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from paper2.restart import matched_neural_runner as runner
from paper2.restart.validation_search import ROOT, atomic_json, read, sha


def run(identifier="r", optimizer="sgd", algorithm="fedavg", objective="sample_weighted"):
    return {"id": identifier, "fold": "outer_0_inner_1", "variant": "fixture", "algorithm": algorithm,
            "aggregation_objective": objective, "outer_test_inference": False, "optimizer": optimizer,
            "schedule": "constant" if optimizer == "sgd" else "cosine", "hidden": 4,
            "learning_rate": .01, "weight_decay": 0. if optimizer == "sgd" else .0001, "rounds": 3, "local_steps": 4}


class MatchedNeuralRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "paper2/results"); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"seed": 42, "batch_size": 2, "fedprox_mu": .01}
        self.fold = {"kind": "inner_selection", "train": [10, 11, 12, 13], "validation": [20, 21], "test": [],
                     "clients": {"a": [10, 11], "b": [12, 13]}, "preprocessing_identity": "schema"}
        self.arrays = {"x_train": np.array([[0., 1.], [1., 0.], [1., 1.], [2., 1.]], np.float32),
                       "y_train": np.array([0., 1., 2., 3.], np.float32),
                       "x_validation": np.array([[.5, .5], [1.5, 1.]], np.float32),
                       "y_validation": np.array([1., 2.], np.float32)}

    def test_exact_state_resume_and_completion_receipt(self):
        item = run(); interrupted = self.root / "interrupted"; full = self.root / "full"
        self.assertIsNone(runner.candidate(interrupted, item, self.settings, self.fold, self.arrays, "token", stop_after=1))
        resumed = runner.candidate(interrupted, item, self.settings, self.fold, self.arrays, "token")
        direct = runner.candidate(full, item, self.settings, self.fold, self.arrays, "token")
        self.assertEqual(resumed["best_round"], direct["best_round"])
        left = torch.load(interrupted / "checkpoint.pt", weights_only=True)["state"]
        right = torch.load(full / "checkpoint.pt", weights_only=True)["state"]
        self.assertTrue(all(torch.equal(left[name], right[name]) for name in left))

    def test_outer_or_overlapping_fold_is_rejected(self):
        bad = dict(self.fold, kind="outer_refit")
        with self.assertRaisesRegex(ValueError, "Inner-only"):
            runner.candidate(self.root / "bad", run(), self.settings, bad, self.arrays, "token")
        retained_test_metadata = dict(self.fold, test=[99])
        self.assertEqual(set(runner._fold_groups(retained_test_metadata, 4)), {"a", "b"})
        overlap = dict(self.fold, test=[10])
        with self.assertRaisesRegex(ValueError, "Inner-only"):
            runner._fold_groups(overlap, 4)

    def test_dry_run_does_not_create_output(self):
        with patch.object(runner, "verify_execution", return_value=({}, self.settings, [run()], self.root / "data")):
            result = runner.execute(self.root / "package", self.root / "output", workers=4)
        self.assertEqual(result["status"], "DRY_RUN_NO_TRAINING")
        self.assertEqual(result["workers"], 4)
        self.assertFalse((self.root / "output").exists())
        for invalid in (0, 9, True, 1.5):
            with self.assertRaises(ValueError):
                runner.execute(self.root / "package", self.root / "output", workers=invalid)

    def test_cuda_mode_fails_closed_without_cuda(self):
        with patch.dict(runner.os.environ, {"PAPER2_DEVICE": "cuda"}), patch.object(runner, "verify_execution", return_value=({}, self.settings, [run()], self.root / "data")):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runner.execute(self.root / "package", self.root / "output")

    def test_valid_cached_completion_repairs_interrupted_journal(self):
        item = run(); folder = self.root / "cached"
        runner.candidate(folder, item, self.settings, self.fold, self.arrays, "token")
        attempts_path = folder / "attempts.json"
        attempts = [{"attempt": 1, "status": "STARTED", "identity": "token"}]
        atomic_json(attempts_path, attempts)
        repaired = runner.reconcile_cached_attempt(folder, attempts_path, attempts)
        self.assertEqual(repaired[-1]["status"], "CACHE_VERIFIED_AFTER_INTERRUPTION")
        self.assertEqual(repaired[-1]["completion_sha256"], sha(folder / "completion.json"))
        self.assertEqual(read(attempts_path), repaired)


if __name__ == "__main__":
    unittest.main()
