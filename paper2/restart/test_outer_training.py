import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from paper2.restart.outer_training import fit, evaluate_once, verify_evaluation
from paper2.restart.validation_search import ROOT


class OuterTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "paper2/results")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.x = np.arange(16, dtype=np.float32).reshape(8, 2) / 16
        self.y = np.arange(8, dtype=np.float32) / 8
        self.fold = {"train": list(range(8)), "clients": {"a": [0, 1, 2], "b": [3, 4, 5, 6, 7]}, "preprocessing_identity": "synthetic"}
        self.settings = {"hidden": 4, "batch_size": 2, "mu": .01}
        self.run = {"algorithm": "fednova", "rounds": 3, "seed": 123, "learning_rate": .001}

    def test_exact_resume_all_neural_methods(self):
        for algorithm in ("centralized", "fedavg", "fedprox", "fednova"):
            run = dict(self.run, algorithm=algorithm)
            a, b = self.root / (algorithm + "a"), self.root / (algorithm + "b")
            fit(a, run, self.settings, self.fold, self.x, self.y, "token")
            self.assertIsNone(fit(b, run, self.settings, self.fold, self.x, self.y, "token", stop_after=1))
            fit(b, run, self.settings, self.fold, self.x, self.y, "token")
            sa = torch.load(a / "model.pt", weights_only=True)
            sb = torch.load(b / "model.pt", weights_only=True)
            self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))

    def test_completed_fit_never_retrains(self):
        fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "token")
        with patch("paper2.restart.outer_training.train_regression_client", side_effect=AssertionError("retrain")):
            fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "token")

    def test_once_only_cache_and_tamper(self):
        fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "token")
        result = evaluate_once(self.root, self.run, self.settings, self.x, self.y, np.arange(8), "token")
        with patch("paper2.restart.outer_training.make_regressor", side_effect=AssertionError("reinference")):
            self.assertEqual(result, evaluate_once(self.root, self.run, self.settings, self.x, self.y, np.arange(8), "token"))
        (self.root / "result.json").write_text("{}")
        with self.assertRaises(ValueError):
            verify_evaluation(self.root, "token")

    def test_ambiguous_inference_stops(self):
        fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "token")
        (self.root / "inference_started.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            evaluate_once(self.root, self.run, self.settings, self.x, self.y, np.arange(8), "token")

    def test_changed_identity(self):
        fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "token", stop_after=1)
        with self.assertRaises(ValueError):
            fit(self.root, self.run, self.settings, self.fold, self.x, self.y, "changed")

    def test_reference_smokes(self):
        references = [{"model": "training_mean"}, {"model": "ridge", "alpha": 1},
                      {"model": "random_forest", "trees": 3, "min_samples_leaf": 1},
                      {"model": "xgboost", "rounds": 3, "max_depth": 2, "learning_rate": .05}]
        for candidate in references:
            folder = self.root / candidate["model"]
            run = {"algorithm": candidate["model"], "seed": 123, "configuration": candidate}
            fit(folder, run, self.settings, self.fold, self.x, self.y, "token")
            result = evaluate_once(folder, run, self.settings, self.x, self.y, np.arange(8), "token")
            self.assertTrue(np.isfinite(result["test"]["rmse"]))

    def test_bad_arrays_and_clients(self):
        with self.assertRaises(ValueError):
            fit(self.root, self.run, self.settings, self.fold, self.x.astype(np.float64), self.y, "token")
        bad = dict(self.fold, clients={"a": [0, 1], "b": [1, 2]})
        with self.assertRaises(ValueError):
            fit(self.root, self.run, self.settings, bad, self.x, self.y, "token")


if __name__ == "__main__":
    unittest.main()
