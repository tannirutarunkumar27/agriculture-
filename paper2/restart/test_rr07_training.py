"""Synthetic RR07 fit/evaluation tests; no project outer arrays are accessed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from paper2.restart import rr07_training as training
from paper2.restart.validation_search import ROOT


def run():
    return {"id": "outer_0_fedavg_sample_weighted_sgd_seed123", "fold": "outer_0", "seed": 123,
            "variant": "fedavg_sample_weighted", "algorithm": "fedavg", "aggregation_objective": "sample_weighted",
            "outer_test_inference": False, "optimizer": "sgd", "schedule": "constant", "hidden": 4,
            "learning_rate": .01, "weight_decay": 0., "rounds": 3, "local_steps": 4}


class Rr07TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "paper2/results"); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "run"; self.settings = {"seed": 42, "batch_size": 2, "fedprox_mu": .01}
        self.fold = {"kind": "outer_refit", "train": [10, 11, 12, 13], "validation": [], "test": [20, 21],
                     "clients": {"a": [10, 11], "b": [12, 13]}, "preprocessing_identity": "schema"}
        self.train = {"x_train": np.array([[0., 1.], [1., 0.], [1., 1.], [2., 1.]], np.float32),
                      "y_train": np.array([0., 1., 2., 3.], np.float32)}
        self.test = (np.array([[.5, .5], [1.5, 1.]], np.float32), np.array([1., 2.], np.float32), np.array([20, 21]))

    def test_resume_then_once_only_evaluation(self):
        item = run()
        self.assertIsNone(training.fit(self.folder, item, self.settings, self.fold, self.train, "token", stop_after=1))
        training.fit(self.folder, item, self.settings, self.fold, self.train, "token")
        result = training.evaluate_once(self.folder, item, *self.test, "token")
        self.assertTrue(result["outer_test_inference"])
        again = training.evaluate_once(self.folder, item, *self.test, "token")
        self.assertEqual(result, again)
        state = torch.load(self.folder / "checkpoint.pt", weights_only=True)
        self.assertEqual(state["done"], 3)

    def test_fit_refuses_test_arrays(self):
        arrays = dict(self.train, x_test=self.test[0])
        with self.assertRaisesRegex(ValueError, "only receive training"):
            training.fit(self.folder, run(), self.settings, self.fold, arrays, "token")


if __name__ == "__main__":
    unittest.main()
