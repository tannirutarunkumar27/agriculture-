"""Synthetic regression optimizer tests; independent of historical classifier tests."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from paper2.federated import clone_state, aggregate
from paper2.restart.regression_training import make_regressor, train_regression_client


class TestRegressionClient(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.model = make_regressor(2, 4)
        self.x, self.y = torch.randn(5, 2), torch.arange(5, dtype=torch.float32)

    def update(self, **kwargs):
        return train_regression_client(self.model, self.x, self.y, "a", 1, "schema", 123, **kwargs)

    def test_step_count_and_global_unchanged(self):
        before = clone_state(self.model)
        result = self.update(batch_size=2)
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.samples, 5)
        for key in before:
            torch.testing.assert_close(before[key], self.model.state_dict()[key], rtol=0, atol=0)

    def test_determinism_and_mu_zero(self):
        a, b = self.update(), self.update(mu=0.)
        for key in a.state:
            torch.testing.assert_close(a.state[key], b.state[key], rtol=0, atol=0)

    def test_nonfinite_inputs_reject(self):
        self.y[0] = float("nan")
        with self.assertRaises(ValueError):
            self.update()

    def test_invalid_optimizer_reject(self):
        for kwargs in ({"learning_rate": -1}, {"learning_rate": float("nan")}, {"batch_size": 0}, {"mu": -1}):
            with self.assertRaises(ValueError):
                self.update(**kwargs)

    def test_all_aggregators_scalar_regression(self):
        a = self.update(batch_size=2)
        b = train_regression_client(self.model, self.x[:3], self.y[:3], "b", 1, "schema", 456, batch_size=2)
        for algorithm in ("fedavg", "fedprox", "fednova"):
            state = aggregate(clone_state(self.model), [a, b], algorithm, 1, "schema", {"a": 5, "b": 3})
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestRegressionClient))
    (args.output/"unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    (args.output/"verification.json").write_text(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}, indent=2), encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
