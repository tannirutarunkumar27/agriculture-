"""Synthetic search/selection/recovery fixtures, never agricultural measurements."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import paper2.restart.validation_search as search


class TestValidationSearch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root/"data"
        self.folder = self.data/"folds"/"outer_0_inner_1"
        self.folder.mkdir(parents=True)
        self.config = search.read(search.ROOT/"paper2/configs/restart_validation_search_v1.json")
        self.config.update(rounds=2, hidden=4, batch_size=2, learning_rates=[.001])
        self.fold = {"kind": "inner_selection", "train": list(range(5)), "validation": [5, 6],
                     "clients": {"a": [0, 1, 2], "b": [3, 4]}, "preprocessing_identity": "toy"}
        rng = np.random.default_rng(42)
        self.arrays = {"x_train": rng.normal(size=(5, 2)).astype(np.float32), "y_train": np.arange(5, dtype=np.float32),
                       "ids_train": np.arange(5), "x_validation": rng.normal(size=(2, 2)).astype(np.float32),
                       "y_validation": np.array([1, 2], np.float32), "ids_validation": np.array([5, 6])}
        np.savez(self.folder/"arrays.npz", **self.arrays)
        search.atomic_json(self.folder/"partition.json", self.fold)
        search.atomic_json(self.folder/"preprocessing.json", {"test_fixture": True})
        self.prepared = {"assets": {"outer_0_inner_1": {name: search.sha(self.folder/name)
                         for name in ("arrays.npz", "partition.json", "preprocessing.json")}}}
        search.atomic_json(self.data/"manifest.json", self.prepared)
        self.plan = search.make_plan(self.config, self.prepared)
        self.manifest = {"data": str(self.data), "test_fixture": True}
        self.run = self.plan[0]

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self, name):
        folder = self.root/name
        folder.mkdir(exist_ok=True)
        return search.train_candidate(folder, self.run, "fixture", self.config, self.fold, self.arrays)

    def test_metrics_units_and_bad_arrays(self):
        self.assertEqual(search.metrics(np.array([1., 2.]), np.array([2., 3.]))["rmse"], 1.)
        for prediction in (np.array([1.]), np.array([np.nan, 1.])):
            with self.assertRaises(ValueError):
                search.metrics(np.array([1., 2.]), prediction)

    def test_disabled_test_and_bad_config(self):
        search.validate_config(self.config)
        bad = copy.deepcopy(self.config)
        bad["test_inference"] = True
        with self.assertRaises(ValueError):
            search.validate_config(bad)

    def test_complete_cached_verification_and_tamper(self):
        result = self.candidate("complete")
        self.assertEqual(search.verify_completed(self.root/"complete", "fixture"), result)
        (self.root/"complete"/"result.json").write_text("{}")
        with self.assertRaises(ValueError):
            search.verify_completed(self.root/"complete", "fixture")

    def test_exact_round_resume(self):
        uninterrupted = self.candidate("whole")
        original = search.atomic_checkpoint
        def interrupt(path, value):
            original(path, value)
            if value["done"] == 1:
                raise RuntimeError("synthetic interruption after round commit")
        with patch.object(search, "atomic_checkpoint", side_effect=interrupt), self.assertRaises(RuntimeError):
            self.candidate("resume")
        resumed = self.candidate("resume")
        self.assertEqual(resumed, uninterrupted)
        a = torch.load(self.root/"whole"/"model.pt", weights_only=True)
        b = torch.load(self.root/"resume"/"model.pt", weights_only=True)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_final_checkpoint_does_not_retrain(self):
        self.candidate("final")
        with patch.object(search, "train_regression_client", side_effect=AssertionError("must not retrain")):
            self.candidate("final")

    def test_wrong_checkpoint_identity_rejected(self):
        self.candidate("changed")
        with self.assertRaises(ValueError):
            search.train_candidate(self.root/"changed", self.run, "different", self.config, self.fold, self.arrays)

    def test_outer_test_fields_rejected(self):
        values = self.arrays | {"x_test": np.zeros((1, 2), np.float32)}
        np.savez(self.folder/"arrays.npz", **values)
        self.prepared["assets"]["outer_0_inner_1"]["arrays.npz"] = search.sha(self.folder/"arrays.npz")
        with self.assertRaises(ValueError):
            search.load_inner(self.data, self.prepared, self.run)

    def test_dry_run_no_writes(self):
        with patch.object(search, "verify_frozen", return_value=(self.manifest, self.config, self.plan)):
            result = search.execute(self.root/"frozen", self.root/"dry")
        self.assertEqual(result["status"], "DRY_RUN_NO_TRAINING")
        self.assertFalse((self.root/"dry").exists())

    def test_bounded_execution_resume_and_overwrite_rejection(self):
        with patch.object(search, "verify_frozen", return_value=(self.manifest, self.config, self.plan)):
            result = search.execute(self.root/"frozen", self.root/"study", max_runs=1, dry_run=False)
            self.assertEqual(result["completed"], 1)
            with self.assertRaises(ValueError):
                search.execute(self.root/"frozen", self.root/"study", dry_run=False)
            result = search.execute(self.root/"frozen", self.root/"study", max_runs=1, resume=True, dry_run=False)
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["executed_this_invocation"], 1)

    def test_failure_stops_and_requires_retry(self):
        with patch.object(search, "verify_frozen", return_value=(self.manifest, self.config, self.plan)):
            with patch.object(search, "train_candidate", side_effect=RuntimeError("toy failure")), self.assertRaises(RuntimeError):
                search.execute(self.root/"frozen", self.root/"failed", dry_run=False)
            with self.assertRaises(ValueError):
                search.execute(self.root/"frozen", self.root/"failed", resume=True, dry_run=False)
            folder = self.root/"failed"/"runs"/self.run["id"]
            self.assertTrue((folder/"failure_attempt_1.json").exists())
            self.assertEqual(search.read(self.root/"failed"/"progress.json")["status"], "FAILED_STOPPED")

    def test_writer_exclusion(self):
        with search.writer_lock(self.root):
            with self.assertRaises((RuntimeError, OSError)):
                with search.writer_lock(self.root):
                    pass

    def test_partial_selection_rejected(self):
        with self.assertRaises(ValueError):
            search.selections([], self.config, self.plan)

    def test_selection_ties_and_refit_median(self):
        results = [{"id": run["id"], "validation": {"rmse": 1.}, "best_round": 2} for run in self.plan]
        selected = search.selections(results, self.config, self.plan)["outer_0"]
        self.assertEqual(selected["common_learning_rate"], .001)
        self.assertEqual(selected["references"]["ridge"]["configuration"]["alpha"], .1)
        self.assertEqual(selected["outer_refit_rounds"]["fedavg"], 2)

    def test_real_frozen_plan_integrity_without_training(self):
        frozen = self.root/"real_freeze"
        result = search.freeze(search.ROOT/"paper2/configs/restart_validation_search_v1.json",
            search.ROOT/"paper2/results/restart_regression_data_v1",
            search.ROOT/"paper2/results/restart_regression_verified_v1/verification.json", frozen)
        self.assertEqual(result["planned_runs"], 1120)
        search.verify_frozen(frozen)
        (frozen/"plan.json").write_text("[]")
        with self.assertRaises(ValueError):
            search.verify_frozen(frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestValidationSearch))
    (args.output/"unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    (args.output/"verification.json").write_text(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "fixtures": "synthetic_only"}, indent=2), encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
