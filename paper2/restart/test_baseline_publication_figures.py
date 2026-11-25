"""Regression tests for baseline figures; no training or inference."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest

from paper2.restart import baseline_publication_figures as figures
from paper2.restart.validation_search import ROOT, atomic_json, read


class BaselineFigureTests(unittest.TestCase):
    def setUp(self):
        self.original = read(ROOT / "paper2/results/restart_outer_analysis_v1/analysis.json")
        self.guarded = read(ROOT / "paper2/results/restart_guard_analysis_v1/analysis.json")

    def test_exact_inventory_counts_and_metrics(self):
        values = figures.extract(self.original, self.guarded)
        self.assertEqual([r["algorithm"] for r in values["overview"]], list(figures.ALGORITHMS))
        self.assertEqual(sum(r["samples"] for r in values["geographic"]), 607)
        self.assertEqual(len(values["geographic"]), 8)
        self.assertEqual(len(values["posthoc_sensitivity"]), 4)
        xgb = values["overview"][0]
        source = next(r for r in self.original["summary"] if r["algorithm"] == "xgboost" and r["panel"] == "all")
        self.assertEqual(xgb["equal_group_rmse"]["mean"], source["equal_group_rmse"]["mean"])

    def test_partial_scope_or_inconsistent_values_rejected(self):
        mutations = []
        bad = copy.deepcopy(self.original); bad["verified_runs"] = 319; mutations.append((bad, self.guarded))
        bad = copy.deepcopy(self.guarded); bad["scope"] = "INDEPENDENT"; mutations.append((self.original, bad))
        bad = copy.deepcopy(self.original); bad["summary"][0]["equal_group_rmse"]["mean"] += 1; mutations.append((bad, self.guarded))
        bad = copy.deepcopy(self.guarded); bad["summary"][0]["equal_group_rmse_mean"] += 1; mutations.append((self.original, bad))
        bad = copy.deepcopy(self.guarded); bad["training_performed"] = True; mutations.append((self.original, bad))
        for original, guarded in mutations:
            with self.assertRaises(ValueError):
                figures.extract(original, guarded)

    def test_three_figures_render_in_three_formats(self):
        values = figures.extract(self.original, self.guarded)
        with tempfile.TemporaryDirectory(prefix="baseline_plot_test_", dir=ROOT / "paper2/results") as temp:
            assets, captions = figures.draw(values, temp)
            self.assertEqual(len(assets), 9)
            self.assertEqual(len(captions), 3)
            self.assertTrue(all((Path(temp) / name).stat().st_size > 500 for name in assets))
            self.assertFalse(figures.plt.get_fignums())

    def test_original_and_posthoc_are_not_merged(self):
        values = figures.extract(self.original, self.guarded)
        self.assertTrue(all("guarded" not in row for row in values["overview"]))
        self.assertIn("POST_HOC_SENSITIVITY", values["guard_scope"])
        self.assertTrue(all(row["algorithm"] in figures.NEURAL for row in values["posthoc_sensitivity"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BaselineFigureTests))
    (args.output / "unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    atomic_json(args.output / "verification.json", {"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun, "errors": len(result.errors), "failures": len(result.failures)})
    print(stream.getvalue()); raise SystemExit(0 if result.wasSuccessful() else 1)
