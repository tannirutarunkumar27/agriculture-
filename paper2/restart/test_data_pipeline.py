"""Synthetic data-engineering fixtures, not scientific validation."""
import argparse
import copy
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from paper2.restart.data_pipeline import (NUMERIC, CATEGORICAL, FIXED, geographic_groups, nested_partitions,
                           check_partition, fit_preprocessor, transform, support_flags)


class TestRegressionPreparation(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"row_id": [str(i) for i in range(8)], "geo_group": [str(i//2) for i in range(8)],
                                 "site": [str(i//2) for i in range(8)], "experiment_id": [str(i//2) for i in range(8)],
                                 "system_species": ["pea"]*8, "target_seed_dm_t_ha": np.arange(8.)})
        for name in NUMERIC:
            self.rows[name] = np.arange(8.)
        for name in CATEGORICAL:
            self.rows[name] = ["a", "b"]*4
        for name in FIXED:
            self.rows[name] = int(name == "contains_pea")

    def test_nested_counts_and_client_exclusion(self):
        folds = nested_partitions(self.rows)
        self.assertEqual(sum(f["kind"] == "outer_refit" for f in folds), 4)
        self.assertEqual(sum(f["kind"] == "inner_selection" for f in folds), 12)
        for fold in folds:
            self.assertNotIn(fold["test_group"], fold["clients"])
            self.assertNotIn(fold["validation_group"], fold["clients"])

    def test_group_and_index_overlap_rejected(self):
        fold = nested_partitions(self.rows)[0]
        fold["train"].append(fold["test"][0])
        with self.assertRaises(ValueError):
            check_partition(self.rows, fold)

    def test_duplicate_rows_rejected(self):
        self.rows.loc[1, "row_id"] = self.rows.row_id.iloc[0]
        with self.assertRaises(ValueError):
            nested_partitions(self.rows)

    def test_train_statistics_and_heldout_changes_do_not_leak(self):
        before = fit_preprocessor(self.rows, [0, 1])
        changed = self.rows.copy()
        changed.loc[2:, list(NUMERIC)] = 1e9
        changed.loc[2:, list(CATEGORICAL)] = "never_seen"
        changed.loc[2:, "target_seed_dm_t_ha"] = 1e9
        after = fit_preprocessor(changed, [0, 1])
        self.assertEqual(before, after)
        self.assertEqual(before["numeric"][NUMERIC[0]]["median"], .5)

    def test_missing_numeric_and_unknown_category(self):
        state = fit_preprocessor(self.rows, [0, 1])
        held = self.rows.iloc[2:3].copy()
        held[NUMERIC[0]] = np.nan
        held[CATEGORICAL[0]] = "unseen"
        array = transform(held, state)
        self.assertTrue(np.isfinite(array).all())
        self.assertEqual(array[0, state["feature_names"].index(NUMERIC[0]+"__missing")], 1)
        self.assertEqual(array[0, state["feature_names"].index(CATEGORICAL[0]+"__unknown")], 1)

    def test_all_missing_training_field_explicit(self):
        self.rows.loc[:1, NUMERIC[0]] = np.nan
        state = fit_preprocessor(self.rows, [0, 1])
        self.assertTrue(state["numeric"][NUMERIC[0]]["all_training_missing"])
        self.assertTrue(np.isfinite(transform(self.rows, state)).all())

    def test_infinity_rejected(self):
        self.rows.loc[0, NUMERIC[0]] = np.inf
        with self.assertRaises(ValueError):
            fit_preprocessor(self.rows, [0, 1])

    def test_geographic_transitive_union(self):
        trials = pd.DataFrame({"site": ["a", "b", "c", "d"], "plot_lat": [0., 0., 0., 10.],
                               "plot_long": [0., .005, .01, 10.]})
        mapping, pairs = geographic_groups(trials, 1.)
        self.assertEqual(mapping["a"], "a+b+c")
        self.assertEqual(mapping["b"], mapping["c"])
        self.assertEqual(mapping["d"], "d")
        self.assertEqual(len(pairs), 6)

    def test_support_uses_training_composition_only(self):
        self.rows.loc[2, "system_species"] = "pea+barley"
        flags = support_flags(self.rows, [0, 1], [2])
        self.assertFalse(flags[0]["all_species_seen"])
        self.assertFalse(flags[0]["composition_seen"])

    def test_splits_are_deterministic(self):
        self.assertEqual(nested_partitions(self.rows), nested_partitions(self.rows.copy()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestRegressionPreparation))
    (args.output/"unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    (args.output/"verification.json").write_text(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}, indent=2), encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
