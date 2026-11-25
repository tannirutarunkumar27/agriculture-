import copy
import unittest

import numpy as np
import pandas as pd

from paper2.data import group_ids, make_partition, prepare_arrays, validate_partition


class TestPartitions(unittest.TestCase):
    def setUp(self):
        self.schema = {"numeric": ["N", "ph"], "categorical": ["soil"], "target": "crop", "region": "district"}
        self.frame = pd.DataFrame({"N": np.arange(90), "ph": np.tile(np.arange(30), 3),
                                   "soil": ["black"]*90, "crop": np.repeat(["a", "b", "c"], 30),
                                   "district": np.tile(["east", "west", "north"], 30)})

    def partition(self, **kwargs):
        return make_partition(self.frame, self.schema, "hash", "toy", min_client_rows=1, **kwargs)

    def test_reproducible_all_strategies(self):
        for strategy in ["iid", "dirichlet", "unequal", "district"]:
            with self.subTest(strategy=strategy):
                a = self.partition(strategy=strategy, clients=3)
                b = self.partition(strategy=strategy, clients=3)
                self.assertEqual(a, b)
                validate_partition(self.frame, self.schema, a)

    def test_same_holdout_across_client_strategies(self):
        a = self.partition(strategy="iid")
        b = self.partition(strategy="dirichlet")
        self.assertEqual(a["splits"], b["splits"])

    def test_profile_duplicate_stays_together(self):
        self.frame = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        p = self.partition()
        for rows in list(p["splits"].values()) + list(p["clients"].values()):
            self.assertEqual(0 in rows, 90 in rows)

    def test_field_stays_together(self):
        self.schema["field"] = "field"
        self.frame["field"] = np.repeat(np.arange(30), 3)
        p = self.partition(clients=3)
        groups = group_ids(self.frame, self.schema)
        sets = [set(groups[rows]) for rows in p["splits"].values()]
        self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])

    def test_train_only_scaling_and_vocabulary(self):
        p = self.partition()
        _, _, original = prepare_arrays(self.frame, self.schema, p)
        changed = self.frame.copy()
        ids = p["splits"]["test"]
        changed.loc[ids, "N"] = 100000
        changed.loc[ids, "soil"] = "unseen"
        x, _, metadata = prepare_arrays(changed, self.schema, p)
        self.assertEqual(original, metadata)
        self.assertTrue(np.all(x[ids, -1] == 0))

    def test_invalid_client_overlap_rejected(self):
        p = self.partition()
        keys = list(p["clients"])
        p["clients"][keys[1]].append(p["clients"][keys[0]][0])
        with self.assertRaises(ValueError):
            validate_partition(self.frame, self.schema, p)

    def test_invalid_holdout_overlap_rejected(self):
        p = self.partition()
        p["splits"]["test"][0] = p["splits"]["train"][0]
        with self.assertRaises(ValueError):
            validate_partition(self.frame, self.schema, p)

    def test_conflicting_profile_labels_rejected(self):
        row = self.frame.iloc[[0]].copy()
        row["crop"] = "b"
        self.frame = pd.concat([self.frame, row], ignore_index=True)
        with self.assertRaises(ValueError):
            self.partition()

    def test_missing_region_cannot_be_invented(self):
        del self.schema["region"]
        with self.assertRaises(ValueError):
            self.partition(strategy="district")

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            self.partition(alpha=float("nan"))
