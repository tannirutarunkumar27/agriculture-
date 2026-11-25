"""Tests for the derived, no-training dataset/client tables."""
import unittest

import pandas as pd

from paper2.restart.data_pipeline import CATEGORICAL, FIXED, NUMERIC, SPECIES
from paper2.restart.dataset_client_validation import FEATURE_UNITS, client_profile


class TestDatasetClientValidation(unittest.TestCase):
    def test_feature_contract_has_exactly_21_inputs(self):
        self.assertEqual(len(NUMERIC + CATEGORICAL + FIXED), 21)
        self.assertEqual(set(FEATURE_UNITS), set(NUMERIC + CATEGORICAL + FIXED))

    def test_client_profile_counts_multilabel_species_not_classes(self):
        rows = pd.DataFrame({
            "experiment_id": ["a", "a", "b"],
            "crop_type": ["IC", "SC", "SC"],
            "target_seed_dm_t_ha": [3., 1., 9.],
            **{f"contains_{name}": [int(name in ("pea", "barley")),
                                      int(name == "pea"), int(name == "lentil")]
               for name in SPECIES},
        })
        profile = client_profile(rows, [0, 1])
        self.assertEqual(profile["n_systems"], 2)
        self.assertEqual(profile["n_experiments"], 1)
        self.assertEqual(profile["intercrop_systems"], 1)
        self.assertEqual(profile["contains_pea_systems"], 2)
        self.assertEqual(profile["contains_barley_systems"], 1)
        self.assertEqual(profile["contains_lentil_systems"], 0)
        self.assertEqual(profile["target_median_t_ha"], 2.)


if __name__ == "__main__":
    unittest.main()
