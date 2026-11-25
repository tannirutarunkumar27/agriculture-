import unittest

import pandas as pd

from paper2.restart.prepare_ranking_covariates import coverage


class RankingCovariateTests(unittest.TestCase):
    def test_complete_covariates_still_require_design_review(self):
        rows = pd.DataFrame([{"experiment_id": "e", "management": "m", "row_id": "r", "crop_type": "IC"}])
        base = {"experiment_id": "e", "management": "m", "species_mix": "a_b", "cultivar_mix": "x_y",
                "density_level": 10, "density_relative": .5, "density_factor": "0.5_0.5", "interrow": .1,
                "mixing_pattern": "within_row", "mixture_design": "substitutive"}
        management = pd.DataFrame([{**base, "species": "a", "cultivar": "x"}, {**base, "species": "b", "cultivar": "y"}])
        report, components = coverage(rows, management)
        self.assertEqual(report["status"], "RANKING_COVARIATES_RECOVERED_DESIGN_REVIEW_REQUIRED")
        self.assertFalse(report["ranking_enabled"])
        self.assertEqual(len(components), 2)

    def test_duplicate_component_fails_closed(self):
        rows = pd.DataFrame([{"experiment_id": "e", "management": "m", "row_id": "r", "crop_type": "SC"}])
        row = {"experiment_id": "e", "management": "m", "species": "a", "species_mix": "a", "cultivar": "x",
               "cultivar_mix": "x", "density_level": 1, "density_relative": 1, "density_factor": "1",
               "interrow": .1, "mixing_pattern": None, "mixture_design": None}
        with self.assertRaisesRegex(ValueError, "unique"):
            coverage(rows, pd.DataFrame([row, row]))


if __name__ == "__main__":
    unittest.main()
