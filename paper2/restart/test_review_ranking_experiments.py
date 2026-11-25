import unittest

import pandas as pd

from paper2.restart.review_ranking_experiments import review_frames


def fixtures(reference="https://doi.org/10.1000/example"):
    trials = pd.DataFrame([{"experiment_id": "e1", "site": "s", "country": "c", "year": 2020}])
    refs = pd.DataFrame([{
        "experiment_id": "e1", "reference linked to original experiments": reference,
    }])
    key = {"experiment_id": "e1", "crop_season": "spring", "sowing_date": "2020-01-01",
           "N_amount": 0, "P_amount": 0, "irrigation_amount": 0.0, "pest_control": "yes"}
    blocks = pd.DataFrame([{**key, "systems": "a;a+b", "alternative_count": 2,
                            "management_count": 2, "status": "candidate_not_causally_certified"}])
    systems = pd.DataFrame([
        {**key, "management": "m1", "crop_type": "SC", "system_species": "a",
         "audit_total_seed_dm_t_ha": 1.0, "component_harvest_dates": "2020-06-01",
         "site": "s", "country": "c", "year": 2020},
        {**key, "management": "m2", "crop_type": "IC", "system_species": "a+b",
         "audit_total_seed_dm_t_ha": 2.0, "component_harvest_dates": "2020-06-01;2020-06-02",
         "site": "s", "country": "c", "year": 2020},
    ])
    common = {"density_level": 10, "density_relative": 1, "density_factor": "1",
              "interrow": .2, "mixing_pattern": "within_row", "mixture_design": "substitutive"}
    components = pd.DataFrame([
        {"experiment_id": "e1", "management": "m1", "species": "a", "cultivar": "ca", **common},
        {"experiment_id": "e1", "management": "m2", "species": "a", "cultivar": "ca", **common},
        {"experiment_id": "e1", "management": "m2", "species": "b", "cultivar": "cb", **common},
    ])
    return trials, refs, blocks, systems, components


class RankingExperimentReviewTests(unittest.TestCase):
    def test_complete_reference_still_fails_closed(self):
        result = review_frames(*fixtures())
        summary = result["summary"]
        self.assertEqual(summary["candidate_blocks"], 1)
        self.assertEqual(summary["candidate_packages"], 2)
        self.assertEqual(summary["blocks_approved_for_ranking"], 0)
        self.assertFalse(summary["ranking_enabled"])
        self.assertIn("original_randomization_not_independently_confirmed",
                      result["block_review"].iloc[0].blocking_reasons)

    def test_missing_reference_is_explicit_blocker(self):
        result = review_frames(*fixtures(reference=None))
        self.assertEqual(result["summary"]["experiments_missing_publication_reference"], 1)
        self.assertIn("publication_reference_missing", result["block_review"].iloc[0].blocking_reasons)

    def test_reference_registry_must_match_trials(self):
        values = list(fixtures())
        values[1] = pd.DataFrame([{"experiment_id": "other", "reference linked to original experiments": None}])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            review_frames(*values)

    def test_changed_block_counts_are_rejected(self):
        values = list(fixtures())
        values[2].loc[0, "management_count"] = 3
        with self.assertRaisesRegex(ValueError, "management count changed"):
            review_frames(*values)


if __name__ == "__main__":
    unittest.main()
