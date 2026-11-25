"""Feature-timing audit tests."""
import unittest

from paper2.restart.feature_timing_registry import registry


class FeatureTimingRegistryTests(unittest.TestCase):
    def test_registry_disables_pre_sowing_claims(self):
        result = registry({"target": "yield", "feature_timing": "recorded", "numeric_features": ["soil_pH", "N_amount"],
                           "categorical_features": ["previous_crop", "pest_control"], "fixed_features": ["contains_wheat"]})
        self.assertFalse(result["pre_sowing_recommendation_enabled"])
        statuses = {row["feature"]: row["status"] for row in result["features"]}
        self.assertEqual(statuses["N_amount"], "RETROSPECTIVE_ONLY")
        self.assertEqual(statuses["pest_control"], "RETROSPECTIVE_ONLY")

    def test_feature_count_is_complete(self):
        result = registry({"target": "yield", "feature_timing": "recorded", "numeric_features": ["a"],
                           "categorical_features": ["b"], "fixed_features": ["c"]})
        self.assertEqual(len(result["features"]), 3)


if __name__ == "__main__":
    unittest.main()
