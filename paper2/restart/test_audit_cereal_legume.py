"""Synthetic audit logic fixtures, not agricultural performance evidence."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

from paper2.restart.audit_cereal_legume import audit, harvest_components, matching_candidates, system_outcomes


class TestFeasibilityAudit(unittest.TestCase):
    def setUp(self):
        self.management = pd.DataFrame([
            {"experiment_id": "e", "management": "m1", "species": "pea", "crop_type": "SC"},
            {"experiment_id": "e", "management": "m2", "species": "pea", "crop_type": "IC"},
            {"experiment_id": "e", "management": "m2", "species": "barley", "crop_type": "IC"}])
        for key, value in {"harvest_date": "2020-07-01", "sowing_date": "2020-03-01", "crop_season": "spring",
                           "N_amount": 0, "P_amount": 0, "irrigation_amount": 0, "pest_control": "yes"}.items():
            self.management[key] = value
        self.traits = self.management.copy()
        self.traits["measurement_date"] = "2020-07-01"
        self.traits["variable"] = "biomass_seed"
        self.traits["value"] = [2., 1., 3.]

    def test_components_and_sum_not_average(self):
        components, counts = harvest_components(self.management, self.traits)
        systems = system_outcomes(components)
        self.assertEqual(counts["selected_component_rows"], 3)
        self.assertEqual(systems.audit_total_seed_dm_t_ha.tolist(), [2., 4.])
        self.assertEqual(len(matching_candidates(systems)), 1)

    def test_early_measurement_excluded(self):
        self.traits.loc[0, "measurement_date"] = "2020-06-01"
        components, counts = harvest_components(self.management, self.traits)
        self.assertEqual(counts["selected_component_rows"], 2)
        self.assertTrue(pd.isna(components.audit_seed_dm_t_ha.iloc[0]))

    def test_duplicate_harvest_rejected(self):
        with self.assertRaises(ValueError):
            harvest_components(self.management, pd.concat([self.traits, self.traits.iloc[:1]], ignore_index=True))

    def test_harvest_metadata_conflict_rejected(self):
        self.traits.loc[0, "harvest_date"] = "2020-07-02"
        with self.assertRaises(ValueError):
            harvest_components(self.management, self.traits)

    def test_unmatched_component_rejected(self):
        self.traits.loc[0, "species"] = "unknown"
        with self.assertRaises(ValueError):
            harvest_components(self.management, self.traits)

    def test_incomplete_intercrop_not_partial_total(self):
        self.traits.loc[2, "value"] = None
        components, _ = harvest_components(self.management, self.traits)
        systems = system_outcomes(components)
        self.assertTrue(pd.isna(systems.audit_total_seed_dm_t_ha.iloc[1]))
        self.assertTrue(matching_candidates(systems).empty)

    def test_management_missing_or_different_blocks_comparison(self):
        for value in (None, 10):
            modified = self.management.copy()
            modified.loc[1:, "irrigation_amount"] = value
            components, _ = harvest_components(modified, self.traits)
            self.assertTrue(matching_candidates(system_outcomes(components)).empty)

    def test_existing_output_refused_before_loading_sources(self):
        with self.assertRaises(FileExistsError):
            audit("unavailable", Path(__file__).parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestFeasibilityAudit))
    (args.output/"unittest.log").write_text(buffer.getvalue(), encoding="utf-8")
    (args.output/"verification.json").write_text(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "fixtures": "synthetic_audit_logic_only", "python": sys.version}, indent=2), encoding="utf-8")
    print(buffer.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
