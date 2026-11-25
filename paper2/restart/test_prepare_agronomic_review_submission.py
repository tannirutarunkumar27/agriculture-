import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from paper2.restart.prepare_agronomic_review_submission import prepare


class AgronomicReviewSubmissionTests(unittest.TestCase):
    def test_submission_kit_is_blank_source_bound_and_not_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "output"; registry_path = root / "registry.json"
            registry_path.write_text(json.dumps({
                "schema_version": 2, "status": "candidate_rules_pending_independent_domain_review",
                "rules": [{
                    "rule_id": "r1", "crop_taxon": "pea", "source_crop_name": "Pea",
                    "feature": "soil_pH", "source_url_or_doi": "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1721",
                    "source_locator": "FAO ECOCROP code 1721; Soil PH Optimal Min/Max",
                    "original_min": 5.5, "original_max": 7.0, "original_unit": "pH scale (-)",
                    "dataset_unit": "-", "transformation": "identity; compare source optimal interval directly with dataset soil_pH",
                    "region": "global", "variety_or_stage": "not specified", "water_regime": "not specified",
                    "confidence": "candidate", "approval_rationale": "pending review", "independent_of_dataset": True,
                    "checked_date": "2026-11-10", "domain_review_status": "pending",
                }],
            }), encoding="utf-8")
            with patch("paper2.restart.prepare_agronomic_review_submission.RESULTS", root):
                report = prepare(output, registry_path)
            decisions = pd.read_csv(output / "domain_review_decisions.csv", keep_default_na=False)
            attestation = json.loads((output / "reviewer_attestation.template.json").read_text(encoding="utf-8"))
        self.assertFalse(report["guidance_enabled"])
        self.assertEqual(decisions.loc[0, "decision"], "")
        self.assertEqual(attestation["reviewer_name"], "")
        self.assertTrue(attestation["registry_sha256"])


if __name__ == "__main__":
    unittest.main()
