import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_agronomic_review_packet import packet_rows, prepare


def registry():
    return {
        "schema_version": 2,
        "status": "candidate_rules_pending_independent_domain_review",
        "rules": [{
            "rule_id": "r1", "crop_taxon": "pea", "source_crop_name": "Pisum sativum",
            "feature": "soil_pH", "source_url_or_doi": "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1721",
            "source_locator": "FAO ECOCROP code 1721, Ecology table, Soil PH row, Optimal Min/Max",
            "original_min": 5.5, "original_max": 7.0, "original_unit": "pH scale (-)",
            "dataset_unit": "-", "transformation": "identity; direct pH comparison", "region": "global",
            "variety_or_stage": "species-level", "water_regime": "not specified", "confidence": "candidate",
            "approval_rationale": "requires independent review", "independent_of_dataset": True,
            "checked_date": "2026-11-09", "domain_review_status": "pending",
        }],
    }


class AgronomicReviewPacketTests(unittest.TestCase):
    def test_packet_preserves_rule_scope_without_approval(self):
        row = packet_rows(registry())[0]
        self.assertEqual(row["optimal_ph_min"], 5.5)
        self.assertIn("no default inclusion", row["required_reviewer_decision"])

    def test_prepare_never_enables_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "output"; config = root / "registry.json"
            import json
            config.write_text(json.dumps(registry()), encoding="utf-8")
            with patch("paper2.restart.prepare_agronomic_review_packet.RESULTS", root):
                report = prepare(output, config)
        self.assertFalse(report["guidance_enabled"])
        self.assertEqual(report["rules"], 1)


if __name__ == "__main__":
    unittest.main()
