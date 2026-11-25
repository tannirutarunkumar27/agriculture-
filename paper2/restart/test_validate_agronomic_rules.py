import unittest
from pathlib import Path

from paper2.restart.validate_agronomic_rules import REGISTRY, REQUIRED, validate_registry


def registry(rule):
    return {"schema_version": 2, "status": "candidate_rules_pending_independent_domain_review", "rules": [rule]}


def rule():
    value = {key: "declared" for key in REQUIRED}
    value |= {
        "rule_id": "r1", "source_crop_name": "Pisum sativum", "feature": "soil_pH",
        "source_url_or_doi": "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1721",
        "source_locator": "FAO ECOCROP code 1721, Ecology table, Soil PH row, Optimal Min/Max",
        "original_min": 5.5, "original_max": 7.0, "original_unit": "pH scale (-)",
        "dataset_unit": "-", "transformation": "identity; direct comparison",
        "independent_of_dataset": True, "checked_date": "2026-10-27", "domain_review_status": "pending",
    }
    return value


class AgronomicRuleValidationTests(unittest.TestCase):
    def test_default_registry_is_current_candidate_bank(self):
        self.assertEqual(REGISTRY.name, "knowledge_registry_v2_candidates.json")
        self.assertTrue(Path(REGISTRY).is_file())

    def test_empty_bank_blocks_guidance(self):
        result = validate_registry({"schema_version": 2,
                                    "status": "candidate_rules_pending_independent_domain_review", "rules": []})
        self.assertEqual(result["status"], "AGRONOMIC_RULES_BLOCKED")
        self.assertFalse(result["guidance_enabled"])

    def test_complete_schema_is_only_pending_domain_review(self):
        result = validate_registry(registry(rule()))
        self.assertEqual(result["status"], "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW")
        self.assertFalse(result["guidance_enabled"])

    def test_declared_approval_requires_reviewer_and_stays_disabled(self):
        value = rule(); value["domain_review_status"] = "approved"
        invalid = validate_registry(registry(value))
        self.assertEqual(invalid["rules_schema_valid"], 0)
        value |= {"domain_reviewer": "Independent agronomist", "domain_review_date": "2026-10-27"}
        declared = validate_registry(registry(value))
        self.assertEqual(declared["rules_domain_approved_declared"], 1)
        self.assertFalse(declared["guidance_enabled"])

    def test_unattributed_rule_fails(self):
        result = validate_registry(registry({"rule_id": "r1"}))
        self.assertEqual(result["rules_schema_valid"], 0)
        self.assertTrue(result["errors"])

    def test_impossible_ph_and_non_fao_source_fail(self):
        value = rule(); value["original_max"] = 15; value["source_url_or_doi"] = "https://example.org/rule"
        result = validate_registry(registry(value))
        errors = result["errors"][0]["errors"]
        self.assertIn("invalid_numeric_bounds", errors)
        self.assertIn("invalid_fao_source_locator", errors)


if __name__ == "__main__":
    unittest.main()
