import unittest

from paper2.restart.agronomic_domain_review_gate import INDEPENDENCE_FIELDS, REVIEW_FIELDS, evaluate, make_template


def registry():
    return {"rules": [{"rule_id": "r1", "crop_taxon": "pea", "feature": "soil_pH"}]}


class AgronomicDomainReviewGateTests(unittest.TestCase):
    def test_blank_template_is_pending(self):
        report, rows = evaluate(registry(), make_template(registry()))
        self.assertEqual(report["status"], "AGRONOMIC_DOMAIN_REVIEW_PENDING")
        self.assertFalse(report["guidance_enabled"])
        self.assertEqual(rows.iloc[0].gate_status, "PENDING_REVIEW")

    def test_complete_declaration_still_does_not_enable_guidance(self):
        reviews = make_template(registry())
        reviews.loc[0, ["reviewer", "reviewer_affiliation", "review_date", "source_check_date", "decision", "rationale"]] = [
            "Independent agronomist", "Independent university", "2026-10-27", "2026-10-27",
            "include", "Source and scope reviewed",
        ]
        reviews.loc[0, INDEPENDENCE_FIELDS] = "true"
        reviews.loc[0, REVIEW_FIELDS] = "true"
        report, _ = evaluate(registry(), reviews)
        self.assertTrue(report["declarations_complete"])
        self.assertFalse(report["guidance_enabled"])

    def test_self_review_or_undisclosed_conflict_is_invalid(self):
        reviews = make_template(registry())
        reviews.loc[0, ["reviewer", "reviewer_affiliation", "review_date", "source_check_date", "decision", "rationale"]] = [
            "Implementation author", "Project team", "2026-10-27", "2026-10-27", "exclude", "Reviewed",
        ]
        report, rows = evaluate(registry(), reviews)
        self.assertEqual(report["status"], "AGRONOMIC_DOMAIN_REVIEW_BLOCKED_INVALID")
        self.assertIn("reviewer_independent_not_true", rows.iloc[0].errors)

    def test_partial_or_changed_identity_is_rejected(self):
        reviews = make_template(registry())
        reviews.loc[0, "rule_id"] = "other"
        with self.assertRaisesRegex(ValueError, "every candidate rule"):
            evaluate(registry(), reviews)


if __name__ == "__main__":
    unittest.main()
