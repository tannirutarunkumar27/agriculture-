import unittest

from paper2.restart.ranking_reference_gate import evaluate


def candidate(**updates):
    row = {
        "experiment_id": "e1",
        "candidate_source": "Study",
        "doi_or_url": "https://doi.org/10.1000/test",
        "source_type": "peer_reviewed_original_article",
        "exact_site_year_link": "confirmed",
        "coverage": "full",
        "evidence_locator": "Methods p. 4",
        "evidence_summary": "Site, year, cultivars and treatments match.",
        "confidence": "high",
        "independent_review_status": "pending",
        "reviewer": "",
        "review_date": "",
        "review_rationale": "",
        "usable_as_original_reference": False,
    }
    row.update(updates)
    return row


def registry(*rows):
    return {"schema_version": "1.0", "checked_date": "2026-10-27", "candidates": list(rows)}


class RankingReferenceGateTests(unittest.TestCase):
    def test_exact_match_without_review_remains_blocked(self):
        report, rows = evaluate(registry(candidate()), {"e1"})
        self.assertEqual(report["status"], "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_UNRESOLVED")
        self.assertFalse(report["ranking_reference_supplement_enabled"])
        self.assertEqual(rows.iloc[0].gate_status, "EXACT_MATCH_PENDING_INDEPENDENT_REVIEW")

    def test_approved_exact_original_reference_can_pass(self):
        row = candidate(
            independent_review_status="approved",
            reviewer="Independent reviewer",
            reviewer_affiliation="Independent university",
            review_date="2026-10-27",
            source_check_date="2026-10-27",
            reviewer_independent=True,
            conflict_of_interest_disclosed=True,
            review_rationale="Verified against methods and treatment table",
            usable_as_original_reference=True,
        )
        report, rows = evaluate(registry(row), {"e1"})
        self.assertTrue(report["ranking_reference_supplement_enabled"])
        self.assertTrue(rows.iloc[0].computed_usable)

    def test_declared_usable_pending_record_is_invalid(self):
        report, rows = evaluate(registry(candidate(usable_as_original_reference=True)), {"e1"})
        self.assertEqual(report["status"], "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_INVALID_RECORDS")
        self.assertIn("declared_usability_disagrees", rows.iloc[0].errors)

    def test_every_source_missing_experiment_requires_a_candidate(self):
        report, _ = evaluate(registry(candidate()), {"e1", "e2"})
        self.assertEqual(report["experiments_without_candidates"], ["e2"])
        self.assertEqual(report["status"], "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_INVALID_RECORDS")

    def test_duplicate_candidates_cannot_enable_reference_supplement(self):
        approved = candidate(independent_review_status="approved", reviewer="Reviewer",
            reviewer_affiliation="Independent university", review_date="2026-10-27",
            source_check_date="2026-10-27", reviewer_independent=True,
            conflict_of_interest_disclosed=True, review_rationale="Verified", usable_as_original_reference=True)
        report, _ = evaluate(registry(approved, candidate(doi_or_url="https://doi.org/10.1000/other")), {"e1"})
        self.assertFalse(report["ranking_reference_supplement_enabled"])
        self.assertEqual(report["duplicate_experiment_candidate_records"], ["e1"])


if __name__ == "__main__":
    unittest.main()
