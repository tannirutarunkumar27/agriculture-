import unittest

import pandas as pd

from paper2.restart.ranking_approval_gate import evaluate, validate_protocol
from paper2.restart.review_ranking_experiments import APPROVAL_FIELDS


def frames(reference=True, decision=""):
    blocks = pd.DataFrame([{"block_id": "b1", "experiment_id": "e1", "reference_present": reference}])
    approval = {"block_id": "b1", "experiment_id": "e1", "reviewer": "",
                "reviewer_affiliation": "", "review_date": "", "source_check_date": "",
                "reviewer_independent": "", "conflict_of_interest_disclosed": "",
                "source_locator": "", "decision": decision, "rationale": ""}
    approval.update({field: "" for field in APPROVAL_FIELDS})
    return blocks, pd.DataFrame([approval])


class RankingApprovalGateTests(unittest.TestCase):
    def test_blank_template_is_pending(self):
        report, rows = evaluate(*frames())
        self.assertEqual(report["status"], "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW")
        self.assertFalse(report["ranking_enabled"])
        self.assertEqual(rows.iloc[0].gate_status, "PENDING_REVIEW")

    def test_complete_include_can_pass(self):
        blocks, approvals = frames(decision="include")
        approvals.loc[0, ["reviewer", "reviewer_affiliation", "review_date", "source_check_date",
                          "reviewer_independent", "conflict_of_interest_disclosed", "source_locator", "rationale"]] = [
            "Independent reviewer", "Independent university", "2026-10-27", "2026-10-27",
            "true", "true", "Methods p. 4", "Comparable randomized treatments",
        ]
        approvals.loc[0, APPROVAL_FIELDS] = "true"
        report, _ = evaluate(blocks, approvals)
        self.assertTrue(report["ranking_enabled"])

    def test_include_without_reference_is_invalid(self):
        blocks, approvals = frames(reference=False, decision="include")
        approvals.loc[0, ["reviewer", "reviewer_affiliation", "review_date", "source_check_date",
                          "reviewer_independent", "conflict_of_interest_disclosed", "source_locator", "rationale"]] = [
            "Independent reviewer", "Independent university", "2026-10-27", "2026-10-27",
            "true", "true", "Methods p. 4", "Reviewed",
        ]
        approvals.loc[0, APPROVAL_FIELDS] = "true"
        report, rows = evaluate(blocks, approvals)
        self.assertEqual(report["status"], "RANKING_APPROVAL_BLOCKED_INVALID_REVIEW")
        self.assertIn("publication_reference_missing", rows.iloc[0].errors)

    def test_partial_block_set_is_rejected(self):
        blocks, approvals = frames()
        approvals.loc[0, "block_id"] = "other"
        with self.assertRaisesRegex(ValueError, "every reviewed block"):
            evaluate(blocks, approvals)

    def test_self_review_is_invalid(self):
        blocks, approvals = frames(decision="exclude")
        approvals.loc[0, ["reviewer", "reviewer_affiliation", "review_date", "source_check_date",
                          "source_locator", "rationale"]] = [
            "Project author", "Project team", "2026-10-27", "2026-10-27", "Workbook", "Excluded",
        ]
        report, rows = evaluate(blocks, approvals)
        self.assertFalse(report["ranking_enabled"])
        self.assertIn("reviewer_independent_not_true", rows.iloc[0].errors)

    def test_modified_protocol_is_rejected(self):
        protocol = {"version": "restricted-observed-ranking-review-v1",
                    "status": "FROZEN_REVIEW_CONTRACT_NOT_APPROVED_FOR_RANKING",
                    "candidate_scope": "only packages actually tested in the source experiment",
                    "utility": {"measure": "profit", "unit": "USD", "economic": True, "nutritional": False},
                    "required_block_approvals": APPROVAL_FIELDS, "prohibited_claims": []}
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
