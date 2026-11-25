import unittest

from paper2.restart.privacy_claim_gate import evaluate


class PrivacyClaimGateTests(unittest.TestCase):
    def test_current_scope_is_blocked(self):
        result = evaluate(dp_training=False, secure_aggregation=False, threat_model=True, release_audit=False,
                          cryptographic_review=False, known_security_issues_resolved=False)
        self.assertEqual(result["status"], "PRIVACY_CLAIM_BLOCKED")
        self.assertFalse(result["privacy_claim_enabled"])

    def test_all_evidence_is_only_pending_review(self):
        result = evaluate(dp_training=True, secure_aggregation=True, threat_model=True, release_audit=True,
                          cryptographic_review=True, known_security_issues_resolved=True)
        self.assertEqual(result["status"], "PRIVACY_CLAIM_ELIGIBLE_PENDING_REVIEW")

    def test_execution_cannot_override_unresolved_cryptographic_issue(self):
        result = evaluate(dp_training=True, secure_aggregation=True, threat_model=True, release_audit=True,
                          cryptographic_review=True, known_security_issues_resolved=False)
        self.assertEqual(result["status"], "PRIVACY_CLAIM_BLOCKED")
        self.assertFalse(result["privacy_claim_enabled"])


if __name__ == "__main__":
    unittest.main()
