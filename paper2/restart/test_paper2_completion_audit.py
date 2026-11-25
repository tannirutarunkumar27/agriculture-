import json
import tempfile
import unittest
from pathlib import Path

from paper2.restart.paper2_completion_audit import build_report, latest_result_file


class Paper2CompletionAuditTests(unittest.TestCase):
    def test_external_blockers_are_not_hidden(self):
        with self.subTest("fixture-backed audit"):
            result = build_report(
                {"verified": 441, "pending": 63},
                {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
                {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False,
                 "missing": ["remote Linux hosts"]},
                {"status": "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW", "rules_schema_valid": 8,
                 "rules_domain_approved_declared": 0},
                {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
                 "blocks_pending": 47, "ranking_enabled": False},
                {"status": "RELEASE_REPRODUCIBILITY_BLOCKED", "release_allowed": False,
                 "blocked": ["clean_linux_reproduction"]},
            )
            self.assertIn("matched_neural_search", result["blockers"])
            self.assertIn("privacy_claim", result["blockers"])
            self.assertIn("rr07_outer_evaluation", result["blockers"])
            self.assertIn("crop_system_ranking", result["blockers"])

    def test_reference_supplement_blockers_are_reported(self):
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            reference_gate={
                "status": "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_UNRESOLVED",
                "source_missing_experiments": 11,
                "approved_usable_reference_supplements": 0,
                "unresolved_experiments": ["e1"],
            },
        )
        ranking = result["checks"]["crop_system_ranking"]
        self.assertEqual(ranking["source_missing_experiments"], 11)
        self.assertEqual(ranking["unresolved_reference_experiments"], ["e1"])

    def test_reviewer_packets_are_exposed_as_evidence_not_approvals(self):
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            artifacts={
                "matched_neural_protocol": "p", "matched_neural_execution": "e",
                "dp_pilot_protocol": "d1", "dp_study_protocol": "d2",
                "dp_pilot_audit": "a1", "dp_study_audit": "a2", "dp_privacy_review_packet": "privacy-review",
                "ranking_review": "rr", "ranking_combined_review_packet": "combined",
                "agronomic_review_packet": "agronomic", "agronomic_review_submission_kit": "submission",
                "ranking_reference_gate": "ref",
                "ranking_gate": "gate", "release_gate": "release", "dependency_license_review_submission_kit": "license-review",
                "dataset_rights_review_submission_kit": "dataset-rights",
                "project_code_license_decision_kit": "code-license",
            },
        )
        self.assertEqual(result["checks"]["independent_agronomic_rules"]["review_packet"], "agronomic")
        self.assertEqual(result["checks"]["independent_agronomic_rules"]["review_submission_kit"], "submission")
        self.assertIn("combined", result["checks"]["crop_system_ranking"]["evidence"])
        self.assertIn("privacy-review", result["checks"]["secure_aggregation"]["evidence"])
        self.assertIn("license-review", result["checks"]["publication_and_clean_reproduction"]["evidence"])
        self.assertIn("dataset-rights", result["checks"]["publication_and_clean_reproduction"]["evidence"])
        self.assertIn("code-license", result["checks"]["publication_and_clean_reproduction"]["evidence"])
        self.assertFalse(result["checks"]["crop_system_ranking"]["claim_allowed"])

    def test_current_protocol_identities_are_reported(self):
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW", "rules_schema_valid": 8,
             "rules_domain_approved_declared": 0},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            {"status": "RELEASE_REPRODUCIBILITY_BLOCKED", "release_allowed": False,
             "blocked": ["clean_linux_reproduction"]},
        )
        neural = result["checks"]["matched_neural_search"]
        self.assertEqual(neural["portable_execution"], "restart_matched_neural_execution_v10")
        self.assertEqual(neural["planned_runs"], 504)
        self.assertFalse(neural["gpu_replication_required"])

    def test_latest_versioned_evidence_is_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in (2, 12):
                path = root / f"restart_gate_v{version}"
                path.mkdir()
                (path / "gate.json").write_text(json.dumps({"version": version}), encoding="utf-8")
            selected = latest_result_file("restart_gate", "gate.json", root)
        self.assertEqual(selected.parent.name, "restart_gate_v12")

    def test_implementation_gaps_are_explicit_blockers(self):
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_RULES_ELIGIBLE_PENDING_DOMAIN_REVIEW", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            gaps=[{"component": "rr07_remote_gpu_runner", "status": "SCAFFOLD_ONLY"}],
        )
        self.assertEqual(result["status"], "PAPER2_IMPLEMENTATION_AND_EXTERNAL_WORK_REMAIN")
        self.assertIn("implementation:rr07_remote_gpu_runner", result["blockers"])
        self.assertTrue(result["checks"]["rr07_outer_evaluation"]["status"].startswith("IMPLEMENTATION_MISSING"))

    def test_complete_cpu_audits_are_reported_without_becoming_gpu_claims(self):
        result = build_report(
            {"verified": 441, "pending": 63}, {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47, "blocks_pending": 47, "ranking_enabled": False},
            cpu_rr06={"status": "PASS_COMPLETE_MATCHED_NEURAL_STUDY", "verified": 504, "device": "cpu"},
            cpu_rr07={"status": "PASS_COMPLETE_RR07_DEVELOPMENT_OUTER_STUDY", "verified": 160, "device": "cpu"},
        )
        self.assertEqual(result["checks"]["matched_neural_search"]["status"], "COMPLETE_AUDITED_CPU")
        self.assertEqual(result["checks"]["rr07_outer_evaluation"]["status"], "COMPLETE_AUDITED_CPU")
        self.assertFalse(result["checks"]["matched_neural_search"]["gpu_replication_required"])
        self.assertTrue(result["checks"]["matched_neural_search"]["gpu_replication_optional"])
        self.assertNotIn("matched_neural_search", result["blockers"])
        self.assertNotIn("rr07_outer_evaluation", result["blockers"])

    def test_terminal_dp_execution_failure_is_visible(self):
        result = build_report(
            {"verified": 441, "pending": 63}, {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47, "blocks_pending": 47, "ranking_enabled": False},
            dp_execution={"status": "ABORTED_BY_OPERATOR_NO_AUTOMATIC_RETRY", "returncode": None,
                          "execution_mode": "local_logical_client_cpu_simulation"},
        )
        self.assertEqual(result["checks"]["secure_aggregation"]["status"], "LOCAL_CPU_EXECUTION_TERMINAL_FAILURE")
        self.assertEqual(result["checks"]["end_to_end_dp_execution"]["status"], "BLOCKED_LOCAL_CPU_EXECUTION_FAILURE")

    def test_audited_wsl_dp_secagg_is_not_reported_as_unexecuted(self):
        pilot = {"status": "PASS_EXECUTED_DP_SECAGG_PILOT_NOT_CERTIFIED", "rounds": 1,
                 "semantic_log_audit": {"passed": True}}
        study = {"status": "PASS_EXECUTED_DP_SECAGG_STUDY_NOT_CERTIFIED", "rounds": 20,
                 "semantic_log_audit": {"passed": True}}
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            dp_pilot_audit=pilot, dp_study_audit=study,
        )
        self.assertEqual(
            result["checks"]["secure_aggregation"]["status"],
            "COMPLETE_AUDITED_WSL_LOGICAL_CLIENTS_NOT_CERTIFIED",
        )
        self.assertNotIn("secure_aggregation", result["blockers"])
        self.assertEqual(
            result["checks"]["end_to_end_dp_execution"]["status"],
            "BLOCKED_APPLICATION_PRIVACY_ACCOUNTING_AND_INDEPENDENT_SECURITY_REVIEW",
        )

    def test_local_smoke_and_free_metadata_are_exposed_without_enabling_claims(self):
        result = build_report(
            {"verified": 441, "pending": 63},
            {"status": "PRIVACY_CLAIM_BLOCKED", "privacy_claim_enabled": False},
            {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False},
            {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "rules_schema_valid": 8},
            {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "blocks_total": 47,
             "blocks_pending": 47, "ranking_enabled": False},
            local_wsl_smoke={"status": "PASS_LOCAL_WSL_LOOPBACK_TLS_AUTH_SMOKE_NOT_DISTRIBUTED_DEPLOYMENT",
                             "online_nodes_observed_from_listing": 7},
            deployment_adapter={"status": "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING"},
        )
        deployment = result["checks"]["external_linux_deployment"]
        self.assertEqual(deployment["local_wsl_smoke"]["online_nodes_observed"], 7)
        self.assertFalse(deployment["local_wsl_smoke"]["counts_as_external_deployment"])
        self.assertTrue(deployment["deployment_adapter"]["semantic_binding_passed"])
        self.assertFalse(deployment["deployment_adapter"]["counts_as_execution"])
        self.assertFalse(result["checks"]["crop_system_ranking"]["claim_allowed"])


if __name__ == "__main__":
    unittest.main()
