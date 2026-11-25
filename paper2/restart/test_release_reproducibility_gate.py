import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from paper2.restart.release_reproducibility_gate import (
    evaluate,
    validate_clean_linux_receipt,
    validate_clean_linux_candidate,
    validate_linux_lock,
)


def values():
    bundle = {"status": "PASS"}
    licensing = {
        "dataset": {"license_identifier": None, "redistribution_approved": False},
        "project_code": {"license_identifier": None, "redistribution_approved": False},
        "derived_artifacts": {},
        "third_party_dependencies": {"license_inventory_complete": False},
    }
    completion = {"checks": {"matched_neural_search": {"status": "PENDING"}}}
    network = {"status": "BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE", "deployment_verified": False}
    ranking = {"status": "RANKING_APPROVAL_BLOCKED_PENDING_REVIEW", "ranking_enabled": False}
    agronomic = {"status": "AGRONOMIC_RULES_BLOCKED", "guidance_enabled": False}
    domain = {"status": "AGRONOMIC_DOMAIN_REVIEW_PENDING", "guidance_enabled": False}
    references = {"status": "RANKING_REFERENCE_SUPPLEMENT_BLOCKED_UNRESOLVED",
                  "ranking_reference_supplement_enabled": False}
    dependency_inventory = {"license_inventory_complete": False}
    return (bundle, bundle, licensing, "not a cross-platform lock", completion, network, ranking,
            agronomic, domain, references, dependency_inventory)


class ReleaseReproducibilityGateTests(unittest.TestCase):
    def test_current_missing_external_evidence_blocks_release(self):
        report = evaluate(*values(), project_license_present=False, linux_lock_present=False,
                          clean_linux_receipt_present=False)
        self.assertEqual(report["status"], "RELEASE_REPRODUCIBILITY_BLOCKED")
        self.assertFalse(report["raw_dataset_in_release_allowed"])
        self.assertIn("clean_linux_reproduction", report["blocked"])

    def test_complete_evidence_can_pass(self):
        args = list(values())
        args[2]["dataset"] = {"license_identifier": "approved", "redistribution_approved": True}
        args[2]["project_code"] = {"license_identifier": "approved", "redistribution_approved": True}
        args[2]["third_party_dependencies"] = {"license_inventory_complete": True}
        args[4] = {"checks": {"matched_neural_search": {"status": "COMPLETE"}}}
        args[5] = {"status": "PASS", "deployment_verified": True}
        args[6] = {"status": "PASS", "ranking_enabled": True}
        args[7] = {"status": "PASS", "guidance_enabled": True}
        args[8] = {"status": "PASS", "guidance_enabled": True}
        args[9] = {"status": "PASS", "ranking_reference_supplement_enabled": True}
        args[10] = {"license_inventory_complete": True}
        report = evaluate(*args, project_license_present=True, linux_lock_present=True,
                          clean_linux_receipt_present=True)
        self.assertTrue(report["release_allowed"])

    def test_bundle_failure_never_passes(self):
        args = list(values())
        args[0] = {"status": "FAIL"}
        report = evaluate(*args, project_license_present=False, linux_lock_present=False,
                          clean_linux_receipt_present=False)
        self.assertIn("rr06_remote_bundle_integrity", report["blocked"])

    def test_audited_cpu_matched_study_satisfies_compute_gate(self):
        args = list(values())
        args[4] = {"checks": {"matched_neural_search": {"status": "COMPLETE_AUDITED_CPU"}}}
        report = evaluate(*args, project_license_present=False, linux_lock_present=False,
                          clean_linux_receipt_present=False)
        self.assertEqual(report["checks"]["matched_neural_results"], "PASS")
        self.assertNotIn("matched_neural_results", report["blocked"])

    def test_linux_lock_rejects_marker_only_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements-linux-lock.txt"
            path.write_text("# STATUS=REMOTE_VERIFIED\n", encoding="utf-8")
            result = validate_linux_lock(path)
        self.assertFalse(result["verified"])
        self.assertIn("missing_linux_platform", result["errors"])

    def test_linux_lock_accepts_standard_multiline_hash_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements-linux-lock.txt"
            required = Path(directory) / "proposed.txt"
            path.write_text(
                "# STATUS=LOCAL_WSL_VERIFIED\n# PLATFORM=LINUX\n"
                "numpy==2.5.1 \\\n    --hash=sha256:" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            required.write_text("numpy==2.5.1\n", encoding="utf-8")
            with patch(
                "paper2.restart.release_reproducibility_gate.PROPOSED_LINUX_LOCK", required
            ):
                result = validate_linux_lock(path)
        self.assertTrue(result["verified"], result["errors"])
        self.assertEqual(result["verification_scope"], "LOCAL_WSL_VERIFIED")

    def test_clean_receipt_is_bound_to_lock_and_required_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            receipt = {
                "schema_version": 1,
                "status": "CLEAN_LINUX_REPRODUCTION_VERIFIED",
                "verified": True,
                "platform": {"system": "Linux"},
                "linux_lock_sha256": "abc",
                "bundle_manifest_sha256": "bundle-abc",
                "checks": {
                    "environment_install": "PASS", "unit_tests": "PASS",
                    "bundle_integrity": "PASS", "recovery": "PASS",
                    "result_retrieval": "PASS",
                },
                "independent_audit": {"status": "APPROVED"},
            }
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertTrue(validate_clean_linux_receipt(path, "abc")["verified"])
            self.assertFalse(validate_clean_linux_receipt(path, "different")["verified"])

    def test_clean_candidate_can_be_evidenced_without_release_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "transcripts").mkdir()
            transcript = root / "transcripts" / "unit_tests.log"
            transcript.write_text("ok\n", encoding="utf-8")
            receipt = {
                "schema_version": 1,
                "status": "CLEAN_LINUX_REPRODUCTION_CANDIDATE_PENDING_INDEPENDENT_AUDIT",
                "verified": False,
                "platform": {"system": "Linux"},
                "linux_lock_sha256": "abc",
                "checks": {name: "PASS" for name in ("environment_install", "unit_tests", "bundle_integrity", "recovery")},
                "command_receipts": {
                    "unit_tests": {"status": "PASS", "transcript_file": "unit_tests.log",
                                   "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()}
                },
            }
            path = root / "receipt_candidate.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            result = validate_clean_linux_candidate(path, "abc")
        self.assertTrue(result["evidence_verified"])
        self.assertFalse(result["release_approval"])

    def test_clean_candidate_rejects_a_different_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "transcripts").mkdir()
            transcript = root / "transcripts" / "check.log"; transcript.write_text("ok\n", encoding="utf-8")
            receipt = {
                "schema_version": 1, "status": "CLEAN_LINUX_REPRODUCTION_CANDIDATE_PENDING_INDEPENDENT_AUDIT",
                "verified": False, "platform": {"system": "Linux"}, "linux_lock_sha256": "abc",
                "bundle_manifest_sha256": "older-bundle",
                "checks": {name: "PASS" for name in ("environment_install", "unit_tests", "bundle_integrity", "recovery")},
                "command_receipts": {"check": {"status": "PASS", "transcript_file": "check.log",
                                      "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()}},
            }
            path = root / "receipt_candidate.json"; path.write_text(json.dumps(receipt), encoding="utf-8")
            result = validate_clean_linux_candidate(path, "abc", "newer-bundle")
        self.assertFalse(result["evidence_verified"])
        self.assertIn("bundle_manifest_sha256", result["errors"])


if __name__ == "__main__":
    unittest.main()
