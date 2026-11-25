import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_dependency_license_review_submission import prepare


class DependencyLicenseReviewSubmissionTests(unittest.TestCase):
    def test_clean_inventory_becomes_blank_nonreleasing_review_kit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "inventory"; source.mkdir(); output = root / "output"
            report = {"environment_scope": "clean_wsl_linux", "missing_from_environment": 0,
                      "locked_version_mismatches": [], "environment_matches_lock": True, "inventory_package_count": 1}
            (source / "license_inventory.json").write_text(json.dumps(report), encoding="utf-8")
            with (source / "dependency_license_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["package", "locked_version", "installed_version_observed", "license_expression_observed", "license_field_observed", "license_classifiers_observed", "license_file_sha256"])
                writer.writeheader(); writer.writerow({"package": "demo", "locked_version": "1", "installed_version_observed": "1", "license_expression_observed": "MIT", "license_field_observed": "", "license_classifiers_observed": "", "license_file_sha256": "x"})
            with patch("paper2.restart.prepare_dependency_license_review_submission.RESULTS", root):
                packet = prepare(output, source / "license_inventory.json")
            with (output / "dependency_license_decisions.csv").open(encoding="utf-8") as stream:
                decisions = list(csv.DictReader(stream))
        self.assertFalse(packet["release_allowed"])
        self.assertEqual(decisions[0]["review_decision"], "")

    def test_incomplete_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "inventory.json"; output = root / "output"
            path.write_text(json.dumps({"environment_scope": "observed_runtime"}), encoding="utf-8")
            with patch("paper2.restart.prepare_dependency_license_review_submission.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "clean-Linux"):
                    prepare(output, path)


if __name__ == "__main__":
    unittest.main()
