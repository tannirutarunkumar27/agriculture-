import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_project_code_license_decision import prepare


class ProjectCodeLicenseDecisionTests(unittest.TestCase):
    def test_missing_license_and_readme_claim_are_recorded_without_selecting_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scope = root / "paper2"; scope.mkdir(); output = root / "output"
            (scope / "main.py").write_text("print('x')\n", encoding="utf-8")
            readme = root / "README.md"; readme.write_text("This project is licensed under the MIT License.\n", encoding="utf-8")
            with patch("paper2.restart.prepare_project_code_license_decision.ROOT", root), patch("paper2.restart.prepare_project_code_license_decision.RESULTS", root):
                packet = prepare(output, scope, readme)
        self.assertFalse(packet["top_level_license_present"])
        self.assertIn("root_readme_claims_mit", packet["license_signal_conflicts"][0])
        self.assertFalse(packet["release_allowed"])

    def test_no_source_files_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scope = root / "paper2"; scope.mkdir(); output = root / "output"; readme = root / "README.md"
            readme.write_text("readme", encoding="utf-8")
            with patch("paper2.restart.prepare_project_code_license_decision.ROOT", root), patch("paper2.restart.prepare_project_code_license_decision.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "No distributable"):
                    prepare(output, scope, readme)

    def test_generated_results_tree_is_excluded_before_file_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scope = root / "paper2"; generated = scope / "results"; generated.mkdir(parents=True)
            (scope / "main.py").write_text("print('x')\n", encoding="utf-8")
            (generated / "ignored.py").write_text("bad", encoding="utf-8")
            readme = root / "README.md"; readme.write_text("readme", encoding="utf-8")
            with patch("paper2.restart.prepare_project_code_license_decision.ROOT", root), patch("paper2.restart.prepare_project_code_license_decision.RESULTS", root):
                packet = prepare(root / "output", scope, readme)
        self.assertEqual(packet["source_file_count"], 1)


if __name__ == "__main__":
    unittest.main()
