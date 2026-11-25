"""Read-only acceptance tests for the versioned regression manuscript revision."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from paper2.restart.revise_regression_manuscript import (
    EXPECTED_SOURCE_SHA256,
    FIRST_CITATION_ORDER,
    SENSITIVITY,
    SOURCE,
    paired_section,
    replace_once,
    revise,
    sha256,
)


class RegressionManuscriptRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))

    def test_archived_source_is_unchanged(self) -> None:
        self.assertEqual(sha256(SOURCE), EXPECTED_SOURCE_SHA256)

    def test_revised_scope_citations_and_equation(self) -> None:
        text, stats = revise(self.source, self.sensitivity)
        body, references = text.split("# References\n", 1)
        self.assertIn("retrospective conditional harvest", body)
        self.assertIn("simulated horizontal federated", body)
        self.assertIn("Gaudio et al. cereal--legume field-experiment dataset [14]", body)
        self.assertIn("f_{\\theta}(\\mathbf{x})=\\mathbf{w}_2", body)
        self.assertNotIn("=mathbf{w}", body)
        self.assertNotIn("Paper 2", body)
        self.assertNotIn("KG-FCR", body)
        self.assertNotIn("privacy-preserving approach", body)
        self.assertNotIn("Author initials, pagination", text)
        self.assertIn("confirmatory superiority test.\n\n## F. Summary", body)
        self.assertFalse(re.search(r"\]\(https?://", body))
        citation_order = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", body)))
        self.assertEqual(citation_order, list(range(1, len(FIRST_CITATION_ORDER) + 1)))
        ref_numbers = [int(n) for n in re.findall(r"(?m)^(\d+)\. ", references)]
        self.assertEqual(ref_numbers, citation_order)
        self.assertAlmostEqual(stats["mean_group_difference_t_ha"], 0.07195272228058455)
        self.assertEqual(stats["rf_lower_group_count"], 3)

    def test_missing_anchor_and_tampered_sensitivity_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected one replacement anchor"):
            replace_once("a a", "a", "b")
        altered = json.loads(json.dumps(self.sensitivity))
        altered["group_sizes"]["Reading"] = 5
        with self.assertRaisesRegex(ValueError, "Paired group count"):
            paired_section(altered)

    def test_source_drift_refused(self) -> None:
        with self.assertRaises(ValueError):
            revise(self.source.replace("## Abstract", "## Summary", 1), self.sensitivity)

    def test_saved_v3_matches_manifest_and_deterministic_revision(self) -> None:
        root = Path(__file__).resolve().parents[2]
        output = root / "paper2/manuscript/PAPER2_REGRESSION_MANUSCRIPT_V3.md"
        manifest = root / "paper2/results/regression_manuscript_v3/manifest.json"
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        expected_text, expected_stats = revise(self.source, self.sensitivity)
        self.assertEqual(output.read_text(encoding="utf-8"), expected_text)
        self.assertEqual(sha256(output), saved["output_sha256"])
        self.assertEqual(sha256(SOURCE), saved["source_sha256"])
        self.assertEqual(sha256(SENSITIVITY), saved["sensitivity_sha256"])
        self.assertEqual(saved["paired_geographic_statistics"], expected_stats)
        self.assertFalse(saved["training_or_inference_performed"])
        self.assertFalse(saved["portable_reproduction_claim"])


if __name__ == "__main__":
    unittest.main()
