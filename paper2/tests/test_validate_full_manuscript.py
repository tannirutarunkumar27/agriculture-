from __future__ import annotations

import json
import unittest
from pathlib import Path

from paper2.restart.validate_full_manuscript import validate


ROOT = Path(__file__).resolve().parents[2]
PAPER2 = ROOT / "paper2"
MANUSCRIPT = PAPER2 / "manuscript"


class FullManuscriptValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (MANUSCRIPT / "PAPER2_FULL_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
        cls.registry = json.loads((PAPER2 / "research" / "restart_regression_closest_work.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((PAPER2 / "results" / "restart_full_manuscript_v1" / "manifest.json").read_text(encoding="utf-8"))

    def test_assembled_draft_passes(self) -> None:
        self.assertEqual(validate(self.text, self.registry, self.manifest, ROOT, MANUSCRIPT), [])

    def test_forbidden_title_term_is_rejected(self) -> None:
        changed = self.text.replace(
            "# Geographically Grouped Federated and Centralized Regression",
            "# Privacy-Preserving Geographically Grouped Federated and Centralized Regression",
            1,
        )
        errors = validate(changed, self.registry, self.manifest, ROOT, MANUSCRIPT)
        self.assertTrue(any("forbidden title term" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
