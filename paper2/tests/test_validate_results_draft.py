from __future__ import annotations

import unittest
from pathlib import Path

from paper2.restart.validate_results_draft import validate


DRAFT = Path(__file__).resolve().parents[1] / "manuscript" / "PAPER2_RESULTS_DRAFT.md"


class ResultsDraftValidationTests(unittest.TestCase):
    def test_checked_draft_passes(self) -> None:
        self.assertEqual(validate(DRAFT.read_text(encoding="utf-8")), [])

    def test_unsupported_superiority_claim_is_rejected(self) -> None:
        text = DRAFT.read_text(encoding="utf-8") + "\nThe model was statistically superior.\n"
        self.assertTrue(any("statistically superior" in error for error in validate(text)))


if __name__ == "__main__":
    unittest.main()
