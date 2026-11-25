from __future__ import annotations

import unittest
from pathlib import Path

from paper2.restart.validate_discussion_conclusion_draft import validate


DRAFT = Path(__file__).resolve().parents[1] / "manuscript" / "PAPER2_DISCUSSION_CONCLUSION_DRAFT.md"


class DiscussionConclusionDraftValidationTests(unittest.TestCase):
    def test_checked_draft_passes(self) -> None:
        self.assertEqual(validate(DRAFT.read_text(encoding="utf-8")), [])

    def test_privacy_overclaim_is_rejected(self) -> None:
        text = DRAFT.read_text(encoding="utf-8") + "\nThe system guarantees end-to-end privacy.\n"
        self.assertTrue(any("guarantees end-to-end privacy" in error for error in validate(text)))


if __name__ == "__main__":
    unittest.main()
