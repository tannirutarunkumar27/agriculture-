from __future__ import annotations

import unittest
from pathlib import Path

from paper2.restart.validate_data_methods_draft import validate


DRAFT = (
    Path(__file__).resolve().parents[1]
    / "manuscript"
    / "PAPER2_DATA_METHODS_DRAFT.md"
)


class DataMethodsDraftValidationTests(unittest.TestCase):
    def test_checked_draft_passes(self) -> None:
        self.assertEqual(validate(DRAFT.read_text(encoding="utf-8")), [])

    def test_line_wrapped_required_fact_is_recognized(self) -> None:
        text = DRAFT.read_text(encoding="utf-8").replace(
            "11 of 56 selected checkpoints", "11 of 56\nselected checkpoints"
        )
        self.assertEqual(validate(text), [])


if __name__ == "__main__":
    unittest.main()
