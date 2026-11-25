"""Guard tests for the complete-study selection finalization gate."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart import finalize_matched_neural_selection as finalizer
from paper2.restart.validation_search import ROOT, atomic_json


class FinalizeSelectionTests(unittest.TestCase):
    def test_partial_audit_is_rejected_before_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            root = Path(temp); package = root / "package"; study = root / "study"; audit = root / "audit"
            package.mkdir(); study.mkdir(); audit.mkdir()
            atomic_json(audit / "audit.json", {"status": "PASS_PARTIAL_MATCHED_NEURAL_STUDY", "verified": 1,
                                                "outer_test_inference": False, "work": []})
            with patch.object(finalizer.runner, "verify_execution", return_value=({}, {}, [{"id": "r"}], root / "data")):
                with self.assertRaisesRegex(ValueError, "Audit"):
                    finalizer.finalize(package, study, audit, root / "out")


if __name__ == "__main__":
    unittest.main()
