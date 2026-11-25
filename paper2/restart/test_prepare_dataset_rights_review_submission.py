import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_dataset_rights_review_submission import prepare


class DatasetRightsReviewSubmissionTests(unittest.TestCase):
    def fixture(self, root: Path):
        licensing = root / "licensing.json"; audit = root / "audit.json"; public = root / "public.json"
        licensing.write_text(json.dumps({"dataset": {"name": "dataset", "version": "2.0.0", "doi": "10.5281/zenodo.14160885",
                                        "record_url": "https://zenodo.org/records/14160885", "license_identifier": None,
                                        "redistribution_approved": False}}), encoding="utf-8")
        audit.write_text(json.dumps({"status": "PASS_DATASET_CLIENT_AUDIT", "systems": 607, "experiments": 37,
                                    "source_manifest_sha256": "a", "regression_rows_sha256": "b"}), encoding="utf-8")
        public.write_text(json.dumps({"status": "PREPARED_NO_TRAINING"}), encoding="utf-8")
        return licensing, audit, public

    def test_packet_is_provenance_bound_and_never_releases_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = self.fixture(root); output = root / "output"
            with patch("paper2.restart.prepare_dataset_rights_review_submission.RESULTS", root):
                packet = prepare(output, *paths)
        self.assertFalse(packet["release_allowed"])
        self.assertIn("raw source files", packet["requested_scopes"])

    def test_already_approved_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); licensing, audit, public = self.fixture(root); output = root / "output"
            value = json.loads(licensing.read_text(encoding="utf-8")); value["dataset"]["redistribution_approved"] = True
            licensing.write_text(json.dumps(value), encoding="utf-8")
            with patch("paper2.restart.prepare_dataset_rights_review_submission.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "unresolved"):
                    prepare(output, licensing, audit, public)


if __name__ == "__main__":
    unittest.main()
