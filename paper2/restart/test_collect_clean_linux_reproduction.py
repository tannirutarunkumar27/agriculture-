"""Contract tests for fail-closed Linux reproduction candidate receipts."""
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from paper2.restart.collect_clean_linux_reproduction import collect


class CleanLinuxCandidateTests(unittest.TestCase):
    def test_non_linux_or_unverified_lock_never_claims_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, lock, output = root / "bundle_manifest.json", root / "lock.txt", root / "output"
            manifest.write_text("{}", encoding="utf-8")
            lock.write_text("# STATUS=PROPOSED_UNVERIFIED\n", encoding="utf-8")
            with patch("paper2.restart.collect_clean_linux_reproduction.RESULTS", root), \
                 patch("paper2.restart.collect_clean_linux_reproduction.platform.system", return_value="Windows"):
                receipt = collect(manifest, lock, output)
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["independent_audit"]["status"], "PENDING")
        self.assertIn("BLOCKED", receipt["status"])
        self.assertEqual(receipt["checks"]["result_retrieval"], "NOT_RUN_REQUIRES_SEPARATE_SSH_RETRIEVAL_RECEIPT")

    def test_run_persists_auditable_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "transcript.log"
            from paper2.restart.collect_clean_linux_reproduction import _run
            result = _run([sys.executable, "-c", "print('ok')"], root, destination)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["transcript_file"], "transcript.log")
            self.assertIn("ok", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
