import tempfile
import unittest
from pathlib import Path

from paper2.restart.recover_local_authenticated_dp_secagg_deployment import recover


class RecoverLocalDeploymentTests(unittest.TestCase):
    def test_writes_failure_only_receipt_with_diagnostic_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); diagnostics = output / "diagnostics"; diagnostics.mkdir()
            (diagnostics / "run.log").write_text("stalled", encoding="utf-8")
            receipt = recover(output, "operator_stopped_stall")
        self.assertTrue(receipt["status"].startswith("ABORTED_"))
        self.assertFalse(receipt["privacy_claim_allowed"])
        self.assertIn("run.log", receipt["diagnostic_sha256"])
