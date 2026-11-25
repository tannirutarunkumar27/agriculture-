"""Regression tests for local DP+SecAgg execution-receipt handling only."""
import json
import tempfile
import unittest
from pathlib import Path

from paper2.restart.run_dp_secagg_local_simulation import MODE, recover_interrupted


class LocalSimulationReceiptTests(unittest.TestCase):
    def test_interrupted_execution_is_closed_once_with_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "execution"
            output.mkdir()
            (output / "manifest.json").write_text(json.dumps({"status": "EXECUTION_STARTED"}), encoding="utf-8")
            result = recover_interrupted(output, "operator stopped a stalled runtime")
            self.assertEqual(result["status"], "ABORTED_BY_OPERATOR_NO_AUTOMATIC_RETRY")
            self.assertEqual(result["execution_mode"], MODE)
            self.assertTrue((output / "completion.json").is_file())
            with self.assertRaises(ValueError):
                recover_interrupted(output, "must not overwrite a receipt")


if __name__ == "__main__":
    unittest.main()
