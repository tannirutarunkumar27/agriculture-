"""RR07 runner guard tests; no outer data access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart import rr07_runner as runner
from paper2.restart.validation_search import ROOT


class Rr07RunnerTests(unittest.TestCase):
    def test_execution_requires_explicit_authorization(self):
        with self.assertRaisesRegex(PermissionError, "explicit authorization"):
            runner.execute(Path("package"), Path("output"))

    def test_dry_run_delegates_to_plan_gate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            with patch.object(runner, "verify_execution", return_value=({}, [{"id": "r"}], Path(temp))):
                result = runner.dry_run(Path(temp) / "plan")
            self.assertEqual(result["status"], "DRY_RUN_NO_OUTER_INFERENCE")
            self.assertFalse(result["outer_test_inference"])

    def test_explicit_cuda_requires_a_cuda_device(self):
        with patch.dict(runner.os.environ, {"PAPER2_DEVICE": "cuda"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runner.execute(Path("package"), Path("output"), allow_outer_inference=True)


if __name__ == "__main__":
    unittest.main()
