"""RR07 execution gate tests; no outer arrays or inference."""
import tempfile
import unittest
from pathlib import Path

from paper2.restart import rr07_execution_gate as gate
from paper2.restart.validation_search import ROOT, atomic_json


class Rr07ExecutionGateTests(unittest.TestCase):
    def test_partial_or_unfrozen_plan_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            root = Path(temp); package = root / "plan"; package.mkdir()
            atomic_json(package / "manifest.json", {"status": "PARTIAL", "runtime": gate.runtime()})
            with self.assertRaisesRegex(ValueError, "not frozen"):
                gate.verify_plan(package)

    def test_scope_guard_rejects_outer_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "paper2/results") as temp:
            root = Path(temp); package = root / "plan"; package.mkdir()
            atomic_json(package / "manifest.json", {"status": "FROZEN_RR07_PLAN_ONLY_NO_OUTER_INFERENCE", "runtime": gate.runtime(),
                "artifacts": {}, "data": str(root), "data_manifest_sha256": "x"})
            atomic_json(root / "manifest.json", {})
            atomic_json(package / "plan.json", [{"id": "r", "outer_test_inference": True}])
            with self.assertRaisesRegex(ValueError, "data binding"):
                gate.verify_plan(package)


if __name__ == "__main__":
    unittest.main()
