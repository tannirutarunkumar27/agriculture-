from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper2.restart.reproducibility_readiness import ASSETS, Asset, ROOT, audit, find_windows_absolute_paths


class ReproducibilityReadinessTests(unittest.TestCase):
    def test_real_local_evidence_passes(self) -> None:
        result = audit(ROOT)
        self.assertEqual(result["local_evidence_status"], "PASS")
        self.assertEqual(result["portable_clean_room_status"], "BLOCKED")
        self.assertEqual(result["verified_asset_count"], len(ASSETS))
        self.assertFalse(result["model_training_performed"])
        self.assertFalse(result["model_inference_performed"])

    def test_missing_required_asset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = audit(Path(temp_dir), (Asset("paper2/missing.json", "fixture"),))
        self.assertEqual(result["local_evidence_status"], "FAIL")
        self.assertTrue(any("missing required asset" in error for error in result["errors"]))

    def test_windows_absolute_path_detection(self) -> None:
        paths = find_windows_absolute_paths(r'{"source": "D:\\research\\input.csv"}')
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].startswith("D:"))


if __name__ == "__main__":
    unittest.main()
