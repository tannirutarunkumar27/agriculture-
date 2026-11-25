from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from paper2.restart.validate_regression_literature import validate_registry


REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "restart_regression_closest_work.json"
)


class RegressionLiteratureRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def test_checked_registry_passes(self) -> None:
        self.assertEqual(validate_registry(self.payload), [])

    def test_numeric_superiority_permission_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["closest_studies"][0]["numerical_comparison_allowed"] = True
        errors = validate_registry(changed)
        self.assertTrue(any("prohibit cross-paper numerical" in error for error in errors))

    def test_missing_privacy_boundary_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["claim_boundaries"] = [
            item
            for item in changed["claim_boundaries"]
            if "formal privacy guarantee" not in item
        ]
        errors = validate_registry(changed)
        self.assertTrue(any("formal privacy guarantee" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
