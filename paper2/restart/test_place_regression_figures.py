"""Fail-closed placement tests for existing audited scientific figures."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from paper2.restart.place_regression_figures import (
    ROOT,
    SOURCE,
    SOURCE_SHA256,
    place,
    sha256,
    verified_figures,
)


class PlaceRegressionFiguresTests(unittest.TestCase):
    def test_source_and_all_selected_assets_match_archives(self) -> None:
        self.assertEqual(sha256(SOURCE), SOURCE_SHA256)
        figures = verified_figures()
        self.assertEqual([figure["number"] for figure in figures], [1, 2, 3, 4, 5])

    def test_placement_uses_existing_pngs_and_stage_safe_captions(self) -> None:
        figures = verified_figures()
        text = place(SOURCE.read_text(encoding="utf-8"), figures)
        self.assertEqual(text.count("**Fig. "), 5)
        self.assertIn("Fig. 3 are explicitly post-hoc", text)
        self.assertIn("Fig. 5 shows the overall matched result", text)
        self.assertNotIn("`posthoc_guard_sensitivity`", text)
        for figure in figures:
            relative = f"../results/{figure['package']}/{figure['stem']}.png"
            self.assertIn(relative, text)
            self.assertTrue((SOURCE.parent / relative).is_file())
        with self.assertRaises(ValueError):
            place("altered source", figures)

    def test_saved_v4_matches_manifest_if_present(self) -> None:
        output = ROOT / "paper2/manuscript/PAPER2_REGRESSION_MANUSCRIPT_V4.md"
        manifest_path = ROOT / "paper2/results/regression_manuscript_v4/manifest.json"
        if not output.exists() and not manifest_path.exists():
            self.skipTest("V4 has not yet been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(sha256(output), manifest["output_sha256"])
        self.assertEqual(sha256(SOURCE), manifest["source_sha256"])
        self.assertFalse(manifest["plot_regeneration"])
        self.assertFalse(manifest["model_training_or_inference"])
        self.assertEqual(output.read_text(encoding="utf-8"),
                         place(SOURCE.read_text(encoding="utf-8"), verified_figures()))


if __name__ == "__main__":
    unittest.main()
