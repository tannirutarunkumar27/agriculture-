from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "paper2" / "manuscript" / "references.bib"
INTRODUCTION_KEYS = {
    "mcmahan2017communication",
    "li2020fedprox",
    "wang2020fednova",
    "delcon2022yield",
    "li2024pruning",
    "mukherjee2025architectures",
}
ADDITIONAL_KEYS = {
    "dey2025flyer",
    "cowlishaw2025balancing",
    "nirosha2025enhancing",
    "utku2026forecasting",
    "koksal2026vertical",
    "bun2016concentrated",
    "bonawitz2017secure",
    "gaudio2024cereallegume",
}


class IntroductionBibliographyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = BIB.read_text(encoding="utf-8")

    def test_all_introduction_keys_are_unique_and_present(self) -> None:
        keys = re.findall(r"^@\w+\{([^,]+),", self.text, flags=re.MULTILINE)
        self.assertEqual(set(keys), INTRODUCTION_KEYS | ADDITIONAL_KEYS)
        self.assertEqual(len(keys), len(set(keys)))

    def test_entries_do_not_use_abbreviated_authors(self) -> None:
        self.assertNotRegex(self.text.lower(), r"\bet\s+al\.?")

    def test_agricultural_entries_retain_persistent_identifiers(self) -> None:
        for doi in (
            "10.1109/DELCON54057.2022.9752836",
            "10.1016/j.eswa.2023.122847",
            "10.1002/spe.3416",
        ):
            self.assertIn(doi, self.text)


if __name__ == "__main__":
    unittest.main()
