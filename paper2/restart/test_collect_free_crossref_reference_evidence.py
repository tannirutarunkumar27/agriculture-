import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.collect_free_crossref_reference_evidence import collect, doi_from_candidate


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload).encode("utf-8")


def opener(request, timeout):
    assert request.full_url == "https://api.crossref.org/v1/works/10.1000/example"
    return Response({"message": {"DOI": "10.1000/example", "title": ["Example study"],
                                 "author": [{"given": "Ada", "family": "Lovelace"}],
                                 "published-print": {"date-parts": [[2020]]},
                                 "container-title": ["Journal"], "type": "journal-article"}})


class CrossrefCandidateEvidenceTests(unittest.TestCase):
    def test_only_recognises_doi_urls_or_bare_dois(self):
        self.assertEqual(doi_from_candidate("https://doi.org/10.1000/ABC"), "10.1000/abc")
        self.assertEqual(doi_from_candidate("doi:10.1000/Test"), "10.1000/test")
        self.assertIsNone(doi_from_candidate("https://example.org/report"))

    def test_writes_candidate_only_evidence_without_approving_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); registry = root / "registry.json"; output = root / "output"
            registry.write_text(json.dumps({"candidates": [
                {"experiment_id": "x", "candidate_source": "Study", "doi_or_url": "https://doi.org/10.1000/example",
                 "exact_site_year_link": "confirmed", "coverage": "full", "independent_review_status": "pending"},
                {"experiment_id": "y", "candidate_source": "Report", "doi_or_url": "https://example.org/report",
                 "exact_site_year_link": "unconfirmed", "coverage": "partial", "independent_review_status": "pending"},
            ]}), encoding="utf-8")
            with patch("paper2.restart.collect_free_crossref_reference_evidence.RESULTS", root):
                result = collect(output, registry, sleep_seconds=0, opener=opener)
        self.assertFalse(result["ranking_enabled"])
        self.assertEqual(result["crossref_retrieved"], 1)
        self.assertEqual(result["rows"][0]["crossref"]["title"], "Example study")
        self.assertEqual(result["rows"][1]["crossref"]["status"], "NO_CROSSREF_DOI")

    def test_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "output"; output.mkdir()
            with patch("paper2.restart.collect_free_crossref_reference_evidence.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "new Paper 2"):
                    collect(output)

    def test_rejects_malformed_candidate_instead_of_silently_skipping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); registry = root / "registry.json"; output = root / "output"
            registry.write_text(json.dumps({"candidates": ["not-an-object"]}), encoding="utf-8")
            with patch("paper2.restart.collect_free_crossref_reference_evidence.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "Candidate 0"):
                    collect(output, registry, sleep_seconds=0, opener=opener)


if __name__ == "__main__":
    unittest.main()
