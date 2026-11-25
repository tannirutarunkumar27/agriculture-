import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from paper2.restart.prepare_ranking_review_packet import prepare


class RankingReviewPacketTests(unittest.TestCase):
    def test_packet_joins_missing_reference_to_block_without_enabling_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "review"; source.mkdir(); output = root / "output"
            reference = root / "references.csv"
            pd.DataFrame([{"experiment_id": "e1", "doi_or_url": "https://doi.org/x", "gate_status": "UNCONFIRMED_CANDIDATE", "computed_usable": False}]).to_csv(reference, index=False)
            pd.DataFrame([{"block_id": "b1", "experiment_id": "e1", "reference_present": False, "blocking_reasons": "publication_reference_missing"}]).to_csv(source / "block_review.csv", index=False)
            pd.DataFrame([{"block_id": "b1", "experiment_id": "e1"}]).to_csv(source / "approval_template.csv", index=False)
            (source / "ranking_protocol.json").write_text(json.dumps({"status": "FROZEN_REVIEW_CONTRACT_NOT_APPROVED_FOR_RANKING"}), encoding="utf-8")
            with patch("paper2.restart.prepare_ranking_review_packet.RESULTS", root):
                report = prepare(output, reference, source)
            row = pd.read_csv(output / "block_reference_worklist.csv").iloc[0]
        self.assertFalse(report["ranking_enabled"])
        self.assertIn("source-reference", row.required_next_action)

    def test_false_string_is_not_coerced_to_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "review"; source.mkdir(); output = root / "output"
            reference = root / "references.csv"
            reference.write_text(
                "experiment_id,doi_or_url,gate_status,computed_usable\n"
                "e1,https://doi.org/x,UNCONFIRMED_CANDIDATE,False\n", encoding="utf-8"
            )
            (source / "block_review.csv").write_text(
                "block_id,experiment_id,reference_present,blocking_reasons\n"
                "b1,e1,False,publication_reference_missing\n", encoding="utf-8"
            )
            (source / "approval_template.csv").write_text(
                "block_id,experiment_id\nb1,e1\n", encoding="utf-8"
            )
            (source / "ranking_protocol.json").write_text("{}", encoding="utf-8")
            with patch("paper2.restart.prepare_ranking_review_packet.RESULTS", root):
                report = prepare(output, reference, source)
        self.assertEqual(report["unresolved_reference_candidates"], 1)
        self.assertEqual(report["blocks_with_missing_frozen_reference"], 1)


if __name__ == "__main__":
    unittest.main()
