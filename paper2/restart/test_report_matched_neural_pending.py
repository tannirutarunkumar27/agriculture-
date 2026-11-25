import tempfile
import unittest
from pathlib import Path

import paper2.restart.report_matched_neural_pending as pending_report
from paper2.restart.validation_search import atomic_json


class PendingReportTests(unittest.TestCase):
    def test_reports_pending_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, study, audit, output = (root / name for name in ("package", "study", "audit", "output"))
            package.mkdir(); study.mkdir(); audit.mkdir()
            protocol = root / "protocol"; protocol.mkdir()
            atomic_json(protocol / "plan.json", [{"id": "a", "fold": 0, "algorithm": "sgd", "aggregation_objective": "sample_weighted"}, {"id": "b", "fold": 1, "algorithm": "adamw", "aggregation_objective": "equal_client"}])
            atomic_json(package / "manifest.json", {"status": "test", "protocol": str(protocol)})
            atomic_json(audit / "audit.json", {"work": [{"id": "a"}]})
            pending_report.RESULTS = root
            result = pending_report.report(package, study, audit, output)
            self.assertEqual(result["pending"], 1)
            self.assertFalse(result["selection_allowed"])


if __name__ == "__main__":
    unittest.main()
