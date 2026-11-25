import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "audit_data.py"
spec = importlib.util.spec_from_file_location("audit_data", MODULE)
audit_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_data)


class TestDataAudit(unittest.TestCase):
    def check_csv(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(text, encoding="utf-8")
            return audit_data.inspect_csv(path)

    def test_counts_missing_and_duplicates(self):
        report = self.check_csv("N,label\n1,rice\n1,rice\n,maize\n")
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["missing_by_column"]["N"], 1)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertEqual(report["class_counts"], {"maize": 1, "rice": 2})

    def test_nasa_sentinel(self):
        report = self.check_csv("-BEGIN HEADER-\nLocation: test\n-END HEADER-\nYEAR,DOY,T2M,RH2M\n2026,285,-999.0,-999\n")
        self.assertEqual(report["usable_weather_values"], 0)
        self.assertEqual(report["missing_by_column"]["T2M"], 1)
        self.assertEqual(len(report["warnings"]), 2)

    def test_malformed_row_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("N,label\n1,rice,extra\n")

    def test_missing_nasa_end_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("-BEGIN HEADER-\nYEAR,DOY,T2M\n2026,285,25\n")

    def test_duplicate_headers_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("N,N\n1,2\n")

    def test_byte_identical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for name in ["a.csv", "b.csv"]:
                (raw / name).write_text("N,label\n1,rice\n", encoding="utf-8")
            report = audit_data.audit(raw)
            self.assertEqual(len(report["duplicate_files"]), 1)


if __name__ == "__main__":
    unittest.main()
