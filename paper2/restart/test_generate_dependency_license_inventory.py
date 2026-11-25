import tempfile
import unittest
from pathlib import Path

from paper2.restart.generate_dependency_license_inventory import inventory
from paper2.restart.generate_dependency_license_inventory import REQUIREMENTS, requirement_names, locked_versions


class DependencyLicenseInventoryTests(unittest.TestCase):
    def test_observed_requirements_include_remote_privacy_runtime(self):
        self.assertIn("cryptography", {name.lower() for name in requirement_names(REQUIREMENTS)})

    def test_inventory_is_pending_and_never_release_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text("numpy==0\nmissing-package==1\n", encoding="utf-8")
            report, rows = inventory(requirements)
        self.assertEqual(report["status"], "DEPENDENCY_LICENSE_INVENTORY_PENDING_REMOTE_LINUX_REVIEW")
        self.assertFalse(report["license_inventory_complete"])
        self.assertFalse(report["release_allowed"])
        self.assertEqual(report["direct_requirement_count"], 2)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(report["inventory_package_count"], len(rows))
        numpy = next(row for row in rows if row["package"].lower() == "numpy")
        self.assertTrue(numpy["direct_requirement"])
        missing = next(row for row in rows if row["package"] == "missing-package")
        self.assertEqual(missing["status"], "NOT_INSTALLED_IN_ENVIRONMENT")
        self.assertNotIn("\\\\", report["requirements_file"])

    def test_lock_parser_retains_exact_version(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.txt"
            lock.write_text("numpy==2.5.1 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
            self.assertEqual(locked_versions(lock), {"numpy": "2.5.1"})


if __name__ == "__main__":
    unittest.main()
