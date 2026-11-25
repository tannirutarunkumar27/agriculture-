import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_authenticated_dp_secagg_adapter import prepare, validate


class AuthenticatedDpSecaggAdapterTests(unittest.TestCase):
    def fixture(self, root: Path):
        app = root / "app"; package = app / "paper2_dp_secagg_deploy"; package.mkdir(parents=True)
        (app / "LICENSE").write_text("all rights", encoding="utf-8")
        (app / "README.md").write_text("readme", encoding="utf-8")
        (app / "pyproject.toml").write_text('''[project]\ndependencies = ["flwr>=1.42.0,<1.43.0"]\n[tool.flwr.app]\nflwr-version-target = "1.42.0"\n[tool.flwr.app.components]\nserverapp = "paper2_dp_secagg_deploy.server_app:app"\nclientapp = "paper2_dp_secagg_deploy.client_app:app"\n[tool.flwr.app.config]\nnum-server-rounds = 20\nlearning-rate = 0.01\nfeatures = 54\nnoise-multiplier = 8.0\nclipping-norm = 1.0\ncentral-dp = true\ndifferential-privacy-enabled = true\nouter-test-evaluation = false\n''', encoding="utf-8")
        for name in ("__init__.py", "protocol.py", "task.py", "client_app.py", "server_app.py"):
            (package / name).write_text("# self contained\n", encoding="utf-8")
        study = root / "study"; study.mkdir()
        run = {"num-server-rounds": 20, "learning-rate": 0.01, "features": 54, "noise-multiplier": 8.0,
               "clipping-norm": 1.0, "central-dp": True, "differential-privacy-enabled": True,
               "outer-test-evaluation": False}
        (study / "run_config.json").write_text(json.dumps(run), encoding="utf-8")
        (study / "manifest.json").write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
        return app, study

    def test_adapter_binds_parameters_without_mutating_frozen_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); app, study = self.fixture(root)
            report = validate(app, study)
        self.assertFalse(report["frozen_study_modified"])
        self.assertFalse(report["data_paths_embedded_in_fab"])
        self.assertEqual(report["node_local_required_config"], ["client-id", "data-root", "fold-id"])

    def test_parameter_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); app, study = self.fixture(root)
            path = app / "pyproject.toml"
            path.write_text(path.read_text(encoding="utf-8").replace("num-server-rounds = 20", "num-server-rounds = 21"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num-server-rounds"):
                validate(app, study)

    def test_writes_candidate_only_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); app, study = self.fixture(root); output = root / "output"
            with patch("paper2.restart.prepare_authenticated_dp_secagg_adapter.RESULTS", root):
                report = prepare(output, app, study)
        self.assertFalse(report["claim_allowed"])


if __name__ == "__main__":
    unittest.main()
