import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "paper2/paper2-dp-secagg-deploy"


class AuthenticatedDpSecaggDeployAppTests(unittest.TestCase):
    def test_manifest_is_self_contained_and_frozen(self):
        config = tomllib.loads((APP / "pyproject.toml").read_text(encoding="utf-8"))
        app = config["tool"]["flwr"]["app"]
        self.assertEqual(app["flwr-version-target"], "1.42.0")
        self.assertEqual(app["config"]["num-server-rounds"], 20)
        self.assertTrue(app["config"]["central-dp"])
        self.assertEqual(app["components"]["serverapp"], "paper2_dp_secagg_deploy.server_app:app")
        self.assertEqual(app["components"]["clientapp"], "paper2_dp_secagg_deploy.client_app:app")
        self.assertIn("flwr>=1.42.0,<1.43.0", config["project"]["dependencies"])
        self.assertNotIn("paper2.secagg_pilot", "\n".join(path.read_text(encoding="utf-8") for path in APP.rglob("*.py")))

    def test_client_requires_node_local_data_configuration(self):
        source = (APP / "paper2_dp_secagg_deploy/client_app.py").read_text(encoding="utf-8")
        self.assertIn('("client-id", "data-root", "fold-id")', source)
        self.assertIn("secaggplus_mod, fixedclipping_mod", source)
        self.assertIn("Evaluation is disabled", source)


if __name__ == "__main__":
    unittest.main()
