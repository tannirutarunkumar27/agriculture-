import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "paper2/docker/compose.yaml"
RUNNER = ROOT / "paper2/restart/run_docker_authenticated_dp_secagg_deployment.sh"


class DockerAuthenticatedDeploymentTests(unittest.TestCase):
    def test_declares_separate_nodes_and_local_only_claims(self):
        compose, runner = COMPOSE.read_text(encoding="utf-8"), RUNNER.read_text(encoding="utf-8")
        superlink = (ROOT / "paper2/docker/scripts/superlink.sh").read_text(encoding="utf-8")
        node = (ROOT / "paper2/docker/scripts/node.sh").read_text(encoding="utf-8")
        self.assertEqual(sum(f"  node-{index}:" in compose for index in range(7)), 7)
        self.assertIn("--enable-supernode-auth", superlink)
        self.assertIn("--isolation subprocess", node)
        self.assertIn("single_laptop_docker_bridge_network", runner)
        self.assertIn("network_deployment_gate_passed': False", runner)
        self.assertIn("privacy_claim_allowed': False", runner)
        self.assertIn("docker compose down --volumes", runner)

    def test_container_entrypoints_use_unix_line_endings(self):
        scripts = sorted((ROOT / "paper2/docker/scripts").glob("*.sh"))
        self.assertEqual(len(scripts), 4)
        for script in scripts:
            self.assertNotIn(b"\r\n", script.read_bytes(), script.name)

    def test_dockerfile_refuses_empty_copied_sources(self):
        dockerfile = (ROOT / "paper2/docker/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("test -s /opt/paper2-app/pyproject.toml", dockerfile)
        self.assertIn("test -s /usr/local/lib/paper2-docker/bootstrap.sh", dockerfile)


if __name__ == "__main__":
    unittest.main()
