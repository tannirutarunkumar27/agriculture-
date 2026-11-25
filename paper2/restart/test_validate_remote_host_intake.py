import unittest

from paper2.restart.validate_remote_host_intake import evaluate


class RemoteHostIntakeTests(unittest.TestCase):
    def test_template_is_blocked(self):
        report = evaluate({
            "status": "PENDING_USER_PROVISIONED_REMOTE_HOST",
            "ssh_host": "user@gpu-host.example", "remote_root": "/srv/paper2",
            "ssh_key_path": "C:/Users/<user>/.ssh/key", "gpu_count": 1, "workers": 1,
        })
        self.assertFalse(report["preflight_allowed"])

    def test_loopback_is_rejected(self):
        report = evaluate({
            "status": "APPROVED_FOR_PREFLIGHT",
            "ssh_host": "user@127.0.0.1", "remote_root": "/srv/paper2",
            "ssh_key_path": "C:/Users/tanne/.ssh/key", "gpu_count": 1, "workers": 1,
        })
        self.assertIn("real_nonloopback_ssh_host_required", report["errors"])

    def test_concrete_remote_host_passes_intake(self):
        report = evaluate({
            "status": "APPROVED_FOR_PREFLIGHT",
            "ssh_host": "research@gpu01.example.org", "remote_root": "/srv/paper2",
            "ssh_key_path": "C:/Users/tanne/.ssh/paper2_gpu_ed25519", "gpu_count": 2, "workers": 2,
        })
        self.assertFalse(report["preflight_allowed"])
        self.assertIn("real_nonloopback_ssh_host_required", report["errors"])

    def test_concrete_nonreserved_host_passes_structural_intake(self):
        report = evaluate({
            "status": "APPROVED_FOR_PREFLIGHT",
            "ssh_host": "research@gpu01.university.edu", "remote_root": "/srv/paper2",
            "ssh_key_path": "C:/Users/tanne/.ssh/paper2_gpu_ed25519",
            "ssh_port": 22, "gpu_count": 2, "workers": 2,
        })
        self.assertTrue(report["preflight_allowed"])

    def test_missing_private_key_blocks_executable_intake(self):
        report = evaluate({
            "status": "APPROVED_FOR_PREFLIGHT",
            "ssh_host": "research@gpu01.university.edu", "remote_root": "/srv/paper2",
            "ssh_key_path": "C:/missing/key", "gpu_count": 1, "workers": 1,
        }, key_exists=False)
        self.assertIn("ssh_private_key_file_not_found", report["errors"])


if __name__ == "__main__":
    unittest.main()
