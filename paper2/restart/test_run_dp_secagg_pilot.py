"""Dry-run checks for the external DP+SecAgg launcher."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from paper2.restart import run_dp_secagg_pilot as runner


PROFILE = {
    "runtime": "flower_managed_local_simulation",
    "num_supernodes": 7,
    "client_resources_num_cpus": 1,
    "client_resources_num_gpus": 0.0,
    "init_args_num_cpus": 10,
    "init_args_num_gpus": 0,
    "managed_superlink_http_port": 39181,
    "runtime_dependency_installation": False,
}


class DpSecAggLauncherTests(unittest.TestCase):
    def test_command_carries_only_declared_run_config(self):
        with patch.object(runner, "read", side_effect=[
            {"noise-multiplier": 8.0, "outer-test-evaluation": False}, PROFILE,
        ]):
            result = runner.command(runner.Path("fixture"))
        self.assertIn("flwr", Path(result[0]).name.lower())
        self.assertIn("outer-test-evaluation=false", result[-3])
        self.assertEqual(result[-2], "--federation-config")
        self.assertIn("num-supernodes=7", result[-1])
        self.assertIn("client-resources-num-gpus=0.0", result[-1])

    def test_linux_uses_flower_from_active_interpreter_environment(self):
        with TemporaryDirectory() as temporary:
            bin_dir = Path(temporary) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            flower = bin_dir / "flwr"
            python.touch()
            flower.touch()
            with patch.object(runner.sys, "executable", str(python)), \
                 patch.object(runner.platform, "system", return_value="Linux"), \
                 patch.object(runner.shutil, "which", return_value=None):
                result = runner.flower_executable()
        self.assertEqual(result, str(flower.resolve()))
        self.assertFalse(result.lower().endswith(".exe"))

    def test_active_bin_dir_preserves_virtual_environment_path(self):
        with TemporaryDirectory() as temporary:
            python = Path(temporary) / "bin" / "python"
            with patch.object(runner.sys, "executable", str(python)):
                result = runner.active_bin_dir()
        self.assertEqual(result, python.absolute().parent)

    def test_superlink_cleanup_discovery_is_scoped_to_frozen_port(self):
        with TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, port in ((101, 39181), (102, 39091)):
                folder = proc / str(pid)
                folder.mkdir()
                (folder / "cmdline").write_bytes(
                    b"python\0/home/user/bin/flower-superlink\0--port\0" + str(port).encode() + b"\0"
                )
            unrelated = proc / "103"
            unrelated.mkdir()
            (unrelated / "cmdline").write_bytes(b"python\0research.py\0--port\0" + b"39181\0")
            result = runner._managed_superlink_pids(39181, proc)
        self.assertEqual(result, [101])

    def test_semantic_audit_rejects_flower_zero_exit_halt(self):
        failed = runner.semantic_log_audit(
            "Secure aggregation commencing.\nERROR : client failure\nSecure aggregation halted.",
            "", 1,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["secure_aggregation_completed"], 0)
        passed = runner.semantic_log_audit(
            "Secure aggregation commencing.\nSecure aggregation completed.", "", 1,
        )
        self.assertTrue(passed["passed"])

    def test_relocated_data_must_match_frozen_hash(self):
        with patch.object(runner, "read", side_effect=[{"data-root": "old"}, {"data_manifest_sha256": "expected"}]), \
             patch.object(runner, "sha", return_value="different"):
            with self.assertRaisesRegex(ValueError, "Relocated data manifest"):
                runner.resolved_run_config(Path("package"), Path("relocated"))

    def test_dry_run_never_creates_output(self):
        with patch.object(runner, "verify", return_value={"status": "PASS"}), \
             patch.object(runner, "execution_profile", return_value=PROFILE), \
             patch.object(runner, "command", return_value=["flwr"]):
            result = runner.launch("package", "output", execute=False)
        self.assertEqual(result["status"], "DRY_RUN_NO_TRAINING")

    def test_native_windows_execution_is_blocked_before_output(self):
        with patch.object(runner, "verify", return_value={"status": "PASS"}), \
             patch.object(runner, "execution_profile", return_value=PROFILE), \
             patch.object(runner.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "WSL2/Linux"):
                runner.launch("package", "output", execute=True)


if __name__ == "__main__":
    unittest.main()
