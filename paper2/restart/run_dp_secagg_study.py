"""Identity-bound launcher for the external 20-round DP+SecAgg+ study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from paper2.restart.freeze_dp_secagg_pilot import RESULTS
from paper2.restart.freeze_dp_secagg_study import verify
from paper2.restart.run_dp_secagg_pilot import (
    active_bin_dir, command, ensure_port_available, execution_profile,
    semantic_log_audit, stop_managed_superlink,
)
from paper2.restart.validation_search import atomic_json, read, sha


def launch(package, output, execute=False, data_root=None):
    package, output = Path(package).resolve(), Path(output).resolve()
    verified = verify(package, data_root)
    profile = execution_profile(package)
    cmd = command(package, data_root)
    if not execute:
        return {"status": "DRY_RUN_NO_TRAINING", "verified": verified, "command": cmd}
    if platform.system() != "Linux":
        raise RuntimeError("Flower/Ray execution is supported only in the local WSL2/Linux profile; native Windows execution is blocked")
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Execution output must be a new Paper 2 results directory")
    port = int(profile["managed_superlink_http_port"])
    ensure_port_available(port)
    output.mkdir(parents=True); root = Path(__file__).resolve().parents[2]
    env = os.environ.copy(); env["FLWR_HOME"] = str(output / "flwr_home"); env["PYTHONPATH"] = str(root)
    env["PATH"] = str(active_bin_dir()) + os.pathsep + env.get("PATH", "")
    env["FLWR_LOCAL_SUPERLINK_HTTP_API_PORT"] = str(port)
    env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    atomic_json(output / "manifest.json", {"status": "EXECUTION_STARTED", "protocol": str(package),
                "protocol_manifest_sha256": sha(package / "manifest.json"), "verified": verified,
                "outer_test_inference": False, "raw_updates_exported": False,
                "platform": platform.platform(), "python_executable": sys.executable,
                "execution_profile": profile, "managed_superlink_http_port": port})
    cleanup = {"port": port, "matched_pids": [], "remaining_pids": [], "clean": False}
    try:
        completed = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, timeout=3600)
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        semantic = semantic_log_audit(stdout, stderr, int(read(package / "run_config.json")["num-server-rounds"]))
        status = "COMPLETE" if returncode == 0 and semantic["passed"] else (
            "FAILED_SEMANTIC_EXECUTION_NO_AUTOMATIC_RETRY" if returncode == 0 else "FAILED_NO_AUTOMATIC_RETRY"
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, returncode = exc.stdout or "", exc.stderr or "", None
        if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
        status = "TIMED_OUT_NO_AUTOMATIC_RETRY"
        semantic = semantic_log_audit(stdout, stderr, int(read(package / "run_config.json")["num-server-rounds"]))
    except KeyboardInterrupt:
        stdout, stderr, returncode = "", "Execution interrupted by operator", None
        status = "ABORTED_BY_OPERATOR_NO_AUTOMATIC_RETRY"
        semantic = semantic_log_audit(stdout, stderr, int(read(package / "run_config.json")["num-server-rounds"]))
    finally:
        cleanup = stop_managed_superlink(port)
    if not cleanup["clean"]:
        status = "FAILED_RUNTIME_CLEANUP_NO_AUTOMATIC_RETRY"
    (output / "flower_stdout.log").write_text(stdout, encoding="utf-8")
    (output / "flower_stderr.log").write_text(stderr, encoding="utf-8")
    atomic_json(output / "completion.json", {"returncode": returncode,
                "status": status,
                "outer_test_inference": False, "raw_updates_exported": False,
                "execution_platform": platform.system(), "timeout_seconds": 3600,
                "managed_superlink_cleanup": cleanup,
                "semantic_log_audit": semantic,
                "log_sha256": {name: sha(output / name) for name in ("flower_stdout.log", "flower_stderr.log")}})
    return read(output / "completion.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); print(json.dumps(launch(args.package, args.output, args.execute, args.data_root), indent=2))
