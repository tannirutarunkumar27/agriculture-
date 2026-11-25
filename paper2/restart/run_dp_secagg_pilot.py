"""Identity-bound launcher for the external DP+SecAgg+ pilot.

The local sandbox may block Flower's loopback preflight. Dry-run is the default
and never launches Flower or creates an execution output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import sys
import time

from paper2.restart.freeze_dp_secagg_pilot import RESULTS, verify
from paper2.restart.paper2_completion_audit import latest_result_file
from paper2.restart.validation_search import atomic_json, read, sha


def resolved_run_config(package: Path, data_root: Path | None = None):
    run = read(package / "run_config.json")
    if data_root is not None:
        data_root = Path(data_root).resolve()
        manifest = read(package / "manifest.json")
        if sha(data_root / "manifest.json") != manifest["data_manifest_sha256"]:
            raise ValueError("Relocated data manifest differs from frozen protocol")
        run["data-root"] = str(data_root)
    return run


def active_bin_dir() -> Path:
    """Return the active environment's script directory without resolving symlinks."""
    return Path(sys.executable).absolute().parent


def flower_executable() -> str:
    """Resolve Flower from the active interpreter without crossing OSes."""
    root = Path(__file__).resolve().parents[2]
    script_name = "flwr.exe" if platform.system() == "Windows" else "flwr"
    # Do not call ``resolve`` here.  On Linux a virtualenv's ``python`` is
    # commonly a symlink into /usr/bin; resolving it would lose the virtualenv
    # and select a global (or missing) Flower entry point.
    candidates = [active_bin_dir() / script_name]
    discovered = shutil.which("flwr")
    if discovered is not None:
        candidates.append(Path(discovered))
    candidates.append(
        root / (".venv/Scripts/flwr.exe" if platform.system() == "Windows" else ".venv/bin/flwr")
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Flower executable {script_name!r} is absent from the active {platform.system()} environment"
    )


def execution_profile(package: Path) -> dict:
    profile = read(Path(package) / "execution_profile.json")
    expected = {
        "runtime": "flower_managed_local_simulation",
        "num_supernodes": 7,
        "client_resources_num_cpus": 1,
        "client_resources_num_gpus": 0.0,
        "init_args_num_cpus": 10,
        "init_args_num_gpus": 0,
        "managed_superlink_http_port": 39181,
        "runtime_dependency_installation": False,
    }
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen Flower execution profile differs from the audited local CPU profile")
    return profile


def federation_config(profile: dict) -> str:
    return " ".join((
        f"num-supernodes={profile['num_supernodes']}",
        f"client-resources-num-cpus={profile['client_resources_num_cpus']}",
        f"client-resources-num-gpus={profile['client_resources_num_gpus']}",
        f"init-args-num-cpus={profile['init_args_num_cpus']}",
        f"init-args-num-gpus={profile['init_args_num_gpus']}",
    ))


def command(package: Path, data_root: Path | None = None):
    run = resolved_run_config(package, data_root)
    profile = execution_profile(package)
    overrides = " ".join(f"{key}={json.dumps(value)}" for key, value in run.items())
    root = Path(__file__).resolve().parents[2]
    return [flower_executable(), "run", str(root / "paper2/secagg_dp_pilot"), "--stream",
            "--run-config", overrides, "--federation-config", federation_config(profile)]


def ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Frozen Flower local port {port} is already in use") from exc


def _managed_superlink_pids(port: int, proc: Path = Path("/proc")) -> list[int]:
    """Find only Flower SuperLinks bound to this frozen local HTTP port."""
    matches = []
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            text_args = [value.decode(errors="replace") for value in args if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(value).name == "flower-superlink" for value in text_args):
            continue
        if any(value == "--port" and index + 1 < len(text_args) and text_args[index + 1] == str(port)
               for index, value in enumerate(text_args)):
            matches.append(int(entry.name))
    return sorted(matches)


def stop_managed_superlink(port: int) -> dict:
    """Stop the exact managed SuperLink Flower documents as persistent."""
    initial = _managed_superlink_pids(port)
    for pid in initial:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    remaining = _managed_superlink_pids(port)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _managed_superlink_pids(port)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    final = _managed_superlink_pids(port)
    return {"port": port, "matched_pids": initial, "remaining_pids": final, "clean": not final}


def semantic_log_audit(stdout: str, stderr: str, rounds: int) -> dict:
    """Reject Flower's zero exit code when the workflow halted internally."""
    combined = stdout + "\n" + stderr
    negative_markers = (
        "ERROR :", "Traceback (most recent call last)", "ClientAppException",
        "Secure aggregation halted.", "Insufficient available nodes.",
    )
    observed_errors = [marker for marker in negative_markers if marker in combined]
    commenced = combined.count("Secure aggregation commencing.")
    completed = combined.count("Secure aggregation completed.")
    passed = not observed_errors and commenced == rounds and completed == rounds
    return {
        "passed": passed,
        "expected_rounds": rounds,
        "secure_aggregation_commenced": commenced,
        "secure_aggregation_completed": completed,
        "negative_markers": observed_errors,
    }


def launch(package, output, execute=False, data_root=None):
    package, output = Path(package).resolve(), Path(output).resolve()
    verified = verify(package, data_root)
    profile = execution_profile(package)
    if not execute:
        return {"status": "DRY_RUN_NO_TRAINING", "verified": verified, "command": command(package, data_root)}
    if platform.system() != "Linux":
        raise RuntimeError("Flower/Ray execution is supported only in the local WSL2/Linux profile; native Windows execution is blocked")
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Execution output must be a new Paper 2 results directory")
    port = int(profile["managed_superlink_http_port"])
    ensure_port_available(port)
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["FLWR_HOME"] = str(output / "flwr_home")
    env["PYTHONPATH"] = str(root)
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
        completed = subprocess.run(command(package, data_root), cwd=root, env=env, text=True, capture_output=True, timeout=900)
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
        "execution_platform": platform.system(), "timeout_seconds": 900,
        "managed_superlink_cleanup": cleanup,
        "semantic_log_audit": semantic,
        "log_sha256": {name: sha(output / name) for name in ("flower_stdout.log", "flower_stderr.log")}})
    return read(output / "completion.json")


def latest_protocol() -> Path:
    """Resolve the current immutable pilot package instead of a stale version."""
    return latest_result_file("restart_dp_secagg_pilot_protocol", "manifest.json").parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=latest_protocol())
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(); print(json.dumps(launch(args.package, args.output, args.execute, args.data_root), indent=2))
